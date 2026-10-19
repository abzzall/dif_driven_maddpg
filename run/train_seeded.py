"""Parameterized canonical trainer for revision multi-seed runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_envs.diff_driven.gym_env.centered_paralelenv.env import (  # noqa: E402
    DiffDriveParallelEnvDone,
)
from rl.maddpg import (  # noqa: E402
    IDDPGWithoutS,
    MADDPGSharedActorCritic,
    MADDPGSharedActorCriticIndependentObs,
)
from rl.mappo import MAPPO  # noqa: E402
from rl import convergence as conv  # noqa: E402

ALGORITHMS = {
    "iddpg_without_s": IDDPGWithoutS,        # canonical: shared decentralized critic (obs_i, action_i)
    "maddpg": MADDPGSharedActorCritic,       # Lowe-2017-style: joint-action centralized critic
    "maddpg_obs": MADDPGSharedActorCriticIndependentObs,  # CTDE: per-agent reward + concat-obs critic (fixes A/B/C)
    "mappo": MAPPO,                          # on-policy CTDE baseline (R2-round2 #3); own loop, no replay
}
from tools.offload_artifacts import ensure_target_root, offload_run_dir  # noqa: E402


SCALES = {
    "full": [1.0, 1.0, 0.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
    "ablation": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
    "nocoll": [1.0, 1.0, 0.0, 10.0, 0.0, 0.0, 1.0, 1.0, 1.0],
}

CSV_SUFFIX = {
    "full": "",
    "ablation": "_ablation",
    "nocoll": "_nocoll",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one canonical IDDPGWithoutS training configuration."
    )
    parser.add_argument("--n", type=int, required=True,
                        help="Number of agents (= number of landmarks).")
    parser.add_argument(
        "--env_size", type=float, default=None,
        help="Arena side length. Default: the config value (20), regardless of "
             "team size -- which means agent density rises with n. Pass "
             "--scale_env to hold density constant instead.",
    )
    parser.add_argument(
        "--scale_env", action="store_true",
        help="Scale the arena as sqrt(n/5) x 20 so area per agent matches the "
             "n=5 setting. Separates 'more agents' from 'more crowded': with a "
             "fixed arena the two are confounded.",
    )
    parser.add_argument(
        "--algorithm",
        choices=sorted(ALGORITHMS),
        default="iddpg_without_s",
        help="Which algorithm to train. 'iddpg_without_s' (default) is the "
             "canonical shared decentralized critic; 'maddpg' is the "
             "Lowe-2017-style joint-action centralized-critic baseline.",
    )
    parser.add_argument("--mode", choices=sorted(SCALES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--v_ang_max",
        choices=["pi9", "pi6", "pi2"],
        default="pi2",
        help="Angular velocity cap. Default pi2, the value used in the published experiments.",
    )
    parser.add_argument(
        "--use_orbit_restart",
        action="store_true",
        help="When training via main_loop, enable orbit-basin restart: at "
             "episode 250 of each attempt, if SR=0 over last 100 episodes "
             "with orbit signature (mean comp4>5 AND mean comp8<-800), "
             "reset actor/critic/targets/buffer/counter and start over. "
             "Max 3 restart events; 4th attempt runs the full episode "
             "budget without further check.",
    )
    parser.add_argument(
        "--orbit_restart_check_ep",
        type=int,
        default=250,
        help="First attempt-local episode at which the orbit-restart check "
             "runs; re-checked every 50 episodes up to the budget. Default 250.",
    )
    parser.add_argument(
        "--orbit_restart_resume_check_ep",
        type=int,
        default=500,
        help="One-time startup restart check episode for an already-far-along "
             "resumed attempt. Default 500.",
    )
    parser.add_argument(
        "--orbit_restart_max",
        type=int,
        default=3,
        help="Max restart events; the (max+1)-th attempt runs the full episode "
             "budget with no early restart. Default 3.",
    )
    parser.add_argument(
        "--orbit_restart_sr_threshold",
        type=float,
        default=0.01,
        help="Restart if last-100 success rate <= this AND coverage < "
             "--orbit_restart_cov_threshold. Default 0.01 (no-traction rule).",
    )
    parser.add_argument(
        "--orbit_restart_cov_threshold",
        type=float,
        default=0.40,
        help="Coverage guard for the restart rule (see --orbit_restart_sr_threshold). "
             "Default 0.40.",
    )
    parser.add_argument(
        "--use_offline_replay",
        action="store_true",
        help="Use MADDPGBase.main_loop with offline_replay_success (HER-style "
             "goal relabeling) instead of train_loop. Addresses sparse-reward "
             "bootstrap failure for higher N. Disclosed as revision-stage enhancement.",
    )
    parser.add_argument(
        "--artifact_root",
        type=Path,
        default=None,
        help="Source root used to preserve relative paths during episode offload.",
    )
    parser.add_argument(
        "--offload_root",
        type=Path,
        default=None,
        help="Mirror this run directory here after each saved episode checkpoint.",
    )
    parser.add_argument(
        "--disable_episode_offload",
        action="store_true",
        help="Disable per-episode artifact mirroring.",
    )
    parser.add_argument(
        "--offload_mode",
        choices=["end", "every", "every_k"],
        default="end",
        help="When to mirror artifacts to --offload_root. "
             "'end' (default): only once, on the final episode of the run — "
             "one offload per run, no per-episode USB cost. "
             "'every': mirror after every episode (old behaviour, slow on USB). "
             "'every_k': mirror every --offload_every episodes.",
    )
    parser.add_argument(
        "--offload_every",
        type=int,
        default=10,
        help="Used only when --offload_mode=every_k. Default 10.",
    )
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="Stop on a plateau of the rolling full-team success rate instead "
             "of at --episodes. Makes the episode count a measured outcome, so "
             "on- and off-policy methods are not budgeted by a constant that "
             "favours one family. Records stop_reason/episodes_used in meta.json. "
             "Applies to mappo and to the DDPG-family loops alike, so the "
             "families share one budget rule.",
    )
    parser.add_argument("--conv_window", type=int, default=conv.DEFAULT_WINDOW,
                        help=f"Rolling SR window. Default {conv.DEFAULT_WINDOW}.")
    parser.add_argument("--conv_min_delta", type=float, default=conv.DEFAULT_MIN_DELTA,
                        help=f"Min SR improvement (pp) counted as progress. "
                             f"Default {conv.DEFAULT_MIN_DELTA}.")
    parser.add_argument("--conv_patience", type=int, default=conv.DEFAULT_PATIENCE,
                        help=f"Episodes without a new best rolling SR before "
                             f"stopping. Default {conv.DEFAULT_PATIENCE}.")
    parser.add_argument("--conv_min_episodes", type=int, default=conv.DEFAULT_MIN_EPISODES,
                        help=f"Never stop before this many episodes. "
                             f"Default {conv.DEFAULT_MIN_EPISODES}.")
    parser.add_argument("--conv_max_episodes", type=int, default=conv.DEFAULT_MAX_EPISODES,
                        help=f"Hard episode cap. Default {conv.DEFAULT_MAX_EPISODES}.")
    parser.add_argument("--conv_max_hours", type=float, default=conv.DEFAULT_MAX_HOURS,
                        help=f"Hard wall-clock cap per run. Default {conv.DEFAULT_MAX_HOURS}.")
    parser.add_argument(
        "--no_step_csv",
        action="store_true",
        help="Disable per-step CSV logging (~25%% of rollout wall-clock). "
             "Per-episode summaries are unaffected.",
    )
    parser.add_argument(
        "--reward",
        choices=["legacy", "new"],
        default="legacy",
        help="'legacy': the inherited 9-component reward on "
             "DiffDriveParallelEnvDone. 'new': the redesigned 6-component "
             "reward on GraphCoverageEnv with flat observations (no done-agent "
             "repulsion, team-completion bonus, rescaled so terminal events "
             "dominate). Forces --no_step_csv, which assumes 9 components.",
    )
    parser.add_argument(
        "--mlp_hidden",
        type=int,
        default=None,
        help="Hidden width for the flat-observation MLP actor/critic. Use to "
             "match the graph encoder's parameter count so an observation-form "
             "comparison is not confounded by model capacity. 512 gives the "
             "actor 1.03x the graph actor's parameters. mappo only.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override the MAPPO actor and critic Adam learning rate. The "
             "canonical submitted setting is 3e-4. MAPPO only.",
    )
    parser.add_argument(
        "--obs",
        choices=["flat", "graph"],
        default="flat",
        help="Observation form when --reward new. 'flat': fixed index-order "
             "concatenation through the MLP networks. 'graph': typed node sets "
             "through the permutation-invariant attention encoder, which also "
             "makes the policy valid at any team size.",
    )
    parser.add_argument(
        "--entity_encoder",
        choices=["attention", "deepsets"],
        default="attention",
        help="Entity encoder for --reward new --obs graph. 'deepsets' uses "
             "typed token transforms plus masked mean pooling and a "
             "parameter-matched 160-wide embedding.",
    )
    parser.add_argument(
        "--no_entity_mask",
        action="store_true",
        help="Attention ablation: retain completed robots and covered "
             "landmarks as attention keys while keeping their status features.",
    )
    parser.add_argument(
        "--require_gpu",
        action="store_true",
        help="Refuse to start unless CUDA is available; never fall back to CPU.",
    )
    parser.add_argument(
        "--done_as_obstacles",
        dest="done_as_obstacles",
        action="store_true",
        default=True,
        help="Legacy reward only: treat finished agents as static obstacles in "
             "reward component 5, as in the published Robotics/JRC experiments "
             "(default). A done agent parks on the landmark it covered, so this "
             "places a repulsion field on every solved goal.",
    )
    parser.add_argument(
        "--no_done_as_obstacles",
        dest="done_as_obstacles",
        action="store_false",
        help="Legacy reward only: drop the done-agent repulsion term. Used for "
             "arm A of the entity-attention MAPPO study.",
    )
    parser.add_argument(
        "--plot_every",
        type=int,
        default=50,
        help="Write learning-curve / loss / trajectory plots every N episodes "
             "(0 disables). The MADDPG loops plotted every episode, producing "
             "~700 PNGs per run. Default 50. mappo only.",
    )
    parser.add_argument(
        "--max_shortfall_plots",
        type=int,
        default=20,
        help="Cap on trajectory plots captured for episodes that end exactly "
             "one agent short (the dominant failure mode). 0 disables. mappo only.",
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Bit-deterministic CUDA. Required for replayability across runs/resumes.
    # CUBLAS_WORKSPACE_CONFIG must be set BEFORE the first CUDA op (we set it
    # here on import-time call before env construction). ~10-30% perf cost.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:
        print(f"use_deterministic_algorithms not available: {exc!r}", flush=True)


def rename_rewards_csv(num_agents: int, mode: str) -> str | None:
    """Move rewards.csv -> result{N}{suffix}.csv at end of run.

    Resume-safe: if a target already exists from a prior completed run in the
    same out_dir, merge by episode_id (target rows for episodes also present
    in source are dropped in favor of source rows; new source episodes are
    appended). This prevents truncating a full result CSV when a run is
    relaunched and only produces a few additional episodes.
    """
    source = Path("rewards.csv")
    target = Path(f"result{num_agents}{CSV_SUFFIX[mode]}.csv")

    if not source.exists():
        return str(target) if target.exists() else None

    if target.exists():
        import pandas as pd
        old = pd.read_csv(target)
        new = pd.read_csv(source)
        if "episode_id" in old.columns and "episode_id" in new.columns:
            new_episodes = set(new["episode_id"].unique())
            old_keep = old[~old["episode_id"].isin(new_episodes)]
            combined = pd.concat([old_keep, new], ignore_index=True)
            combined = combined.sort_values(["episode_id", "timestep"]
                                            if "timestep" in combined.columns
                                            else ["episode_id"]).reset_index(drop=True)
        else:
            combined = pd.concat([old, new], ignore_index=True)
        combined.to_csv(target, index=False)
        source.unlink()
    else:
        source.rename(target)
    return str(target)


def acquire_run_lock(out_dir: Path) -> Path | None:
    """Write .run_all.lock so concurrent manual offloads skip this dir.

    Honours an existing lock only if its pid is still alive; stale locks
    (process gone) are overwritten. Returns the lock path on success,
    None on failure (in which case the caller proceeds without a lock —
    manual offload may then race with us, but training will still finish).
    """
    import socket
    lock_path = out_dir / ".run_all.lock"
    if lock_path.exists():
        try:
            prior = json.loads(lock_path.read_text())
            prior_pid = int(prior.get("pid") or prior.get("owner_pid") or -1)
            if prior_pid > 0 and prior_pid != os.getpid():
                try:
                    os.kill(prior_pid, 0)
                    # Another live process owns this run dir — bail.
                    print(
                        f"WARN: .run_all.lock held by live pid={prior_pid}; "
                        f"another trainer is already in {out_dir}",
                        flush=True,
                    )
                    return None
                except ProcessLookupError:
                    pass  # stale, overwrite
                except PermissionError:
                    print(
                        f"WARN: .run_all.lock pid={prior_pid} alive but not ours; "
                        "proceeding cautiously",
                        flush=True,
                    )
                    return None
        except Exception:
            pass  # malformed, overwrite
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": "run/train_seeded.py",
    }
    try:
        lock_path.write_text(json.dumps(payload))
        return lock_path
    except OSError as exc:
        print(f"WARN: could not write .run_all.lock: {exc!r}", flush=True)
        return None


def release_run_lock(lock_path: Path | None) -> None:
    if lock_path is None:
        return
    try:
        # Only remove if we still own it.
        payload = json.loads(lock_path.read_text())
        if int(payload.get("pid") or -1) == os.getpid():
            lock_path.unlink()
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"WARN: could not release .run_all.lock: {exc!r}", flush=True)


def update_meta(meta_path: Path, extra: dict) -> None:
    if meta_path.exists():
        with meta_path.open("r") as f:
            meta = json.load(f)
    else:
        meta = {}
    meta.update(extra)
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


def parse_v_ang_max(value: str) -> torch.Tensor:
    if value == "pi9":
        return torch.pi / 9
    if value == "pi6":
        return torch.pi / 6
    if value == "pi2":
        return torch.pi / 2
    raise ValueError(f"Unsupported --v_ang_max: {value}")


def make_episode_offload_callback(
    *,
    out_dir: Path,
    artifact_root: Path | None,
    offload_root: Path | None,
    disabled: bool,
    mode: str = "end",
    every: int = 10,
) -> Callable[[int, bool], None] | None:
    if disabled or offload_root is None:
        return None

    source_root = (artifact_root or out_dir.parent.parent).expanduser().resolve()
    target_root = ensure_target_root(offload_root, dry_run=False)
    if target_root is None:
        print(f"Episode offload disabled: target root unavailable: {offload_root}", flush=True)
        return None

    every = max(1, int(every))

    def should_run(episodes_completed: int, finished: bool) -> bool:
        if finished:
            return True
        if mode == "end":
            return False
        if mode == "every_k":
            return (episodes_completed % every) == 0
        return True  # mode == "every"

    def callback(episodes_completed: int, finished: bool) -> None:
        if not should_run(episodes_completed, finished):
            return
        try:
            result = offload_run_dir(
                out_dir,
                source_root=source_root,
                target_root=target_root,
                keep_local_result_csv=True,
                prune_incomplete_snapshots=False,
                include_running=True,
                dry_run=False,
                progress=False,
            )
            print(
                "Episode offload "
                f"episode={episodes_completed} finished={finished} "
                f"status={result.status} copied={result.copied_files} "
                f"removed={result.removed_files} target={result.target}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"Episode offload failed at episode={episodes_completed}: {exc!r}",
                flush=True,
            )

    return callback


def read_restart_count(out_dir: Path) -> int:
    """Read restart_count from a prior orbit-restart state, if any."""
    try:
        with open(out_dir / "restart_state.json") as f:
            return int(json.load(f).get("restart_count") or 0)
    except Exception:
        return 0


def main() -> None:
    args = parse_args()
    if args.require_gpu and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for this run, but torch.cuda.is_available() is false. "
            "Resolve the NVIDIA driver/library state before launching training."
        )
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # On a restart relaunch (this run already triggered >=1 orbit restart) we do
    # NOT re-seed: re-seeding would reproduce attempt 1's init exactly. Skipping
    # set_seeds lets the fresh rebuild diverge — same config, different stream.
    restart_count = read_restart_count(out_dir)
    if restart_count > 0:
        print(
            f"Orbit-restart relaunch (restart_count={restart_count}): "
            "skipping set_seeds so this attempt diverges from the seeded attempt.",
            flush=True,
        )
    else:
        set_seeds(args.seed)

    start_time = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    os.chdir(out_dir)

    lock_path = acquire_run_lock(out_dir)
    if lock_path is None:
        # Another live trainer owns this dir — refuse to step on it.
        print(f"ERROR: {out_dir} already locked by another trainer; aborting.", flush=True)
        sys.exit(2)

    v_ang_max = parse_v_ang_max(args.v_ang_max)
    import config as _cfg
    if args.env_size is not None:
        env_size = float(args.env_size)
    elif args.scale_env:
        # Constant area per agent relative to the n=5 reference.
        env_size = float(_cfg.env_size) * (args.n / 5.0) ** 0.5
    else:
        env_size = float(_cfg.env_size)
    print(f"env_size={env_size:.2f} for n={args.n} "
          f"(area/agent={env_size**2/args.n:.1f})", flush=True)
    if args.reward == "new":
        # Redesigned reward (6 components) with flat observations: the control
        # arm that isolates the reward change from the graph encoder.
        from custom_envs.diff_driven.graph_coverage_env import (  # noqa: E402
            GraphCoverageEnv, DEFAULT_SCALES,
        )
        env = GraphCoverageEnv(
            num_agents=args.n,
            num_obstacles=0,
            v_ang_max=v_ang_max,
            env_size=env_size,
            flat_observations=(args.obs == "flat"),
        )
        env.entity_encoder = args.entity_encoder
        env.use_entity_mask = not args.no_entity_mask
        reward_scales = list(DEFAULT_SCALES)
    else:
        env = DiffDriveParallelEnvDone(
            num_agents=args.n,
            num_obstacles=0,
            v_ang_max=v_ang_max,
            done_as_obstacles=args.done_as_obstacles,
        )
        reward_scales = SCALES[args.mode]
    algo_cls = ALGORITHMS[args.algorithm]
    algo_kwargs = dict(batch_size=128, replay_buffer_size=50000)
    if args.algorithm == "mappo" and args.mlp_hidden is not None:
        algo_kwargs["mlp_hidden"] = args.mlp_hidden
    if args.learning_rate is not None:
        if args.algorithm != "mappo":
            raise SystemExit("--learning_rate is currently supported only for mappo")
        if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise SystemExit("--learning_rate must be a finite positive value")
        algo_kwargs["learning_rate"] = args.learning_rate
    maddpg = algo_cls(env, reward_scales=reward_scales, **algo_kwargs)

    meta_extra = {
        "n": args.n,
        "mode": args.mode,
        "seed": args.seed,
        "algorithm": args.algorithm,
        "episodes_requested": args.episodes,
        "v_ang_max": args.v_ang_max,
        "v_ang_max_float": float(v_ang_max),
        "use_offline_replay": bool(args.use_offline_replay),
        "use_orbit_restart": bool(args.use_orbit_restart),
        "training_loop": "main_loop" if args.use_offline_replay else "train_loop",
        "out_dir": str(out_dir),
        "command_start_iso": start_iso,
        "budget_protocol": "convergence" if args.convergence else "episode_cap",
        "log_step_csv": not args.no_step_csv,
        "reward_design": args.reward,
        "observation": args.obs if args.reward == "new" else "flat",
        "entity_encoder": args.entity_encoder if args.obs == "graph" else None,
        "use_entity_mask": (
            not args.no_entity_mask if args.obs == "graph" else None
        ),
        "reward_scales_used": list(reward_scales),
        "done_as_obstacles": bool(args.done_as_obstacles) if args.reward == "legacy" else False,
        "mlp_hidden": args.mlp_hidden,
        "learning_rate": args.learning_rate if args.learning_rate is not None else 3e-4,
        "env_size": env_size,
        "area_per_agent": env_size ** 2 / args.n,
    }
    post_episode_callback = make_episode_offload_callback(
        out_dir=out_dir,
        artifact_root=args.artifact_root,
        offload_root=args.offload_root,
        disabled=args.disable_episode_offload,
        mode=args.offload_mode,
        every=args.offload_every,
    )

    if args.no_step_csv:
        if not hasattr(maddpg, "log_step_csv"):
            raise SystemExit(
                f"--no_step_csv is not supported by --algorithm {args.algorithm} "
                "(only mappo gates per-step CSV logging)."
            )
        maddpg.log_step_csv = False

    if hasattr(maddpg, "plot_every"):
        maddpg.plot_every = args.plot_every
        maddpg.max_shortfall_plots = args.max_shortfall_plots
        maddpg.plot_shortfall = args.max_shortfall_plots > 0

    monitor = None
    if args.convergence:
        monitor = conv.ConvergenceMonitor(
            window=args.conv_window,
            min_delta=args.conv_min_delta,
            patience=args.conv_patience,
            min_episodes=args.conv_min_episodes,
            max_episodes=args.conv_max_episodes,
            max_hours=args.conv_max_hours,
        )

    try:
        loop_fn = maddpg.main_loop if args.use_offline_replay else maddpg.train_loop
        loop_kwargs = dict(
            start_training_after=500,
            train_each=100,
            patience=256,
            min_episodes_before_early_stop=10000,
            score_avg_window=256,
            max_steps=500,
            n_games=args.episodes,
            meta_extra=meta_extra,
            post_episode_callback=post_episode_callback,
        )
        if monitor is not None:
            loop_kwargs["convergence"] = monitor
            # min_episodes_before_early_stop stays at its huge default so the
            # loops' own score-based early stop never fires: the convergence
            # monitor is the single stopping rule, identical across families.
        if args.use_orbit_restart:
            if not args.use_offline_replay:
                raise SystemExit(
                    "--use_orbit_restart requires --use_offline_replay "
                    "(orbit restart is only implemented in main_loop)."
                )
            loop_kwargs["orbit_restart"] = True
            loop_kwargs["orbit_restart_check_ep"] = args.orbit_restart_check_ep
            loop_kwargs["orbit_restart_resume_check_ep"] = args.orbit_restart_resume_check_ep
            loop_kwargs["orbit_restart_max"] = args.orbit_restart_max
            loop_kwargs["orbit_restart_sr_threshold"] = args.orbit_restart_sr_threshold
            loop_kwargs["orbit_restart_cov_threshold"] = args.orbit_restart_cov_threshold
        loop_fn(**loop_kwargs)
    except BaseException as exc:
        update_meta(
            Path("meta.json"),
            {
                **meta_extra,
                "command_end_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "command_wall_seconds": time.time() - start_time,
                "launcher_status": "failed",
                "launcher_error": repr(exc),
            },
        )
        raise
    else:
        renamed_csv = rename_rewards_csv(args.n, args.mode)
        update_meta(
            Path("meta.json"),
            {
                **meta_extra,
                "command_end_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "command_wall_seconds": time.time() - start_time,
                "launcher_status": "finished",
                "result_csv": renamed_csv,
            },
        )
    finally:
        release_run_lock(lock_path)


if __name__ == "__main__":
    main()
