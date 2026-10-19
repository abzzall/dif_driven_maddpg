#!/usr/bin/env python3
"""Frozen-policy held-out evaluation, driven entirely by each run's meta.json.

The manuscript's primary endpoint. Replaces training-window statistics, which are
model-selection quantities computed on the same stream used for stopping decisions.

Why this exists rather than extending run/eval_policy.py: that script hardcodes
``DiffDriveParallelEnvDone``, loads only ``GaussianActor``, restricts ``--n`` to
4/5/6, defaults ``v_ang_max`` to pi/9 rather than the pi/2 used throughout this
study, and indexes the legacy nine-component reward. It cannot evaluate the
entity-structured arm at all, and its permissive defaults would silently
evaluate a *different* environment from the one a checkpoint was trained in.

Protocol (fixed before any numbers were produced):
  * checkpoint  : the final saved checkpoint of each run; evaluation results
                  never inform checkpoint selection
  * scenarios   : a common bank of episode seeds, identical across every arm, so
                  all policies face the same initial conditions
  * actions     : deterministic (Gaussian mean) by default; ``--stochastic``
                  samples instead, reported only as supplementary
  * unit        : the trained seed. Evaluation episodes are repeated measures of
                  one policy, never independent replicates.

Metrics per episode: full-team success, fractional coverage, steps taken,
horizon failure, and path length. Completion time is meaningful only for
successes; failures are right-censored at the horizon. Path length is *not* a
time-to-event variable and is therefore never censored -- it is reported over
all episodes and separately over successes.
"""

from __future__ import annotations

import argparse
import os
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config as _cfg  # noqa: E402
from custom_envs.diff_driven.gym_env.centered_paralelenv.env import (  # noqa: E402
    DiffDriveParallelEnvDone,
)
from custom_envs.diff_driven.graph_coverage_env import GraphCoverageEnv  # noqa: E402
from rl.mappo import MAPPO  # noqa: E402


def build_from_meta(run_dir: Path):
    """Reconstruct the exact env + policy a checkpoint was trained in."""
    meta = json.loads((run_dir / "meta.json").read_text())

    n = int(meta.get("n") or meta.get("num_agents"))
    v_ang = float(meta.get("v_ang_max_float", math.pi / 2))
    # Older runs predate the env_size field; they used the config default.
    env_size = meta.get("env_size")
    env_size = float(env_size) if env_size is not None else float(_cfg.env_size)
    reward_design = meta.get("reward_design", "legacy")
    observation = meta.get("observation", "flat")
    mlp_hidden = meta.get("mlp_hidden")
    scales = meta.get("reward_scales_used")

    if reward_design == "new":
        env = GraphCoverageEnv(
            num_agents=n, num_obstacles=0, v_ang_max=v_ang, env_size=env_size,
            flat_observations=(observation != "graph"),
        )
        # New entity-encoder ablations are reconstructed from metadata.  The
        # defaults preserve compatibility with runs created before these keys
        # existed (attention with completed-entity masking enabled).
        if observation == "graph":
            env.entity_encoder = meta.get("entity_encoder") or "attention"
            env.use_entity_mask = bool(meta.get("use_entity_mask", True))
    else:
        env = DiffDriveParallelEnvDone(
            num_agents=n, num_obstacles=0, v_ang_max=v_ang, env_size=env_size,
            done_as_obstacles=bool(meta.get("done_as_obstacles", False)),
        )
    if scales is None:
        raise SystemExit(f"{run_dir}: meta.json lacks reward_scales_used")

    kwargs = {}
    if mlp_hidden is not None:
        kwargs["mlp_hidden"] = int(mlp_hidden)
    algo = MAPPO(env, list(scales), **kwargs)

    names = ("gnn_actor.pth", "gnn_critic.pth") if observation == "graph" else \
            ("mappo_actor.pth", "mappo_critic.pth")
    required = [run_dir / name for name in names]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: missing checkpoint files: {missing}")

    # Checkpoint filenames differ by architecture; MAPPO picks them by graph_mode.
    import os
    cwd = os.getcwd()
    try:
        os.chdir(run_dir)          # checkpoints are loaded by bare filename
        algo.load_checkpoint()
    finally:
        os.chdir(cwd)
    return env, algo, meta


@torch.no_grad()
def run_episode(env, algo, seed: int, max_steps: int, stochastic: bool):
    _, obs = env.reset_tensor(seed=seed)
    N = env.num_agents
    prev = env.agent_pos.clone()
    path = torch.zeros(N, device=env.device)
    steps = 0
    for _ in range(max_steps):
        # Call the actor directly rather than MAPPO.choose_actions: the latter
        # coerces its argument with torch.tensor(), which cannot accept the
        # dict of node tensors used by the entity-structured arm. Both actor
        # classes expose the same act(obs, deterministic=...) interface.
        raw, _ = algo.actor.act(obs, deterministic=not stochastic)
        raw = raw.clamp(-algo.max_action, algo.max_action)
        _, obs, _, dones = env.step_tensor(raw)
        path += (env.agent_pos - prev).norm(dim=1)
        prev = env.agent_pos.clone()
        steps += 1
        if bool(dones.all()):
            break
    done_count = int(env.dones.sum().item())
    return dict(
        success=int(done_count == N),
        coverage=done_count / N,
        steps=steps,
        horizon_fail=int(steps >= max_steps and done_count < N),
        path_len=float(path.sum().item()),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="run directories to evaluate")
    ap.add_argument("--episodes", type=int, default=100,
                    help="held-out episodes per checkpoint (default 100)")
    ap.add_argument("--seed0", type=int, default=900000,
                    help="first scenario seed; the bank is seed0..seed0+episodes-1 "
                         "and is IDENTICAL for every arm")
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of using the distribution mean "
                         "(supplementary only)")
    ap.add_argument("--out", type=Path,
                    default=Path(os.environ.get("PPO_STUDY_ROOT", Path.home() / "dif_driven_ppo_study")) / "heldout_eval.csv")
    args = ap.parse_args()

    bank = list(range(args.seed0, args.seed0 + args.episodes))
    if args.out.exists():
        import pandas as _pd
        existing_df = _pd.read_csv(args.out)
        if existing_df["run"].duplicated().any():
            raise SystemExit(f"{args.out} already contains duplicate run rows; repair it first")
        completed = set(existing_df["run"].astype(str))
    else:
        completed = set()
    rows = []
    for run_dir in args.runs:
        if not (run_dir / "meta.json").exists():
            print(f"SKIP {run_dir} (no meta.json)"); continue
        if run_dir.name in completed:
            print(f"SKIP {run_dir} (already present in {args.out})"); continue
        env, algo, meta = build_from_meta(run_dir)
        eps = [run_episode(env, algo, s, args.max_steps, args.stochastic) for s in bank]
        succ = np.array([e["success"] for e in eps], float)
        cov = np.array([e["coverage"] for e in eps], float)
        stp = np.array([e["steps"] for e in eps], float)
        hf = np.array([e["horizon_fail"] for e in eps], float)
        pl = np.array([e["path_len"] for e in eps], float)
        m = succ.astype(bool)
        row = dict(
            run=run_dir.name,
            reward=meta.get("reward_design"), obs=meta.get("observation"),
            entity_encoder=meta.get("entity_encoder"),
            use_entity_mask=meta.get("use_entity_mask"),
            mlp_hidden=meta.get("mlp_hidden"), n=meta.get("n"),
            episodes=len(eps),
            success_pct=100 * succ.mean(),
            coverage_pct=100 * cov.mean(),
            horizon_fail_pct=100 * hf.mean(),
            # completion time is defined only for successes; failures are
            # right-censored at the horizon and excluded from this mean
            completion_steps_success=float(stp[m].mean()) if m.any() else float("nan"),
            path_len_all=float(pl.mean()),
            path_len_success=float(pl[m].mean()) if m.any() else float("nan"),
        )
        rows.append(row)
        # Append incrementally: a crash mid-batch must not discard completed
        # checkpoints (the first version wrote only at the end).
        args.out.parent.mkdir(parents=True, exist_ok=True)
        import pandas as _pd
        _pd.DataFrame([row]).to_csv(args.out, mode="a",
                                    header=not args.out.exists(), index=False)
        completed.add(run_dir.name)
        print(f"{row['run']:22s} success {row['success_pct']:5.1f}%  "
              f"cov {row['coverage_pct']:5.1f}%  horizonfail {row['horizon_fail_pct']:5.1f}%  "
              f"t_succ {row['completion_steps_success']:6.1f}  "
              f"path_all {row['path_len_all']:7.1f}  path_succ {row['path_len_success']:7.1f}",
              flush=True)
        env.delete() if hasattr(env, "delete") else None
        del env, algo
        torch.cuda.empty_cache()

    print(f"\nwrote {args.out}  (protocol: {args.episodes} episodes, "
          f"seeds {bank[0]}..{bank[-1]}, "
          f"{'stochastic' if args.stochastic else 'deterministic'})")


if __name__ == "__main__":
    main()
