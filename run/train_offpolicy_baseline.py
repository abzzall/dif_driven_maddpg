#!/usr/bin/env python3
"""Train isolated from-scratch MADDPG or IDDPG controls."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.offpolicy_scratch import OffPolicyBaseline, ReplayBuffer
from custom_envs.diff_driven.graph_coverage_env import GraphCoverageEnv, DEFAULT_SCALES
from rl.convergence import ConvergenceMonitor


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def atomic_torch(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algorithm", choices=("maddpg_scratch", "iddpg_scratch"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--conv_max_hours", type=float, default=3.0)
    ap.add_argument("--conv_max_episodes", type=int, default=5000)
    ap.add_argument("--require_gpu", action="store_true")
    args = ap.parse_args()

    if args.require_gpu and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but unavailable")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if meta_path.exists():
        try:
            if json.loads(meta_path.read_text()).get("launcher_status") == "finished":
                print(f"SKIP {out.name}: already finished", flush=True)
                return
        except Exception:
            pass

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    start_wall = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    env = GraphCoverageEnv(
        num_agents=args.n, num_obstacles=0, v_ang_max=float(np.pi / 2),
        env_size=20.0, flat_observations=True,
    )
    kind = "maddpg" if args.algorithm == "maddpg_scratch" else "iddpg"
    agent = OffPolicyBaseline(kind, args.n, env.obs_dim, env.action_dim, device)
    buffer = ReplayBuffer(50000, args.n, env.obs_dim, env.action_dim)
    scales = torch.tensor(DEFAULT_SCALES, device=device)
    monitor = ConvergenceMonitor(
        window=200, min_delta=2.0, patience=400, min_episodes=600,
        max_episodes=args.conv_max_episodes, max_hours=args.conv_max_hours,
    )

    total_steps = 0
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    peak_gpu = 0
    stopped = False
    meta = {
        "n": args.n, "num_agents": args.n, "seed": args.seed,
        "algorithm": args.algorithm, "algorithm_family": kind.upper(),
        "implementation": "isolated_from_scratch",
        "reward_design": "new", "observation": "flat",
        "reward_scales_used": list(DEFAULT_SCALES),
        "v_ang_max": "pi2", "v_ang_max_float": float(np.pi / 2),
        "env_size": 20.0, "area_per_agent": 400.0 / args.n,
        "budget_protocol": "convergence", "convergence_window": 200,
        "convergence_min_delta": 2.0, "convergence_patience": 400,
        "convergence_min_episodes": 600,
        "convergence_max_episodes": args.conv_max_episodes,
        "convergence_max_hours": args.conv_max_hours,
        "replay_capacity": 50000, "batch_size": 128,
        "start_training_after": 1000, "updates_per_step": 1,
        "gamma": 0.95, "tau": 0.01, "actor_lr": 1e-3, "critic_lr": 1e-3,
        "exploration_noise_start": 0.20, "exploration_noise_end": 0.05,
        "parameter_count": agent.parameter_count,
        "resource_cpu_affinity_count": 1,
        "resource_memory_max_bytes": 2147483648,
        "resource_swap_max_bytes": 0,
        "device": str(device), "cuda_available": torch.cuda.is_available(),
        "command_start_iso": start_iso, "out_dir": str(out),
        "finished": False, "launcher_status": None,
    }

    try:
        for episode in range(args.conv_max_episodes):
            _, obs = env.reset_tensor()
            done = torch.zeros(args.n, dtype=torch.bool, device=device)
            episode_steps = 0
            while episode_steps < 500 and not bool(done.all()):
                frac = min(total_steps / 200000.0, 1.0)
                noise = 0.20 + frac * (0.05 - 0.20)
                action = agent.act(obs, noise)
                action = action * (~done).unsqueeze(-1)
                _, next_obs, _, next_done = env.step_tensor(action)
                reward = (env.current_rewards * scales).sum(dim=-1)
                buffer.add(obs, action, reward, next_obs, next_done)
                obs, done = next_obs, next_done
                total_steps += 1
                episode_steps += 1
                if buffer.size >= 1000:
                    losses = agent.update(buffer.sample(128, device))
                    actor_losses.append(losses.actor)
                    critic_losses.append(losses.critic)

            elapsed = time.time() - start_wall
            success = bool(done.all())
            stopped = monitor.update(success, elapsed_hours=elapsed / 3600.0)
            if torch.cuda.is_available():
                peak_gpu = max(peak_gpu, int(torch.cuda.max_memory_allocated()))
            meta.update({
                "episodes_completed": episode + 1,
                "total_steps": total_steps,
                "total_train_seconds": elapsed,
                "peak_gpu_bytes": peak_gpu,
                "peak_gpu_gb": peak_gpu / 1e9,
                "final_actor_loss": actor_losses[-1] if actor_losses else None,
                "final_critic_loss": critic_losses[-1] if critic_losses else None,
                **monitor.summary(),
                "last_save_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
            atomic_json(meta_path, meta)
            print(
                f"[{args.algorithm}] ep {episode} success={int(success)} "
                f"done={int(done.sum())}/{args.n} steps={total_steps} "
                f"noise={noise:.3f}", flush=True,
            )
            if (episode + 1) % 25 == 0 or stopped:
                atomic_torch(out / "baseline_checkpoint.pth", agent.checkpoint())
            if stopped:
                break

        atomic_torch(out / "baseline_checkpoint.pth", agent.checkpoint())
        meta.update({
            "finished": True, "launcher_status": "finished",
            "command_end_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command_wall_seconds": time.time() - start_wall,
        })
        atomic_json(meta_path, meta)
    except BaseException as exc:
        meta.update({
            "finished": False, "launcher_status": "failed",
            "launcher_error": repr(exc),
            "command_end_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command_wall_seconds": time.time() - start_wall,
        })
        atomic_json(meta_path, meta)
        raise


if __name__ == "__main__":
    main()
