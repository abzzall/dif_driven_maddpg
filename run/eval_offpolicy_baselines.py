#!/usr/bin/env python3
"""Deterministic common-seed evaluation for isolated off-policy baselines."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.offpolicy_scratch import OffPolicyBaseline
from custom_envs.diff_driven.graph_coverage_env import GraphCoverageEnv


@torch.no_grad()
def evaluate(run_dir: Path, seeds: list[int], max_steps: int) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    n = int(meta["n"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GraphCoverageEnv(
        num_agents=n, num_obstacles=0,
        v_ang_max=float(meta.get("v_ang_max_float", math.pi / 2)),
        env_size=float(meta.get("env_size", 20.0)), flat_observations=True,
    )
    kind = "maddpg" if meta["algorithm"] == "maddpg_scratch" else "iddpg"
    agent = OffPolicyBaseline(kind, n, env.obs_dim, env.action_dim, device)
    checkpoint = torch.load(run_dir / "baseline_checkpoint.pth", map_location=device,
                            weights_only=False)
    agent.load_actor(checkpoint)
    agent.actor.eval()

    episodes = []
    for seed in seeds:
        _, obs = env.reset_tensor(seed=seed)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        prev = env.agent_pos.clone()
        path = torch.zeros(n, device=device)
        steps = 0
        while steps < max_steps and not bool(done.all()):
            action = agent.act(obs, noise_std=0.0) * (~done).unsqueeze(-1)
            _, obs, _, done = env.step_tensor(action)
            path += (env.agent_pos - prev).norm(dim=1)
            prev = env.agent_pos.clone()
            steps += 1
        completed = int(done.sum())
        episodes.append({
            "success": int(completed == n), "coverage": completed / n,
            "steps": steps, "horizon_fail": int(steps >= max_steps and completed < n),
            "path_len": float(path.sum()),
        })

    success = np.array([e["success"] for e in episodes], dtype=float)
    coverage = np.array([e["coverage"] for e in episodes], dtype=float)
    steps = np.array([e["steps"] for e in episodes], dtype=float)
    horizon = np.array([e["horizon_fail"] for e in episodes], dtype=float)
    paths = np.array([e["path_len"] for e in episodes], dtype=float)
    mask = success.astype(bool)
    return {
        "run": run_dir.name, "algorithm": meta["algorithm"],
        "reward": "new", "obs": "flat", "n": n, "episodes": len(seeds),
        "success_pct": 100 * success.mean(), "coverage_pct": 100 * coverage.mean(),
        "horizon_fail_pct": 100 * horizon.mean(),
        "completion_steps_success": float(steps[mask].mean()) if mask.any() else float("nan"),
        "path_len_all": float(paths.mean()),
        "path_len_success": float(paths[mask].mean()) if mask.any() else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=900000)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    completed = set()
    if args.out.exists():
        old = pd.read_csv(args.out)
        if old["run"].duplicated().any():
            raise SystemExit(f"duplicate rows in {args.out}")
        completed = set(old["run"].astype(str))

    for run in args.runs:
        if run.name in completed:
            print(f"SKIP {run.name}: already evaluated", flush=True)
            continue
        row = evaluate(run, seeds, args.max_steps)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(
            args.out, mode="a", header=not args.out.exists(), index=False,
        )
        completed.add(run.name)
        print(
            f"{run.name}: success={row['success_pct']:.1f}% "
            f"coverage={row['coverage_pct']:.1f}%", flush=True,
        )


if __name__ == "__main__":
    main()
