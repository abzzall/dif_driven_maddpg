#!/usr/bin/env python3
"""Reproduce zero-shot cardinality evaluation for n=5-trained graph policies.

The actor and critic parameters are loaded from final n=5 checkpoints, while a
fresh GraphCoverageEnv is constructed at each requested evaluation cardinality.
No retraining or checkpoint selection is performed. Results are one aggregate
row per training seed and evaluation cardinality; the default scenario bank is
the same held-out bank used by eval_frozen.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_envs.diff_driven.graph_coverage_env import GraphCoverageEnv  # noqa: E402
from rl.mappo import MAPPO  # noqa: E402
from run.eval_frozen import run_episode  # noqa: E402


def build_at_n(run_dir: Path, eval_n: int, env_size: float | None = None):
    meta = json.loads((run_dir / "meta.json").read_text())
    if meta.get("reward_design") != "new" or meta.get("observation") != "graph":
        raise ValueError(f"{run_dir}: zero-shot cardinality requires a graph/new checkpoint")
    required = [run_dir / "gnn_actor.pth", run_dir / "gnn_critic.pth"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: missing checkpoint files: {missing}")
    env = GraphCoverageEnv(
        num_agents=eval_n,
        num_obstacles=0,
        v_ang_max=float(meta.get("v_ang_max_float", math.pi / 2)),
        env_size=float(meta.get("env_size", 20.0)) if env_size is None else float(env_size),
        flat_observations=False,
    )
    algo = MAPPO(env, list(meta["reward_scales_used"]))
    cwd = os.getcwd()
    try:
        os.chdir(run_dir)
        algo.load_checkpoint()
    finally:
        os.chdir(cwd)
    return env, algo, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--eval-n", nargs="+", type=int, default=[4, 5, 6])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=900000)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--env-size", type=float, default=None,
                    help="override the arena size recorded in meta.json; use to control "
                         "robot density independently of team size")
    ap.add_argument("--out", type=Path,
                    default=Path(os.environ.get("PPO_STUDY_ROOT", Path.home() / "dif_driven_ppo_study")) / "zeroshot_cardinality.csv")
    args = ap.parse_args()

    bank = range(args.seed0, args.seed0 + args.episodes)
    rows = []
    for run_dir in args.runs:
        train_seed = int(run_dir.name.rsplit("seed", 1)[1])
        for eval_n in args.eval_n:
            env, algo, _ = build_at_n(run_dir, eval_n, args.env_size)
            eps = [run_episode(env, algo, s, args.max_steps, False) for s in bank]
            rows.append({
                "seed": train_seed,
                "eval_n": eval_n,
                "env_size": float(env.env_size),
                "success": 100 * np.mean([e["success"] for e in eps]),
                "coverage": 100 * np.mean([e["coverage"] for e in eps]),
                "horizon_fail": 100 * np.mean([e["horizon_fail"] for e in eps]),
            })
            del env, algo
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows).sort_values(["seed", "eval_n"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows; seeds {args.seed0}..{args.seed0 + args.episodes - 1})")


if __name__ == "__main__":
    main()
