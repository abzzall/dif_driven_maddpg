#!/usr/bin/env python3
"""Matched frozen-policy trajectories: two arms, identical initial conditions.

Existing trajectory PNGs were produced during training, with exploration noise,
from different episodes and different initial states. They cannot be compared
geometrically and must not be presented as deployment behaviour.

This script evaluates two frozen policies deterministically from the *same*
scenario seed, so the only difference between panels is the controller. Output
is illustrative, not causal evidence: it shows what a policy does on one
scenario, chosen by a declared rule, not why one arm outperforms another.

Selection rule (declare in the caption): the scenario is chosen by seed, given
on the command line. Do not scan seeds for a flattering example.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run.eval_frozen import build_from_meta  # noqa: E402


@torch.no_grad()
def rollout(run_dir: Path, seed: int, max_steps: int = 500):
    env, algo, meta = build_from_meta(run_dir)
    _, obs = env.reset_tensor(seed=seed)
    N = env.num_agents
    pos = [env.agent_pos.clone().cpu().numpy()]
    for _ in range(max_steps):
        raw, _ = algo.actor.act(obs, deterministic=True)
        raw = raw.clamp(-algo.max_action, algo.max_action)
        _, obs, _, dones = env.step_tensor(raw)
        pos.append(env.agent_pos.clone().cpu().numpy())
        if bool(dones.all()):
            break
    traj = np.stack(pos)                       # [T+1, N, 2]
    lm = env.landmarks.cpu().numpy()
    done_count = int(env.dones.sum().item())
    path = float(np.linalg.norm(np.diff(traj, axis=0), axis=2).sum())
    info = dict(steps=traj.shape[0] - 1, done=done_count, N=N, path=path)
    del env, algo
    torch.cuda.empty_cache()
    return traj, lm, info


def draw(ax, traj, lm, info, title):
    T, N, _ = traj.shape
    # time-coloured segments so direction of travel is readable in print
    for i in range(N):
        pts = traj[:, i, :].reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis",
                            norm=plt.Normalize(0, T), linewidth=1.6, alpha=.9)
        lc.set_array(np.arange(T))
        ax.add_collection(lc)
    ax.scatter(traj[0, :, 0], traj[0, :, 1], c="tab:blue", s=55, zorder=5,
               marker="o", label="start")
    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], c="tab:green", s=55, zorder=5,
               marker="s", label="end")
    ax.scatter(lm[:, 0], lm[:, 1], c="crimson", s=90, marker="x", zorder=6,
               linewidths=2, label="landmark")
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.autoscale_view()
    ax.set_title(f"{title}\n{info['done']}/{info['N']} covered · "
                 f"{info['steps']} steps · path {info['path']:.0f}", fontsize=10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--seed", type=int, required=True,
                    help="scenario seed; declare it in the caption")
    ap.add_argument("--out", type=Path,
                    default=Path(os.environ.get("PPO_STUDY_ROOT", Path.home() / "dif_driven_ppo_study")) / "figures/matched_trajectory.png")
    args = ap.parse_args()

    ta, la, ia = rollout(args.run_a, args.seed)
    tb, lb, ib = rollout(args.run_b, args.seed)
    assert np.allclose(la, lb), "landmark sets differ — scenarios are not matched"

    fig, axs = plt.subplots(1, 2, figsize=(13, 6.4), sharex=True, sharey=True)
    draw(axs[0], ta, la, ia, args.run_a.name)
    draw(axs[1], tb, lb, ib, args.run_b.name)
    axs[0].legend(loc="upper right", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap="viridis"); sm.set_array([])
    fig.colorbar(sm, ax=axs, fraction=.02, pad=.02, label="episode progress")
    fig.suptitle(f"Matched deterministic rollouts, scenario seed {args.seed} "
                 f"(identical initial conditions; illustrative, not causal)", fontsize=11)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out.with_suffix("." + ext), dpi=200, bbox_inches="tight")
    print(f"wrote {args.out.with_suffix('.pdf')} (+png)")
    print(f"  {args.run_a.name}: {ia}")
    print(f"  {args.run_b.name}: {ib}")


if __name__ == "__main__":
    main()
