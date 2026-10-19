#!/usr/bin/env python3
"""Create the corrected training-diagnostics figure.

Panel (a) plots rolling full-team success against the training episode index.
The episode axis is commensurate with both the 200-episode smoothing window and
the episode-based policy-update schedule, so it does not mix per-episode
smoothing with a per-transition abscissa. Panel (b) reports the total
environment transitions consumed by each run under the common plateau rule.
Neither panel isolates representation-dependent sample efficiency; they are
training diagnostics for the common stopping protocol.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PATTERN = re.compile(r"^\[MAPPO\] ep (\d+) score \S+ done (\d+)/(\d+) steps (\d+)")
WINDOW = 200
ARMS = [
    ("C", range(1, 11), "#2e8b57", "C: masked entity attention"),
    ("B_wide", range(1, 11), "#d97706", "B_wide: wider fixed index"),
    ("B", range(1, 11), "#1769aa", "B: fixed index"),
    ("A", range(1, 5), "#6b7280", "A: contextual reference"),
]


def load_run(roots, arm, seed):
    for root in roots:
        path = Path(root) / f"{arm}_n5_seed{seed}.log"
        if not path.exists():
            continue
        records = {}
        for line in path.open():
            match = PATTERN.match(line)
            if match:
                episode, done, team, steps = (int(g) for g in match.groups())
                records[episode] = (done, team, steps)
        if records:
            order = sorted(records)
            done = np.array([records[i][0] for i in order])
            team = records[order[0]][1]
            steps = np.array([records[i][2] for i in order])
            return done, team, steps
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", action="append", required=True,
                        help="directory of training logs; repeatable, searched in order")
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4),
                             gridspec_kw={"width_ratios": [1.75, 1.0]})

    budgets = {}
    for index, (arm, seeds, color, label) in enumerate(ARMS):
        first = True
        totals = []
        for seed in seeds:
            run = load_run(args.log_root, arm, seed)
            if run is None:
                print(f"warning: missing {arm} seed {seed}")
                continue
            done, team, steps = run
            totals.append(steps[-1])
            if len(done) < WINDOW:
                continue
            rolling = np.convolve((done == team).astype(float),
                                  np.ones(WINDOW) / WINDOW, "valid") * 100
            axes[0].plot(np.arange(WINDOW, WINDOW + len(rolling)), rolling,
                         color=color, linewidth=1.3, alpha=0.85,
                         label=label if first else None)
            first = False
        budgets[arm] = np.asarray(totals, dtype=float)

        values = budgets[arm] / 1000.0
        offsets = np.linspace(-0.11, 0.11, len(values))
        axes[1].scatter(np.full(len(values), index) + offsets, values, s=34,
                        color=color, edgecolor="white", linewidth=0.6, zorder=3)
        axes[1].plot([index - 0.24, index + 0.24], [values.mean()] * 2,
                     color=color, linewidth=2.0, zorder=4)

    axes[0].set_xlabel("training episode")
    axes[0].set_ylabel(f"full-team success (%), rolling {WINDOW} episodes")
    axes[0].set_title("a   Training progress on the episode axis",
                      loc="left", fontweight="bold")
    axes[0].set_ylim(-2, 102)
    axes[0].grid(alpha=0.25, linewidth=0.7)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7.5, loc="upper right", frameon=False)

    axes[1].set_xticks(range(len(ARMS)), [arm for arm, *_ in ARMS])
    axes[1].set_ylabel("environment transitions consumed ($10^3$)")
    axes[1].set_title("b   Training budget at stopping", loc="left", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    for tick, (_, _, color, _) in zip(axes[1].get_xticklabels(), ARMS):
        tick.set_color(color)
        tick.set_fontweight("bold")

    control = np.concatenate([budgets[a] for a in ("B", "B_wide", "A")])
    axes[1].axhline(control.min() / 1000.0, color="black", linewidth=0.8,
                    linestyle="--", alpha=0.55, zorder=1)
    axes[1].annotate("lowest control run", xy=(3.35, control.min() / 1000.0),
                     fontsize=6.8, ha="right", va="bottom", color="black", alpha=0.75)

    fig.tight_layout()
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(args.output_stem.with_suffix(suffix), dpi=200, bbox_inches="tight")
    print("max C =", budgets["C"].max(), " min control =", control.min())


if __name__ == "__main__":
    main()
