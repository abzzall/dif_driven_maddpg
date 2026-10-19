#!/usr/bin/env python3
"""Figure: team size versus workspace scale for the ten configuration-C policies."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--output-stem", type=Path, required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))

    ax = axes[0]
    for cond, colour, label in (("fixed_arena", "#2e8b57", "fixed workspace (side 20)"),
                                ("fixed_density", "#c2413b", "fixed density (side $20\\sqrt{n/5}$)")):
        g = d[d.condition == cond].groupby("team_size").success_pct
        m, s = g.mean(), g.std()
        ax.errorbar(m.index, m.values, yerr=s.values, color=colour, marker="o",
                    markersize=4.5, linewidth=1.7, capsize=3, label=label)
    ax.set_xlabel("team size (robots)")
    ax.set_ylabel("held-out full-team success (%)")
    ax.set_title("a   Team size under two workspace rules", loc="left", fontweight="bold")
    ax.set_ylim(0, 104)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    ax = axes[1]
    cells = [("n=5\nside 20.0", ("fixed_arena", 5), "#2e8b57"),
             ("n=8\nside 20.0", ("fixed_arena", 8), "#2e8b57"),
             ("n=5\nside 25.3", ("scalecontrol", 5), "#c2413b"),
             ("n=8\nside 25.3", ("fixed_density", 8), "#c2413b")]
    for i, (lab, (cond, n), colour) in enumerate(cells):
        v = d[(d.condition == cond) & (d.team_size == n)].success_pct.values
        ax.scatter(np.full(len(v), i) + np.linspace(-0.11, 0.11, len(v)), v, s=32,
                   color=colour, edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=colour, linewidth=2.2, zorder=4)
    ax.set_xticks(range(len(cells)), [c[0] for c in cells], fontsize=7.5)
    ax.set_ylabel("held-out full-team success (%)")
    ax.set_title("b   Workspace scale dominates team size", loc="left", fontweight="bold")
    ax.set_ylim(0, 104)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suf in (".pdf", ".png"):
        fig.savefig(a.output_stem.with_suffix(suf), dpi=200, bbox_inches="tight")
    print("ok")

if __name__ == "__main__":
    main()
