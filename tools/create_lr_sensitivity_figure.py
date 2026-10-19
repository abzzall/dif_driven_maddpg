#!/usr/bin/env python3
"""Figure: held-out success across the learning-rate sensitivity grid."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"C": "#2e8b57", "B_wide": "#d97706", "B": "#1769aa"}
LABELS = {"C": "C: masked entity attention", "B_wide": "B_wide: wider fixed index",
          "B": "B: fixed index"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--output-stem", type=Path, required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    rates = sorted(d.learning_rate.unique())
    x = np.arange(len(rates))
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for k, arm in enumerate(("C", "B_wide", "B")):
        off = (k - 1) * 0.13
        means = []
        for i, r in enumerate(rates):
            v = d[(d.configuration == arm) & (np.isclose(d.learning_rate, r))].success_pct.values
            means.append(v.mean())
            ax.scatter(np.full(len(v), i + off) + np.linspace(-0.035, 0.035, len(v)), v,
                       s=30, color=COLORS[arm], edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot(x + off, means, color=COLORS[arm], linewidth=1.8, marker="D",
                markersize=5, zorder=4, label=LABELS[arm])
    ax.set_xticks(x, [f"{r:g}" for r in rates])
    ax.set_xlabel("actor and critic learning rate")
    ax.set_ylabel("held-out full-team success (%)")
    ax.set_ylim(-3, 104)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="center left")
    ax.annotate("principal setting", xy=(1, 30), fontsize=7, ha="center",
                va="bottom", color="grey")
    fig.tight_layout()
    for suf in (".pdf", ".png"):
        fig.savefig(a.output_stem.with_suffix(suf), dpi=200, bbox_inches="tight")
    print(d.groupby(["configuration", "learning_rate"]).success_pct.mean().round(1).to_dict())

if __name__ == "__main__":
    main()
