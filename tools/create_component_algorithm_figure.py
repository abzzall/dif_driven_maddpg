#!/usr/bin/env python3
"""Create the manuscript figure for component and algorithm controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_panel(ax, series, title):
    colors = [item[2] for item in series]
    for index, (label, values, color) in enumerate(series):
        values = np.asarray(values, dtype=float)
        offsets = np.linspace(-0.10, 0.10, len(values))
        ax.scatter(
            np.full(len(values), index) + offsets,
            values,
            s=42,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        # Horizontal bar at the group mean, matching the manuscript caption.
        # Sample SD is reported in Table~\ref{tab:additional_results}; drawing it
        # here as well would imply a symmetric spread for arms pinned at the floor.
        ax.plot(
            [index - 0.22, index + 0.22],
            [values.mean()] * 2,
            color=color,
            linewidth=2.2,
            solid_capstyle="butt",
            zorder=4,
        )
    ax.set_xticks(range(len(series)), [item[0] for item in series])
    ax.set_ylim(-4, 105)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    for tick, color in zip(ax.get_xticklabels(), colors):
        tick.set_color(color)
        tick.set_fontweight("bold")
        tick.set_fontsize(7.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--primary-csv", type=Path, required=True,
                        help="heldout_eval.csv; supplies the C and B cohorts so their seed "
                             "counts are never hardcoded")
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()

    ablation = pd.read_csv(args.ablation_csv)
    baseline = pd.read_csv(args.baseline_csv)
    primary = pd.read_csv(args.primary_csv)
    primary["arm"] = primary.run.str.extract(r"^(C|B_wide|B|A)_n5")[0]
    c_vals = primary.loc[primary.arm == "C", "success_pct"].to_numpy()
    b_vals = primary.loc[primary.arm == "B", "success_pct"].to_numpy()
    d = ablation.loc[ablation.run.str.startswith("D_"), "success_pct"].to_numpy()
    e = ablation.loc[ablation.run.str.startswith("E_"), "success_pct"].to_numpy()
    iddpg = baseline.loc[baseline.algorithm == "iddpg_scratch", "success_pct"].to_numpy()
    maddpg = baseline.loc[baseline.algorithm == "maddpg_scratch", "success_pct"].to_numpy()

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), sharey=True)
    draw_panel(
        axes[0],
        [
            ("C\nMasked attn.", c_vals, "#1769aa"),
            ("D\nDeep Sets + mask", d, "#2e8b57"),
            ("E\nAttention, no mask", e, "#d97706"),
        ],
        "a   Entity-encoder components",
    )
    draw_panel(
        axes[1],
        [
            ("B\nMAPPO", b_vals, "#6b7280"),
            ("IDDPG", iddpg, "#7c3aed"),
            ("MADDPG", maddpg, "#c2413b"),
        ],
        "b   Flat-observation algorithms",
    )
    axes[0].set_ylabel("Full-team success (%)")
    fig.text(
        0.5,
        0.012,
        "Coloured points: trained seeds    Horizontal bars: group mean",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.0)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
