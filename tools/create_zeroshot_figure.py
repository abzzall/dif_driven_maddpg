#!/usr/bin/env python3
"""Create the zero-shot cardinality-transfer figure from the ten C policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--train-n", type=int, default=5)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    sizes = sorted(data.eval_n.unique())

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    for seed, group in data.groupby("seed"):
        group = group.sort_values("eval_n")
        ax.plot(group.eval_n, group.success, marker="o", markersize=4,
                linewidth=1.2, alpha=0.75, color="#2e8b57",
                markeredgecolor="white", markeredgewidth=0.6)

    means = data.groupby("eval_n").success.mean()
    ax.plot(means.index, means.values, color="black", linewidth=2.0,
            marker="D", markersize=5, zorder=5, label="mean of ten policies")
    ax.axvline(args.train_n, color="grey", linestyle="--", linewidth=1.0,
               alpha=0.8, zorder=1)
    ax.annotate("training team size", xy=(args.train_n, 2), fontsize=7.5,
                ha="center", va="bottom", color="grey",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    ax.set_xticks(sizes)
    ax.set_xlabel("evaluation team size (robots)")
    ax.set_ylabel("held-out full-team success (%)")
    ax.set_ylim(0, 103)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    fig.tight_layout()
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(args.output_stem.with_suffix(suffix), dpi=200, bbox_inches="tight")
    print(means.round(1).to_dict())


if __name__ == "__main__":
    main()
