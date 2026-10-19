#!/usr/bin/env python3
"""Seed-level aggregation of frozen held-out evaluation.

The statistical unit is the trained seed. Evaluation episodes are repeated
measures of a single policy and are never treated as independent replicates, so
no test here consumes the per-episode counts.

Reports, per arm: mean, SD, median, min, max and every individual seed value;
and, per contrast: the absolute difference in percentage points, the exact
two-sided Mann-Whitney permutation p-value with its attainable floor, and the
rank-biserial correlation. Effect size and the raw seed values are the primary
evidence; the p-value is reported alongside them, not in place of them.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ARM_ORDER = ["C", "B_wide", "B", "A"]


def arm_of(run: str) -> str:
    return run.rsplit("_n", 1)[0]


def exact_mw(x, y):
    """Standard exact two-sided Mann–Whitney and rank-biserial effect size.

    SciPy's exact method is used so unequal group sizes and the two-sided tail
    convention are handled consistently. The nominal floor is reported only
    as a sample-size reference; ties can affect attainable probabilities.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    test = mannwhitneyu(x, y, alternative="two-sided", method="exact")
    obs = float(test.statistic)
    rbc = 2 * obs / (n1 * n2) - 1          # +1 = every x above every y
    from math import comb
    nominal_floor = 2 / comb(n1 + n2, n1)
    return float(test.pvalue), nominal_floor, rbc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path(os.environ.get("PPO_STUDY_ROOT", Path.home() / "dif_driven_ppo_study")) / "heldout_eval.csv")
    ap.add_argument("--metric", default="success_pct")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["arm"] = df["run"].map(arm_of)
    # Run identifiers now include seed 10; parse the complete numeric suffix
    # rather than only the final character.
    df["seed"] = df["run"].str.extract(r"_seed(\d+)$", expand=False).astype(int)
    arms = [a for a in ARM_ORDER if a in set(df.arm)]

    print("=" * 92)
    print(f"HELD-OUT DETERMINISTIC EVALUATION — {int(df.episodes.iloc[0])} common episodes/checkpoint")
    print("=" * 92)
    cols = ["success_pct", "coverage_pct", "horizon_fail_pct",
            "completion_steps_success", "path_len_all", "path_len_success"]
    print(f"{'arm':8s} {'n':>2s} " + " ".join(f"{c.replace('_pct','').replace('_',' '):>16s}" for c in cols))
    for a in arms:
        d = df[df.arm == a]
        print(f"{a:8s} {len(d):>2d} " + " ".join(f"{d[c].mean():16.1f}" for c in cols))

    print(f"\n--- per-seed {args.metric} ---")
    for a in arms:
        d = df[df.arm == a].sort_values("seed")
        v = d[args.metric].to_numpy(float)
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        print(f"{a:8s} mean {v.mean():6.2f} ± {sd:5.2f}   median {np.median(v):6.2f}   "
              f"range [{v.min():.1f}, {v.max():.1f}]   seeds {list(np.round(v, 1))}")

    if "C" in arms:
        print(f"\n--- contrasts vs C ({args.metric}), seed-level, unpaired ---")
        c = df[df.arm == "C"][args.metric].to_numpy(float)
        for a in arms:
            if a == "C":
                continue
            o = df[df.arm == a][args.metric].to_numpy(float)
            p, floor, rbc = exact_mw(c, o)
            sep = "complete" if c.min() > o.max() else "overlapping"
            print(f"  C vs {a:7s} Δ={c.mean()-o.mean():+6.1f} pp   p={p:.4f} (floor {floor:.4f})   "
                  f"rank-biserial={rbc:+.2f}   separation: {sep}")

        print("\nDecision rule ([[PAPER_STRATEGY]] §6):")
        bw = df[df.arm == "B_wide"][args.metric].to_numpy(float) if "B_wide" in arms else np.array([])
        if bw.size:
            strong = c.mean() >= 80 and c.min() > bw.max()
            mod = 60 <= c.mean() < 80 and c.min() > bw.max()
            print(f"  C mean = {c.mean():.1f}%   every C seed > every B_wide seed: "
                  f"{'YES' if c.min() > bw.max() else 'NO'} "
                  f"(min C {c.min():.1f} vs max B_wide {bw.max():.1f})")
            print("  → " + ("STRONG outcome: submit the planned reliability paper."
                            if strong else
                            "MODERATE outcome: submit with weaker wording."
                            if mod else
                            "MIXED/NEGATIVE: reassess before submission (see §6)."))

    print(f"\n--- all pairwise contrasts ({args.metric}); non-significance is not equivalence ---")
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            x = df[df.arm == a][args.metric].to_numpy(float)
            y = df[df.arm == b][args.metric].to_numpy(float)
            p, floor, rbc = exact_mw(x, y)
            print(f"  {a:7s} vs {b:7s} Δ={x.mean()-y.mean():+6.1f} pp   "
                  f"p={p:.4f} (nominal floor {floor:.4f})   rank-biserial={rbc:+.2f}")


if __name__ == "__main__":
    main()
