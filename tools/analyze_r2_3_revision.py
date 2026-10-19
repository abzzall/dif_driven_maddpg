#!/usr/bin/env python3
"""Combine the submitted and R2.3 cohorts, recompute statistics, and figures."""

from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "review_round1_runs" / "r2_3_additional_seeds"
ORIGINAL = ROOT / "paper_revision2" / "Supplementary" / "heldout_eval.csv"
NEW = R2 / "additional_seeds_heldout_eval.csv"
COMBINED = R2 / "heldout_eval_seeds1_10.csv"
STATS_JSON = R2 / "revision_statistics.json"
REPORT = R2 / "revision_statistics.md"
FIGURES = R2 / "figures"

ARM_ORDER = ["C", "B_wide", "B", "A"]
COLORS = {"C": "#2CA02C", "B_wide": "#DD8452", "B": "#4C72B0", "A": "#8C8C8C"}
PARAMS = {"C": 566_917, "B_wide": 685_577, "B": 207_131, "A": 138_866}


def arm_of(run: str) -> str:
    return run.rsplit("_n", 1)[0]


def seed_of(run: str) -> int:
    match = re.search(r"_seed(\d+)$", run)
    if not match:
        raise ValueError(f"cannot parse seed from {run}")
    return int(match.group(1))


def exact_label_permutation(x, y):
    """Tie-aware exhaustive label-permutation Mann--Whitney calculation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pooled = np.concatenate([x, y])
    ranks = rankdata(pooled, method="average")
    n1, n2 = len(x), len(y)

    def u_for(indices):
        return float(ranks[list(indices)].sum() - n1 * (n1 + 1) / 2)

    observed = u_for(range(n1))
    total = math.comb(n1 + n2, n1)
    low = high = 0
    eps = 1e-12
    for indices in itertools.combinations(range(n1 + n2), n1):
        u = u_for(indices)
        low += u <= observed + eps
        high += u >= observed - eps
    p = min(1.0, 2.0 * min(low / total, high / total))
    rbc = 2.0 * observed / (n1 * n2) - 1.0
    return {"u": observed, "p": p, "rank_biserial": rbc,
            "assignments": total, "nominal_floor": 2.0 / total}


def holm_adjust(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    m = len(values)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    old = pd.read_csv(ORIGINAL)
    new = pd.read_csv(NEW)
    if len(new) != 15:
        raise SystemExit(f"expected 15 new evaluation rows, found {len(new)}")
    new["arm"] = new.run.map(arm_of)
    new["seed"] = new.run.map(seed_of)
    expected = {(arm, seed) for seed in range(6, 11) for arm in ("C", "B", "B_wide")}
    actual = set(zip(new.arm, new.seed))
    if actual != expected or new.run.duplicated().any():
        raise SystemExit(f"new evaluation population mismatch: missing={expected-actual}, extra={actual-expected}")
    if not (new.episodes == 100).all():
        raise SystemExit("new evaluation did not use 100 episodes for every checkpoint")

    old["arm"] = old.run.map(arm_of)
    old["seed"] = old.run.map(seed_of)
    keep_old = old[(old.arm == "A") | ((old.arm.isin(["C", "B", "B_wide"])) & (old.seed <= 5))]
    combined = pd.concat([keep_old, new], ignore_index=True, sort=False)
    if combined.run.duplicated().any() or len(combined) != 34:
        raise SystemExit("combined evaluation must contain 34 unique runs")
    order = {arm: i for i, arm in enumerate(ARM_ORDER)}
    combined["_order"] = combined.arm.map(order)
    combined = combined.sort_values(["_order", "seed"]).drop(columns="_order")
    data_cols = [c for c in combined.columns if c not in ("arm", "seed")]
    combined[data_cols].to_csv(COMBINED, index=False)

    metric_cols = ["success_pct", "coverage_pct", "horizon_fail_pct",
                   "completion_steps_success", "path_len_all", "path_len_success"]
    summaries = {}
    for arm in ARM_ORDER:
        d = combined[combined.arm == arm].sort_values("seed")
        summaries[arm] = {
            "n": len(d),
            "seeds": d.seed.astype(int).tolist(),
            "success_values": d.success_pct.tolist(),
            "metrics": {
                col: {"mean": float(d[col].mean()),
                      "sd": float(d[col].std(ddof=1)) if len(d) > 1 else 0.0}
                for col in metric_cols
            },
            "success_median": float(d.success_pct.median()),
            "success_min": float(d.success_pct.min()),
            "success_max": float(d.success_pct.max()),
            "success_cv": float(d.success_pct.std(ddof=1) / d.success_pct.mean()),
        }

    ablation = pd.read_csv(ROOT / "paper_revision2" / "Supplementary" / "ablation_heldout_eval.csv")
    baseline = pd.read_csv(ROOT / "paper_revision2" / "Supplementary" / "baseline_heldout_eval.csv")
    groups = {
        **{arm: combined.loc[combined.arm == arm, "success_pct"].to_numpy() for arm in ARM_ORDER},
        "D": ablation.loc[ablation.run.str.startswith("D_"), "success_pct"].to_numpy(),
        "E": ablation.loc[ablation.run.str.startswith("E_"), "success_pct"].to_numpy(),
        "IDDPG": baseline.loc[baseline.algorithm == "iddpg_scratch", "success_pct"].to_numpy(),
        "MADDPG": baseline.loc[baseline.algorithm == "maddpg_scratch", "success_pct"].to_numpy(),
    }
    pairs = [("C", "B_wide"), ("C", "B"), ("C", "A"), ("B_wide", "B"),
             ("C", "D"), ("C", "E"), ("D", "E"),
             ("C", "IDDPG"), ("C", "MADDPG")]
    contrasts = {}
    for left, right in pairs:
        result = exact_label_permutation(groups[left], groups[right])
        result["delta_pp"] = float(groups[left].mean() - groups[right].mean())
        result["separation"] = bool(groups[left].min() > groups[right].max())
        contrasts[f"{left}_vs_{right}"] = result
    family = [contrasts[k]["p"] for k in ("C_vs_D", "C_vs_E", "D_vs_E")]
    for key, adj in zip(("C_vs_D", "C_vs_E", "D_vs_E"), holm_adjust(family)):
        contrasts[key]["holm_p"] = adj

    output = {"summaries": summaries, "contrasts": contrasts}
    STATS_JSON.write_text(json.dumps(output, indent=2) + "\n")

    lines = ["# R2.3 combined held-out analysis", ""]
    for arm in ARM_ORDER:
        s = summaries[arm]
        m = s["metrics"]["success_pct"]
        lines.append(f"- {arm}: n={s['n']}, {m['mean']:.2f} ± {m['sd']:.2f}%, "
                     f"range {s['success_min']:.1f}–{s['success_max']:.1f}%, "
                     f"values {s['success_values']}")
    lines += ["", "## Contrasts", ""]
    for name, value in contrasts.items():
        holm = f", Holm p={value['holm_p']:.6g}" if "holm_p" in value else ""
        lines.append(f"- {name}: Δ={value['delta_pp']:+.2f} pp, p={value['p']:.6g}, "
                     f"r={value['rank_biserial']:+.3f}, complete={value['separation']}{holm}")
    REPORT.write_text("\n".join(lines) + "\n")

    # Main primary-result figure: every seed and the capacity control.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for i, arm in enumerate(ARM_ORDER):
        d = combined[combined.arm == arm].sort_values("seed")
        offsets = np.linspace(-0.16, 0.16, len(d)) if len(d) > 1 else [0]
        ax1.scatter(i + offsets, d.success_pct, s=58, color=COLORS[arm], zorder=3)
        ax1.hlines(d.success_pct.mean(), i - .28, i + .28, color=COLORS[arm], lw=3)
    ax1.set_xticks(range(len(ARM_ORDER)), ARM_ORDER)
    ax1.set_ylabel("held-out full-team success (%)")
    ax1.set_ylim(-3, 103)
    ax1.set_title("(a) per-seed held-out success")
    ax1.grid(alpha=.3, axis="y")
    for arm in ARM_ORDER:
        d = combined[combined.arm == arm].success_pct
        ax2.errorbar(PARAMS[arm] / 1e3, d.mean(), yerr=d.std(ddof=1), fmt="o", ms=10,
                     color=COLORS[arm], capsize=4)
        ax2.annotate(arm, (PARAMS[arm] / 1e3, d.mean()), xytext=(8, 5),
                     textcoords="offset points")
    ax2.set_xlabel("trainable parameters (thousands)")
    ax2.set_ylabel("held-out success (%)")
    ax2.set_ylim(-3, 108)
    ax2.set_title("(b) capacity control")
    ax2.grid(alpha=.3)
    fig.tight_layout()
    save(fig, "figure2_heldout_success_capacity")

    # Coverage versus full completion, and horizon failure.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(ARM_ORDER)); width = .38
    coverage = [combined.loc[combined.arm == a, "coverage_pct"].mean() for a in ARM_ORDER]
    success = [combined.loc[combined.arm == a, "success_pct"].mean() for a in ARM_ORDER]
    horizon = [combined.loc[combined.arm == a, "horizon_fail_pct"].mean() for a in ARM_ORDER]
    ax1.bar(x-width/2, coverage, width, label="coverage", color="#4C72B0")
    ax1.bar(x+width/2, success, width, label="full-team success", color="#2CA02C")
    ax1.set_xticks(x, ARM_ORDER); ax1.set_ylabel("%"); ax1.set_ylim(0, 105)
    ax1.set_title("(a) coverage versus completion"); ax1.legend(); ax1.grid(alpha=.3, axis="y")
    ax2.bar(ARM_ORDER, horizon, color=[COLORS[a] for a in ARM_ORDER])
    ax2.set_ylabel("%"); ax2.set_ylim(0, 105)
    ax2.set_title("(b) episodes reaching the 500-step horizon"); ax2.grid(alpha=.3, axis="y")
    fig.tight_layout()
    save(fig, "figure3_coverage_horizon")

    # Across-seed coefficient of variation.
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    cvs = [summaries[a]["success_cv"] for a in ARM_ORDER]
    bars = ax.bar(ARM_ORDER, cvs, color=[COLORS[a] for a in ARM_ORDER])
    for bar, value in zip(bars, cvs):
        ax.annotate(f"{value:.2f}", (bar.get_x()+bar.get_width()/2, value),
                    xytext=(0, 4), textcoords="offset points", ha="center")
    ax.set_ylabel("coefficient of variation (SD / mean)")
    ax.set_title("Across-seed reliability, held-out success")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    save(fig, "figure4_reliability_cv")

    # Component and algorithm context with updated C and B anchors.
    component = {
        "C": groups["C"], "D": groups["D"], "E": groups["E"],
    }
    algorithm = {
        "C": groups["C"], "B": groups["B"],
        "IDDPG": groups["IDDPG"], "MADDPG": groups["MADDPG"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, values, title in zip(axes, (component, algorithm),
                                 ("(a) entity-encoder controls", "(b) algorithmic context")):
        for i, (label, vals) in enumerate(values.items()):
            offsets = np.linspace(-.14, .14, len(vals)) if len(vals) > 1 else [0]
            color = COLORS.get(label, "#8172B2")
            ax.scatter(i + offsets, vals, color=color, s=52, zorder=3)
            ax.hlines(np.mean(vals), i-.25, i+.25, color=color, lw=3)
        ax.set_xticks(range(len(values)), list(values))
        ax.set_ylim(-3, 103); ax.set_ylabel("held-out full-team success (%)")
        ax.set_title(title); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    save(fig, "figure_component_algorithm_controls")

    print(REPORT.read_text())
    print(f"wrote {COMBINED}, {STATS_JSON}, {REPORT}, and figures in {FIGURES}")


if __name__ == "__main__":
    main()
