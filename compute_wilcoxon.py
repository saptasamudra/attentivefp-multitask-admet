"""
compute_wilcoxon.py
-------------------
Runs pooled Wilcoxon signed-rank test comparing plain GCN vs MoE-GCN
on TDC ADMET datasets.

Requires:
    results_gcn_tdc.json       (GCN baseline, from run_gcn_tdc_baseline.py)
    results_moegcn_tdc_v2.json (MoE-GCN, from run_moegcn_tdc_benchmark.py)

Usage:
    python compute_wilcoxon.py
    python compute_wilcoxon.py --gcn results_gcn_tdc.json --moe results_moegcn_tdc_v2.json
"""

import json
import argparse
import numpy as np
from scipy.stats import wilcoxon

# ── Metric direction ──────────────────────────────────────────────────────────
HIGHER_IS_BETTER = {
    "AUROC": True,
    "spearman": True,
    "mae": False,
    "rmse": False,
}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sign_correct(score, metric):
    """Return score such that higher is always better."""
    if HIGHER_IS_BETTER.get(metric, True):
        return score
    return -score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcn", default="results_gcn_tdc.json")
    parser.add_argument("--moe", default="results_moegcn_tdc_v2.json")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    gcn_results = load(args.gcn)
    moe_results = load(args.moe)

    # ── Find common datasets ──────────────────────────────────────────────────
    common = sorted(set(gcn_results.keys()) & set(moe_results.keys()))
    print(f"Common datasets: {len(common)}")
    print(f"GCN total: {len(gcn_results)} | MoE total: {len(moe_results)}")

    # ── Per-dataset comparison ────────────────────────────────────────────────
    print(f"\n{'─'*85}")
    print(f"{'Dataset':<40} {'Metric':<10} {'GCN':>8} {'MoE-GCN':>8} {'Delta':>8}  Winner")
    print(f"{'─'*85}")

    all_gcn_scores = []
    all_moe_scores = []
    moe_wins = gcn_wins = ties = 0

    for ds in common:
        g = gcn_results[ds]
        m = moe_results[ds]
        metric = g.get("metric", "AUROC").lower()
        higher = HIGHER_IS_BETTER.get(metric, True)

        gcn_mean = g["mean"]
        moe_mean = m["mean"]

        # sign-corrected delta: positive = MoE better
        if higher:
            delta = moe_mean - gcn_mean
        else:
            delta = gcn_mean - moe_mean  # lower MAE → MoE better → positive

        if delta > 0.001:
            winner = "MoE ✓"
            moe_wins += 1
        elif delta < -0.001:
            winner = "GCN ✗"
            gcn_wins += 1
        else:
            winner = "≈ tie"
            ties += 1

        print(f"  {ds:<38} {metric:<10} {gcn_mean:>8.4f} {moe_mean:>8.4f} {delta:>+8.4f}  {winner}")

        # Collect per-seed pairs for Wilcoxon
        gcn_seeds = g.get("seeds", [])
        moe_seeds = m.get("seeds", [])

        # align seed count — use minimum available
        n = min(len(gcn_seeds), len(moe_seeds))
        if n == 0:
            continue

        for i in range(n):
            gs = sign_correct(gcn_seeds[i], metric)
            ms = sign_correct(moe_seeds[i], metric)
            all_gcn_scores.append(gs)
            all_moe_scores.append(ms)

    print(f"{'─'*85}")
    print(f"\nDataset-level: MoE wins={moe_wins}, GCN wins={gcn_wins}, ties={ties}")

    # ── Pooled Wilcoxon ───────────────────────────────────────────────────────
    all_gcn_scores = np.array(all_gcn_scores)
    all_moe_scores = np.array(all_moe_scores)
    n_pairs = len(all_gcn_scores)

    print(f"\n{'='*57}")
    print(f"  POOLED WILCOXON SIGNED-RANK TEST")
    print(f"{'='*57}")
    print(f"  Total paired observations: {n_pairs}")
    print(f"  (datasets × seeds, sign-corrected so higher=better)")

    if n_pairs < 10:
        print("  ⚠️  Too few pairs for reliable Wilcoxon — run more seeds")
        return

    diffs = all_moe_scores - all_gcn_scores
    nonzero = np.sum(diffs != 0)
    print(f"  Non-zero differences: {nonzero}/{n_pairs}")

    stat, p_value = wilcoxon(all_moe_scores, all_gcn_scores, alternative="greater")
    _, p_two = wilcoxon(all_moe_scores, all_gcn_scores, alternative="two-sided")

    print(f"\n  H0: MoE-GCN ≤ GCN (no improvement)")
    print(f"  H1: MoE-GCN > GCN (MoE improves)")
    print(f"\n  Wilcoxon statistic : {stat:.1f}")
    print(f"  p-value (one-sided) : {p_value:.4f}")
    print(f"  p-value (two-sided) : {p_two:.4f}")
    print(f"  Alpha               : {args.alpha}")

    if p_value < args.alpha:
        print(f"\n  ✅ SIGNIFICANT: MoE-GCN significantly outperforms plain GCN")
        print(f"     (p={p_value:.4f} < α={args.alpha})")
    else:
        print(f"\n  ✗ NOT SIGNIFICANT: Cannot reject H0")
        print(f"     (p={p_value:.4f} ≥ α={args.alpha})")

    # ── Effect size (rank-biserial correlation) ───────────────────────────────
    n = n_pairs
    r = 1 - (2 * stat) / (n * (n + 1))
    print(f"\n  Effect size (rank-biserial r): {r:.4f}")
    if abs(r) >= 0.5:
        effect = "large"
    elif abs(r) >= 0.3:
        effect = "medium"
    else:
        effect = "small"
    print(f"  Effect magnitude: {effect}")

    # ── Per-metric breakdown ──────────────────────────────────────────────────
    print(f"\n{'─'*57}")
    print("  PER-METRIC BREAKDOWN")
    print(f"{'─'*57}")

    for metric_name in ["AUROC", "mae", "spearman"]:
        metric_key = metric_name.lower()
        gcn_m, moe_m = [], []
        for ds in common:
            g = gcn_results[ds]
            m = moe_results[ds]
            if g.get("metric", "").lower() != metric_key:
                continue
            gs_seeds = g.get("seeds", [])
            ms_seeds = m.get("seeds", [])
            n = min(len(gs_seeds), len(ms_seeds))
            for i in range(n):
                gcn_m.append(sign_correct(gs_seeds[i], metric_key))
                moe_m.append(sign_correct(ms_seeds[i], metric_key))

        if len(gcn_m) < 6:
            continue

        gcn_m = np.array(gcn_m)
        moe_m = np.array(moe_m)
        try:
            _, p = wilcoxon(moe_m, gcn_m, alternative="greater")
            mean_delta = np.mean(moe_m - gcn_m)
            sig = "✅ sig" if p < args.alpha else "✗ n.s."
            print(f"  {metric_name:<10}  n={len(gcn_m):>4}  "
                  f"mean_delta={mean_delta:>+7.4f}  p={p:.4f}  {sig}")
        except Exception as e:
            print(f"  {metric_name:<10}  Wilcoxon failed: {e}")

    print(f"\n{'='*57}")
    print("  JOURNAL TARGET GUIDANCE")
    print(f"{'='*57}")
    if p_value < 0.05 and moe_wins > gcn_wins:
        print("  → Journal of Cheminformatics (primary target)")
        print("    MoE significantly outperforms GCN with statistical evidence.")
    elif p_value < 0.1 or moe_wins >= gcn_wins:
        print("  → Machine Learning: Science & Technology (fallback)")
        print("    Results are marginal — borderline significance or mixed wins.")
    else:
        print("  → Reconsider framing: MoE-GCN does not consistently beat GCN on TDC.")
        print("    Consider focusing on MoleculeNet results + expert specialization story.")


if __name__ == "__main__":
    main()
