"""
compute_statistics.py
---------------------
Computes the pooled Wilcoxon signed-rank test and effect sizes
that Prof. Li asked for. Reads your existing result files directly.

Run AFTER results_gcn_tdc.json exists:
    python compute_statistics.py

Or run now to see MoleculeNet-only stats (n=10):
    python compute_statistics.py --molnet_only
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
import argparse

# ── File paths ────────────────────────────────────────────────────────────────
FILES = {
    "moe_gcn_classif":   "results_moegcn_classif.json",
    "moe_gcn_regr":      "results_moegcn_regr.json",
    "moe_dmpnn_classif": "results_moedmpnn_classif.json",
    "moe_dmpnn_regr":    "results_moedmpnn_regr.json",
    "dmpnn_classif":     "results_dmpnn_classif.json",
    "dmpnn_regr":        "results_dmpnn_regr.json",
    "attfp":             "results_attentivefp.json",
    "moe_gcn_tdc":       "results_tdc.json",          # MoE-GCN on TDC
    "gcn_tdc":           "results_gcn_tdc.json",       # plain GCN on TDC (new)
}

# For RMSE metrics, lower is better → convert to "improvement = baseline - moe"
# For AUC / Spearman, higher is better → "improvement = moe - baseline"
HIGHER_IS_BETTER = {"roc_auc": True, "AUROC": True, "spearman": True,
                    "mae": False, "rmse": False}

def load(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def get_mean(data, dataset_key):
    """Extract mean score from a result dict, trying both original and lowercase keys."""
    for k in [dataset_key, dataset_key.upper(), dataset_key.capitalize()]:
        if k in data:
            v = data[k]
            if isinstance(v, dict):
                return float(v.get("mean", v.get("score", 0))), v.get("metric", "?")
            elif isinstance(v, (int, float)):
                return float(v), "?"
    return None, None

def rank_biserial(n, W):
    """Rank-biserial correlation from Wilcoxon W statistic."""
    max_W = n * (n + 1) / 2
    return (2 * W / max_W) - 1

def run_wilcoxon(pairs, label=""):
    """
    pairs: list of (moe_score, baseline_score, metric, dataset_name)
    Returns stats dict.
    """
    if len(pairs) < 4:
        print(f"  ⚠️  Only {len(pairs)} pairs — too few for reliable test")
        return None

    # Compute signed differences (always: positive = MoE wins)
    diffs = []
    wins, losses, ties = 0, 0, 0
    for moe, base, metric, name in pairs:
        higher = HIGHER_IS_BETTER.get(metric, True)
        diff = (moe - base) if higher else (base - moe)  # positive = MoE better
        diffs.append(diff)
        if diff > 1e-6:   wins   += 1
        elif diff < -1e-6: losses += 1
        else:              ties   += 1

    diffs = np.array(diffs)
    n = len(diffs)

    # Wilcoxon signed-rank (one-sided: MoE > baseline)
    try:
        stat, p_two = wilcoxon(diffs, alternative="two-sided")
        _, p_one    = wilcoxon(diffs, alternative="greater")
        r = rank_biserial(n, stat)
    except Exception as e:
        print(f"  Wilcoxon error: {e}")
        stat, p_two, p_one, r = float("nan"), 1.0, 1.0, 0.0

    # Effect size interpretation
    abs_r = abs(r)
    effect = "negligible" if abs_r < 0.1 else \
             "small"      if abs_r < 0.3 else \
             "medium"     if abs_r < 0.5 else "large"

    # Mean % improvement (only on datasets where MoE wins)
    winning_diffs = [d for d in diffs if d > 1e-6]
    mean_gain = np.mean(winning_diffs) if winning_diffs else 0.0

    print(f"\n  ── {label} (n={n}) ──")
    print(f"  Wins/Losses/Ties : {wins}/{losses}/{ties}")
    print(f"  Wilcoxon W       : {stat:.1f}")
    print(f"  p-value (two-sided): {p_two:.4f}")
    print(f"  p-value (one-sided, MoE > baseline): {p_one:.4f}")
    print(f"  Rank-biserial r  : {r:+.3f}  [{effect} effect]")
    print(f"  Significant at α=0.05: {'YES ✅' if p_one < 0.05 else 'NO ❌'}")
    print(f"  Mean gain on winning datasets: {mean_gain:+.4f}")

    print(f"\n  Per-dataset breakdown:")
    for (moe, base, metric, name), diff in zip(pairs, diffs):
        flag = "✅" if diff > 1e-6 else ("❌" if diff < -1e-6 else "—")
        print(f"    {flag} {name:<35} MoE={moe:.4f}  Base={base:.4f}  Δ={diff:+.4f}")

    return {
        "n": n, "wins": wins, "losses": losses, "ties": ties,
        "W": stat, "p_two": p_two, "p_one": p_one,
        "rank_biserial_r": r, "effect_size": effect,
        "significant_one_sided": p_one < 0.05,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--molnet_only", action="store_true",
                        help="Only run MoleculeNet stats (don't need GCN TDC file)")
    args = parser.parse_args()

    print("Loading result files...")
    data = {k: load(v) for k, v in FILES.items()}

    missing = [k for k, v in data.items() if v is None]
    if missing:
        print(f"⚠️  Missing files: {missing}")
        if "gcn_tdc" in missing and not args.molnet_only:
            print("    → results_gcn_tdc.json not found.")
            print("    → Run: python run_gcn_tdc_baseline.py")
            print("    → Or run with --molnet_only to see partial stats now\n")

    # ── MoleculeNet: MoE-GCN vs DMPNN ────────────────────────────────────────
    print("\n" + "="*60)
    print("  MOLECULENET: MoE-GCN vs DMPNN (n=10)")
    print("="*60)

    molnet_gcn_vs_dmpnn = []
    molnet_datasets = {
        "BBBP": "roc_auc", "BACE": "roc_auc", "Tox21": "roc_auc",
        "ToxCast": "roc_auc", "SIDER": "roc_auc", "ClinTox": "roc_auc",
        "HIV": "roc_auc", "ESOL": "rmse", "FreeSolv": "rmse", "Lipo": "rmse",
    }

    if data["moe_gcn_classif"] and data["moe_gcn_regr"] and \
       data["dmpnn_classif"] and data["dmpnn_regr"]:
        for ds, metric in molnet_datasets.items():
            moe_src  = data["moe_gcn_classif"] if metric == "roc_auc" else data["moe_gcn_regr"]
            base_src = data["dmpnn_classif"]   if metric == "roc_auc" else data["dmpnn_regr"]
            moe_score, _  = get_mean(moe_src, ds)
            base_score, _ = get_mean(base_src, ds)
            if moe_score is not None and base_score is not None:
                molnet_gcn_vs_dmpnn.append((moe_score, base_score, metric, ds))

        run_wilcoxon(molnet_gcn_vs_dmpnn, "MoE-GCN vs DMPNN — MoleculeNet")
    else:
        print("  ❌ Missing MoleculeNet result files")

    # ── MoleculeNet: MoE-DMPNN vs DMPNN ──────────────────────────────────────
    print("\n" + "="*60)
    print("  MOLECULENET: MoE-DMPNN vs DMPNN (n=10)")
    print("="*60)

    molnet_moedmpnn_vs_dmpnn = []
    if data["moe_dmpnn_classif"] and data["moe_dmpnn_regr"] and \
       data["dmpnn_classif"] and data["dmpnn_regr"]:
        for ds, metric in molnet_datasets.items():
            moe_src  = data["moe_dmpnn_classif"] if metric == "roc_auc" else data["moe_dmpnn_regr"]
            base_src = data["dmpnn_classif"]     if metric == "roc_auc" else data["dmpnn_regr"]
            moe_score, _  = get_mean(moe_src, ds)
            base_score, _ = get_mean(base_src, ds)
            if moe_score is not None and base_score is not None:
                molnet_moedmpnn_vs_dmpnn.append((moe_score, base_score, metric, ds))

        run_wilcoxon(molnet_moedmpnn_vs_dmpnn, "MoE-DMPNN vs DMPNN — MoleculeNet")

    if args.molnet_only or data["gcn_tdc"] is None:
        print("\n  (Skipping TDC pooled analysis — run without --molnet_only once GCN TDC results exist)")
        return

    # ── TDC: MoE-GCN vs plain GCN ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TDC ADMET: MoE-GCN vs plain GCN (n=22)")
    print("="*60)

    tdc_pairs = []
    for ds_key, moe_entry in data["moe_gcn_tdc"].items():
        gcn_entry = data["gcn_tdc"].get(ds_key)
        if gcn_entry is None:
            print(f"  ⚠️  No plain GCN result for {ds_key}, skipping")
            continue
        moe_score = moe_entry["mean"]
        gcn_score = gcn_entry["mean"]
        metric    = moe_entry.get("metric", "AUROC")
        tdc_pairs.append((moe_score, gcn_score, metric, ds_key))

    tdc_stats = run_wilcoxon(tdc_pairs, "MoE-GCN vs plain GCN — TDC ADMET")

    # ── POOLED: MoleculeNet + TDC ─────────────────────────────────────────────
    if molnet_gcn_vs_dmpnn and tdc_pairs:
        print("\n" + "="*60)
        print("  POOLED: MoleculeNet + TDC (n=32)")
        print("="*60)
        pooled = molnet_gcn_vs_dmpnn + tdc_pairs
        pooled_stats = run_wilcoxon(pooled, "MoE-GCN pooled MoleculeNet+TDC")

        print("\n" + "="*60)
        print("  PAPER-READY SUMMARY FOR SECTION 5.5")
        print("="*60)
        if pooled_stats:
            sig = pooled_stats["significant_one_sided"]
            r   = pooled_stats["rank_biserial_r"]
            p   = pooled_stats["p_one"]
            w   = pooled_stats["wins"]
            n   = pooled_stats["n"]
            print(f"""
  Suggested text:
  
  \"Pooled Wilcoxon signed-rank tests across all {n} datasets (10 MoleculeNet
  + 22 TDC ADMET) yield W={pooled_stats['W']:.0f}, p={p:.3f} (one-sided, H1: MoE-GCN > GCN),
  rank-biserial r={r:+.3f} ({pooled_stats['effect_size']} effect size).
  MoE-GCN outperforms the plain GCN baseline on {w}/{n} datasets
  ({'statistically significant' if sig else 'not statistically significant at α=0.05'}).\"
            """)


if __name__ == "__main__":
    main()
