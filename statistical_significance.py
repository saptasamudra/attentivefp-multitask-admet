"""
statistical_significance.py
============================
Statistical significance testing for Pharma-MoE vs Standard MoE.

Tests:
1. Paired t-test per dataset (5 seeds)
2. Wilcoxon signed-rank per dataset
3. Overall Wilcoxon across all datasets (performance gains)
4. Cohen's d effect size per dataset
5. Summary table ready for paper

Reads from pharma_moe_results.json (already generated).
If seed-level data not in JSON, re-reads from raw output.

Run: python statistical_significance.py
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon, norm
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("  Statistical Significance Testing")
print("  Pharma-Guided MoE-GCN vs Standard MoE-GCN")
print("="*70)

# ── Per-seed results (from full benchmark runs) ────────────────────────
# These are the raw per-seed scores from pharma_moe_gcn.py output
# Pharma-MoE scores then Standard-MoE scores per seed

PER_SEED_DATA = {
    "solubility_aqsoldb": {
        "display": "Solubility", "metric": "mae",
        "pharma": [0.8190, 0.8086, 0.8788, 0.8375, 0.8103],
        "standard":[0.8926, 0.8841, 0.9801, 0.8602, 0.8663],
    },
    "caco2_wang": {
        "display": "Caco-2", "metric": "mae",
        "pharma": [0.3326, 0.3510, 0.3827, 0.3708, 0.3722],
        "standard":[0.3543, 0.3768, 0.3292, 0.3908, 0.4158],
    },
    "lipophilicity_astrazeneca": {
        "display": "Lipophilicity", "metric": "mae",
        "pharma": [0.5559, 0.5849, 0.5603, 0.5617, 0.5560],
        "standard":[0.5328, 0.5581, 0.5317, 0.5726, 0.5480],
    },
    "ld50_zhu": {
        "display": "LD50", "metric": "mae",
        "pharma": [0.6264, 0.6306, 0.6355, 0.6667, 0.6493],
        "standard":[0.6647, 0.6266, 0.6324, 0.6778, 0.6676],
    },
    "vdss_lombardo": {
        "display": "VDss", "metric": "spearman",
        "pharma": [0.5268, 0.4662, 0.5407, 0.5425, 0.5155],
        "standard":[0.5685, 0.4484, 0.5454, 0.5003, 0.4342],
    },
    "half_life_obach": {
        "display": "Half-Life", "metric": "spearman",
        "pharma": [0.2906, 0.3536, 0.3335, 0.3396, 0.3801],
        "standard":[0.2653, 0.2774, 0.2905, 0.3134, 0.3519],
    },
    "clearance_microsome_az": {
        "display": "CL-Microsome", "metric": "spearman",
        "pharma": [0.5555, 0.5460, 0.1607, 0.1681, 0.5558],
        "standard":[0.5354, 0.5415, 0.5549, 0.5398, 0.5293],
    },
    "clearance_hepatocyte_az": {
        "display": "CL-Hepatocyte", "metric": "spearman",
        "pharma": [0.3601, 0.3042, 0.2904, 0.3448, 0.3616],
        "standard":[0.3703, 0.3391, 0.3358, 0.3526, 0.3526],
    },
    "ppbr_az": {
        "display": "PPBR", "metric": "mae",
        "pharma": [9.5898, 9.4531, 9.9843, 9.7496, 10.4136],
        "standard":[9.5234, 9.3485, 9.2029, 9.1355, 9.6230],
    },
    "hia_hou": {
        "display": "HIA", "metric": "auroc",
        "pharma": [0.7617, 0.9267, 0.9494, 0.9556, 0.9737],
        "standard":[0.9263, 0.9498, 0.9428, 0.9580, 0.9584],
    },
    "bbb_martins": {
        "display": "BBB", "metric": "auroc",
        "pharma": [0.8628, 0.8758, 0.8818, 0.8751, 0.8586],
        "standard":[0.8627, 0.8455, 0.8497, 0.8365, 0.8235],
    },
    "herg": {
        "display": "hERG", "metric": "auroc",
        "pharma": [0.7848, 0.7760, 0.7521, 0.7535, 0.7501],
        "standard":[0.6641, 0.7268, 0.7333, 0.7119, 0.6691],
    },
    "ames": {
        "display": "AMES", "metric": "auroc",
        "pharma": [0.8489, 0.8415, 0.8417, 0.8371, 0.8379],
        "standard":[0.8376, 0.8382, 0.8363, 0.8408, 0.8478],
    },
    "dili": {
        "display": "DILI", "metric": "auroc",
        "pharma": [0.9500, 0.9387, 0.8983, 0.9257, 0.9296],
        "standard":[0.9226, 0.9187, 0.9248, 0.9152, 0.9187],
    },
}


def higher_is_better(metric):
    return metric in ("spearman", "auroc")


def cohens_d(a, b):
    """Cohen's d for paired samples."""
    diff = np.array(a) - np.array(b)
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))


def effect_size_label(d):
    d = abs(d)
    if d >= 0.8: return "large"
    if d >= 0.5: return "medium"
    if d >= 0.2: return "small"
    return "negligible"


# ── Per-dataset tests ─────────────────────────────────────────────────

results = {}
all_gains = []  # for overall test

print(f"\n{'─'*70}")
print(f"  Per-Dataset Significance Tests (n=5 seeds each)")
print(f"{'─'*70}")
print(f"  {'Dataset':<16} {'Gain':>7} {'t-stat':>8} {'p(t)':>8} "
      f"{'p(W)':>8} {'d':>7} {'Effect':<10} Sig")
print("  "+"-"*75)

for dataset, data in PER_SEED_DATA.items():
    pharma   = np.array(data["pharma"])
    standard = np.array(data["standard"])
    metric   = data["metric"]
    hib      = higher_is_better(metric)
    display  = data["display"]

    # Compute per-seed gains (positive = pharma wins)
    if hib:
        gains = pharma - standard
    else:
        gains = standard - pharma  # lower is better for MAE

    mean_gain = float(np.mean(gains))
    all_gains.append(mean_gain)

    # Paired t-test
    try:
        t_stat, p_t = ttest_rel(pharma, standard)
        # One-sided: pharma better than standard
        p_t_one = float(p_t) / 2 if (hib and t_stat > 0) or \
                                     (not hib and t_stat < 0) \
                  else 1.0 - float(p_t)/2
    except:
        t_stat, p_t, p_t_one = 0, 1, 1

    # Wilcoxon signed-rank
    try:
        w_stat, p_w = wilcoxon(pharma, standard, alternative='greater' if hib else 'less')
    except:
        w_stat, p_w = 0, 1.0

    # Cohen's d
    d = cohens_d(
        pharma if hib else -pharma,
        standard if hib else -standard
    )
    eff = effect_size_label(d)

    sig = ("***" if p_t_one < 0.001 else
           "**"  if p_t_one < 0.01  else
           "*"   if p_t_one < 0.05  else
           "ns")

    # Metric-specific gain display
    if metric == "mae":
        gain_pct = (float(np.mean(standard)) - float(np.mean(pharma))) / abs(float(np.mean(standard))) * 100
    else:
        gain_pct = (float(np.mean(pharma)) - float(np.mean(standard))) / abs(float(np.mean(standard))) * 100

    print(f"  {display:<16} {gain_pct:>+6.1f}% {float(t_stat):>8.3f} "
          f"{p_t_one:>8.4f} {float(p_w):>8.4f} "
          f"{d:>7.3f} {eff:<10} {sig}")

    results[dataset] = {
        "display":      display,
        "metric":       metric,
        "mean_pharma":  round(float(np.mean(pharma)), 4),
        "mean_standard":round(float(np.mean(standard)), 4),
        "gain_pct":     round(gain_pct, 2),
        "t_stat":       round(float(t_stat), 4),
        "p_ttest_one":  round(p_t_one, 6),
        "p_wilcoxon":   round(float(p_w), 6),
        "cohens_d":     round(d, 4),
        "effect_size":  eff,
        "significant_p05": p_t_one < 0.05,
        "significant_p01": p_t_one < 0.01,
    }


# ── Overall significance test ──────────────────────────────────────────

print(f"\n{'─'*70}")
print(f"  Overall Tests (across all 14 datasets)")
print(f"{'─'*70}")

all_gains = np.array(all_gains)
print(f"\n  Mean gain across 14 datasets: {np.mean(all_gains):+.4f}")
print(f"  Std:                          {np.std(all_gains):.4f}")
print(f"  Datasets with positive gain:  {(all_gains > 0).sum()}/14")

# One-sample t-test: mean gain > 0
from scipy.stats import ttest_1samp
t_overall, p_overall = ttest_1samp(all_gains, 0)
sig_overall = ("***" if p_overall/2 < 0.001 else
               "**"  if p_overall/2 < 0.01  else
               "*"   if p_overall/2 < 0.05  else "ns")
print(f"\n  One-sample t-test (H0: mean gain = 0):")
print(f"  t={t_overall:.3f}, p={p_overall/2:.4f} {sig_overall}")

# Wilcoxon signed-rank: overall gains vs 0
try:
    w_all, pw_all = wilcoxon(all_gains)
    sig_w = ("***" if pw_all < 0.001 else
             "**"  if pw_all < 0.01  else
             "*"   if pw_all < 0.05  else "ns")
    print(f"  Wilcoxon signed-rank (H0: median gain = 0):")
    print(f"  W={w_all:.1f}, p={pw_all:.4f} {sig_w}")
except Exception as e:
    print(f"  Wilcoxon: {e}")

# Regression vs classification split
reg_datasets  = ["solubility_aqsoldb","caco2_wang","lipophilicity_astrazeneca",
                  "ld50_zhu","vdss_lombardo","half_life_obach",
                  "clearance_microsome_az","clearance_hepatocyte_az","ppbr_az"]
clf_datasets  = ["hia_hou","bbb_martins","herg","ames","dili"]

reg_gains = np.array([results[d]["gain_pct"] for d in reg_datasets if d in results])
clf_gains = np.array([results[d]["gain_pct"] for d in clf_datasets if d in results])

print(f"\n  Regression endpoints (n={len(reg_gains)}):")
print(f"  Mean gain: {np.mean(reg_gains):+.1f}%  "
      f"Positive: {(reg_gains>0).sum()}/{len(reg_gains)}")

print(f"\n  Classification endpoints (n={len(clf_gains)}):")
print(f"  Mean gain: {np.mean(clf_gains):+.1f}%  "
      f"Positive: {(clf_gains>0).sum()}/{len(clf_gains)}")

# Mann-Whitney between regression and classification gains
from scipy.stats import mannwhitneyu
try:
    mw_stat, mw_p = mannwhitneyu(reg_gains, clf_gains, alternative='two-sided')
    print(f"\n  Mann-Whitney (regression vs classification gains):")
    print(f"  U={mw_stat:.1f}, p={mw_p:.4f} "
          f"({'significant' if mw_p<0.05 else 'not significant'})")
except Exception as e:
    print(f"  Mann-Whitney: {e}")

# Specialization gains overall Wilcoxon
spec_gains = np.array([
    53.7, 66.2, 193.2, 39.6, 176.3, -26.2, -100.0,
    159.5, 0.0, 34.6, 47.7, 40.9, 27.9, 65.3
])
print(f"\n  Specialization (eta²) improvement across 14 datasets:")
print(f"  Mean: {np.mean(spec_gains):+.1f}%  "
      f"Positive: {(spec_gains>0).sum()}/14")
try:
    w_spec, pw_spec = wilcoxon(spec_gains)
    print(f"  Wilcoxon (H0: median spec gain = 0):")
    print(f"  W={w_spec:.1f}, p={pw_spec:.4f} "
          f"{'***' if pw_spec<0.001 else '**' if pw_spec<0.01 else '*' if pw_spec<0.05 else 'ns'}")
except Exception as e:
    print(f"  Wilcoxon spec: {e}")


# ── Paper-ready stats table ───────────────────────────────────────────

print(f"\n{'─'*70}")
print(f"  Paper-Ready Statistics (paste into Methods section)")
print(f"{'─'*70}")

sig_p05 = sum(1 for v in results.values() if v["significant_p05"] and v["gain_pct"] > 0)
sig_p01 = sum(1 for v in results.values() if v["significant_p01"] and v["gain_pct"] > 0)
pos_gains = sum(1 for v in results.values() if v["gain_pct"] > 0)

print(f"""
  Datasets with significant improvement (p<0.05, one-sided): {sig_p05}/14
  Datasets with significant improvement (p<0.01, one-sided): {sig_p01}/14
  Datasets with positive mean gain:                           {pos_gains}/14
  Overall one-sample t-test: t={t_overall:.3f}, p={p_overall/2:.4f} {sig_overall}
  Mean specialization improvement: {np.mean(spec_gains):+.1f}% (Wilcoxon p={pw_spec:.4f})
""")

# Save
with open("statistical_significance_results.json","w") as f:
    json.dump({
        "per_dataset": results,
        "overall": {
            "mean_gain": round(float(np.mean(all_gains)),4),
            "t_stat": round(float(t_overall),4),
            "p_ttest_one": round(float(p_overall/2),6),
            "significant_overall": float(p_overall/2) < 0.05,
            "positive_datasets": int((all_gains>0).sum()),
            "sig_p05_datasets": sig_p05,
            "sig_p01_datasets": sig_p01,
        }
    }, f, indent=2)
print("  [SAVED] statistical_significance_results.json")
print("="*70)
