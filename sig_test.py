import json
import numpy as np
from scipy.stats import wilcoxon, ttest_rel

d = json.load(open('ablation_routing_results.json'))

print("="*70)
print("PAIRED SIGNIFICANCE TESTS: MoE-GCN vs Dense baselines")
print("="*70)

for baseline_key, baseline_label in [('Dense-uniform (no route)', 'Dense-uniform'),
                                       ('Dense-wide (no route)', 'Dense-wide')]:
    moe_all, base_all = [], []
    print(f"\n--- MoE-GCN vs {baseline_label} ---")
    for dataset, results in d.items():
        moe_seeds = results['MoE-GCN (routing)']['seeds']
        base_seeds = results[baseline_key]['seeds']
        n = min(len(moe_seeds), len(base_seeds))
        moe_seeds, base_seeds = moe_seeds[:n], base_seeds[:n]
        moe_all.extend(moe_seeds)
        base_all.extend(base_seeds)
        diff = np.mean(moe_seeds) - np.mean(base_seeds)
        print(f"  {dataset:25s} MoE={np.mean(moe_seeds):.4f}  {baseline_label}={np.mean(base_seeds):.4f}  diff={diff:+.4f}")

    moe_all, base_all = np.array(moe_all), np.array(base_all)
    t_stat, t_p = ttest_rel(moe_all, base_all)
    try:
        w_stat, w_p = wilcoxon(moe_all, base_all)
    except ValueError as e:
        w_stat, w_p = None, None
        print(f"  [Wilcoxon failed: {e}]")

    print(f"\n  Pooled n={len(moe_all)} pairs")
    print(f"  Paired t-test:      t={t_stat:.3f}, p={t_p:.4f}")
    if w_p is not None:
        print(f"  Wilcoxon signed-rank: W={w_stat:.3f}, p={w_p:.4f}")

print("\n" + "="*70)
print("Interpretation: p > 0.05 supports 'comparable performance' framing.")
print("p < 0.05 means one config is significantly better — report that instead.")
print("="*70)