"""
Phase 1.4: Statistical Rigor — Significance Testing
===================================================
Run paired t-tests and Wilcoxon signed-rank tests across models.

Compares:
- All baselines vs MoE variants
- MoE K=2 vs K=4 vs K=8
- Significance threshold: p < 0.05
"""

import numpy as np
from scipy import stats
import pandas as pd


class StatisticalTester:
    """Paired statistical tests for model comparison."""
    
    def __init__(self, seed_results_dict):
        """
        Args:
            seed_results_dict: {
                'Model_Name': {
                    'ESOL': [0.95, 0.97, 0.93],  # [seed1, seed2, seed3]
                    'BACE': [0.80, 0.82, 0.78],
                    ...
                },
                ...
            }
        """
        self.results = seed_results_dict
    
    def paired_ttest(self, model1, model2, dataset_name):
        """
        Paired t-test: are model1 and model2 significantly different on this dataset?
        
        Returns:
            t_stat, p_value, mean_diff
        """
        scores1 = np.array(self.results[model1][dataset_name])
        scores2 = np.array(self.results[model2][dataset_name])
        
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        mean_diff = np.mean(scores2) - np.mean(scores1)
        
        return t_stat, p_value, mean_diff
    
    def wilcoxon_test(self, model1, model2, dataset_name):
        """
        Wilcoxon signed-rank test (non-parametric alternative).
        """
        scores1 = np.array(self.results[model1][dataset_name])
        scores2 = np.array(self.results[model2][dataset_name])
        
        stat, p_value = stats.wilcoxon(scores1, scores2)
        mean_diff = np.mean(scores2) - np.mean(scores1)
        
        return stat, p_value, mean_diff
    
    def generate_comparison_table(self, model_pairs, datasets, test_type='ttest', alpha=0.05):
        """
        Generate significance table for multiple model pairs.
        
        Args:
            model_pairs: list of tuples [('Baseline_GIN', 'MoE_K4'), ...]
            datasets: list of dataset names
            test_type: 'ttest' or 'wilcoxon'
            alpha: significance threshold (default 0.05)
        
        Returns:
            DataFrame with p-values and significance markers
        """
        results_list = []
        
        for model1, model2 in model_pairs:
            row = {'Comparison': f"{model1} vs {model2}"}
            
            for ds in datasets:
                if test_type == 'ttest':
                    t_stat, p_val, mean_diff = self.paired_ttest(model1, model2, ds)
                else:
                    t_stat, p_val, mean_diff = self.wilcoxon_test(model1, model2, ds)
                
                sig_marker = '***' if p_val < alpha else ('*' if p_val < 0.1 else 'ns')
                row[ds] = f"{p_val:.4f} {sig_marker}"
            
            results_list.append(row)
        
        return pd.DataFrame(results_list)
    
    def print_detailed_comparison(self, model1, model2, datasets, test_type='ttest', alpha=0.05):
        """Print detailed pairwise comparison."""
        print(f"\n{'='*80}")
        print(f"  {test_type.upper()} COMPARISON: {model1} vs {model2}")
        print(f"{'='*80}")
        print(f"  {'Dataset':<12} {'Model1 Mean':>12} {'Model2 Mean':>12} "
              f"{'Diff':>10} {'p-value':>10} {'Sig':>6}")
        print(f"  {'-'*80}")
        
        significant_count = 0
        
        for ds in datasets:
            if test_type == 'ttest':
                t_stat, p_val, mean_diff = self.paired_ttest(model1, model2, ds)
            else:
                t_stat, p_val, mean_diff = self.wilcoxon_test(model1, model2, ds)
            
            m1_mean = np.mean(self.results[model1][ds])
            m2_mean = np.mean(self.results[model2][ds])
            sig_marker = '***' if p_val < alpha else ('*' if p_val < 0.1 else 'ns')
            
            if p_val < alpha:
                significant_count += 1
            
            print(f"  {ds:<12} {m1_mean:>12.4f} {m2_mean:>12.4f} "
                  f"{mean_diff:>10.4f} {p_val:>10.4f} {sig_marker:>6}")
        
        print(f"  {'-'*80}")
        print(f"  Significant at α={alpha}: {significant_count}/{len(datasets)}")
        print(f"  {'='*80}\n")
    
    def confidence_intervals(self, model_name, datasets, confidence=0.95):
        """
        Compute 95% CI for each model/dataset combination.
        """
        print(f"\n{'='*70}")
        print(f"  {confidence*100:.0f}% CONFIDENCE INTERVALS: {model_name}")
        print(f"{'='*70}")
        print(f"  {'Dataset':<12} {'Mean':>10} {'95% CI':>25}")
        print(f"  {'-'*50}")
        
        for ds in datasets:
            scores = np.array(self.results[model_name][ds])
            mean = np.mean(scores)
            sem = stats.sem(scores)  # standard error of mean
            ci = sem * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
            
            print(f"  {ds:<12} {mean:>10.4f} [{mean-ci:.4f}, {mean+ci:.4f}]")
        
        print(f"{'='*70}\n")


# ─ Example usage ─
if __name__ == '__main__':
    # Mock data (3 seeds for each model/dataset)
    seed_results = {
        'Baseline_GIN': {
            'ESOL': [0.95, 0.97, 0.93],
            'BACE': [0.80, 0.82, 0.78],
            'ClinTox': [0.85, 0.88, 0.83],
        },
        'MoE_K4': {
            'ESOL': [0.92, 0.94, 0.91],
            'BACE': [0.82, 0.84, 0.80],
            'ClinTox': [0.88, 0.90, 0.86],
        },
    }
    
    tester = StatisticalTester(seed_results)
    
    # Paired t-test
    print("Paired t-test:")
    t, p, diff = tester.paired_ttest('Baseline_GIN', 'MoE_K4', 'ESOL')
    print(f"  t={t:.3f}, p={p:.4f}, diff={diff:.4f}")
    
    # Detailed comparison
    tester.print_detailed_comparison('Baseline_GIN', 'MoE_K4',
                                      ['ESOL', 'BACE', 'ClinTox'])
    
    # Confidence intervals
    tester.confidence_intervals('Baseline_GIN', ['ESOL', 'BACE', 'ClinTox'])
    
    print("StatisticalTester module ready for integration.")
