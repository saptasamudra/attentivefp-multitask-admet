"""
GNN Baseline Results Aggregation Script
Computes mean ± std across seeds for comparison with D-MPNN

Following Professor's Phase 1 Requirements:
- Statistical significance testing (5 seeds minimum)
- Formatted output matching D-MPNN baseline table

Author: Sapta (林恩)
Date: 2026-04-01
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse


def aggregate_results(results_file, output_csv=None):
    """Aggregate results across seeds"""
    
    # Load results
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    # Group by model and dataset
    grouped = defaultdict(lambda: defaultdict(list))
    
    for result in all_results:
        model = result['model']
        dataset = result['dataset']
        test_metric = result['test']
        grouped[model][dataset].append(test_metric)
    
    # Compute statistics
    summary = []
    
    for model in sorted(grouped.keys()):
        for dataset in sorted(grouped[model].keys()):
            metrics = np.array(grouped[model][dataset])
            
            # Find metric name
            for r in all_results:
                if r['model'] == model and r['dataset'] == dataset:
                    metric_name = r['metric']
                    task_type = r['task_type']
                    break
            
            summary.append({
                'Model': model,
                'Dataset': dataset,
                'Task': task_type,
                'Metric': metric_name,
                'Mean': np.mean(metrics),
                'Std': np.std(metrics),
                'Seeds': len(metrics),
                'Raw': metrics.tolist()
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(summary)
    
    # Print formatted results
    print("\n" + "="*100)
    print("GNN BASELINE RESULTS - Aggregated Across Seeds")
    print("="*100)
    print()
    
    for model in sorted(df['Model'].unique()):
        model_df = df[df['Model'] == model]
        
        print(f"\n{'='*100}")
        print(f"Model: {model}")
        print(f"{'='*100}")
        print(f"{'Dataset':<15} {'Task':<15} {'Metric':<10} {'Mean':<12} {'Std':<12} {'Seeds':<8}")
        print(f"{'-'*100}")
        
        for _, row in model_df.iterrows():
            print(f"{row['Dataset']:<15} {row['Task']:<15} {row['Metric']:<10} "
                  f"{row['Mean']:<12.4f} {row['Std']:<12.4f} {row['Seeds']:<8}")
        
        print()
    
    # Regression summary table
    print("\n" + "="*100)
    print("REGRESSION TASKS SUMMARY")
    print("="*100)
    reg_df = df[df['Task'] == 'regression'].copy()
    reg_pivot = reg_df.pivot_table(
        index='Dataset', 
        columns='Model', 
        values='Mean', 
        aggfunc='first'
    )
    print(reg_pivot.to_string())
    print()
    
    # Classification summary table
    print("\n" + "="*100)
    print("CLASSIFICATION TASKS SUMMARY")
    print("="*100)
    cls_df = df[df['Task'] == 'classification'].copy()
    cls_pivot = cls_df.pivot_table(
        index='Dataset',
        columns='Model',
        values='Mean',
        aggfunc='first'
    )
    print(cls_pivot.to_string())
    print()
    
    # Seed breakdown for each experiment
    print("\n" + "="*100)
    print("SEED BREAKDOWN (for statistical significance testing)")
    print("="*100)
    
    for _, row in df.iterrows():
        seeds_str = " | ".join([f"{x:.4f}" for x in row['Raw']])
        print(f"{row['Model']:<10} {row['Dataset']:<15} {seeds_str}")
    
    print()
    
    # Save to CSV if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")
        print()
    
    # Compare with D-MPNN (if available)
    print("\n" + "="*100)
    print("COMPARISON WITH D-MPNN BASELINE")
    print("="*100)
    print()
    print("D-MPNN results (from your completed runs):")
    print("  ESOL:     RMSE 0.962 ± 0.069")
    print("  FreeSolv: RMSE 2.802 ± 0.182")
    print("  Lipo:     RMSE 0.812 ± 0.016")
    print("  BACE:     AUC  0.791 ± 0.031")
    print("  BBBP:     AUC  0.879 ± 0.033")
    print("  HIV:      AUC  0.781 ± 0.024")
    print("  ClinTox:  AUC  0.922 ± 0.014")
    print("  Tox21:    AUC  0.810 ± 0.024")
    print("  SIDER:    AUC  0.600 ± 0.000")
    print()
    print("Compare these with your GNN results above.")
    print("="*100)
    print()
    
    return df


def check_statistical_significance(df, model1, model2, dataset, alpha=0.05):
    """Perform paired t-test between two models on same dataset"""
    from scipy import stats
    
    # Get raw metrics
    metrics1 = df[(df['Model'] == model1) & (df['Dataset'] == dataset)]['Raw'].values[0]
    metrics2 = df[(df['Model'] == model2) & (df['Dataset'] == dataset)]['Raw'].values[0]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(metrics1, metrics2)
    
    significant = p_value < alpha
    
    print(f"\nPaired t-test: {model1} vs {model2} on {dataset}")
    print(f"  {model1}: {np.mean(metrics1):.4f} ± {np.std(metrics1):.4f}")
    print(f"  {model2}: {np.mean(metrics2):.4f} ± {np.std(metrics2):.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant at α={alpha}: {'YES' if significant else 'NO'}")
    
    return significant, p_value


def main():
    parser = argparse.ArgumentParser(description='Aggregate GNN baseline results')
    parser.add_argument('--results_file', type=str, 
                       default='./baselines/results/gnn_results.json',
                       help='Path to results JSON file')
    parser.add_argument('--output_csv', type=str,
                       default='./baselines/results/gnn_summary.csv',
                       help='Path to save summary CSV')
    parser.add_argument('--stat_test', action='store_true',
                       help='Run statistical significance tests')
    
    args = parser.parse_args()
    
    # Aggregate results
    df = aggregate_results(args.results_file, args.output_csv)
    
    # Statistical tests (if requested)
    if args.stat_test:
        print("\n" + "="*100)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*100)
        
        # Example: Test GIN vs GCN on BBBP
        if len(df[df['Dataset'] == 'BBBP']) >= 2:
            models = df['Model'].unique()
            if len(models) >= 2:
                try:
                    check_statistical_significance(df, models[0], models[1], 'BBBP')
                except:
                    print("Not enough data for statistical test")
        
        print()


if __name__ == '__main__':
    main()
