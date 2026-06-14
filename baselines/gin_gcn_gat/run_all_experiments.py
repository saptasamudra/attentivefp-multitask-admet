"""
Simple Python script to run all GNN baseline experiments sequentially
No batch file needed - pure Python

Author: Sapta
Date: 2026-04-01
"""

import os
import subprocess
import time
from datetime import datetime

# Configuration
MODELS = ['GIN', 'GCN', 'GAT']
DATASETS = ['ESOL', 'FreeSolv', 'Lipo', 'BACE', 'BBBP', 'HIV', 'ClinTox', 'Tox21', 'SIDER']
SEEDS = [0, 1, 2, 3, 4]
DATA_DIR = r'D:\molprop_project\temp\chemprop_data'
# CHANGE THIS to match your data location
DATA_DIR = r'D:\molprop_project\temp\chemprop_data'

# Output directories
SAVE_DIR = r'baselines\saved_models'
RESULTS_FILE = r'baselines\results\gnn_results.json'
LOG_DIR = r'logs'

# Create directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def run_experiment(model, dataset, seed):
    """Run single experiment"""
    
    # Build command
    cmd = [
        'python', 'gnn_baselines.py',
        '--model', model,
        '--dataset', dataset,
        '--seed', str(seed),
        '--data_dir', DATA_DIR,
        '--save_dir', SAVE_DIR,
        '--results_file', RESULTS_FILE,
        '--hidden_dim', '300',
        '--num_layers', '3',
        '--dropout', '0.0',
        '--epochs', '100',
        '--batch_size', '64',
        '--lr', '1e-3',
        '--patience', '20'
    ]
    
    # Log file
    log_file = os.path.join(LOG_DIR, f'{model}_{dataset}_seed{seed}.log')
    
    print(f"\n{'='*80}")
    print(f"Running: {model} on {dataset} (seed {seed})")
    print(f"Log: {log_file}")
    print(f"{'='*80}")
    
    # Run experiment
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        if result.returncode == 0:
            print(f"✓ SUCCESS: {model} on {dataset} (seed {seed})")
            return True
        else:
            print(f"✗ FAILED: {model} on {dataset} (seed {seed})")
            print(f"  Check log: {log_file}")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {model} on {dataset} (seed {seed})")
        print(f"  {str(e)}")
        return False

def main():
    """Run all experiments"""
    
    print("\n" + "="*80)
    print("GNN BASELINE EXPERIMENTS - Sequential Execution")
    print("="*80)
    print(f"\nTotal experiments: {len(MODELS) * len(DATASETS) * len(SEEDS)}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Seeds: {', '.join(map(str, SEEDS))}")
    print(f"\nData directory: {DATA_DIR}")
    print(f"Results will be saved to: {RESULTS_FILE}")
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    input("\nPress Enter to start, or Ctrl+C to cancel...")
    
    # Track progress
    total_jobs = len(MODELS) * len(DATASETS) * len(SEEDS)
    completed_jobs = 0
    failed_jobs = 0
    start_time = time.time()
    
    # Run all experiments
    for model in MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                completed_jobs += 1
                
                print(f"\n[{completed_jobs}/{total_jobs}] {model} | {dataset} | seed {seed}")
                
                success = run_experiment(model, dataset, seed)
                
                if not success:
                    failed_jobs += 1
                
                # Estimate remaining time
                if completed_jobs > 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed_jobs
                    remaining = avg_time * (total_jobs - completed_jobs)
                    
                    print(f"\nProgress: {completed_jobs}/{total_jobs} "
                          f"({completed_jobs/total_jobs*100:.1f}%)")
                    print(f"Elapsed: {elapsed/3600:.2f} hours")
                    print(f"Estimated remaining: {remaining/3600:.2f} hours")
    
    # Final summary
    elapsed_total = time.time() - start_time
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"Total jobs: {total_jobs}")
    print(f"Successful: {completed_jobs - failed_jobs}")
    print(f"Failed: {failed_jobs}")
    print(f"Total time: {elapsed_total/3600:.2f} hours")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {RESULTS_FILE}")
    
    if failed_jobs > 0:
        print(f"\n⚠ {failed_jobs} jobs failed. Check logs in: {LOG_DIR}")
    
    print("="*80)
    
    input("\nPress Enter to exit...")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiments cancelled by user.")
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        input("Press Enter to exit...")
