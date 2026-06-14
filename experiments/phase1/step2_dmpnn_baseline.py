"""
Phase 1, Step 2: D-MPNN Baseline using Chemprop
All 9 MoleculeNet datasets with scaffold split
"""
import os
import pandas as pd
import numpy as np
from chemprop.train import make_predictions
from chemprop.data import get_data, get_task_names, MoleculeDataLoader, MoleculeDataset
from chemprop.data.utils import get_data, split_data
from chemprop.utils import build_optimizer, build_lr_scheduler, makedirs
from chemprop.nn_utils import param_count
from chemprop.train import predict
from chemprop.args import TrainArgs, PredictArgs
import torch
import warnings
warnings.filterwarnings('ignore')

# Dataset configurations
DATASETS = {
    'ESOL': {
        'path': 'data/ESOL/esol/raw/delaney-processed.csv',
        'smiles_col': 'smiles',
        'target_cols': ['measured log solubility in mols per litre'],
        'task_type': 'regression'
    },
    'FreeSolv': {
        'path': 'data/FreeSolv/freesolv/raw/SAMPL.csv',
        'smiles_col': 'smiles',
        'target_cols': ['expt'],
        'task_type': 'regression'
    },
    'Lipo': {
        'path': 'data/Lipo/lipo/raw/Lipophilicity.csv',
        'smiles_col': 'smiles',
        'target_cols': ['exp'],
        'task_type': 'regression'
    },
    'BACE': {
        'path': 'data/BACE/bace/raw/bace.csv',
        'smiles_col': 'mol',
        'target_cols': ['Class'],
        'task_type': 'classification'
    },
    'BBBP': {
        'path': 'data/BBBP/bbbp/raw/bbbp.csv',
        'smiles_col': 'smiles',
        'target_cols': ['p_np'],
        'task_type': 'classification'
    },
    'HIV': {
        'path': 'data/HIV/hiv/raw/HIV.csv',
        'smiles_col': 'smiles',
        'target_cols': ['HIV_active'],
        'task_type': 'classification'
    },
    'ClinTox': {
        'path': 'data/ClinTox/clintox/raw/clintox.csv',
        'smiles_col': 'smiles',
        'target_cols': ['FDA_APPROVED', 'CT_TOX'],
        'task_type': 'classification'
    },
    'Tox21': {
        'path': 'data/Tox21/tox21/raw/tox21.csv',
        'smiles_col': 'smiles',
        'target_cols': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
                        'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
                        'SR-HSE', 'SR-MMP', 'SR-p53'],
        'task_type': 'classification'
    }
}

def prepare_chemprop_csv(df, smiles_col, target_cols, output_path):
    """Prepare CSV in Chemprop format: smiles, target1, target2, ..."""
    data = df[[smiles_col] + target_cols].copy()
    data.columns = ['smiles'] + target_cols
    data.to_csv(output_path, index=False)
    return output_path

print("="*70)
print("Phase 1 Step 2: D-MPNN Baseline (Chemprop)")
print("="*70)

# Create temp directory for Chemprop CSVs
os.makedirs('temp/chemprop_data', exist_ok=True)
os.makedirs('temp/chemprop_models', exist_ok=True)
os.makedirs('results/phase1/step2', exist_ok=True)

all_results = {}

for dataset_name, config in DATASETS.items():
    print(f"\n{'='*70}")
    print(f"Processing {dataset_name}...")
    print(f"{'='*70}")
    
    # Load and prepare data
    df = pd.read_csv(config['path'])
    csv_path = f"temp/chemprop_data/{dataset_name}.csv"
    prepare_chemprop_csv(df, config['smiles_col'], config['target_cols'], csv_path)
    
    print(f"Dataset: {len(df)} molecules, {len(config['target_cols'])} task(s)")
    print(f"Running D-MPNN with scaffold split...")
    
    # Run Chemprop training via command line (easiest approach)
    save_dir = f"temp/chemprop_models/{dataset_name}"
    
    cmd = (
        f"chemprop_train "
        f"--data_path {csv_path} "
        f"--dataset_type {config['task_type']} "
        f"--save_dir {save_dir} "
        f"--split_type scaffold_balanced "
        f"--epochs 30 "
        f"--quiet "
    )
    
    if config['task_type'] == 'classification':
        cmd += "--metric auc "
    else:
        cmd += "--metric rmse "
    
    # Run training
    print(f"Command: {cmd}")
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print(f"✓ Training complete for {dataset_name}")
        
        # Read results from Chemprop's output
        # Results are saved in save_dir/test_scores.csv
        try:
            scores_df = pd.read_csv(f"{save_dir}/test_scores.csv")
            print(scores_df)
            all_results[dataset_name] = scores_df
        except:
            print(f"Could not read results for {dataset_name}")
    else:
        print(f"✗ Training failed for {dataset_name}")

print("\n" + "="*70)
print("D-MPNN Baseline Complete!")
print("="*70)