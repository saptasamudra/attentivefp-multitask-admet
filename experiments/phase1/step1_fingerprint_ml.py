"""
Phase 1, Step 1: Fingerprint + ML Baseline
All 9 MoleculeNet datasets
Models: Random Forest + XGBoost
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
import os
warnings.filterwarnings('ignore')

# Dataset configurations
DATASETS = {
    # Regression tasks
    'ESOL': {
        'path': 'data/ESOL/esol/raw/delaney-processed.csv',
        'smiles_col': 'smiles',
        'label_cols': ['measured log solubility in mols per litre'],
        'task': 'regression'
    },
    'FreeSolv': {
        'path': 'data/FreeSolv/freesolv/raw/SAMPL.csv',
        'smiles_col': 'smiles',
        'label_cols': ['expt'],
        'task': 'regression'
    },
    'Lipo': {
        'path': 'data/Lipo/lipo/raw/Lipophilicity.csv',
        'smiles_col': 'smiles',
        'label_cols': ['exp'],
        'task': 'regression'
    },
    
    # Single-task classification
    'BACE': {
        'path': 'data/BACE/bace/raw/bace.csv',
        'smiles_col': 'mol',
        'label_cols': ['Class'],
        'task': 'classification'
    },
    'BBBP': {
        'path': 'data/BBBP/bbbp/raw/bbbp.csv',
        'smiles_col': 'smiles',
        'label_cols': ['p_np'],
        'task': 'classification'
    },
    'HIV': {
        'path': 'data/HIV/hiv/raw/HIV.csv',
        'smiles_col': 'smiles',
        'label_cols': ['HIV_active'],
        'task': 'classification'
    },
    
    # Multi-task classification
    'ClinTox': {
        'path': 'data/ClinTox/clintox/raw/clintox.csv',
        'smiles_col': 'smiles',
        'label_cols': ['FDA_APPROVED', 'CT_TOX'],
        'task': 'classification'
    },
    'Tox21': {
        'path': 'data/Tox21/tox21/raw/tox21.csv',
        'smiles_col': 'smiles',
        'label_cols': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
                       'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
                       'SR-HSE', 'SR-MMP', 'SR-p53'],
        'task': 'classification'
    },
    'SIDER': {
        'path': 'data/SIDER/sider/raw/sider.csv',
        'smiles_col': 'smiles',
        'label_cols': None,  # Will use all columns except smiles
        'task': 'classification'
    }
}

def compute_fingerprints(smiles_list):
    """Convert SMILES to ECFP4 fingerprints"""
    fps = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(np.array(fp))
            valid_indices.append(i)
    return np.array(fps), valid_indices

def train_single_task(X_train, X_test, y_train, y_test, task_type, model_type='RF'):
    """Train and evaluate single task"""
    if task_type == 'classification':
        if model_type == 'RF':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss')
        
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        
        # Handle cases where test set has only one class
        if len(np.unique(y_test)) < 2:
            return np.nan
        
        return roc_auc_score(y_test, y_pred)
    
    else:  # regression
        if model_type == 'RF':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return np.sqrt(mean_squared_error(y_test, y_pred))

# Main execution
print("="*70)
print("Phase 1 Step 1: Fingerprint + ML Baseline (All 9 Datasets)")
print("="*70)

all_results = {}

for dataset_name, config in DATASETS.items():
    print(f"\n{'='*70}")
    print(f"Processing {dataset_name}...")
    print(f"{'='*70}")
    
    # Load data
    df = pd.read_csv(config['path'])
    
    # Get label columns
    if config['label_cols'] is None:
        label_cols = [c for c in df.columns if c != config['smiles_col']]
    else:
        label_cols = config['label_cols']
    
    print(f"Loaded: {len(df)} molecules, {len(label_cols)} task(s)")
    
    # Compute fingerprints
    print("Computing ECFP4 fingerprints...")
    smiles = df[config['smiles_col']].values
    X, valid_indices = compute_fingerprints(smiles)
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    print(f"Valid: {len(X)} molecules")
    
    # Process each task
    task_scores = {'RF': [], 'XGB': []}
    
    for task_col in label_cols:
        y = df_valid[task_col].values
        
        # Remove NaN labels (common in multi-task datasets)
        mask = ~pd.isna(y)
        if mask.sum() == 0:
            continue
        
        X_task = X[mask]
        y_task = y[mask]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_task, y_task, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Train RF
        score_rf = train_single_task(X_train, X_test, y_train, y_test, 
                                     config['task'], 'RF')
        if not np.isnan(score_rf):
            task_scores['RF'].append(score_rf)
        
        # Train XGB
        score_xgb = train_single_task(X_train, X_test, y_train, y_test, 
                                      config['task'], 'XGB')
        if not np.isnan(score_xgb):
            task_scores['XGB'].append(score_xgb)
    
    # Average across tasks
    results = {
        'RF': np.mean(task_scores['RF']) if task_scores['RF'] else np.nan,
        'XGB': np.mean(task_scores['XGB']) if task_scores['XGB'] else np.nan,
        'n_tasks': len(label_cols),
        'valid_tasks': len(task_scores['RF'])
    }
    all_results[dataset_name] = results
    
    # Print results
    metric = "AUC-ROC" if config['task'] == 'classification' else "RMSE"
    print(f"\nResults (mean {metric} across {results['valid_tasks']} tasks):")
    print(f"  Random Forest:  {results['RF']:.4f}")
    print(f"  XGBoost:        {results['XGB']:.4f}")

# Summary table
print("\n" + "="*70)
print("SUMMARY: Fingerprint + ML Baseline Results")
print("="*70)
print(f"{'Dataset':<12} {'Task':<15} {'RF':<10} {'XGB':<10} {'Metric'}")
print("-"*70)

for dataset_name, config in DATASETS.items():
    task = config['task']
    metric = "AUC" if task == 'classification' else "RMSE"
    rf_score = all_results[dataset_name]['RF']
    xgb_score = all_results[dataset_name]['XGB']
    print(f"{dataset_name:<12} {task:<15} {rf_score:<10.4f} {xgb_score:<10.4f} {metric}")

# Save results
os.makedirs('results/phase1/step1', exist_ok=True)
with open('results/phase1/step1/all9_results.txt', 'w') as f:
    f.write("Fingerprint + ML Baseline Results (All 9 Datasets)\n")
    f.write("="*70 + "\n\n")
    for dataset_name, config in DATASETS.items():
        task = config['task']
        metric = "AUC-ROC" if task == 'classification' else "RMSE"
        rf_score = all_results[dataset_name]['RF']
        xgb_score = all_results[dataset_name]['XGB']
        n_tasks = all_results[dataset_name]['n_tasks']
        f.write(f"{dataset_name} ({task}, {n_tasks} task(s)):\n")
        f.write(f"  RF:  {metric} = {rf_score:.4f}\n")
        f.write(f"  XGB: {metric} = {xgb_score:.4f}\n\n")

print("\n✓ Results saved to results/phase1/step1/all9_results.txt")
print("="*70)