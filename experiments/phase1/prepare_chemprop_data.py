"""
Prepare CSVs for Chemprop (SMILES must be first column)
"""
import pandas as pd
import os

# Dataset configurations
DATASETS = {
    'ESOL': {
        'input': 'data/ESOL/esol/raw/delaney-processed.csv',
        'smiles_col': 'smiles',
        'target_cols': ['measured log solubility in mols per litre'],
    },
    'FreeSolv': {
        'input': 'data/FreeSolv/freesolv/raw/SAMPL.csv',
        'smiles_col': 'smiles',
        'target_cols': ['expt'],
    },
    'Lipo': {
        'input': 'data/Lipo/lipo/raw/Lipophilicity.csv',
        'smiles_col': 'smiles',
        'target_cols': ['exp'],
    },
    'BACE': {
        'input': 'data/BACE/bace/raw/bace.csv',
        'smiles_col': 'mol',
        'target_cols': ['Class'],
    },
    'BBBP': {
        'input': 'data/BBBP/bbbp/raw/bbbp.csv',
        'smiles_col': 'smiles',
        'target_cols': ['p_np'],
    },
    'HIV': {
        'input': 'data/HIV/hiv/raw/HIV.csv',
        'smiles_col': 'smiles',
        'target_cols': ['HIV_active'],
    },
    'ClinTox': {
        'input': 'data/ClinTox/clintox/raw/clintox.csv',
        'smiles_col': 'smiles',
        'target_cols': ['FDA_APPROVED', 'CT_TOX'],
    },
    'Tox21': {
        'input': 'data/Tox21/tox21/raw/tox21.csv',
        'smiles_col': 'smiles',
        'target_cols': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
                        'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
                        'SR-HSE', 'SR-MMP', 'SR-p53'],
    },
    'SIDER': {
        'input': 'data/SIDER/sider/raw/sider.csv',
        'smiles_col': 'smiles',
        'target_cols': None,  # Use all columns except smiles
    }
}

os.makedirs('temp/chemprop_data', exist_ok=True)

print("Preparing Chemprop-compatible CSVs...")
print("="*60)

for dataset_name, config in DATASETS.items():
    print(f"\n{dataset_name}...")
    
    # Load data
    df = pd.read_csv(config['input'])
    
    # Get target columns
    if config['target_cols'] is None:
        target_cols = [c for c in df.columns if c != config['smiles_col']]
    else:
        target_cols = config['target_cols']
    
    # Reorder: smiles first, then targets
    cols = [config['smiles_col']] + target_cols
    df_chemprop = df[cols].copy()
    
    # Rename smiles column to 'smiles' (Chemprop standard)
    df_chemprop.rename(columns={config['smiles_col']: 'smiles'}, inplace=True)
    
    # Save
    output_path = f"temp/chemprop_data/{dataset_name}.csv"
    df_chemprop.to_csv(output_path, index=False)
    
    print(f"  ✓ Saved: {output_path}")
    print(f"    Shape: {df_chemprop.shape}, Columns: {df_chemprop.columns.tolist()[:3]}...")

print("\n" + "="*60)
print("✓ All CSVs prepared for Chemprop!")
print("Files saved in: temp/chemprop_data/")