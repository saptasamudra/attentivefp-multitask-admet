"""
prepare_grover_data.py — Convert MoleculeNet datasets to GROVER CSV format
Run from: D:\molprop_project (moe_admet env is fine for this)
Run: python prepare_grover_data.py
"""

import os
import numpy as np
import pandas as pd
from torch_geometric.datasets import MoleculeNet
import warnings
warnings.filterwarnings("ignore")

DATA_ROOT  = "./data"
GROVER_DIR = "./grover_data"
os.makedirs(GROVER_DIR, exist_ok=True)

DATASETS = [
    {"name": "BBBP",    "tasks": ["p_np"],           "type": "classif"},
    {"name": "BACE",    "tasks": ["Class"],           "type": "classif"},
    {"name": "Tox21",   "tasks": ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase","NR-ER","NR-ER-LBD",
                                   "NR-PPAR-gamma","SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"], "type": "classif"},
    {"name": "SIDER",   "tasks": [f"task_{i}" for i in range(27)],  "type": "classif"},
    {"name": "ClinTox", "tasks": ["FDA_APPROVED","CT_TOX"],          "type": "classif"},
    {"name": "HIV",     "tasks": ["HIV_active"],                     "type": "classif"},
    {"name": "ESOL",    "tasks": ["measured log solubility in mols per litre"], "type": "regr"},
    {"name": "FreeSolv","tasks": ["expt"],                           "type": "regr"},
    {"name": "Lipo",    "tasks": ["exp"],                            "type": "regr"},
]

def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict
    scaffolds = defaultdict(list)
    for i in range(len(dataset)):
        try:
            smi = dataset.smiles[i]
            mol = Chem.MolFromSmiles(smi)
            sc  = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else smi
        except:
            sc = str(i)
        scaffolds[sc].append(i)
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(dataset)
    train_cutoff = int(n * frac_train)
    val_cutoff   = int(n * (frac_train + frac_val))
    train_idx, val_idx, test_idx = [], [], []
    for s in scaffold_sets:
        if len(train_idx) < train_cutoff: train_idx.extend(s)
        elif len(val_idx) < (val_cutoff - train_cutoff): val_idx.extend(s)
        else: test_idx.extend(s)
    return train_idx, val_idx, test_idx

for ds in DATASETS:
    name   = ds["name"]
    tasks  = ds["tasks"]
    print(f"Processing {name}...")

    dataset = MoleculeNet(root=DATA_ROOT, name=name)
    smiles  = dataset.smiles
    labels  = dataset.data.y.numpy()

    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    # Build dataframe
    df = pd.DataFrame({"smiles": smiles})
    for t, task in enumerate(tasks):
        if t < labels.shape[1]:
            df[task] = labels[:, t]

    # Scaffold split
    train_idx, val_idx, test_idx = scaffold_split(dataset)

    ds_dir = os.path.join(GROVER_DIR, name)
    os.makedirs(ds_dir, exist_ok=True)

    df.iloc[train_idx].to_csv(f"{ds_dir}/train.csv", index=False)
    df.iloc[val_idx].to_csv(f"{ds_dir}/val.csv",   index=False)
    df.iloc[test_idx].to_csv(f"{ds_dir}/test.csv",  index=False)
    df.to_csv(f"{ds_dir}/full.csv", index=False)

    print(f"  {name}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"  Saved to {ds_dir}/")

print("\nAll datasets prepared!")
print(f"Now run feature extraction from grover directory.")
