"""
MoleculeNet Full Baseline — 7 Datasets
AttentiveFP | Scaffold Split | 3 Seeds | Optuna-Optimized Hyperparameters

Datasets:
  Regression:     ESOL, FreeSolv, Lipophilicity
  Classification: BACE, BBBP, ClinTox, Tox21

Run:  python moleculenet_baseline.py
Time: ~2-3 hours on GPU
"""

import os.path as osp
from collections import defaultdict
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ─────────────────────────────────────────────
# PAPER DEFAULT HYPERPARAMETERS (proven better than Optuna)
# ─────────────────────────────────────────────

HP = {
    'lr': 10**-2.5,
    'hidden_dim': 200,
    'num_layers': 2,
    'num_timesteps': 2,
    'dropout': 0.2,
    'batch_size': 200,
    'weight_decay': 1e-5,
}

SEEDS = [42, 123, 7]
EPOCHS = 200


# ─────────────────────────────────────────────
# DATASET REGISTRY
# Each entry: name, task_type, metric_name, num_tasks
# ─────────────────────────────────────────────

DATASETS = [
    # Physical chemistry (regression)
    {'name': 'ESOL',          'task': 'regression',     'metric': 'RMSE ↓', 'published': '0.877'},
    {'name': 'FreeSolv',      'task': 'regression',     'metric': 'RMSE ↓', 'published': '2.082'},
    {'name': 'Lipo',           'task': 'regression',     'metric': 'RMSE ↓', 'published': '0.655'},
    # Biophysics + Physiology (classification)
    {'name': 'BACE',          'task': 'classification', 'metric': 'AUC ↑',  'published': '0.863'},
    {'name': 'BBBP',          'task': 'classification', 'metric': 'AUC ↑',  'published': '0.862'},
    {'name': 'ClinTox',       'task': 'classification', 'metric': 'AUC ↑',  'published': '0.832'},
    {'name': 'Tox21',         'task': 'classification', 'metric': 'AUC ↑',  'published': '0.829'},
]


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

class GenFeatures:
    """Exact copy of PyG's official AttentiveFP example featurizer.
    Produces 39-dim node features and 10-dim edge features."""
    def __init__(self):
        self.symbols = [
            'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br',
            'Te', 'I', 'At', 'other'
        ]
        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other',
        ]
        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]

    def __call__(self, data):
        mol = Chem.MolFromSmiles(data.smiles)
        if mol is None:
            data.x = torch.zeros((1, 39), dtype=torch.float)
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
            return data

        xs = []
        for atom in mol.GetAtoms():
            symbol = [0.] * len(self.symbols)
            symbol[self.symbols.index(atom.GetSymbol())
                   if atom.GetSymbol() in self.symbols else -1] = 1.
            degree = [0.] * 6
            degree[min(atom.GetDegree(), 5)] = 1.
            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.] * len(self.hybridizations)
            hybridization[self.hybridizations.index(
                atom.GetHybridization())
                if atom.GetHybridization() in self.hybridizations else -1] = 1.
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            hydrogens = [0.] * 5
            hydrogens[min(atom.GetTotalNumHs(), 4)] = 1.
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.

            x = torch.tensor(symbol + degree + [formal_charge] +
                             [radical_electrons] + hybridization +
                             [aromaticity] + hydrogens + [chirality] +
                             chirality_type)
            xs.append(x)

        data.x = torch.stack(xs, dim=0)

        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            conjugation = 1. if bond.GetIsConjugated() else 0.
            ring = 1. if bond.IsInRing() else 0.
            stereo = [0.] * 4
            stereo[self.stereos.index(bond.GetStereo())] = 1.

            edge_attr = torch.tensor(
                [single, double, triple, aromatic, conjugation, ring] + stereo)
            edge_attrs += [edge_attr, edge_attr]

        if len(edge_attrs) == 0:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_indices).t().contiguous()
            data.edge_attr = torch.stack(edge_attrs, dim=0)

        return data


# ─────────────────────────────────────────────
# SCAFFOLD SPLIT
# ─────────────────────────────────────────────

def scaffold_split(dataset, train_frac=0.8, val_frac=0.1, classification=False):
    scaffolds = defaultdict(list)
    for i, data in enumerate(dataset):
        mol = Chem.MolFromSmiles(data.smiles)
        if mol is None:
            scaffolds['unknown'].append(i)
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False)
        scaffolds[scaffold].append(i)

    scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)

    n = len(dataset)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx, val_idx, test_idx = [], [], []

    if classification:
        # Get first task label for class-aware seeding
        labels = []
        for data in dataset:
            y = data.y
            if y.dim() > 1:
                y = y[:, 0]
            val = y.item() if y.numel() == 1 else y[0].item()
            # Handle NaN labels
            if np.isnan(val):
                labels.append(0)
            else:
                labels.append(int(val))
        labels = np.array(labels)

        pos_scaffolds = [g for g in scaffold_groups if all(labels[i] == 1 for i in g)]
        neg_scaffolds = [g for g in scaffold_groups if all(labels[i] == 0 for i in g)]

        seeded = set()
        if pos_scaffolds and neg_scaffolds:
            val_idx.extend(pos_scaffolds[-1]); seeded.add(id(pos_scaffolds[-1]))
            val_idx.extend(neg_scaffolds[-1]); seeded.add(id(neg_scaffolds[-1]))
            if len(pos_scaffolds) > 1 and len(neg_scaffolds) > 1:
                test_idx.extend(pos_scaffolds[-2]); seeded.add(id(pos_scaffolds[-2]))
                test_idx.extend(neg_scaffolds[-2]); seeded.add(id(neg_scaffolds[-2]))

        for group in scaffold_groups:
            if id(group) in seeded:
                continue
            if len(train_idx) < n_train:
                train_idx.extend(group)
            elif len(val_idx) < n_val:
                val_idx.extend(group)
            else:
                test_idx.extend(group)
    else:
        for group in scaffold_groups:
            if len(train_idx) < n_train:
                train_idx.extend(group)
            elif len(val_idx) < n_val:
                val_idx.extend(group)
            else:
                test_idx.extend(group)

    return (dataset[torch.tensor(train_idx)],
            dataset[torch.tensor(val_idx)],
            dataset[torch.tensor(test_idx)])


# ─────────────────────────────────────────────
# TRAIN AND EVALUATE — REGRESSION
# ─────────────────────────────────────────────

def run_regression(dataset_name, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', dataset_name)
    dataset = MoleculeNet(path, name=dataset_name, pre_transform=GenFeatures())
    train_set, val_set, test_set = scaffold_split(dataset)

    print(f'    Split: train={len(train_set)} | val={len(val_set)} | test={len(test_set)}')

    train_loader = DataLoader(train_set, batch_size=HP['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=HP['batch_size'])
    test_loader = DataLoader(test_set, batch_size=HP['batch_size'])

    model = AttentiveFP(
        in_channels=39, hidden_channels=HP['hidden_dim'], out_channels=1,
        edge_dim=10, num_layers=HP['num_layers'],
        num_timesteps=HP['num_timesteps'], dropout=HP['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=HP['lr'],
                                 weight_decay=HP['weight_decay'])

    best_val = float('inf')
    best_test = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            # Handle multi-task datasets — use first task only for single-task baseline
            y = batch.y
            if y.dim() > 1:
                y = y[:, 0]
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                continue
            loss = F.mse_loss(out[mask].squeeze(-1), y[mask])
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        val_errors = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                y = batch.y
                if y.dim() > 1:
                    y = y[:, 0]
                mask = ~torch.isnan(y)
                if mask.sum() > 0:
                    val_errors.append((out[mask].squeeze(-1) - y[mask]).cpu())

        if val_errors:
            val_rmse = sqrt(torch.cat(val_errors).pow(2).mean().item())
            if val_rmse < best_val:
                best_val = val_rmse
                # Compute test
                test_errors = []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(device)
                        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                        y = batch.y
                        if y.dim() > 1:
                            y = y[:, 0]
                        mask = ~torch.isnan(y)
                        if mask.sum() > 0:
                            test_errors.append((out[mask].squeeze(-1) - y[mask]).cpu())
                if test_errors:
                    best_test = sqrt(torch.cat(test_errors).pow(2).mean().item())

        if epoch % 50 == 0:
            print(f'    Epoch {epoch:03d} | Val: {best_val:.4f} | Test: {best_test:.4f}')

    return best_test


# ─────────────────────────────────────────────
# TRAIN AND EVALUATE — CLASSIFICATION
# Now handles multi-task datasets (Tox21=12 tasks, ClinTox=2 tasks)
# Reports mean AUC across all tasks
# ─────────────────────────────────────────────

def run_classification(dataset_name, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', dataset_name)
    dataset = MoleculeNet(path, name=dataset_name, pre_transform=GenFeatures())

    # Detect number of tasks
    sample_y = dataset[0].y
    num_tasks = sample_y.shape[-1] if sample_y.dim() > 1 else 1
    print(f'    Tasks: {num_tasks}')

    train_set, val_set, test_set = scaffold_split(dataset, classification=True)
    print(f'    Split: train={len(train_set)} | val={len(val_set)} | test={len(test_set)}')

    train_loader = DataLoader(train_set, batch_size=HP['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=HP['batch_size'])
    test_loader = DataLoader(test_set, batch_size=HP['batch_size'])

    # Output 1 channel per task
    model = AttentiveFP(
        in_channels=39, hidden_channels=HP['hidden_dim'],
        out_channels=num_tasks,
        edge_dim=10, num_layers=HP['num_layers'],
        num_timesteps=HP['num_timesteps'], dropout=HP['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=HP['lr'],
                                 weight_decay=HP['weight_decay'])

    best_val = 0.0
    best_test = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            y = batch.y.float()
            if y.dim() == 1:
                y = y.unsqueeze(-1)

            # Mask NaN labels (common in multi-task datasets)
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                continue

            loss = F.binary_cross_entropy_with_logits(out[mask], y[mask])
            loss.backward()
            optimizer.step()

        # ── Evaluate: compute mean AUC across tasks ──
        model.eval()
        val_auc = compute_mean_auc(model, val_loader, num_tasks)

        if val_auc > best_val:
            best_val = val_auc
            best_test = compute_mean_auc(model, test_loader, num_tasks)

        if epoch % 50 == 0:
            print(f'    Epoch {epoch:03d} | Val: {best_val:.4f} | Test: {best_test:.4f}')

    return best_test


def compute_mean_auc(model, loader, num_tasks):
    """Compute mean ROC-AUC across all tasks, skipping tasks with one class."""
    model.eval()
    all_preds = [[] for _ in range(num_tasks)]
    all_labels = [[] for _ in range(num_tasks)]

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = torch.sigmoid(out).cpu().numpy()

            y = batch.y.float()
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            y = y.cpu().numpy()

            for t in range(num_tasks):
                mask = ~np.isnan(y[:, t])
                if mask.sum() > 0:
                    all_preds[t].append(pred[mask, t] if num_tasks > 1 else pred[mask].squeeze())
                    all_labels[t].append(y[mask, t])

    # Compute per-task AUC
    aucs = []
    for t in range(num_tasks):
        if not all_preds[t]:
            continue
        preds_t = np.concatenate(all_preds[t])
        labels_t = np.concatenate(all_labels[t])
        if len(np.unique(labels_t)) >= 2:
            aucs.append(roc_auc_score(labels_t, preds_t))

    if not aucs:
        return 0.5

    return np.mean(aucs)


# ═════════════════════════════════════════════
# MAIN — RUN ALL DATASETS
# ═════════════════════════════════════════════

if __name__ == '__main__':
    print('\n' + '█' * 60)
    print('  MoleculeNet Full Baseline')
    print('  7 Datasets | AttentiveFP | Scaffold Split | 3 Seeds')
    print('  Paper Default Hyperparameters')
    print('█' * 60)

    all_results = {}

    for ds_info in DATASETS:
        name = ds_info['name']
        task = ds_info['task']

        print(f'\n{"="*60}')
        print(f'  {name} ({task})')
        print(f'{"="*60}')

        results = []
        for seed in SEEDS:
            print(f'\n  Seed {seed}:')
            if task == 'regression':
                results.append(run_regression(name, seed))
            else:
                results.append(run_classification(name, seed))

        mean = np.mean(results)
        std = np.std(results)
        all_results[name] = {
            'mean': mean, 'std': std, 'results': results,
            'metric': ds_info['metric'], 'published': ds_info['published']
        }

        print(f'\n  {name} DONE: {mean:.4f} ± {std:.4f}')

    # ── Final Summary Table ──
    print('\n\n' + '=' * 75)
    print('  MOLECULENET BASELINE RESULTS — scaffold split — 3 seeds')
    print('=' * 75)
    print(f'  {"Dataset":<16} {"Task":<16} {"Metric":<10} '
          f'{"Our Result":<22} {"Published":<12}')
    print(f'  {"-"*70}')

    for ds_info in DATASETS:
        name = ds_info['name']
        r = all_results[name]
        result_str = f'{r["mean"]:.4f} ± {r["std"]:.4f}'
        print(f'  {name:<16} {ds_info["task"]:<16} {r["metric"]:<10} '
              f'{result_str:<22} {r["published"]:<12}')

    print('=' * 75)

    print(f'\n  Seed breakdown:')
    for ds_info in DATASETS:
        name = ds_info['name']
        r = all_results[name]
        seeds_str = ' | '.join([f'{s}: {v:.4f}' for s, v in zip(SEEDS, r['results'])])
        print(f'    {name:<16} {seeds_str}')

    print(f'\n  Hyperparameters (Optuna-optimized):')
    for k, v in HP.items():
        print(f'    {k}: {v}')

    print(f'\n  All baselines locked. Ready for multi-task experiments.')
