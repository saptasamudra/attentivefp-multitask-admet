"""
Multi-Task AttentiveFP — 7 MoleculeNet Datasets
One shared encoder, 7 task-specific heads
Scaffold split | 3 seeds

Datasets:
  Regression:     ESOL (1), FreeSolv (1), Lipo (1)
  Classification: BACE (1), BBBP (1), ClinTox (2), Tox21 (12)

Run:  python multitask_7dataset.py
Time: ~3-4 hours on GPU
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
from torch.nn import Linear, ModuleDict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ─────────────────────────────────────────────
# CONFIG
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

# Dataset registry: name, task type, number of output tasks, loss weight
DATASETS = {
    'ESOL':    {'task': 'regression',     'num_tasks': 1,  'weight': 0.5, 'metric': 'RMSE ↓'},
    'FreeSolv':{'task': 'regression',     'num_tasks': 1,  'weight': 0.5, 'metric': 'RMSE ↓'},
    'Lipo':    {'task': 'regression',     'num_tasks': 1,  'weight': 0.5, 'metric': 'RMSE ↓'},
    'BACE':    {'task': 'classification', 'num_tasks': 1,  'weight': 1.0, 'metric': 'AUC ↑'},
    'BBBP':    {'task': 'classification', 'num_tasks': 1,  'weight': 1.0, 'metric': 'AUC ↑'},
    'ClinTox': {'task': 'classification', 'num_tasks': 2,  'weight': 1.0, 'metric': 'AUC ↑'},
    'Tox21':   {'task': 'classification', 'num_tasks': 12, 'weight': 1.0, 'metric': 'AUC ↑'},
}


# ─────────────────────────────────────────────
# FEATURE ENGINEERING (official PyG version)
# ─────────────────────────────────────────────

class GenFeatures:
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
        labels = []
        for data in dataset:
            y = data.y
            if y.dim() > 1:
                y = y[:, 0]
            val = y.item() if y.numel() == 1 else y[0].item()
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
# MULTI-TASK MODEL — 7 HEADS
# ─────────────────────────────────────────────

class AttentiveFPMultiTask7(torch.nn.Module):
    """
    Shared AttentiveFP encoder + 7 dataset-specific output heads.
    """
    def __init__(self, hp, dataset_info):
        super().__init__()
        h = hp['hidden_dim']
        self.encoder = AttentiveFP(
            in_channels=39,
            hidden_channels=h,
            out_channels=h,
            edge_dim=10,
            num_layers=hp['num_layers'],
            num_timesteps=hp['num_timesteps'],
            dropout=hp['dropout'],
        )
        # One head per dataset
        self.heads = ModuleDict()
        for name, info in dataset_info.items():
            self.heads[name] = Linear(h, info['num_tasks'])

    def forward(self, x, edge_index, edge_attr, batch):
        return self.encoder(x, edge_index, edge_attr, batch)

    def predict(self, h, dataset_name):
        return self.heads[dataset_name](h)


# ─────────────────────────────────────────────
# LOAD ALL DATASETS
# ─────────────────────────────────────────────

def load_all_datasets():
    base_path = osp.dirname(osp.abspath(__file__))
    featurizer = GenFeatures()
    all_data = {}

    for name, info in DATASETS.items():
        print(f'  Loading {name}...')
        path = osp.join(base_path, 'data', name)
        dataset = MoleculeNet(path, name=name, pre_transform=featurizer)
        is_clf = info['task'] == 'classification'
        train_set, val_set, test_set = scaffold_split(
            dataset, classification=is_clf)

        all_data[name] = {
            'train': train_set,
            'val': val_set,
            'test': test_set,
            'info': info,
        }
        print(f'    {name}: train={len(train_set)} | val={len(val_set)} | test={len(test_set)}')

    return all_data


# ─────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────

def eval_regression(model, loader, dataset_name):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = model.predict(h, dataset_name)
            y = batch.y
            if y.dim() > 1:
                y = y[:, 0]
            mask = ~torch.isnan(y)
            if mask.sum() > 0:
                errors.append((pred[mask].squeeze(-1) - y[mask]).cpu())
    if not errors:
        return float('inf')
    return sqrt(torch.cat(errors).pow(2).mean().item())


def eval_classification(model, loader, dataset_name, num_tasks):
    model.eval()
    all_preds = [[] for _ in range(num_tasks)]
    all_labels = [[] for _ in range(num_tasks)]

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = torch.sigmoid(model.predict(h, dataset_name)).cpu().numpy()
            y = batch.y.float()
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            y = y.cpu().numpy()

            for t in range(num_tasks):
                mask = ~np.isnan(y[:, t])
                if mask.sum() > 0:
                    all_preds[t].append(pred[mask, t] if num_tasks > 1 else pred[mask].squeeze())
                    all_labels[t].append(y[mask, t])

    aucs = []
    for t in range(num_tasks):
        if not all_preds[t]:
            continue
        p = np.concatenate(all_preds[t])
        l = np.concatenate(all_labels[t])
        if len(np.unique(l)) >= 2:
            aucs.append(roc_auc_score(l, p))
    return np.mean(aucs) if aucs else 0.5


# ─────────────────────────────────────────────
# RUN ONE SEED
# ─────────────────────────────────────────────

def run_multitask(seed, all_data):
    print(f'\n{"="*60}')
    print(f'  Multi-task 7-dataset | Seed {seed}')
    print(f'{"="*60}')

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create data loaders
    loaders = {}
    for name, data in all_data.items():
        loaders[name] = {
            'train': DataLoader(list(data['train']),
                                batch_size=HP['batch_size'], shuffle=True),
            'val': DataLoader(list(data['val']),
                              batch_size=HP['batch_size']),
            'test': DataLoader(list(data['test']),
                               batch_size=HP['batch_size']),
        }

    # Build model
    model = AttentiveFPMultiTask7(HP, DATASETS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HP['lr'],
                                 weight_decay=HP['weight_decay'])

    best_val_score = float('-inf')
    best_results = {}

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Train on each dataset sequentially
        for name, info in DATASETS.items():
            w = info['weight']
            for batch in loaders[name]['train']:
                batch = batch.to(device)
                optimizer.zero_grad()

                h = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                pred = model.predict(h, name)

                if info['task'] == 'regression':
                    y = batch.y
                    if y.dim() > 1:
                        y = y[:, 0]
                    mask = ~torch.isnan(y)
                    if mask.sum() == 0:
                        continue
                    loss = w * F.mse_loss(pred[mask].squeeze(-1), y[mask])
                else:
                    y = batch.y.float()
                    if y.dim() == 1:
                        y = y.unsqueeze(-1)
                    mask = ~torch.isnan(y)
                    if mask.sum() == 0:
                        continue
                    loss = w * F.binary_cross_entropy_with_logits(pred[mask], y[mask])

                loss.backward()
                optimizer.step()

        # Evaluate every 20 epochs
        if epoch % 20 == 0 or epoch == EPOCHS:
            results = {}
            val_score = 0.0

            for name, info in DATASETS.items():
                if info['task'] == 'regression':
                    val_metric = eval_regression(model, loaders[name]['val'], name)
                    test_metric = eval_regression(model, loaders[name]['test'], name)
                    # For combined score: negate RMSE (lower is better)
                    val_score -= val_metric
                else:
                    nt = info['num_tasks']
                    val_metric = eval_classification(model, loaders[name]['val'], name, nt)
                    test_metric = eval_classification(model, loaders[name]['test'], name, nt)
                    # For combined score: add AUC (higher is better)
                    val_score += val_metric

                results[name] = {'val': val_metric, 'test': test_metric}

            if val_score > best_val_score:
                best_val_score = val_score
                best_results = {k: v['test'] for k, v in results.items()}

            if epoch % 50 == 0:
                print(f'\n    Epoch {epoch:03d}:')
                for name in DATASETS:
                    print(f'      {name:<10} Val: {results[name]["val"]:.4f} | '
                          f'Test: {results[name]["test"]:.4f}')

    print(f'\n    Seed {seed} DONE — Best test results:')
    for name, val in best_results.items():
        print(f'      {name:<10} {val:.4f}')

    return best_results


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

if __name__ == '__main__':
    print('\n' + '█' * 60)
    print('  Multi-Task AttentiveFP — 7 MoleculeNet Datasets')
    print('  Shared encoder + 7 heads | Scaffold Split | 3 Seeds')
    print('█' * 60)

    # Load all datasets once
    print('\nLoading datasets...')
    all_data = load_all_datasets()

    # Run 3 seeds
    all_seed_results = []
    for seed in SEEDS:
        results = run_multitask(seed, all_data)
        all_seed_results.append(results)

    # ── Final Summary ──
    print('\n\n' + '=' * 75)
    print('  MULTI-TASK RESULTS — 7 datasets — scaffold split — 3 seeds')
    print('=' * 75)
    print(f'  {"Dataset":<12} {"Task":<16} {"MT Result":<22} '
          f'{"ST Baseline":<22} {"Improved?":<10}')
    print(f'  {"-"*70}')

    # Single-task baselines from last night's run
    st_baselines = {
        'ESOL': 1.0365, 'FreeSolv': 2.2363, 'Lipo': 0.6514,
        'BACE': 0.8918, 'BBBP': 0.6471, 'ClinTox': 0.8742, 'Tox21': 0.7286,
    }

    for name, info in DATASETS.items():
        values = [r[name] for r in all_seed_results]
        mean = np.mean(values)
        std = np.std(values)
        st = st_baselines[name]

        if info['task'] == 'regression':
            improved = '✓' if mean < st else '✗'
        else:
            improved = '✓' if mean > st else '✗'

        result_str = f'{mean:.4f} ± {std:.4f}'
        print(f'  {name:<12} {info["task"]:<16} {result_str:<22} '
              f'{st:<22.4f} {improved:<10}')

    print('=' * 75)

    print(f'\n  Task weights used:')
    for name, info in DATASETS.items():
        print(f'    {name}: {info["weight"]}')

    print(f'\n  Seed breakdown:')
    for name in DATASETS:
        vals = [f'{r[name]:.4f}' for r in all_seed_results]
        print(f'    {name:<12} {" | ".join(f"Seed {s}: {v}" for s, v in zip(SEEDS, vals))}')

    print(f'\n  Next: tune task weights with Optuna to improve weak datasets.')
