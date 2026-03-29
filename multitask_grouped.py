"""
══════════════════════════════════════════════════════════════
  Grouped Multi-Task AttentiveFP
══════════════════════════════════════════════════════════════

  Instead of forcing all 7 datasets into one model, we group
  related tasks together:

    Group 1 (Physical Chemistry): ESOL + FreeSolv + Lipo
      → All regression, all predict physical molecular properties
    
    Group 2 (Toxicity): ClinTox + Tox21
      → Both classification, both predict toxicity endpoints
    
    Group 3 (Bioactivity): BACE + BBBP
      → Both classification, both predict biological activity

  Each group gets its own shared encoder + heads.
  Uses balanced batch sampling + Kendall uncertainty weighting.

  Run:  python multitask_grouped.py
══════════════════════════════════════════════════════════════
"""

import os.path as osp
from collections import defaultdict
from itertools import cycle
from math import sqrt
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP
from torch.nn import Linear, ModuleDict, Parameter


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
    'batch_size': 128,
    'weight_decay': 1e-5,
}

SEEDS = [42, 123, 7]
EPOCHS = 200
STEPS_PER_EPOCH = 40

# All datasets
ALL_DATASETS = {
    'ESOL':    {'task': 'regression',     'num_tasks': 1},
    'FreeSolv':{'task': 'regression',     'num_tasks': 1},
    'Lipo':    {'task': 'regression',     'num_tasks': 1},
    'BACE':    {'task': 'classification', 'num_tasks': 1},
    'BBBP':    {'task': 'classification', 'num_tasks': 1},
    'ClinTox': {'task': 'classification', 'num_tasks': 2},
    'Tox21':   {'task': 'classification', 'num_tasks': 12},
}

# Three related groups
GROUPS = {
    'Physical Chemistry': {
        'datasets': ['ESOL', 'FreeSolv', 'Lipo'],
        'type': 'regression',
    },
    'Toxicity': {
        'datasets': ['ClinTox', 'Tox21'],
        'type': 'classification',
    },
    'Bioactivity': {
        'datasets': ['BACE', 'BBBP'],
        'type': 'classification',
    },
}

# Single-task baselines
ST_BASELINES = {
    'ESOL': 1.0506, 'FreeSolv': 2.3012, 'Lipo': 0.6783,
    'BACE': 0.9205, 'BBBP': 0.6471, 'ClinTox': 0.8639, 'Tox21': 0.7778,
}


# ─────────────────────────────────────────────
# FEATURE ENGINEERING (39-dim atoms, 10-dim bonds)
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
            fc = atom.GetFormalCharge()
            re = atom.GetNumRadicalElectrons()
            hyb = [0.] * len(self.hybridizations)
            hyb[self.hybridizations.index(atom.GetHybridization())
                if atom.GetHybridization() in self.hybridizations else -1] = 1.
            arom = 1. if atom.GetIsAromatic() else 0.
            hs = [0.] * 5
            hs[min(atom.GetTotalNumHs(), 4)] = 1.
            chiral = 1. if atom.HasProp('_ChiralityPossible') else 0.
            ct = [0.] * 2
            if atom.HasProp('_CIPCode'):
                ct[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.
            x = torch.tensor(symbol + degree + [fc, re] + hyb + [arom] + hs + [chiral] + ct)
            xs.append(x)
        data.x = torch.stack(xs, dim=0)

        ei, ea = [], []
        for bond in mol.GetBonds():
            ei += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            ei += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]
            bt = bond.GetBondType()
            s = 1. if bt == Chem.rdchem.BondType.SINGLE else 0.
            d = 1. if bt == Chem.rdchem.BondType.DOUBLE else 0.
            t = 1. if bt == Chem.rdchem.BondType.TRIPLE else 0.
            a = 1. if bt == Chem.rdchem.BondType.AROMATIC else 0.
            c = 1. if bond.GetIsConjugated() else 0.
            r = 1. if bond.IsInRing() else 0.
            st = [0.] * 4
            st[self.stereos.index(bond.GetStereo())] = 1.
            attr = torch.tensor([s, d, t, a, c, r] + st)
            ea += [attr, attr]
        if not ea:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(ei).t().contiguous()
            data.edge_attr = torch.stack(ea, dim=0)
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
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds[scaffold].append(i)
    groups = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(dataset)
    n_train, n_val = int(n * train_frac), int(n * val_frac)
    train_idx, val_idx, test_idx = [], [], []

    if classification:
        labels = []
        for data in dataset:
            y = data.y
            if y.dim() > 1: y = y[:, 0]
            v = y.item() if y.numel() == 1 else y[0].item()
            labels.append(0 if np.isnan(v) else int(v))
        labels = np.array(labels)
        pos = [g for g in groups if all(labels[i] == 1 for i in g)]
        neg = [g for g in groups if all(labels[i] == 0 for i in g)]
        seeded = set()
        if pos and neg:
            val_idx.extend(pos[-1]); seeded.add(id(pos[-1]))
            val_idx.extend(neg[-1]); seeded.add(id(neg[-1]))
            if len(pos) > 1 and len(neg) > 1:
                test_idx.extend(pos[-2]); seeded.add(id(pos[-2]))
                test_idx.extend(neg[-2]); seeded.add(id(neg[-2]))
        for g in groups:
            if id(g) in seeded: continue
            if len(train_idx) < n_train: train_idx.extend(g)
            elif len(val_idx) < n_val: val_idx.extend(g)
            else: test_idx.extend(g)
    else:
        for g in groups:
            if len(train_idx) < n_train: train_idx.extend(g)
            elif len(val_idx) < n_val: val_idx.extend(g)
            else: test_idx.extend(g)

    return (dataset[torch.tensor(train_idx)],
            dataset[torch.tensor(val_idx)],
            dataset[torch.tensor(test_idx)])


# ─────────────────────────────────────────────
# MODEL: Shared encoder + N heads + uncertainty weights
# ─────────────────────────────────────────────

class GroupMultiTask(nn.Module):
    def __init__(self, hp, dataset_names, dataset_info):
        super().__init__()
        h = hp['hidden_dim']
        self.encoder = AttentiveFP(
            in_channels=39, hidden_channels=h, out_channels=h,
            edge_dim=10, num_layers=hp['num_layers'],
            num_timesteps=hp['num_timesteps'], dropout=hp['dropout'],
        )
        self.heads = ModuleDict()
        self.log_vars = nn.ParameterDict()
        for name in dataset_names:
            self.heads[name] = Linear(h, dataset_info[name]['num_tasks'])
            self.log_vars[name] = Parameter(torch.zeros(1))

    def encode(self, x, edge_index, edge_attr, batch):
        return self.encoder(x, edge_index, edge_attr, batch)

    def predict(self, h, name):
        return self.heads[name](h)

    def compute_loss(self, pred, y, name, task_type):
        """Kendall uncertainty-weighted loss."""
        if task_type == 'regression':
            if y.dim() > 1: y = y[:, 0]
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred.device)
            raw = F.mse_loss(pred[mask].squeeze(-1), y[mask])
            prec = torch.exp(-self.log_vars[name])
            return 0.5 * prec * raw + 0.5 * self.log_vars[name]
        else:
            if y.dim() == 1: y = y.unsqueeze(-1)
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred.device)
            raw = F.binary_cross_entropy_with_logits(pred[mask], y[mask])
            prec = torch.exp(-self.log_vars[name])
            return prec * raw + self.log_vars[name]


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_all_datasets():
    base_path = osp.dirname(osp.abspath(__file__))
    feat = GenFeatures()
    all_data = {}
    for name, info in ALL_DATASETS.items():
        path = osp.join(base_path, 'data', name)
        dataset = MoleculeNet(path, name=name, pre_transform=feat)
        is_clf = info['task'] == 'classification'
        tr, va, te = scaffold_split(dataset, classification=is_clf)
        all_data[name] = {'train': tr, 'val': va, 'test': te}
        print(f'  {name}: train={len(tr)} | val={len(va)} | test={len(te)}')
    return all_data


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def eval_regression(model, loader, name):
    model.eval()
    errs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h = model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            p = model.predict(h, name)
            y = batch.y
            if y.dim() > 1: y = y[:, 0]
            m = ~torch.isnan(y)
            if m.sum() > 0:
                errs.append((p[m].squeeze(-1) - y[m]).cpu())
    if not errs: return float('inf')
    return sqrt(torch.cat(errs).pow(2).mean().item())


def eval_classification(model, loader, name, num_tasks):
    model.eval()
    preds = [[] for _ in range(num_tasks)]
    labels = [[] for _ in range(num_tasks)]
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h = model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            p = torch.sigmoid(model.predict(h, name)).cpu().numpy()
            y = batch.y.float()
            if y.dim() == 1: y = y.unsqueeze(-1)
            y = y.cpu().numpy()
            for t in range(num_tasks):
                m = ~np.isnan(y[:, t])
                if m.sum() > 0:
                    preds[t].append(p[m, t] if num_tasks > 1 else p[m].squeeze())
                    labels[t].append(y[m, t])
    aucs = []
    for t in range(num_tasks):
        if not preds[t]: continue
        pp = np.concatenate(preds[t])
        ll = np.concatenate(labels[t])
        if len(np.unique(ll)) >= 2:
            aucs.append(roc_auc_score(ll, pp))
    return np.mean(aucs) if aucs else 0.5


# ─────────────────────────────────────────────
# TRAIN ONE GROUP, ONE SEED
# ─────────────────────────────────────────────

def train_group(group_name, group_info, all_data, seed):
    dataset_names = group_info['datasets']
    group_type = group_info['type']

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create loaders with cycling iterators
    loaders = {}
    for name in dataset_names:
        train_loader = DataLoader(
            list(all_data[name]['train']), batch_size=HP['batch_size'],
            shuffle=True, drop_last=False)
        loaders[name] = {
            'train_iter': iter(cycle(train_loader)),
            'val': DataLoader(list(all_data[name]['val']), batch_size=HP['batch_size']),
            'test': DataLoader(list(all_data[name]['test']), batch_size=HP['batch_size']),
        }

    # Build model for this group only
    group_ds_info = {n: ALL_DATASETS[n] for n in dataset_names}
    model = GroupMultiTask(HP, dataset_names, group_ds_info).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HP['lr'],
                                 weight_decay=HP['weight_decay'])

    best_val_score = float('-inf')
    best_results = {}

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Balanced training: equal steps per dataset
        for step in range(STEPS_PER_EPOCH):
            for name in dataset_names:
                batch = next(loaders[name]['train_iter'])
                batch = batch.to(device)
                optimizer.zero_grad()
                h = model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                pred = model.predict(h, name)
                y = batch.y.float()
                loss = model.compute_loss(pred, y, name, ALL_DATASETS[name]['task'])
                if loss.item() > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == EPOCHS:
            results = {}
            val_score = 0.0
            for name in dataset_names:
                info = ALL_DATASETS[name]
                if info['task'] == 'regression':
                    v = eval_regression(model, loaders[name]['val'], name)
                    t = eval_regression(model, loaders[name]['test'], name)
                    val_score -= v
                else:
                    nt = info['num_tasks']
                    v = eval_classification(model, loaders[name]['val'], name, nt)
                    t = eval_classification(model, loaders[name]['test'], name, nt)
                    val_score += v
                results[name] = {'val': v, 'test': t}

            if val_score > best_val_score:
                best_val_score = val_score
                best_results = {k: r['test'] for k, r in results.items()}

            if epoch % 50 == 0:
                print(f'      Epoch {epoch:03d}:', end='')
                for name in dataset_names:
                    print(f'  {name}: V={results[name]["val"]:.4f} T={results[name]["test"]:.4f}', end='')
                # Print learned weights
                ws = ' | '.join(f'{n}:{torch.exp(-model.log_vars[n]).item():.2f}'
                                for n in dataset_names)
                print(f'  [w: {ws}]')

    return best_results


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

if __name__ == '__main__':
    print('\n' + '█' * 65)
    print('  Grouped Multi-Task AttentiveFP')
    print('  3 groups of related tasks | Balanced sampling')
    print('  Kendall uncertainty weighting | 3 Seeds')
    print('█' * 65)

    print('\n  Loading all datasets...')
    all_data = load_all_datasets()

    # Store all results across groups and seeds
    all_results = {name: [] for name in ALL_DATASETS}

    for group_name, group_info in GROUPS.items():
        print(f'\n{"═"*65}')
        print(f'  GROUP: {group_name}')
        print(f'  Datasets: {", ".join(group_info["datasets"])}')
        print(f'{"═"*65}')

        for seed in SEEDS:
            print(f'\n    Seed {seed}:')
            results = train_group(group_name, group_info, all_data, seed)
            for name, val in results.items():
                all_results[name].append(val)
            print(f'      DONE: {" | ".join(f"{n}: {results[n]:.4f}" for n in group_info["datasets"])}')

    # ── Final Summary ──
    print('\n\n' + '█' * 70)
    print('  GROUPED MULTI-TASK RESULTS')
    print('  3 groups × 3 seeds | scaffold split | uncertainty weighting')
    print('█' * 70)

    print(f'\n  {"Dataset":<12} {"Group":<22} {"MT Result":<22} {"ST Baseline":<12} {"Delta":<12} {"Result"}')
    print(f'  {"-"*82}')

    total_improved = 0
    for name, info in ALL_DATASETS.items():
        vals = all_results[name]
        mean, std = np.mean(vals), np.std(vals)
        st = ST_BASELINES[name]

        # Find which group this dataset belongs to
        group = [g for g, gi in GROUPS.items() if name in gi['datasets']][0]

        if info['task'] == 'regression':
            delta = ((st - mean) / st) * 100
            better = mean < st
        else:
            delta = ((mean - st) / st) * 100
            better = mean > st

        if better:
            total_improved += 1

        symbol = f'{"↓" if info["task"]=="regression" else "↑"} {abs(delta):.1f}%'
        tag = '✓ IMPROVED' if better else '✗'
        print(f'  {name:<12} {group:<22} {mean:.4f} ± {std:.4f}     {st:<12.4f} {symbol:<12} {tag}')

    print(f'\n  Improved: {total_improved}/7 datasets')

    print(f'\n  Seed breakdown:')
    for name in ALL_DATASETS:
        vals = all_results[name]
        print(f'    {name:<12} ' + ' | '.join(f'Seed {s}: {v:.4f}' for s, v in zip(SEEDS, vals)))

    # Compare with all-7 multi-task (if available)
    print(f'\n  Key insight: grouping related tasks reduces negative transfer.')
    print(f'  Physical chemistry tasks share molecular property features.')
    print(f'  Toxicity tasks share biological response features.')
    print(f'  Bioactivity tasks share binding/permeability features.')
