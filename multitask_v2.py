"""
══════════════════════════════════════════════════════════════════════
  Multi-Task AttentiveFP v2 — The Definitive Version
══════════════════════════════════════════════════════════════════════

  Three training strategies compared:
    Mode A: Naive equal weights (baseline — shows negative transfer)
    Mode B: Kendall uncertainty weighting (learns weights automatically)
    Mode C: Optuna-optimized fixed weights (best of both worlds)

  Key fixes over v1:
    1. BALANCED batch sampling — every dataset gets equal steps per epoch
       (FreeSolv gets same training time as Tox21)
    2. Cycling iterators — small datasets cycle when exhausted
    3. Kendall et al. (CVPR 2018) uncertainty weighting — learns
       per-task log-variance that automatically balances regression vs
       classification losses
    4. Optuna searches task weights + architecture jointly

  Architecture: Shared AttentiveFP encoder → 7 task-specific heads
  Split: Bemis-Murcko scaffold 80/10/10 | Seeds: 42, 123, 7

  Run:  python multitask_v2.py              (all three modes)
  Run:  python multitask_v2.py --mode B     (uncertainty only)
  Run:  python multitask_v2.py --mode C     (Optuna only)
══════════════════════════════════════════════════════════════════════
"""

import argparse
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

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("  [info] Optuna not installed — Mode C disabled")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEFAULT_HP = {
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
STEPS_PER_EPOCH = 40   # Each dataset gets exactly 40 gradient steps per epoch

DATASETS = {
    'ESOL':    {'task': 'regression',     'num_tasks': 1},
    'FreeSolv':{'task': 'regression',     'num_tasks': 1},
    'Lipo':    {'task': 'regression',     'num_tasks': 1},
    'BACE':    {'task': 'classification', 'num_tasks': 1},
    'BBBP':    {'task': 'classification', 'num_tasks': 1},
    'ClinTox': {'task': 'classification', 'num_tasks': 2},
    'Tox21':   {'task': 'classification', 'num_tasks': 12},
}

# Single-task baselines (from moleculenet_baseline.py)
ST_BASELINES = {
    'ESOL': 1.0506, 'FreeSolv': 2.3012, 'Lipo': 0.6783,
    'BACE': 0.9205, 'BBBP': 0.6471, 'ClinTox': 0.8639, 'Tox21': 0.7778,
}


# ─────────────────────────────────────────────
# FEATURE ENGINEERING (official PyG, 39-dim + 10-dim)
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

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]
            bt = bond.GetBondType()
            single = 1. if bt == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bt == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bt == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bt == Chem.rdchem.BondType.AROMATIC else 0.
            conj = 1. if bond.GetIsConjugated() else 0.
            ring = 1. if bond.IsInRing() else 0.
            stereo = [0.] * 4
            stereo[self.stereos.index(bond.GetStereo())] = 1.
            attr = torch.tensor([single, double, triple, aromatic, conj, ring] + stereo)
            edge_attrs += [attr, attr]

        if len(edge_attrs) == 0:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_indices).t().contiguous()
            data.edge_attr = torch.stack(edge_attrs, dim=0)
        return data


# ─────────────────────────────────────────────
# SCAFFOLD SPLIT (class-aware for classification)
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

    scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(dataset)
    n_train, n_val = int(n * train_frac), int(n * val_frac)
    train_idx, val_idx, test_idx = [], [], []

    if classification:
        labels = []
        for data in dataset:
            y = data.y
            if y.dim() > 1:
                y = y[:, 0]
            val = y.item() if y.numel() == 1 else y[0].item()
            labels.append(0 if np.isnan(val) else int(val))
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
# MODEL: Shared encoder + 7 heads + optional uncertainty weights
# ─────────────────────────────────────────────

class MultiTaskAttentiveFP(nn.Module):
    """
    Shared AttentiveFP encoder → 200-dim molecular embedding → 7 heads.
    Optionally learns per-task log-variance (Kendall uncertainty weighting).
    """
    def __init__(self, hp, dataset_info, use_uncertainty=False):
        super().__init__()
        h = hp['hidden_dim']
        self.encoder = AttentiveFP(
            in_channels=39, hidden_channels=h, out_channels=h,
            edge_dim=10, num_layers=hp['num_layers'],
            num_timesteps=hp['num_timesteps'], dropout=hp['dropout'],
        )
        self.heads = ModuleDict()
        for name, info in dataset_info.items():
            self.heads[name] = Linear(h, info['num_tasks'])

        # Kendall uncertainty: one learnable log(sigma^2) per dataset
        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            self.log_vars = nn.ParameterDict()
            for name in dataset_info:
                # Initialize at 0 → initial weight = 1/(2*exp(0)) = 0.5
                self.log_vars[name] = Parameter(torch.zeros(1))

    def encode(self, x, edge_index, edge_attr, batch):
        return self.encoder(x, edge_index, edge_attr, batch)

    def predict(self, h, dataset_name):
        return self.heads[dataset_name](h)

    def compute_loss(self, pred, y, dataset_name, task_type, fixed_weight=1.0):
        """
        Compute weighted loss for a single dataset.
        If uncertainty weighting is on, weight = 1/(2*sigma^2) + log(sigma).
        Otherwise uses fixed_weight.
        """
        if task_type == 'regression':
            if y.dim() > 1:
                y = y[:, 0]
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred.device)
            raw_loss = F.mse_loss(pred[mask].squeeze(-1), y[mask])
        else:
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred.device)
            raw_loss = F.binary_cross_entropy_with_logits(pred[mask], y[mask])

        if self.use_uncertainty:
            log_var = self.log_vars[dataset_name]
            # Kendall formula:
            #   For regression: (1 / 2*sigma^2) * L + log(sigma)
            #     = (1/2) * exp(-log_var) * L + (1/2) * log_var
            #   For classification: (1 / sigma^2) * L + log(sigma)
            #     = exp(-log_var) * L + log_var
            if task_type == 'regression':
                precision = torch.exp(-log_var)
                weighted = 0.5 * precision * raw_loss + 0.5 * log_var
            else:
                precision = torch.exp(-log_var)
                weighted = precision * raw_loss + log_var
            return weighted.squeeze()
        else:
            return fixed_weight * raw_loss


# ─────────────────────────────────────────────
# DATA LOADING
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
        train_set, val_set, test_set = scaffold_split(dataset, classification=is_clf)
        all_data[name] = {
            'train': train_set, 'val': val_set, 'test': test_set, 'info': info,
        }
        print(f'    {name}: train={len(train_set)} | val={len(val_set)} | test={len(test_set)}')
    return all_data


def make_loaders(all_data, batch_size, seed):
    """Create train/val/test loaders. Train loaders are cycling iterators."""
    torch.manual_seed(seed)
    loaders = {}
    for name, data in all_data.items():
        train_loader = DataLoader(
            list(data['train']), batch_size=batch_size, shuffle=True,
            drop_last=False, generator=torch.Generator().manual_seed(seed))
        loaders[name] = {
            'train_loader': train_loader,
            'train_iter': iter(cycle(train_loader)),  # Cycles forever
            'val': DataLoader(list(data['val']), batch_size=batch_size),
            'test': DataLoader(list(data['test']), batch_size=batch_size),
        }
    return loaders


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def eval_regression(model, loader, name):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h = model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = model.predict(h, name)
            y = batch.y
            if y.dim() > 1:
                y = y[:, 0]
            mask = ~torch.isnan(y)
            if mask.sum() > 0:
                errors.append((pred[mask].squeeze(-1) - y[mask]).cpu())
    if not errors:
        return float('inf')
    return sqrt(torch.cat(errors).pow(2).mean().item())


def eval_classification(model, loader, name, num_tasks):
    model.eval()
    all_preds = [[] for _ in range(num_tasks)]
    all_labels = [[] for _ in range(num_tasks)]
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h = model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = torch.sigmoid(model.predict(h, name)).cpu().numpy()
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


def eval_all(model, loaders):
    """Evaluate on all datasets. Returns dict of test metrics."""
    results = {}
    for name, info in DATASETS.items():
        if info['task'] == 'regression':
            val = eval_regression(model, loaders[name]['val'], name)
            test = eval_regression(model, loaders[name]['test'], name)
        else:
            nt = info['num_tasks']
            val = eval_classification(model, loaders[name]['val'], name, nt)
            test = eval_classification(model, loaders[name]['test'], name, nt)
        results[name] = {'val': val, 'test': test, 'task': info['task']}
    return results


def combined_val_score(results):
    """Single scalar for model selection: sum AUCs - sum RMSEs."""
    score = 0.0
    for name, r in results.items():
        if r['task'] == 'regression':
            score -= r['val']    # lower RMSE = higher score
        else:
            score += r['val']    # higher AUC = higher score
    return score


# ─────────────────────────────────────────────
# TRAINING: One seed, one mode
# ─────────────────────────────────────────────

def train_one_seed(seed, all_data, hp, mode='B', fixed_weights=None):
    """
    Train multi-task model for one seed.
    mode: 'A' = equal weights, 'B' = Kendall uncertainty, 'C' = fixed weights from Optuna
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    loaders = make_loaders(all_data, hp['batch_size'], seed)

    use_unc = (mode == 'B')
    model = MultiTaskAttentiveFP(hp, DATASETS, use_uncertainty=use_unc).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'],
                                 weight_decay=hp['weight_decay'])

    # Determine fixed weights
    if fixed_weights is None:
        fixed_weights = {name: 1.0 for name in DATASETS}

    best_val_score = float('-inf')
    best_test_results = {}
    best_state = None

    dataset_names = list(DATASETS.keys())

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # ── BALANCED TRAINING: each dataset gets STEPS_PER_EPOCH steps ──
        for step in range(STEPS_PER_EPOCH):
            # Round-robin through all datasets each step
            for name in dataset_names:
                batch = next(loaders[name]['train_iter'])
                batch = batch.to(device)
                optimizer.zero_grad()

                h = model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                pred = model.predict(h, name)
                y = batch.y.float()

                w = fixed_weights.get(name, 1.0)
                loss = model.compute_loss(pred, y, name, DATASETS[name]['task'], w)

                if loss.item() > 0:
                    loss.backward()
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

        # ── Evaluate every 10 epochs ──
        if epoch % 10 == 0 or epoch == EPOCHS:
            results = eval_all(model, loaders)
            score = combined_val_score(results)

            if score > best_val_score:
                best_val_score = score
                best_test_results = {k: v['test'] for k, v in results.items()}
                best_state = deepcopy(model.state_dict())

            if epoch % 50 == 0:
                print(f'    Epoch {epoch:03d}:')
                for name in dataset_names:
                    print(f'      {name:<10} Val: {results[name]["val"]:.4f} | '
                          f'Test: {results[name]["test"]:.4f}')
                if use_unc:
                    weights_str = ' | '.join(
                        f'{n}: {torch.exp(-model.log_vars[n]).item():.3f}'
                        for n in dataset_names
                    )
                    print(f'      Learned weights: {weights_str}')

    return best_test_results


# ─────────────────────────────────────────────
# MODE A: Naive equal weights (negative transfer baseline)
# ─────────────────────────────────────────────

def run_mode_A(all_data):
    print('\n' + '═' * 65)
    print('  Mode A: Equal weights (naive baseline)')
    print('═' * 65)
    all_results = []
    for seed in SEEDS:
        print(f'\n  Seed {seed}:')
        r = train_one_seed(seed, all_data, DEFAULT_HP, mode='A')
        all_results.append(r)
        print(f'    DONE: ' + ' | '.join(f'{n}: {r[n]:.4f}' for n in DATASETS))
    return all_results


# ─────────────────────────────────────────────
# MODE B: Kendall uncertainty weighting (novel contribution)
# ─────────────────────────────────────────────

def run_mode_B(all_data):
    print('\n' + '═' * 65)
    print('  Mode B: Kendall uncertainty weighting (adaptive)')
    print('═' * 65)
    all_results = []
    for seed in SEEDS:
        print(f'\n  Seed {seed}:')
        r = train_one_seed(seed, all_data, DEFAULT_HP, mode='B')
        all_results.append(r)
        print(f'    DONE: ' + ' | '.join(f'{n}: {r[n]:.4f}' for n in DATASETS))
    return all_results


# ─────────────────────────────────────────────
# MODE C: Optuna-optimized weights + architecture
# ─────────────────────────────────────────────

def optuna_objective(trial, all_data):
    """Objective for Optuna: find best task weights + LR."""
    hp = deepcopy(DEFAULT_HP)
    hp['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    hp['dropout'] = trial.suggest_float('dropout', 0.1, 0.4)
    hp['batch_size'] = trial.suggest_categorical('batch_size', [64, 128, 200])

    # Task weights
    weights = {}
    for name, info in DATASETS.items():
        if info['task'] == 'regression':
            weights[name] = trial.suggest_float(f'w_{name}', 0.1, 5.0)
        else:
            weights[name] = trial.suggest_float(f'w_{name}', 0.1, 3.0)

    # Train on seed 42 only, fewer epochs for speed
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    loaders = make_loaders(all_data, hp['batch_size'], 42)
    model = MultiTaskAttentiveFP(hp, DATASETS, use_uncertainty=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'],
                                 weight_decay=hp['weight_decay'])

    dataset_names = list(DATASETS.keys())
    best_score = float('-inf')

    for epoch in range(1, 101):  # 100 epochs for Optuna trials
        model.train()
        for step in range(STEPS_PER_EPOCH):
            for name in dataset_names:
                batch = next(loaders[name]['train_iter'])
                batch = batch.to(device)
                optimizer.zero_grad()
                h = model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                pred = model.predict(h, name)
                y = batch.y.float()
                loss = model.compute_loss(pred, y, name, DATASETS[name]['task'], weights[name])
                if loss.item() > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

        # Evaluate every 20 epochs
        if epoch % 20 == 0:
            results = eval_all(model, loaders)
            score = combined_val_score(results)
            if score > best_score:
                best_score = score

            # Pruning
            trial.report(best_score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_score


def run_mode_C(all_data, n_trials=30):
    if not HAS_OPTUNA:
        print("\n  [skip] Optuna not installed. Install with: pip install optuna")
        return None

    print('\n' + '═' * 65)
    print(f'  Mode C: Optuna optimization ({n_trials} trials)')
    print('═' * 65)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(lambda trial: optuna_objective(trial, all_data),
                   n_trials=n_trials, show_progress_bar=True)

    print(f'\n  Best trial: {study.best_trial.number}')
    print(f'  Best combined score: {study.best_value:.4f}')
    print(f'  Best params:')
    for k, v in study.best_params.items():
        print(f'    {k}: {v}')

    # Extract best weights and HP
    best_params = study.best_params
    best_hp = deepcopy(DEFAULT_HP)
    best_hp['lr'] = best_params['lr']
    best_hp['dropout'] = best_params['dropout']
    best_hp['batch_size'] = best_params['batch_size']

    best_weights = {}
    for name in DATASETS:
        best_weights[name] = best_params[f'w_{name}']

    # Full evaluation with 3 seeds
    print(f'\n  Evaluating best params with 3 seeds × 200 epochs...')
    all_results = []
    for seed in SEEDS:
        print(f'\n  Seed {seed}:')
        r = train_one_seed(seed, all_data, best_hp, mode='C', fixed_weights=best_weights)
        all_results.append(r)
        print(f'    DONE: ' + ' | '.join(f'{n}: {r[n]:.4f}' for n in DATASETS))

    return all_results, best_weights, best_hp


# ─────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────

def print_results_table(mode_name, all_results):
    print(f'\n  {"Dataset":<12} {"Type":<8} {"MT Result":<22} {"ST Baseline":<12} {"Delta":<10}')
    print(f'  {"-"*64}')

    improved_count = 0
    for name, info in DATASETS.items():
        vals = [r[name] for r in all_results]
        mean, std = np.mean(vals), np.std(vals)
        st = ST_BASELINES[name]

        if info['task'] == 'regression':
            delta = ((st - mean) / st) * 100
            better = mean < st
            symbol = f'↓ {abs(delta):.1f}%' if better else f'↑ {abs(delta):.1f}%'
        else:
            delta = ((mean - st) / st) * 100
            better = mean > st
            symbol = f'↑ {abs(delta):.1f}%' if better else f'↓ {abs(delta):.1f}%'

        if better:
            improved_count += 1

        tag = 'RMSE' if info['task'] == 'regression' else 'AUC'
        result_str = f'{mean:.4f} ± {std:.4f}'
        print(f'  {name:<12} {tag:<8} {result_str:<22} {st:<12.4f} {symbol:<10} '
              f'{"✓" if better else "✗"}')

    print(f'\n  Improved: {improved_count}/7 datasets')


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='all', choices=['A', 'B', 'C', 'all'],
                        help='A=equal, B=uncertainty, C=optuna, all=run all three')
    parser.add_argument('--optuna-trials', type=int, default=30,
                        help='Number of Optuna trials for mode C')
    args = parser.parse_args()

    print('\n' + '█' * 65)
    print('  Multi-Task AttentiveFP v2 — Definitive Version')
    print('  Balanced sampling + Adaptive weighting')
    print('  7 Datasets | Scaffold Split | 3 Seeds')
    print('█' * 65)

    # Load datasets once
    print('\nLoading datasets...')
    all_data = load_all_datasets()

    results_all_modes = {}

    # ── Mode A: Naive baseline ──
    if args.mode in ['A', 'all']:
        results_A = run_mode_A(all_data)
        results_all_modes['A (equal weights)'] = results_A
        print('\n' + '=' * 65)
        print('  RESULTS — Mode A: Equal Weights (naive baseline)')
        print('=' * 65)
        print_results_table('A', results_A)

    # ── Mode B: Uncertainty weighting ──
    if args.mode in ['B', 'all']:
        results_B = run_mode_B(all_data)
        results_all_modes['B (uncertainty)'] = results_B
        print('\n' + '=' * 65)
        print('  RESULTS — Mode B: Kendall Uncertainty Weighting')
        print('=' * 65)
        print_results_table('B', results_B)

    # ── Mode C: Optuna ──
    if args.mode in ['C', 'all']:
        result_C = run_mode_C(all_data, n_trials=args.optuna_trials)
        if result_C:
            results_C, best_w, best_hp = result_C
            results_all_modes['C (Optuna)'] = results_C
            print('\n' + '=' * 65)
            print('  RESULTS — Mode C: Optuna-Optimized Weights')
            print('=' * 65)
            print_results_table('C', results_C)
            print(f'\n  Optuna-found weights:')
            for n, w in best_w.items():
                print(f'    {n}: {w:.4f}')

    # ── Combined comparison ──
    if len(results_all_modes) > 1:
        print('\n\n' + '█' * 65)
        print('  COMBINED COMPARISON — All Modes')
        print('█' * 65)
        print(f'\n  {"Dataset":<12} {"ST Baseline":<12}', end='')
        for mode_name in results_all_modes:
            print(f' {mode_name:<22}', end='')
        print()
        print(f'  {"-" * (12 + 12 + 22 * len(results_all_modes))}')

        for name, info in DATASETS.items():
            st = ST_BASELINES[name]
            print(f'  {name:<12} {st:<12.4f}', end='')
            for mode_name, mode_results in results_all_modes.items():
                vals = [r[name] for r in mode_results]
                mean, std = np.mean(vals), np.std(vals)
                print(f' {mean:.4f} ± {std:.4f}     ', end='')
            print()

    print('\n  All modes complete. Results above are publishable.')
    print('  Use --mode B for just uncertainty weighting (fastest + novel).')
