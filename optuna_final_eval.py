"""
Final Evaluation — Optuna-Optimized Hyperparameters
Runs both single-task ESOL and multi-task (ESOL+BACE) with 3 seeds
Uses the best hyperparameters found by Optuna

Run:  python optuna_final_eval.py
Time: ~25 min on GPU
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
from torch.nn import Linear


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ─────────────────────────────────────────────
# OPTUNA-OPTIMIZED HYPERPARAMETERS
# From single-task Optuna (Trial 2):
# ─────────────────────────────────────────────

OPTUNA_HP = {
    'lr': 0.00340,
    'hidden_dim': 200,
    'num_layers': 2,
    'num_timesteps': 3,      # was 2 in paper → Optuna found 3 is better
    'dropout': 0.374,         # was 0.2 in paper → Optuna found higher is better
    'batch_size': 200,
    'weight_decay': 1e-5,
}

# Multi-task weights (from manual ablation — proven to work)
MT_WEIGHTS = {
    'w_esol': 0.5,
    'w_bace': 1.0,
}

SEEDS = [42, 123, 7]
EPOCHS = 200


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
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
            raise ValueError(f'Invalid SMILES: {data.smiles}')
        xs = []
        for atom in mol.GetAtoms():
            symbol = [0.] * len(self.symbols)
            symbol[self.symbols.index(atom.GetSymbol())
                   if atom.GetSymbol() in self.symbols else -1] = 1.
            degree = [0.] * 6
            degree[atom.GetDegree()] = 1.
            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.] * len(self.hybridizations)
            hybridization[self.hybridizations.index(
                atom.GetHybridization()
            ) if atom.GetHybridization() in self.hybridizations else -1] = 1.
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            hydrogens = [0.] * 5
            hydrogens[atom.GetTotalNumHs()] = 1.
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            xs.append(symbol + degree + [formal_charge] +
                      [radical_electrons] + hybridization +
                      [aromaticity] + hydrogens + [chirality])
        data.x = torch.tensor(xs, dtype=torch.float).view(-1, 39)

        edge_attrs = []
        for bond in mol.GetBonds():
            bond_type = bond.GetBondTypeAsDouble()
            stereo = [0.] * 4
            stereo[self.stereos.index(bond.GetStereo())] = 1.
            is_aromatic = 1. if bond.GetIsAromatic() else 0.
            is_conjugated = 1. if bond.GetIsConjugated() else 0.
            is_in_ring = 1. if bond.IsInRing() else 0.
            attr = [bond_type] + stereo + [is_aromatic, is_conjugated, is_in_ring]
            edge_attrs += [attr, attr]
        if len(edge_attrs) == 0:
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
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
        labels = np.array([
            int(data.y.item() if data.y.numel() == 1 else data.y[0].item())
            for data in dataset
        ])
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
# MULTI-TASK MODEL
# ─────────────────────────────────────────────

class AttentiveFPMultiTask(torch.nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.encoder = AttentiveFP(
            in_channels=39,
            hidden_channels=hp['hidden_dim'],
            out_channels=hp['hidden_dim'],
            edge_dim=10,
            num_layers=hp['num_layers'],
            num_timesteps=hp['num_timesteps'],
            dropout=hp['dropout'],
        )
        self.esol_head = Linear(hp['hidden_dim'], 1)
        self.bace_head = Linear(hp['hidden_dim'], 1)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder(x, edge_index, edge_attr, batch)
        return self.esol_head(h), self.bace_head(h)


# ─────────────────────────────────────────────
# LOAD DATASETS
# ─────────────────────────────────────────────

print('Loading datasets...')
base_path = osp.dirname(osp.abspath(__file__))
esol_dataset = MoleculeNet(osp.join(base_path, 'data', 'ESOL'), name='ESOL', pre_transform=GenFeatures())
bace_dataset = MoleculeNet(osp.join(base_path, 'data', 'BACE'), name='BACE', pre_transform=GenFeatures())

esol_train, esol_val, esol_test = scaffold_split(esol_dataset)
bace_train, bace_val, bace_test = scaffold_split(bace_dataset, classification=True)

print(f'  ESOL: train={len(esol_train)} | val={len(esol_val)} | test={len(esol_test)}')
print(f'  BACE: train={len(bace_train)} | val={len(bace_val)} | test={len(bace_test)}')


# ─────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────

def eval_rmse(model, loader, single_task=True):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if single_task:
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                out, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            errors.append((out.squeeze(-1) - batch.y.squeeze(-1)).cpu())
    return sqrt(torch.cat(errors).pow(2).mean().item())


def eval_auc(model, loader, single_task=True):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if single_task:
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                _, out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y = batch.y.float()
            if y.dim() > 1:
                y = y[:, 0]
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                continue
            pred = torch.sigmoid(out[mask].squeeze(-1))
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y[mask].cpu().numpy())
    if not all_preds:
        return 0.5
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, preds)


# ═════════════════════════════════════════════
# PART 1: SINGLE-TASK ESOL WITH OPTUNA PARAMS
# ═════════════════════════════════════════════

def run_esol_optuna(seed):
    print(f'\n  ESOL Optuna | Seed {seed}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    hp = OPTUNA_HP
    train_loader = DataLoader(esol_train, batch_size=hp['batch_size'], shuffle=True)
    val_loader = DataLoader(esol_val, batch_size=hp['batch_size'])
    test_loader = DataLoader(esol_test, batch_size=hp['batch_size'])

    model = AttentiveFP(
        in_channels=39, hidden_channels=hp['hidden_dim'], out_channels=1,
        edge_dim=10, num_layers=hp['num_layers'],
        num_timesteps=hp['num_timesteps'], dropout=hp['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'],
                                 weight_decay=hp['weight_decay'])

    best_val = float('inf')
    best_test = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.mse_loss(out.squeeze(-1), batch.y.squeeze(-1))
            loss.backward()
            optimizer.step()

        val_rmse = eval_rmse(model, val_loader)
        if val_rmse < best_val:
            best_val = val_rmse
            best_test = eval_rmse(model, test_loader)

        if epoch % 50 == 0:
            print(f'    Epoch {epoch:03d} | Val: {val_rmse:.4f} | Best Test: {best_test:.4f}')

    print(f'    Done → Test RMSE: {best_test:.4f}')
    return best_test


# ═════════════════════════════════════════════
# PART 2: MULTI-TASK WITH OPTUNA ARCH + MANUAL WEIGHTS
# ═════════════════════════════════════════════

def run_multitask_optuna(seed):
    print(f'\n  Multi-task Optuna | Seed {seed}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    hp = OPTUNA_HP
    w_esol = MT_WEIGHTS['w_esol']
    w_bace = MT_WEIGHTS['w_bace']

    esol_loader = DataLoader(esol_train, batch_size=hp['batch_size'], shuffle=True)
    bace_loader = DataLoader(bace_train, batch_size=hp['batch_size'], shuffle=True)
    val_esol_loader = DataLoader(esol_val, batch_size=hp['batch_size'])
    val_bace_loader = DataLoader(bace_val, batch_size=hp['batch_size'])
    test_esol_loader = DataLoader(esol_test, batch_size=hp['batch_size'])
    test_bace_loader = DataLoader(bace_test, batch_size=hp['batch_size'])

    model = AttentiveFPMultiTask(hp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'],
                                 weight_decay=hp['weight_decay'])

    best_val_score = float('inf')
    best_test_esol = float('inf')
    best_test_bace = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Train ESOL batches
        for batch in esol_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            esol_pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = w_esol * F.mse_loss(esol_pred.squeeze(-1), batch.y.squeeze(-1))
            loss.backward()
            optimizer.step()

        # Train BACE batches
        for batch in bace_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, bace_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            bace_y = batch.y.float()
            if bace_y.dim() > 1:
                bace_y = bace_y[:, 0]
            nan_mask = ~torch.isnan(bace_y)
            if nan_mask.sum() == 0:
                continue
            loss = w_bace * F.binary_cross_entropy_with_logits(
                bace_pred[nan_mask].squeeze(-1), bace_y[nan_mask])
            loss.backward()
            optimizer.step()

        # Evaluate
        if epoch % 10 == 0 or epoch == EPOCHS:
            val_esol = eval_rmse(model, val_esol_loader, single_task=False)
            val_bace = eval_auc(model, val_bace_loader, single_task=False)
            val_score = val_esol - val_bace

            if val_score < best_val_score:
                best_val_score = val_score
                best_test_esol = eval_rmse(model, test_esol_loader, single_task=False)
                best_test_bace = eval_auc(model, test_bace_loader, single_task=False)

        if epoch % 50 == 0:
            print(f'    Epoch {epoch:03d} | ESOL: {best_test_esol:.4f} | BACE: {best_test_bace:.4f}')

    print(f'    Done → ESOL: {best_test_esol:.4f} | BACE: {best_test_bace:.4f}')
    return best_test_esol, best_test_bace


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

if __name__ == '__main__':
    print('\n' + '█' * 60)
    print('  Final Evaluation — Optuna-Optimized Hyperparameters')
    print('  3 seeds | scaffold split | 200 epochs')
    print('█' * 60)

    # ── Part 1: Single-task ESOL ──
    print('\n' + '=' * 60)
    print('  PART 1: Single-task ESOL (Optuna-optimized)')
    print('=' * 60)

    esol_results = []
    for seed in SEEDS:
        esol_results.append(run_esol_optuna(seed))

    esol_mean = np.mean(esol_results)
    esol_std = np.std(esol_results)

    # ── Part 2: Multi-task ──
    print('\n' + '=' * 60)
    print('  PART 2: Multi-task ESOL+BACE (Optuna arch + w=0.5,1.0)')
    print('=' * 60)

    mt_esol_results = []
    mt_bace_results = []
    for seed in SEEDS:
        e, b = run_multitask_optuna(seed)
        mt_esol_results.append(e)
        mt_bace_results.append(b)

    mt_esol_mean = np.mean(mt_esol_results)
    mt_esol_std = np.std(mt_esol_results)
    mt_bace_mean = np.mean(mt_bace_results)
    mt_bace_std = np.std(mt_bace_results)

    # ── Final Summary ──
    print('\n\n' + '=' * 70)
    print('  FINAL RESULTS — Optuna-Optimized')
    print('=' * 70)
    print(f'  {"Model":<35} {"ESOL RMSE ↓":<20} {"BACE AUC ↑":<20}')
    print(f'  {"-"*65}')
    print(f'  {"Paper defaults (baseline)":<35} {"0.9848 ± 0.0049":<20} {"0.9558 ± 0.0083":<20}')
    print(f'  {"Manual MT w=(0.5,1.0)":<35} {"0.8688 ± 0.0303":<20} {"0.9612 ± 0.0275":<20}')
    print(f'  {"-"*65}')
    print(f'  {"Optuna single-task ESOL":<35} '
          f'{esol_mean:.4f} ± {esol_std:.4f}      {"—":<20}')
    print(f'  {"Optuna MT w=(0.5,1.0)":<35} '
          f'{mt_esol_mean:.4f} ± {mt_esol_std:.4f}      '
          f'{mt_bace_mean:.4f} ± {mt_bace_std:.4f}')
    print('=' * 70)

    print(f'\n  Optuna hyperparameters used:')
    for k, v in OPTUNA_HP.items():
        print(f'    {k}: {v}')

    print(f'\n  Seed breakdown (single-task ESOL):')
    for s, r in zip(SEEDS, esol_results):
        print(f'    Seed {s:>3d} → RMSE: {r:.4f}')

    print(f'\n  Seed breakdown (multi-task):')
    for s, e, b in zip(SEEDS, mt_esol_results, mt_bace_results):
        print(f'    Seed {s:>3d} → ESOL: {e:.4f} | BACE: {b:.4f}')

    print(f'\n  These are your publishable numbers.')
