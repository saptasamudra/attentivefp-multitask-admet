"""
Optuna Hyperparameter Optimization — ESOL Single-Task
Learn how Optuna works before scaling to multi-task.

Run:  python optuna_esol_simple.py
Time: ~30-40 min for 20 trials on GPU
"""

import os.path as osp
from collections import defaultdict
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
import optuna
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ─────────────────────────────────────────────
# FEATURE ENGINEERING (same as baseline)
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
# SCAFFOLD SPLIT (same as baseline)
# ─────────────────────────────────────────────

def scaffold_split(dataset, train_frac=0.8, val_frac=0.1):
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
# LOAD DATASET ONCE (outside objective so it's not reloaded every trial)
# ─────────────────────────────────────────────

print('Loading ESOL dataset...')
path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'ESOL')
dataset = MoleculeNet(path, name='ESOL', pre_transform=GenFeatures())
train_set, val_set, test_set = scaffold_split(dataset)
print(f'  Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}')


# ─────────────────────────────────────────────
# THE OBJECTIVE FUNCTION
# This is the heart of Optuna — it gets called once per trial
# Optuna suggests hyperparameters, you train, return val score
# ─────────────────────────────────────────────

def objective(trial):
    """
    One Optuna trial = one full training run with suggested hyperparameters.
    Returns validation RMSE (lower is better).
    """

    # ── Step 1: Optuna suggests hyperparameters ──
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 200, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    num_timesteps = trial.suggest_int('num_timesteps', 1, 3)
    dropout = trial.suggest_float('dropout', 0.05, 0.4)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 200])

    # Print what this trial is trying
    print(f'\n  Trial {trial.number}: lr={lr:.5f}, hidden={hidden_dim}, '
          f'layers={num_layers}, timesteps={num_timesteps}, '
          f'dropout={dropout:.3f}, batch={batch_size}')

    # ── Step 2: Set seed for reproducibility ──
    torch.manual_seed(42)
    np.random.seed(42)

    # ── Step 3: Create data loaders with suggested batch size ──
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # ── Step 4: Build model with suggested architecture ──
    model = AttentiveFP(
        in_channels=39,
        hidden_channels=hidden_dim,
        out_channels=1,
        edge_dim=10,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # ── Step 5: Train for 150 epochs (fewer than full 200 to save time) ──
    best_val_rmse = float('inf')

    for epoch in range(1, 151):
        # Train
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.mse_loss(out.squeeze(-1), batch.y.squeeze(-1))
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        errors = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                errors.append((out.squeeze(-1) - batch.y.squeeze(-1)).cpu())

        val_rmse = sqrt(torch.cat(errors).pow(2).mean().item())

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse

        # ── Step 6: PRUNING — tell Optuna the intermediate result ──
        # Optuna checks: is this trial hopeless compared to others?
        trial.report(val_rmse, epoch)
        if trial.should_prune():
            print(f'    Pruned at epoch {epoch} (val RMSE: {val_rmse:.4f})')
            raise optuna.TrialPruned()

    print(f'    Finished → Best Val RMSE: {best_val_rmse:.4f}')
    return best_val_rmse


# ─────────────────────────────────────────────
# RUN THE STUDY
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print('\n' + '█' * 52)
    print('  Optuna Hyperparameter Search — ESOL')
    print('  20 trials | seed 42 | scaffold split')
    print('█' * 52)

    # Create study — minimize RMSE
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        study_name='esol_optimization'
    )

    # Run 20 trials (takes ~30-40 min on GPU)
    study.optimize(objective, n_trials=20)

    # ── Results ──
    print('\n\n' + '=' * 52)
    print('  OPTUNA RESULTS — ESOL')
    print('=' * 52)

    print(f'\n  Best trial: #{study.best_trial.number}')
    print(f'  Best Val RMSE: {study.best_value:.4f}')
    print(f'\n  Best hyperparameters:')
    for key, value in study.best_params.items():
        print(f'    {key}: {value}')

    print(f'\n  Comparison:')
    print(f'    Hardcoded baseline:  0.9848 (your current result)')
    print(f'    Optuna best:         {study.best_value:.4f}')
    improvement = ((0.9848 - study.best_value) / 0.9848) * 100
    if improvement > 0:
        print(f'    Improvement:         {improvement:.1f}%')
    else:
        print(f'    No improvement (baseline hyperparams were already good)')

    # ── Show all trials sorted by score ──
    print(f'\n  All trials (sorted by val RMSE):')
    trials = sorted(study.trials, key=lambda t: t.value if t.value else float('inf'))
    for t in trials[:10]:
        status = 'PRUNED' if t.state == optuna.trial.TrialState.PRUNED else 'DONE'
        val = f'{t.value:.4f}' if t.value else 'N/A'
        print(f'    Trial {t.number:>2d} | RMSE: {val} | {status}')

    print(f'\n  Total trials: {len(study.trials)}')
    print(f'  Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}')
    print(f'  Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}')
    print(f'\n  Use these best params in your final model.')
