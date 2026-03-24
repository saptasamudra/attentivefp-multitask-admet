"""
Optuna Hyperparameter Optimization — Multi-Task AttentiveFP
Searches: lr, hidden_dim, layers, timesteps, dropout, batch_size, w_esol, w_bace

Run overnight:  python optuna_multitask.py
Time: ~3-5 hours for 50 trials on GPU
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
from sklearn.metrics import roc_auc_score

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP
from torch.nn import Linear


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


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
        labels = []
        for data in dataset:
            y = data.y.item() if data.y.numel() == 1 else data.y[0].item()
            labels.append(int(y))
        labels = np.array(labels)

        pos_scaffolds = [g for g in scaffold_groups if all(labels[i] == 1 for i in g)]
        neg_scaffolds = [g for g in scaffold_groups if all(labels[i] == 0 for i in g)]

        seeded = set()
        if pos_scaffolds and neg_scaffolds:
            val_idx.extend(pos_scaffolds[-1])
            seeded.add(id(pos_scaffolds[-1]))
            val_idx.extend(neg_scaffolds[-1])
            seeded.add(id(neg_scaffolds[-1]))
            if len(pos_scaffolds) > 1 and len(neg_scaffolds) > 1:
                test_idx.extend(pos_scaffolds[-2])
                seeded.add(id(pos_scaffolds[-2]))
                test_idx.extend(neg_scaffolds[-2])
                seeded.add(id(neg_scaffolds[-2]))

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
    def __init__(self, hidden_channels, num_layers, num_timesteps, dropout):
        super().__init__()
        self.encoder = AttentiveFP(
            in_channels=39,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=10,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
        self.esol_head = Linear(hidden_channels, 1)
        self.bace_head = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder(x, edge_index, edge_attr, batch)
        return self.esol_head(h), self.bace_head(h)


# ─────────────────────────────────────────────
# LOAD DATASETS ONCE
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

def eval_esol(model, loader):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            esol_pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            errors.append((esol_pred.squeeze(-1) - batch.y.squeeze(-1)).cpu())
    return sqrt(torch.cat(errors).pow(2).mean().item())


def eval_bace(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, bace_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y = batch.y.float()
            if y.dim() > 1:
                y = y[:, 0]
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                continue
            pred = torch.sigmoid(bace_pred[mask].squeeze(-1))
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y[mask].cpu().numpy())

    if not all_preds:
        return 0.5
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, preds)


# ─────────────────────────────────────────────
# OBJECTIVE FUNCTION — 7 hyperparameters
# ─────────────────────────────────────────────

def objective(trial):
    # ── Optuna suggests all 7 hyperparameters ──
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 200, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    num_timesteps = trial.suggest_int('num_timesteps', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 200])
    w_esol = trial.suggest_float('w_esol', 0.1, 2.0)
    w_bace = trial.suggest_float('w_bace', 0.1, 2.0)

    print(f'\n  Trial {trial.number}: lr={lr:.5f}, hidden={hidden_dim}, '
          f'layers={num_layers}, ts={num_timesteps}, drop={dropout:.3f}, '
          f'batch={batch_size}, w_esol={w_esol:.2f}, w_bace={w_bace:.2f}')

    torch.manual_seed(42)
    np.random.seed(42)

    # Separate loaders — no need to tag data, much more reliable
    esol_train_loader = DataLoader(list(esol_train), batch_size=batch_size, shuffle=True)
    bace_train_loader = DataLoader(list(bace_train), batch_size=batch_size, shuffle=True)
    val_esol_loader = DataLoader(list(esol_val), batch_size=batch_size)
    val_bace_loader = DataLoader(list(bace_val), batch_size=batch_size)

    model = AttentiveFPMultiTask(
        hidden_channels=hidden_dim,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_combined_score = float('inf')

    for epoch in range(1, 201):
        # ── Train: alternate between ESOL and BACE batches ──
        model.train()

        esol_iter = iter(esol_train_loader)
        bace_iter = iter(bace_train_loader)

        # Train on all ESOL batches
        for batch in esol_iter:
            batch = batch.to(device)
            optimizer.zero_grad()
            esol_pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            esol_loss = w_esol * F.mse_loss(
                esol_pred.squeeze(-1), batch.y.squeeze(-1))
            esol_loss.backward()
            optimizer.step()

        # Train on all BACE batches
        for batch in bace_iter:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, bace_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            bace_y = batch.y.float()
            if bace_y.dim() > 1:
                bace_y = bace_y[:, 0]
            nan_mask = ~torch.isnan(bace_y)
            if nan_mask.sum() == 0:
                continue
            bace_loss = w_bace * F.binary_cross_entropy_with_logits(
                bace_pred[nan_mask].squeeze(-1), bace_y[nan_mask])
            bace_loss.backward()
            optimizer.step()

        # ── Evaluate every 10 epochs ──
        if epoch % 10 == 0 or epoch == 1:
            val_esol_rmse = eval_esol(model, val_esol_loader)
            val_bace_auc = eval_bace(model, val_bace_loader)

            # Combined score: lower is better
            # Normalize: RMSE around 0.8-1.0, AUC around 0.8-1.0
            # Score = ESOL_RMSE - BACE_AUC (lower = better ESOL + higher BACE)
            combined = val_esol_rmse - val_bace_auc

            if combined < best_combined_score:
                best_combined_score = combined

            # Report to Optuna for pruning
            trial.report(combined, epoch)
            if trial.should_prune():
                print(f'    Pruned at epoch {epoch} '
                      f'(ESOL: {val_esol_rmse:.4f}, BACE: {val_bace_auc:.4f})')
                raise optuna.TrialPruned()

    # Final evaluation
    final_esol = eval_esol(model, val_esol_loader)
    final_bace = eval_bace(model, val_bace_loader)
    final_score = final_esol - final_bace

    print(f'    Done → ESOL: {final_esol:.4f} | BACE: {final_bace:.4f} | '
          f'Score: {final_score:.4f}')

    # Store individual metrics for later analysis
    trial.set_user_attr('val_esol_rmse', final_esol)
    trial.set_user_attr('val_bace_auc', final_bace)

    return final_score


# ─────────────────────────────────────────────
# RUN THE STUDY
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print('\n' + '█' * 56)
    print('  Optuna Multi-Task Optimization')
    print('  50 trials | 7 hyperparameters | scaffold split')
    print('█' * 56)

    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=30,
        ),
        study_name='multitask_optimization'
    )

    study.optimize(objective, n_trials=50)

    # ── Results ──
    print('\n\n' + '=' * 60)
    print('  OPTUNA MULTI-TASK RESULTS')
    print('=' * 60)

    best = study.best_trial
    print(f'\n  Best trial: #{best.number}')
    print(f'  Best combined score: {best.value:.4f}')
    print(f'  Val ESOL RMSE: {best.user_attrs.get("val_esol_rmse", "N/A")}')
    print(f'  Val BACE AUC:  {best.user_attrs.get("val_bace_auc", "N/A")}')

    print(f'\n  Best hyperparameters:')
    for key, value in best.params.items():
        if isinstance(value, float):
            print(f'    {key}: {value:.6f}')
        else:
            print(f'    {key}: {value}')

    print(f'\n  Comparison with manual results:')
    print(f'    {"Method":<30} {"ESOL RMSE":<15} {"BACE AUC":<15}')
    print(f'    {"-"*55}')
    print(f'    {"Single-task baseline":<30} {"0.9848":<15} {"0.9558":<15}')
    print(f'    {"Manual MT w=(0.5,1.0)":<30} {"0.8688":<15} {"0.9612":<15}')

    best_esol = best.user_attrs.get("val_esol_rmse", "?")
    best_bace = best.user_attrs.get("val_bace_auc", "?")
    if isinstance(best_esol, float):
        print(f'    {"Optuna best":<30} {best_esol:<15.4f} {best_bace:<15.4f}')
    else:
        print(f'    {"Optuna best":<30} {best_esol:<15} {best_bace:<15}')

    # ── Top 10 trials ──
    print(f'\n  Top 10 trials:')
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)
    for t in completed[:10]:
        esol = t.user_attrs.get("val_esol_rmse", "?")
        bace = t.user_attrs.get("val_bace_auc", "?")
        if isinstance(esol, float):
            print(f'    Trial {t.number:>2d} | ESOL: {esol:.4f} | BACE: {bace:.4f} | '
                  f'Score: {t.value:.4f}')

    print(f'\n  Total trials: {len(study.trials)}')
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f'  Pruned: {pruned}')
    print(f'  Completed: {len(completed)}')

    print(f'\n  Next: take these best params and run 3 seeds for final results.')
    print(f'  Update your paper Table 1 with the Optuna-optimized numbers.')
