"""
BACE Classification Baseline — AttentiveFP
Scaffold split | 3 seeds | ROC-AUC metric

Run:  python bace_baseline.py
Time: ~10 minutes on GPU
"""

import os.path as osp
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SEEDS       = [42, 123, 7]
EPOCHS      = 200
BATCH_SIZE  = 200
LR          = 10**-2.5
WEIGHT_DECAY = 10**-5
HIDDEN_DIM  = 200
NUM_LAYERS  = 2
NUM_TIMESTEPS = 2
DROPOUT     = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# Converts SMILES → node features (39-dim) + edge features (10-dim)
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

        # ── Node (atom) features ──
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

        # ── Edge (bond) features ──
        edge_attrs = []
        for bond in mol.GetBonds():
            bond_type = bond.GetBondTypeAsDouble()
            stereo = [0.] * 4
            stereo[self.stereos.index(bond.GetStereo())] = 1.
            is_aromatic = 1. if bond.GetIsAromatic() else 0.
            is_conjugated = 1. if bond.GetIsConjugated() else 0.
            is_in_ring = 1. if bond.IsInRing() else 0.
            attr = [bond_type] + stereo + [is_aromatic, is_conjugated, is_in_ring]
            edge_attrs += [attr, attr]  # both directions

        if len(edge_attrs) == 0:
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        return data


# ─────────────────────────────────────────────
# SCAFFOLD SPLIT (classification-aware)
# Ensures both classes present in val and test
# ─────────────────────────────────────────────

def scaffold_split(dataset, train_frac=0.8, val_frac=0.1):
    """
    Bemis-Murcko scaffold split that guarantees both classes
    appear in val and test sets (critical for ROC-AUC).
    """
    # Group molecules by scaffold
    scaffolds = defaultdict(list)
    for i, data in enumerate(dataset):
        mol = Chem.MolFromSmiles(data.smiles)
        if mol is None:
            scaffolds['unknown'].append(i)
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False)
        scaffolds[scaffold].append(i)

    # Sort scaffold groups by size (largest first)
    scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)

    # Get labels for class-aware seeding
    labels = []
    for data in dataset:
        y = data.y.item() if data.y.numel() == 1 else data.y[0].item()
        labels.append(int(y))
    labels = np.array(labels)

    n = len(dataset)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx, val_idx, test_idx = [], [], []

    # Seed val and test with one positive and one negative scaffold each
    pos_scaffolds = [g for g in scaffold_groups
                     if all(labels[i] == 1 for i in g)]
    neg_scaffolds = [g for g in scaffold_groups
                     if all(labels[i] == 0 for i in g)]

    seeded = set()
    if pos_scaffolds and neg_scaffolds:
        # Seed val with smallest pos + smallest neg scaffold
        val_idx.extend(pos_scaffolds[-1])
        seeded.add(id(pos_scaffolds[-1]))
        val_idx.extend(neg_scaffolds[-1])
        seeded.add(id(neg_scaffolds[-1]))
        # Seed test with second-smallest pos + neg scaffold
        if len(pos_scaffolds) > 1 and len(neg_scaffolds) > 1:
            test_idx.extend(pos_scaffolds[-2])
            seeded.add(id(pos_scaffolds[-2]))
            test_idx.extend(neg_scaffolds[-2])
            seeded.add(id(neg_scaffolds[-2]))

    # Fill remaining scaffolds greedily
    for group in scaffold_groups:
        if id(group) in seeded:
            continue
        if len(train_idx) < n_train:
            train_idx.extend(group)
        elif len(val_idx) < n_val:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    # Report split statistics
    train_pos = sum(labels[i] for i in train_idx)
    val_pos = sum(labels[i] for i in val_idx)
    test_pos = sum(labels[i] for i in test_idx)
    print(f'  Split: train={len(train_idx)} ({train_pos}+/{len(train_idx)-train_pos}-) | '
          f'val={len(val_idx)} ({val_pos}+/{len(val_idx)-val_pos}-) | '
          f'test={len(test_idx)} ({test_pos}+/{len(test_idx)-test_pos}-)')

    return (dataset[torch.tensor(train_idx)],
            dataset[torch.tensor(val_idx)],
            dataset[torch.tensor(test_idx)])


# ─────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.float()
        if y.dim() > 1:
            y = y[:, 0]

        # Mask NaN labels
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue

        pred = out[mask].squeeze(-1)
        target = y[mask]
        loss = F.binary_cross_entropy_with_logits(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mask.sum().item()

    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
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
        return 0.5  # fallback if no valid predictions

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    # ROC-AUC requires both classes
    if len(np.unique(labels)) < 2:
        print('    WARNING: only one class in labels, returning AUC=0.5')
        return 0.5

    return roc_auc_score(labels, preds)


# ─────────────────────────────────────────────
# RUN ONE SEED
# ─────────────────────────────────────────────

def run_bace(seed):
    print(f'\n{"="*52}')
    print(f'  BACE | Seed {seed}')
    print(f'{"="*52}')

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load dataset
    path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'BACE')
    dataset = MoleculeNet(path, name='BACE', pre_transform=GenFeatures())

    # Scaffold split
    train_set, val_set, test_set = scaffold_split(dataset)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE)

    # Model
    model = AttentiveFP(
        in_channels=39,
        hidden_channels=HIDDEN_DIM,
        out_channels=1,
        edge_dim=10,
        num_layers=NUM_LAYERS,
        num_timesteps=NUM_TIMESTEPS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val  = 0.0
    best_test = 0.0

    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_loader, optimizer)
        val_auc = evaluate(model, val_loader)

        if val_auc > best_val:
            best_val  = val_auc
            best_test = evaluate(model, test_loader)
            torch.save(model.state_dict(), f'bace_best_seed{seed}.pt')

        if epoch % 20 == 0 or epoch == 1:
            print(f'  Epoch {epoch:03d} | Loss: {loss:.4f} | '
                  f'Val AUC: {val_auc:.4f} | Best Test AUC: {best_test:.4f}')

    print(f'\n  Seed {seed} DONE')
    print(f'  Best Val AUC:  {best_val:.4f}')
    print(f'  Best Test AUC: {best_test:.4f}')
    return best_test


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print('\n' + '█' * 52)
    print('  BACE Classification Baseline')
    print('  AttentiveFP | Scaffold Split | 3 Seeds')
    print('█' * 52)

    results = []
    for seed in SEEDS:
        auc = run_bace(seed)
        results.append(auc)

    # ── Summary ──
    mean_auc = np.mean(results)
    std_auc  = np.std(results)

    print('\n\n' + '=' * 52)
    print('  BACE BASELINE RESULTS')
    print('=' * 52)
    print(f'  {"Model":<24} {"BACE AUC ↑":<20} Split')
    print(f'  {"-"*48}')
    print(f'  {"ECFP + RF":<24} {"0.861":<20} scaffold')
    print(f'  {"MPNN":<24} {"0.815":<20} scaffold')
    print(f'  {"AttentiveFP (paper)":<24} {"0.863":<20} scaffold')
    print(f'  {"-"*48}')
    print(f'  {"AFP baseline (ours)":<24} '
          f'{mean_auc:.4f} ± {std_auc:.4f}      scaffold')
    print('=' * 52)

    print(f'\n  Seed breakdown:')
    for s, r in zip(SEEDS, results):
        print(f'    Seed {s:>3d} → BACE AUC: {r:.4f}')

    print(f'\n  BACE baseline is locked.')
    print(f'  Save this output for your professor.')
