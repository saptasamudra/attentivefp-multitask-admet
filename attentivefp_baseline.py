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
            single   = 1. if bond_type == Chem.rdchem.BondType.SINGLE   else 0.
            double   = 1. if bond_type == Chem.rdchem.BondType.DOUBLE   else 0.
            triple   = 1. if bond_type == Chem.rdchem.BondType.TRIPLE   else 0.
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
            data.edge_attr  = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_indices).t().contiguous()
            data.edge_attr  = torch.stack(edge_attrs, dim=0)
        return data


# ─────────────────────────────────────────────
# SCAFFOLD SPLIT — REGRESSION VERSION
# For continuous labels (ESOL)
# Groups molecules by scaffold, assigns groups
# to splits greedily largest→train
# ─────────────────────────────────────────────

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=False)


def scaffold_split(dataset, val_frac=0.1, test_frac=0.1, seed=42):
    scaffolds = defaultdict(list)
    for idx, data in enumerate(dataset):
        scaffolds[get_scaffold(data.smiles)].append(idx)

    scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)

    train_size = (1 - val_frac - test_frac) * len(dataset)
    val_size   = val_frac * len(dataset)

    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_groups:
        if len(train_idx) < train_size:
            train_idx.extend(group)
        elif len(val_idx) < val_size:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    rng = np.random.RandomState(seed)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return dataset[train_idx], dataset[val_idx], dataset[test_idx]


# ─────────────────────────────────────────────
# SCAFFOLD SPLIT — CLASSIFICATION VERSION
# For binary labels (BACE)
# Same scaffold grouping, but checks that both
# classes are present in val and test splits.
# If a split ends up with only one class,
# we move scaffold groups between splits to fix it.
# ─────────────────────────────────────────────

def scaffold_split_clf(dataset, val_frac=0.1, test_frac=0.1, seed=42):
    # Step 1: get scaffold groups
    scaffolds = defaultdict(list)
    for idx, data in enumerate(dataset):
        scaffolds[get_scaffold(data.smiles)].append(idx)

    scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)

    # Step 2: get label for each molecule (0 or 1)
    labels = {}
    for idx, data in enumerate(dataset):
        y = data.y.item()
        labels[idx] = int(y) if not np.isnan(y) else -1

    # Step 3: initial greedy assignment
    train_size = (1 - val_frac - test_frac) * len(dataset)
    val_size   = val_frac * len(dataset)

    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_groups:
        if len(train_idx) < train_size:
            train_idx.extend(group)
        elif len(val_idx) < val_size:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    # Step 4: check class balance in val and test
    def has_both_classes(idx_list):
        vals = [labels[i] for i in idx_list if labels[i] != -1]
        return len(set(vals)) >= 2

    # Step 5: if val or test only has one class,
    # swap scaffold groups from train to fix it
    # Re-run split grouping scaffold-by-scaffold
    # with priority to balance val/test first
    if not has_both_classes(val_idx) or not has_both_classes(test_idx):
        # Separate scaffold groups by dominant label
        pos_groups, neg_groups, mixed_groups = [], [], []
        for group in scaffold_groups:
            group_labels = [labels[i] for i in group if labels[i] != -1]
            if not group_labels:
                mixed_groups.append(group)
                continue
            pos_ratio = sum(group_labels) / len(group_labels)
            if pos_ratio == 1.0:
                pos_groups.append(group)
            elif pos_ratio == 0.0:
                neg_groups.append(group)
            else:
                mixed_groups.append(group)

        # Ensure val and test each get at least one pos and one neg group
        train_idx, val_idx, test_idx = [], [], []

        # Seed val and test with one positive and one negative group each
        if len(pos_groups) >= 2 and len(neg_groups) >= 2:
            val_idx.extend(pos_groups[0])
            val_idx.extend(neg_groups[0])
            test_idx.extend(pos_groups[1])
            test_idx.extend(neg_groups[1])
            remaining = pos_groups[2:] + neg_groups[2:] + mixed_groups
        else:
            # fallback: use mixed groups to seed
            val_idx.extend(mixed_groups[0])
            test_idx.extend(mixed_groups[1])
            remaining = scaffold_groups[2:]

        # Fill remaining into splits by size
        for group in sorted(remaining, key=len, reverse=True):
            if len(train_idx) < train_size:
                train_idx.extend(group)
            elif len(val_idx) < val_size:
                val_idx.extend(group)
            else:
                test_idx.extend(group)

    # Step 6: shuffle within splits
    rng = np.random.RandomState(seed)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    # Step 7: print class distribution for verification
    def class_dist(idx_list, name):
        vals = [labels[i] for i in idx_list if labels[i] != -1]
        pos = sum(vals)
        neg = len(vals) - pos
        print(f'    {name}: {len(idx_list)} mols | pos={pos} neg={neg}')

    class_dist(train_idx, 'Train')
    class_dist(val_idx,   'Val  ')
    class_dist(test_idx,  'Test ')

    return dataset[train_idx], dataset[val_idx], dataset[test_idx]


# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
base_path = osp.dirname(osp.realpath(__file__))


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train(model, loader, optimizer, task='regression'):
    model.train()
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        if task == 'regression':
            loss = F.mse_loss(out, data.y)
        else:
            y    = data.y.view(-1)
            pred = out.view(-1)
            mask = ~torch.isnan(y)
            loss = F.binary_cross_entropy_with_logits(pred[mask], y[mask])

        loss.backward()
        optimizer.step()
        total_loss     += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    return total_loss / total_examples


# ─────────────────────────────────────────────
# EVALUATE
# regression  → RMSE  (lower  = better)
# classification → ROC-AUC (higher = better)
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, task='regression'):
    model.eval()

    if task == 'regression':
        mse = []
        for data in loader:
            data = data.to(device)
            out  = model(data.x, data.edge_index, data.edge_attr, data.batch)
            mse.append(F.mse_loss(out, data.y, reduction='none').cpu())
        return float(torch.cat(mse, dim=0).mean().sqrt())

    else:
        all_preds, all_labels = [], []
        for data in loader:
            data = data.to(device)
            out  = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = torch.sigmoid(out).cpu().numpy().flatten()
            y    = data.y.cpu().numpy().flatten()
            mask = ~np.isnan(y)
            all_preds.extend(pred[mask].tolist())
            all_labels.extend(y[mask].tolist())

        # Guard: if only one class present return 0.5 (random)
        if len(set(all_labels)) < 2:
            return 0.5
        return roc_auc_score(all_labels, all_preds)


# ─────────────────────────────────────────────
# ESOL — REGRESSION
# ─────────────────────────────────────────────

def run_esol(seed):
    print(f'\n{"="*52}')
    print(f'  ESOL | Seed {seed}')
    print(f'{"="*52}')
    torch.manual_seed(seed)
    np.random.seed(seed)

    path    = osp.join(base_path, 'data', 'ESOL')
    dataset = MoleculeNet(path, name='ESOL', pre_transform=GenFeatures())
    train_set, val_set, test_set = scaffold_split(dataset, seed=seed)
    print(f'  Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}')

    train_loader = DataLoader(train_set, batch_size=200, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=200)
    test_loader  = DataLoader(test_set,  batch_size=200)

    model = AttentiveFP(
        in_channels=39, hidden_channels=200, out_channels=1,
        edge_dim=10, num_layers=2, num_timesteps=2, dropout=0.2
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=10**-2.5, weight_decay=10**-5)

    best_val  = float('inf')
    best_test = float('inf')

    for epoch in range(1, 201):
        train(model, train_loader, optimizer, task='regression')
        val_rmse = evaluate(model, val_loader, task='regression')

        if val_rmse < best_val:
            best_val  = val_rmse
            best_test = evaluate(model, test_loader, task='regression')
            torch.save(model.state_dict(), f'esol_best_seed{seed}.pt')

        if epoch % 20 == 0 or epoch == 1:
            print(f'  Epoch {epoch:03d} | Val RMSE: {val_rmse:.4f} | '
                  f'Best Test RMSE: {best_test:.4f}')

    print(f'  Seed {seed} → Best Val: {best_val:.4f} | '
          f'Best Test RMSE: {best_test:.4f}')
    return best_test


# ─────────────────────────────────────────────
# BACE — BINARY CLASSIFICATION
# ─────────────────────────────────────────────

def run_bace(seed):
    print(f'\n{"="*52}')
    print(f'  BACE | Seed {seed}')
    print(f'{"="*52}')
    torch.manual_seed(seed)
    np.random.seed(seed)

    path    = osp.join(base_path, 'data', 'BACE')
    dataset = MoleculeNet(path, name='BACE', pre_transform=GenFeatures())
    train_set, val_set, test_set = scaffold_split_clf(dataset, seed=seed)

    train_loader = DataLoader(train_set, batch_size=200, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=200)
    test_loader  = DataLoader(test_set,  batch_size=200)

    model = AttentiveFP(
        in_channels=39, hidden_channels=200, out_channels=1,
        edge_dim=10, num_layers=2, num_timesteps=2, dropout=0.2
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=10**-2.5, weight_decay=10**-5)

    best_val  = 0.0
    best_test = 0.0

    for epoch in range(1, 201):
        train(model, train_loader, optimizer, task='classification')
        val_auc = evaluate(model, val_loader, task='classification')

        if val_auc > best_val:
            best_val  = val_auc
            best_test = evaluate(model, test_loader, task='classification')
            torch.save(model.state_dict(), f'bace_best_seed{seed}.pt')

        if epoch % 20 == 0 or epoch == 1:
            print(f'  Epoch {epoch:03d} | Val AUC: {val_auc:.4f} | '
                  f'Best Test AUC: {best_test:.4f}')

    print(f'  Seed {seed} → Best Val: {best_val:.4f} | '
          f'Best Test AUC: {best_test:.4f}')
    return best_test


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

seeds = [42, 123, 7]

print('\n' + '█'*52)
print('  DATASET 1: ESOL  (regression,      RMSE ↓)')
print('█'*52)
esol_results = [run_esol(s) for s in seeds]

print('\n' + '█'*52)
print('  DATASET 2: BACE  (classification,  AUC  ↑)')
print('█'*52)
bace_results = [run_bace(s) for s in seeds]

# ─────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────

esol_mean, esol_std = np.mean(esol_results), np.std(esol_results)
bace_mean, bace_std = np.mean(bace_results), np.std(bace_results)

print('\n\n' + '='*62)
print('  BASELINE RESULTS — scaffold split — 3 seeds')
print('='*62)
print(f'  {"Model":<24} {"ESOL RMSE ↓":<20} {"BACE AUC ↑"}')
print(f'  {"-"*58}')
print(f'  {"ECFP + RF":<24} {"1.074":<20} {"0.861"}')
print(f'  {"MPNN":<24} {"1.167":<20} {"0.815"}')
print(f'  {"AttentiveFP (paper)":<24} {"0.877":<20} {"0.863"}')
print(f'  {"-"*58}')
print(f'  {"AFP baseline (ours)":<24} '
      f'{esol_mean:.4f} ± {esol_std:.4f}    '
      f'{bace_mean:.4f} ± {bace_std:.4f}')
print('='*62)

print(f'\n  Seed breakdown:')
for s, e, b in zip(seeds, esol_results, bace_results):
    print(f'    Seed {s:>3d} → ESOL: {e:.4f}  |  BACE AUC: {b:.4f}')

print(f'\n  ESOL is locked. BACE is locked.')
print(f'  Next step: implement multi-task head and beat these numbers.')

