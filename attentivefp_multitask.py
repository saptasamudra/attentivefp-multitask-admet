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
from torch.nn import Linear


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
# SCAFFOLD SPLITS
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
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return dataset[train_idx], dataset[val_idx], dataset[test_idx]


def scaffold_split_clf(dataset, val_frac=0.1, test_frac=0.1, seed=42):
    scaffolds = defaultdict(list)
    for idx, data in enumerate(dataset):
        scaffolds[get_scaffold(data.smiles)].append(idx)
    scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)
    labels = {}
    for idx, data in enumerate(dataset):
        y = data.y.item()
        labels[idx] = int(y) if not np.isnan(y) else -1
    train_size = (1 - val_frac - test_frac) * len(dataset)
    val_size   = val_frac * len(dataset)
    train_idx, val_idx, test_idx = [], [], []
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
    if len(pos_groups) >= 2 and len(neg_groups) >= 2:
        val_idx.extend(pos_groups[0]); val_idx.extend(neg_groups[0])
        test_idx.extend(pos_groups[1]); test_idx.extend(neg_groups[1])
        remaining = pos_groups[2:] + neg_groups[2:] + mixed_groups
    else:
        val_idx.extend(mixed_groups[0]); test_idx.extend(mixed_groups[1])
        remaining = scaffold_groups[2:]
    for group in sorted(remaining, key=len, reverse=True):
        if len(train_idx) < train_size:
            train_idx.extend(group)
        elif len(val_idx) < val_size:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
    rng = np.random.RandomState(seed)
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return dataset[train_idx], dataset[val_idx], dataset[test_idx]


# ─────────────────────────────────────────────
# MULTI-TASK MODEL
# ─────────────────────────────────────────────

class AttentiveFPMultiTask(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim,
                 num_layers, num_timesteps, dropout):
        super().__init__()
        self.encoder = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
        self.esol_head = Linear(hidden_channels, 1)
        self.bace_head = Linear(hidden_channels, 1)
        self.dropout   = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        mol_repr  = self.encoder(x, edge_index, edge_attr, batch)
        mol_repr  = F.dropout(mol_repr, p=self.dropout,
                              training=self.training)
        esol_pred = self.esol_head(mol_repr)
        bace_pred = self.bace_head(mol_repr)
        return esol_pred, bace_pred


# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
base_path = osp.dirname(osp.realpath(__file__))


def tag_dataset(dataset, task_id):
    tagged = []
    for data in dataset:
        data.task = task_id
        tagged.append(data)
    return tagged


# ─────────────────────────────────────────────
# WEIGHTED TRAIN
# loss = w_esol * esol_loss + w_bace * bace_loss
# ─────────────────────────────────────────────

def train_multitask(model, loader, optimizer, w_esol=1.0, w_bace=1.0):
    model.train()
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        esol_pred, bace_pred = model(
            data.x, data.edge_index, data.edge_attr, data.batch)
        esol_mask = (data.task == 0)
        bace_mask = (data.task == 1)
        loss = torch.tensor(0.0, device=device)
        if esol_mask.sum() > 0:
            loss = loss + w_esol * F.mse_loss(
                esol_pred[esol_mask], data.y[esol_mask])
        if bace_mask.sum() > 0:
            bace_y   = data.y[bace_mask].view(-1)
            bace_out = bace_pred[bace_mask].view(-1)
            nan_mask = ~torch.isnan(bace_y)
            if nan_mask.sum() > 0:
                loss = loss + w_bace * F.binary_cross_entropy_with_logits(
                    bace_out[nan_mask], bace_y[nan_mask])
        if loss.item() > 0:
            loss.backward()
            optimizer.step()
        total_loss     += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    return total_loss / total_examples


@torch.no_grad()
def evaluate_esol(model, loader):
    model.eval()
    mse = []
    for data in loader:
        data = data.to(device)
        esol_pred, _ = model(
            data.x, data.edge_index, data.edge_attr, data.batch)
        mse.append(F.mse_loss(esol_pred, data.y, reduction='none').cpu())
    return float(torch.cat(mse, dim=0).mean().sqrt())


@torch.no_grad()
def evaluate_bace(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(device)
        _, bace_pred = model(
            data.x, data.edge_index, data.edge_attr, data.batch)
        pred = torch.sigmoid(bace_pred).cpu().numpy().flatten()
        y    = data.y.cpu().numpy().flatten()
        mask = ~np.isnan(y)
        all_preds.extend(pred[mask].tolist())
        all_labels.extend(y[mask].tolist())
    if len(set(all_labels)) < 2:
        return 0.5
    return roc_auc_score(all_labels, all_preds)


# ─────────────────────────────────────────────
# RUN ONE WEIGHT COMBINATION ACROSS 3 SEEDS
# ─────────────────────────────────────────────

def run_weight_combo(w_esol, w_bace, esol_data, bace_data,
                     seeds=[42, 123, 7]):
    print(f'\n{"▓"*52}')
    print(f'  w_esol={w_esol}  w_bace={w_bace}')
    print(f'{"▓"*52}')

    esol_results, bace_results = [], []

    for seed in seeds:
        print(f'\n  --- Seed {seed} ---')
        torch.manual_seed(seed)
        np.random.seed(seed)

        esol_train, esol_val, esol_test = scaffold_split(
            esol_data, seed=seed)
        bace_train, bace_val, bace_test = scaffold_split_clf(
            bace_data, seed=seed)

        combined_train = (tag_dataset(list(esol_train), 0) +
                          tag_dataset(list(bace_train), 1))
        train_loader     = DataLoader(combined_train, batch_size=200,
                                      shuffle=True)
        esol_val_loader  = DataLoader(list(esol_val),  batch_size=200)
        esol_test_loader = DataLoader(list(esol_test), batch_size=200)
        bace_val_loader  = DataLoader(list(bace_val),  batch_size=200)
        bace_test_loader = DataLoader(list(bace_test), batch_size=200)

        model = AttentiveFPMultiTask(
            in_channels=39, hidden_channels=200, edge_dim=10,
            num_layers=2, num_timesteps=2, dropout=0.2
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=10**-2.5, weight_decay=10**-5)

        best_esol_val  = float('inf')
        best_esol_test = float('inf')
        best_bace_val  = 0.0
        best_bace_test = 0.0

        for epoch in range(1, 201):
            train_multitask(model, train_loader, optimizer,
                            w_esol=w_esol, w_bace=w_bace)
            esol_val_rmse = evaluate_esol(model, esol_val_loader)
            bace_val_auc  = evaluate_bace(model, bace_val_loader)

            if esol_val_rmse < best_esol_val:
                best_esol_val  = esol_val_rmse
                best_esol_test = evaluate_esol(model, esol_test_loader)

            if bace_val_auc > best_bace_val:
                best_bace_val  = bace_val_auc
                best_bace_test = evaluate_bace(model, bace_test_loader)

            if epoch % 40 == 0 or epoch == 1:
                print(f'  Ep {epoch:03d} | '
                      f'ESOL: {esol_val_rmse:.4f} '
                      f'(test:{best_esol_test:.4f}) | '
                      f'BACE: {bace_val_auc:.4f} '
                      f'(test:{best_bace_test:.4f})')

        torch.save(model.state_dict(),
                   f'mt_w{w_esol}_{w_bace}_seed{seed}.pt')
        print(f'  Seed {seed} → ESOL: {best_esol_test:.4f} | '
              f'BACE: {best_bace_test:.4f}')
        esol_results.append(best_esol_test)
        bace_results.append(best_bace_test)

    return (np.mean(esol_results), np.std(esol_results),
            np.mean(bace_results), np.std(bace_results))


# ─────────────────────────────────────────────
# WEIGHT SEARCH
# Three combos → becomes ablation Table 2
# (1.0, 1.0) already known from v1
# (1.0, 2.0) upweight BACE
# (0.5, 1.0) downweight ESOL
# ─────────────────────────────────────────────

# Load datasets once — cached after first run
esol_path = osp.join(base_path, 'data', 'ESOL')
esol_data = MoleculeNet(esol_path, name='ESOL', pre_transform=GenFeatures())
bace_path = osp.join(base_path, 'data', 'BACE')
bace_data = MoleculeNet(bace_path, name='BACE', pre_transform=GenFeatures())

weight_combos = [
    (1.0, 1.0),   # equal — v1 result
    (1.0, 2.0),   # upweight BACE
    (0.5, 1.0),   # downweight ESOL
]

all_results = {}
for w_esol, w_bace in weight_combos:
    em, es, bm, bs = run_weight_combo(
        w_esol, w_bace, esol_data, bace_data)
    all_results[(w_esol, w_bace)] = (em, es, bm, bs)

# ─────────────────────────────────────────────
# FINAL TABLE
# ─────────────────────────────────────────────

print('\n\n' + '='*70)
print('  FULL RESULTS — scaffold split — 3 seeds')
print('='*70)
print(f'  {"Model":<32} {"ESOL RMSE ↓":<22} {"BACE AUC ↑"}')
print(f'  {"-"*66}')
print(f'  {"ECFP + RF":<32} {"1.074":<22} {"0.861"}')
print(f'  {"MPNN":<32} {"1.167":<22} {"0.815"}')
print(f'  {"AttentiveFP (paper)":<32} {"0.877":<22} {"0.863"}')
print(f'  {"AFP single-task (ours)":<32} '
      f'{"0.9791 ± 0.0238":<22} {"0.9708 ± 0.0145"}')
print(f'  {"-"*66}')

for (we, wb), (em, es, bm, bs) in all_results.items():
    label = f'AFP-MT w=({we},{wb})'
    print(f'  {label:<32} '
          f'{em:.4f} ± {es:.4f}       '
          f'{bm:.4f} ± {bs:.4f}')

print('='*70)

# Find best combo
print('\n  Per-combo improvement over single-task baseline:')
for (we, wb), (em, es, bm, bs) in all_results.items():
    esol_ok = '✓' if em < 0.9791 else '✗'
    bace_ok = '✓' if bm > 0.9708 else '✗'
    print(f'  w=({we},{wb}) → '
          f'ESOL {esol_ok} {em:.4f}  |  '
          f'BACE {bace_ok} {bm:.4f}')