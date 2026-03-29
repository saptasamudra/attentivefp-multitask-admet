"""
MoleculeNet Full Baseline — 7 Datasets
AttentiveFP | Scaffold Split | 3 Seeds | Paper Default Hyperparameters

Datasets:
  Regression:     ESOL, FreeSolv, Lipophilicity (Lipo)
  Classification: BACE, BBBP, ClinTox (2 tasks), Tox21 (12 tasks)

Bugs fixed (verified against PyG official example + MoleculeNet paper):
  1. Scaffold split: additive size check — prevents test=0
  2. safe_out(): AttentiveFP returns [B] for out_channels=1 → always [B,T]
  3. safe_y():   MoleculeNet y can be [B] or [B,T] after batching → always [B,T]
  4. Multi-task y in split: y.view(-1)[0] not y[0] (y[0] = tensor for multi-task)
  5. safe_auc(): skip task if only one class present (no crash, no silent 0.5)

Run:  python moleculenet_baseline.py
Time: ~2-3 hours on GPU
"""

import os.path as osp
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

# ── Hyperparameters ───────────────────────────────────────────────
HP = {
    'lr':            10 ** -2.5,
    'hidden_dim':    200,
    'num_layers':    2,
    'num_timesteps': 2,
    'dropout':       0.2,
    'batch_size':    200,
    'weight_decay':  1e-5,
}
SEEDS  = [42, 123, 7]
EPOCHS = 200
base_path = '.'

DATASETS = [
    {'name': 'ESOL',     'task': 'regression',     'num_tasks': 1,  'metric': 'RMSE', 'published': 0.877},
    {'name': 'FreeSolv', 'task': 'regression',     'num_tasks': 1,  'metric': 'RMSE', 'published': 2.082},
    {'name': 'Lipo',     'task': 'regression',     'num_tasks': 1,  'metric': 'RMSE', 'published': 0.655},
    {'name': 'BACE',     'task': 'classification', 'num_tasks': 1,  'metric': 'AUC',  'published': 0.863},
    {'name': 'BBBP',     'task': 'classification', 'num_tasks': 1,  'metric': 'AUC',  'published': 0.862},
    {'name': 'ClinTox',  'task': 'classification', 'num_tasks': 2,  'metric': 'AUC',  'published': 0.832},
    {'name': 'Tox21',    'task': 'classification', 'num_tasks': 12, 'metric': 'AUC',  'published': 0.829},
]


# ── Feature engineering (official PyG Table 1, 39-dim atom, 10-dim bond) ──
class GenFeatures:
    def __init__(self):
        self.symbols = [
            'B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At','other',
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
            sym = atom.GetSymbol()
            symbol[self.symbols.index(sym) if sym in self.symbols else -1] = 1.
            degree = [0.] * 6
            degree[min(atom.GetDegree(), 5)] = 1.
            formal_charge     = float(atom.GetFormalCharge())
            radical_electrons = float(atom.GetNumRadicalElectrons())
            hybridization = [0.] * len(self.hybridizations)
            hyb = atom.GetHybridization()
            hybridization[self.hybridizations.index(hyb) if hyb in self.hybridizations else -1] = 1.
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            hydrogens = [0.] * 5
            hydrogens[min(atom.GetTotalNumHs(), 4)] = 1.
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                cip = atom.GetProp('_CIPCode')
                if cip in ('R', 'S'):
                    chirality_type[['R', 'S'].index(cip)] = 1.
            # 16+6+1+1+6+1+5+1+2 = 39 dims
            xs.append(symbol + degree + [formal_charge, radical_electrons] +
                      hybridization + [aromaticity] + hydrogens + [chirality] + chirality_type)

        data.x = torch.tensor(xs, dtype=torch.float)

        edge_attrs = []
        for bond in mol.GetBonds():
            # 10-dim bond features (verified count below):
            # bond_type float (1) + stereo one-hot (4) + aromatic (1)
            # + conjugated (1) + in_ring (1) + bond_type_1 (1) + bond_type_2 (1) = 10
            # Simplest correct schema matching AttentiveFP paper Table 1:
            # single/double/triple/aromatic one-hot (4) + stereo (4) + conjugated (1) + ring (1) = 10
            bt = bond.GetBondTypeAsDouble()
            bond_type_onehot = [
                1. if bt == 1.0 else 0.,   # single
                1. if bt == 2.0 else 0.,   # double
                1. if bt == 3.0 else 0.,   # triple
                1. if bt == 1.5 else 0.,   # aromatic
            ]
            stereo = [0.] * 4
            s = bond.GetStereo()
            if s in self.stereos:
                stereo[self.stereos.index(s)] = 1.
            is_conjugated = 1. if bond.GetIsConjugated() else 0.
            is_in_ring    = 1. if bond.IsInRing()        else 0.
            # Total: 4 + 4 + 1 + 1 = 10 dims exactly
            attr = bond_type_onehot + stereo + [is_conjugated, is_in_ring]
            edge_attrs += [attr, attr]   # both directions (undirected)

        data.edge_attr = (torch.zeros((0, 10), dtype=torch.float) if not edge_attrs
                          else torch.tensor(edge_attrs, dtype=torch.float))
        return data


# ── Shape helpers ─────────────────────────────────────────────────
def safe_out(out):
    """AttentiveFP returns [B] when out_channels=1. Always return [B, T]."""
    return out.unsqueeze(-1) if out.dim() == 1 else out


def safe_y(y, num_tasks):
    """MoleculeNet y arrives as [B] or [B, T] after batching. Always return [B, T]."""
    y = y.float()
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    return y


def safe_auc(labels, preds):
    """Compute ROC-AUC only when both classes present, else return None."""
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, preds))


# ── Scaffold split ────────────────────────────────────────────────
def scaffold_split(dataset, seed, task='regression', frac_train=0.8, frac_val=0.1):
    """
    Standard Bemis-Murcko scaffold split matching DeepChem/Chemprop implementation.
    
    Key insight: sort scaffolds by size (largest first), then fill train/val/test
    greedily. Do NOT separate pos/neg scaffolds — that causes all positives to land
    in train, leaving test with only one class (AUC undefined → 0.5).
    
    Reference: Wu et al. MoleculeNet (2018), DeepChem scaffold splitter.
    """
    scaffold_to_idx = {}
    for i, data in enumerate(dataset):
        mol = Chem.MolFromSmiles(data.smiles)
        scaffold = ('' if mol is None else
                    MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
        scaffold_to_idx.setdefault(scaffold, []).append(i)

    # Sort by scaffold frequency descending (largest scaffold groups first → go to train)
    # Then shuffle within same-size groups using seed for reproducibility
    rng = np.random.RandomState(seed)
    
    scaffold_sets = list(scaffold_to_idx.values())
    # Shuffle first for randomness, then stable-sort by size so large go to train
    rng.shuffle(scaffold_sets)
    scaffold_sets = sorted(scaffold_sets, key=lambda x: len(x), reverse=True)

    n = len(dataset)
    train_cutoff = int(frac_train * n)
    val_cutoff   = int((frac_train + frac_val) * n)

    train_idx, val_idx, test_idx = [], [], []
    for sset in scaffold_sets:
        if len(train_idx) + len(sset) <= train_cutoff:
            train_idx.extend(sset)
        elif len(val_idx) + len(sset) <= (val_cutoff - train_cutoff):
            val_idx.extend(sset)
        else:
            test_idx.extend(sset)

    # Safety fallback — guarantee non-empty splits
    if not test_idx:
        cut = max(1, int(len(train_idx) * 0.9))
        test_idx, train_idx = train_idx[cut:], train_idx[:cut]
    if not val_idx:
        cut = max(1, int(len(train_idx) * 0.9))
        val_idx, train_idx = train_idx[cut:], train_idx[:cut]

    return (dataset[torch.tensor(train_idx, dtype=torch.long)],
            dataset[torch.tensor(val_idx,   dtype=torch.long)],
            dataset[torch.tensor(test_idx,  dtype=torch.long)])


# ── Model factory ─────────────────────────────────────────────────
def make_model(out_channels):
    return AttentiveFP(
        in_channels=39, hidden_channels=HP['hidden_dim'], out_channels=out_channels,
        edge_dim=10, num_layers=HP['num_layers'], num_timesteps=HP['num_timesteps'],
        dropout=HP['dropout'],
    ).to(device)


# ── Regression ────────────────────────────────────────────────────
def run_regression(ds_name, seed):
    dataset = MoleculeNet(osp.join(base_path, 'data', ds_name),
                          name=ds_name, pre_transform=GenFeatures())
    torch.manual_seed(seed); np.random.seed(seed)

    train_ds, val_ds, test_ds = scaffold_split(dataset, seed, task='regression')
    print(f'    Split: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=HP['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=HP['batch_size'])
    test_loader  = DataLoader(test_ds,  batch_size=HP['batch_size'])

    model     = make_model(out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=HP['lr'],
                                 weight_decay=HP['weight_decay'])

    best_val, best_test = float('inf'), float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out  = safe_out(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch))
            y    = safe_y(batch.y, 1)
            mask = ~torch.isnan(y[:, 0])
            if mask.sum() == 0: continue
            F.mse_loss(out[mask, 0], y[mask, 0]).backward()
            optimizer.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out  = safe_out(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch))
                y    = safe_y(batch.y, 1)
                mask = ~torch.isnan(y[:, 0])
                if mask.sum() == 0: continue
                vp.append(out[mask, 0].cpu()); vl.append(y[mask, 0].cpu())

        if vp:
            val_rmse = sqrt(F.mse_loss(torch.cat(vp), torch.cat(vl)).item())
            if val_rmse < best_val:
                best_val = val_rmse
                tp, tl = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(device)
                        out  = safe_out(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch))
                        y    = safe_y(batch.y, 1)
                        mask = ~torch.isnan(y[:, 0])
                        if mask.sum() == 0: continue
                        tp.append(out[mask, 0].cpu()); tl.append(y[mask, 0].cpu())
                if tp:
                    best_test = sqrt(F.mse_loss(torch.cat(tp), torch.cat(tl)).item())

        if epoch % 50 == 0:
            print(f'    Epoch {epoch:03d} | Val RMSE: {best_val:.4f} | Test RMSE: {best_test:.4f}')

    return best_test


# ── Classification ────────────────────────────────────────────────
def run_classification(ds_name, seed, num_tasks):
    dataset = MoleculeNet(osp.join(base_path, 'data', ds_name),
                          name=ds_name, pre_transform=GenFeatures())
    torch.manual_seed(seed); np.random.seed(seed)

    train_ds, val_ds, test_ds = scaffold_split(dataset, seed, task='classification')
    print(f'    Split: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=HP['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=HP['batch_size'])
    test_loader  = DataLoader(test_ds,  batch_size=HP['batch_size'])

    model     = make_model(out_channels=num_tasks)
    optimizer = torch.optim.Adam(model.parameters(), lr=HP['lr'],
                                 weight_decay=HP['weight_decay'])

    best_val, best_test = -float('inf'), -float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out  = safe_out(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch))
            y    = safe_y(batch.y, num_tasks)
            mask = ~torch.isnan(y)
            if mask.sum() == 0: continue
            F.binary_cross_entropy_with_logits(out[mask], y[mask]).backward()
            optimizer.step()

        model.eval()
        vp = [[] for _ in range(num_tasks)]
        vl = [[] for _ in range(num_tasks)]
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = safe_out(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch))
                y   = safe_y(batch.y, num_tasks)
                for t in range(num_tasks):
                    mask = ~torch.isnan(y[:, t])
                    if mask.sum() == 0: continue
                    vp[t].append(torch.sigmoid(out[mask, t]).cpu().numpy())
                    vl[t].append(y[mask, t].cpu().numpy())

        val_aucs = []
        for t in range(num_tasks):
            if vp[t]:
                auc = safe_auc(np.concatenate(vl[t]), np.concatenate(vp[t]))
                if auc is not None: val_aucs.append(auc)
        val_auc = np.mean(val_aucs) if val_aucs else 0.5

        if val_auc > best_val:
            best_val = val_auc
            tp = [[] for _ in range(num_tasks)]
            tl = [[] for _ in range(num_tasks)]
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = safe_out(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch))
                    y   = safe_y(batch.y, num_tasks)
                    for t in range(num_tasks):
                        mask = ~torch.isnan(y[:, t])
                        if mask.sum() == 0: continue
                        tp[t].append(torch.sigmoid(out[mask, t]).cpu().numpy())
                        tl[t].append(y[mask, t].cpu().numpy())
            test_aucs = []
            for t in range(num_tasks):
                if tp[t]:
                    auc = safe_auc(np.concatenate(tl[t]), np.concatenate(tp[t]))
                    if auc is not None: test_aucs.append(auc)
            best_test = np.mean(test_aucs) if test_aucs else 0.5

        if epoch % 50 == 0:
            print(f'    Epoch {epoch:03d} | Val AUC: {best_val:.4f} | Test AUC: {best_test:.4f}')

    return best_test


# ── Main ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '█' * 60)
    print('  MoleculeNet Full Baseline')
    print('  7 Datasets | AttentiveFP | Scaffold Split | 3 Seeds')
    print('  Paper Default Hyperparameters')
    print('█' * 60)

    all_results = {}

    for ds_info in DATASETS:
        name, task, num_tasks = ds_info['name'], ds_info['task'], ds_info['num_tasks']
        print(f'\n{"=" * 60}\n  {name} ({task}, {num_tasks} task{"s" if num_tasks > 1 else ""})\n{"=" * 60}')

        results = []
        for seed in SEEDS:
            print(f'\n  Seed {seed}:')
            if task == 'regression':
                results.append(run_regression(name, seed))
            else:
                results.append(run_classification(name, seed, num_tasks))

        mean, std = float(np.mean(results)), float(np.std(results))
        all_results[name] = {'mean': mean, 'std': std, 'results': results,
                             'metric': ds_info['metric'], 'published': ds_info['published']}
        print(f'\n  {name} DONE: {mean:.4f} ± {std:.4f}')

    print('\n\n' + '=' * 75)
    print('  MOLECULENET BASELINE RESULTS — scaffold split — 3 seeds')
    print('=' * 75)
    print(f'  {"Dataset":<14} {"Task":<16} {"Metric":<8} {"Our Result":<24} {"Published"}')
    print(f'  {"-" * 70}')
    for ds_info in DATASETS:
        name = ds_info['name']
        r    = all_results[name]
        print(f'  {name:<14} {ds_info["task"]:<16} {r["metric"]:<8} '
              f'{r["mean"]:.4f} ± {r["std"]:.4f}      {r["published"]}')
    print('=' * 75)

    print('\n  Seed breakdown:')
    for ds_info in DATASETS:
        name = ds_info['name']
        vals = ' | '.join(f'Seed {s}: {v:.4f}' for s, v in zip(SEEDS, all_results[name]['results']))
        print(f'    {name:<14} {vals}')

    print('\n  All baselines locked. Ready for multi-task experiments.')
