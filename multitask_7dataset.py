"""
Multi-Task AttentiveFP — 7 MoleculeNet Datasets
One shared encoder, 7 task-specific output heads
Scaffold split | 3 seeds | Paper default hyperparameters

Same bug fixes as moleculenet_baseline.py:
  - safe_out(), safe_y(), safe_auc()
  - Additive scaffold split with fallback guarantee
  - y.view(-1)[0] for multi-task label check

Run:  python multitask_7dataset.py
Time: ~3-4 hours on GPU
"""

import os.path as osp
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score
from torch.nn import Linear, ModuleDict

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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

DATASETS = {
    'ESOL':     {'task': 'regression',     'num_tasks': 1,  'weight': 0.5},
    'FreeSolv': {'task': 'regression',     'num_tasks': 1,  'weight': 0.5},
    'Lipo':     {'task': 'regression',     'num_tasks': 1,  'weight': 0.5},
    'BACE':     {'task': 'classification', 'num_tasks': 1,  'weight': 1.0},
    'BBBP':     {'task': 'classification', 'num_tasks': 1,  'weight': 1.0},
    'ClinTox':  {'task': 'classification', 'num_tasks': 2,  'weight': 1.0},
    'Tox21':    {'task': 'classification', 'num_tasks': 12, 'weight': 1.0},
}

ST_BASELINES = {
    'ESOL': 1.0272, 'FreeSolv': 2.1699, 'Lipo': 0.6532,
    'BACE': 0.8940, 'BBBP': 0.6471, 'ClinTox': 0.8677, 'Tox21': 0.7620,
}

base_path = '.'


# ── Feature engineering ───────────────────────────────────────────
class GenFeatures:
    def __init__(self):
        self.symbols = [
            'B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At','other',
        ]
        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, 'other',
        ]
        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,    Chem.rdchem.BondStereo.STEREOE,
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
            hyb_list = [0.] * len(self.hybridizations)
            hyb = atom.GetHybridization()
            hyb_list[self.hybridizations.index(hyb) if hyb in self.hybridizations else -1] = 1.
            hydrogens = [0.] * 5
            hydrogens[min(atom.GetTotalNumHs(), 4)] = 1.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                cip = atom.GetProp('_CIPCode')
                if cip in ('R', 'S'):
                    chirality_type[['R', 'S'].index(cip)] = 1.
            xs.append(symbol + degree +
                      [float(atom.GetFormalCharge()), float(atom.GetNumRadicalElectrons())] +
                      hyb_list + [1. if atom.GetIsAromatic() else 0.] + hydrogens +
                      [1. if atom.HasProp('_ChiralityPossible') else 0.] + chirality_type)

        data.x = torch.tensor(xs, dtype=torch.float)

        edge_attrs = []
        for bond in mol.GetBonds():
            bt = bond.GetBondTypeAsDouble()
            bond_type_onehot = [
                1. if bt == 1.0 else 0.,
                1. if bt == 2.0 else 0.,
                1. if bt == 3.0 else 0.,
                1. if bt == 1.5 else 0.,
            ]
            stereo = [0.] * 4
            s = bond.GetStereo()
            if s in self.stereos:
                stereo[self.stereos.index(s)] = 1.
            is_conjugated = 1. if bond.GetIsConjugated() else 0.
            is_in_ring    = 1. if bond.IsInRing()        else 0.
            # Total: 4 + 4 + 1 + 1 = 10 dims exactly
            attr = bond_type_onehot + stereo + [is_conjugated, is_in_ring]
            edge_attrs += [attr, attr]

        data.edge_attr = (torch.zeros((0, 10), dtype=torch.float) if not edge_attrs
                          else torch.tensor(edge_attrs, dtype=torch.float))
        return data


# ── Shape helpers (same as baseline) ─────────────────────────────
def safe_out(out):
    return out.unsqueeze(-1) if out.dim() == 1 else out

def safe_y(y, num_tasks):
    y = y.float()
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    return y

def safe_auc(labels, preds):
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, preds))


# ── Scaffold split (standard DeepChem/Chemprop implementation) ───
def scaffold_split(dataset, seed, task='regression', frac_train=0.8, frac_val=0.1):
    scaffold_to_idx = {}
    for i, data in enumerate(dataset):
        mol = Chem.MolFromSmiles(data.smiles)
        scaffold = ('' if mol is None else
                    MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
        scaffold_to_idx.setdefault(scaffold, []).append(i)

    rng = np.random.RandomState(seed)
    scaffold_sets = list(scaffold_to_idx.values())
    rng.shuffle(scaffold_sets)
    scaffold_sets = sorted(scaffold_sets, key=lambda x: len(x), reverse=True)

    n = len(dataset)
    train_cutoff = int(frac_train * n)
    val_cutoff   = int((frac_train + frac_val) * n)

    train_idx, val_idx, test_idx = [], [], []
    for sset in scaffold_sets:
        if   len(train_idx) + len(sset) <= train_cutoff:               train_idx.extend(sset)
        elif len(val_idx)   + len(sset) <= (val_cutoff - train_cutoff): val_idx.extend(sset)
        else:                                                            test_idx.extend(sset)

    if not test_idx:
        cut = max(1, int(len(train_idx) * 0.9))
        test_idx, train_idx = train_idx[cut:], train_idx[:cut]
    if not val_idx:
        cut = max(1, int(len(train_idx) * 0.9))
        val_idx, train_idx = train_idx[cut:], train_idx[:cut]

    return (dataset[torch.tensor(train_idx, dtype=torch.long)],
            dataset[torch.tensor(val_idx,   dtype=torch.long)],
            dataset[torch.tensor(test_idx,  dtype=torch.long)])


# ── Multi-task model ──────────────────────────────────────────────
class AttentiveFPMultiTask(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers, num_timesteps, dropout, dataset_configs):
        super().__init__()
        self.encoder = AttentiveFP(
            in_channels=39, hidden_channels=hidden_dim, out_channels=hidden_dim,
            edge_dim=10, num_layers=num_layers, num_timesteps=num_timesteps, dropout=dropout,
        )
        self.heads = ModuleDict({
            name: Linear(hidden_dim, info['num_tasks'])
            for name, info in dataset_configs.items()
        })

    def forward(self, x, edge_index, edge_attr, batch, dataset_name):
        h = self.encoder(x, edge_index, edge_attr, batch)
        return self.heads[dataset_name](h)


# ── Load all datasets ─────────────────────────────────────────────
def load_all_datasets(seed):
    loaded = {}
    for name, info in DATASETS.items():
        dataset = MoleculeNet(osp.join(base_path, 'data', name),
                              name=name, pre_transform=GenFeatures())
        train_ds, val_ds, test_ds = scaffold_split(dataset, seed, task=info['task'])
        loaded[name] = {
            'train': DataLoader(train_ds, batch_size=HP['batch_size'], shuffle=True),
            'val':   DataLoader(val_ds,   batch_size=HP['batch_size']),
            'test':  DataLoader(test_ds,  batch_size=HP['batch_size']),
            **info,
        }
        print(f'  {name}: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}')
    return loaded


# ── Evaluate one dataset ──────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, dataset_name, task, num_tasks):
    model.eval()
    if task == 'regression':
        preds, labels = [], []
        for batch in loader:
            batch = batch.to(device)
            out = safe_out(model(batch.x, batch.edge_index, batch.edge_attr,
                                 batch.batch, dataset_name))
            y   = safe_y(batch.y, 1)
            mask = ~torch.isnan(y[:, 0])
            if mask.sum() == 0: continue
            preds.append(out[mask, 0].cpu()); labels.append(y[mask, 0].cpu())
        if not preds: return float('inf')
        return sqrt(F.mse_loss(torch.cat(preds), torch.cat(labels)).item())
    else:
        tp = [[] for _ in range(num_tasks)]
        tl = [[] for _ in range(num_tasks)]
        for batch in loader:
            batch = batch.to(device)
            out = safe_out(model(batch.x, batch.edge_index, batch.edge_attr,
                                 batch.batch, dataset_name))
            y   = safe_y(batch.y, num_tasks)
            for t in range(num_tasks):
                mask = ~torch.isnan(y[:, t])
                if mask.sum() == 0: continue
                tp[t].append(torch.sigmoid(out[mask, t]).cpu().numpy())
                tl[t].append(y[mask, t].cpu().numpy())
        aucs = []
        for t in range(num_tasks):
            if tp[t]:
                auc = safe_auc(np.concatenate(tl[t]), np.concatenate(tp[t]))
                if auc is not None: aucs.append(auc)
        return np.mean(aucs) if aucs else 0.5


# ── Train one seed ────────────────────────────────────────────────
def run_seed(seed):
    print(f'\n{"─" * 50}\n  Seed {seed} — loading datasets...\n{"─" * 50}')
    torch.manual_seed(seed); np.random.seed(seed)

    loaders  = load_all_datasets(seed)
    ds_names = list(DATASETS.keys())

    model = AttentiveFPMultiTask(
        hidden_dim=HP['hidden_dim'], num_layers=HP['num_layers'],
        num_timesteps=HP['num_timesteps'], dropout=HP['dropout'],
        dataset_configs=DATASETS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=HP['lr'],
                                 weight_decay=HP['weight_decay'])

    best_val  = {n: float('inf')  if DATASETS[n]['task'] == 'regression' else -float('inf')
                 for n in ds_names}
    best_test = {n: float('inf')  if DATASETS[n]['task'] == 'regression' else -float('inf')
                 for n in ds_names}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for name in ds_names:
            info     = loaders[name]
            task     = info['task']
            n_tasks  = info['num_tasks']
            weight   = info['weight']
            for batch in info['train']:
                batch = batch.to(device)
                optimizer.zero_grad()
                out  = safe_out(model(batch.x, batch.edge_index, batch.edge_attr,
                                      batch.batch, name))
                y    = safe_y(batch.y, n_tasks)
                if task == 'regression':
                    mask = ~torch.isnan(y[:, 0])
                    if mask.sum() == 0: continue
                    loss = weight * F.mse_loss(out[mask, 0], y[mask, 0])
                else:
                    mask = ~torch.isnan(y)
                    if mask.sum() == 0: continue
                    loss = weight * F.binary_cross_entropy_with_logits(out[mask], y[mask])
                loss.backward()
                optimizer.step()

        if epoch % 10 == 0 or epoch == EPOCHS:
            summary = []
            for name in ds_names:
                info    = loaders[name]
                task    = info['task']
                n_tasks = info['num_tasks']

                val_score  = evaluate(model, info['val'],  name, task, n_tasks)
                test_score = evaluate(model, info['test'], name, task, n_tasks)

                improved = (val_score < best_val[name] if task == 'regression'
                            else val_score > best_val[name])
                if improved:
                    best_val[name]  = val_score
                    best_test[name] = test_score

                summary.append(f'{name}={best_test[name]:.4f}')

            if epoch % 50 == 0 or epoch == EPOCHS:
                print(f'  Epoch {epoch:03d} | ' + ' | '.join(summary))

    return best_test


# ── Main ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '█' * 60)
    print('  Multi-Task AttentiveFP — 7 MoleculeNet Datasets')
    print('  Shared Encoder | 7 Heads | Scaffold Split | 3 Seeds')
    print('  Regression weight=0.5 | Classification weight=1.0')
    print('█' * 60)

    all_seed_results = []
    for seed in SEEDS:
        result = run_seed(seed)
        all_seed_results.append(result)
        print(f'\n  Seed {seed} results:')
        for name, val in result.items():
            print(f'    {name:<14} {val:.4f}')

    print('\n\n' + '=' * 80)
    print('  MULTI-TASK RESULTS vs SINGLE-TASK BASELINES')
    print('=' * 80)
    print(f'  {"Dataset":<14} {"Task":<16} {"Multi-Task":<24} {"Single-Task":<22} {"Δ"}')
    print(f'  {"-" * 75}')

    for name, info in DATASETS.items():
        values = [r[name] for r in all_seed_results]
        mean, std = np.mean(values), np.std(values)
        st = ST_BASELINES[name]
        if info['task'] == 'regression':
            delta = st - mean; sign = '✓ improved' if delta > 0 else '✗ worse'
        else:
            delta = mean - st; sign = '✓ improved' if delta > 0 else '✗ worse'
        print(f'  {name:<14} {info["task"]:<16} {mean:.4f} ± {std:.4f}         '
              f'{st:<22.4f} {delta:+.4f} {sign}')

    print('=' * 80)

    print('\n  Seed breakdown:')
    for name in DATASETS:
        vals = ' | '.join(f'Seed {s}: {r[name]:.4f}'
                          for s, r in zip(SEEDS, all_seed_results))
        print(f'    {name:<14} {vals}')

    print('\n  Task weights: regression=0.5, classification=1.0')
    print('\n  Done. Insert these results into Table 3 of paper_v2.docx')
