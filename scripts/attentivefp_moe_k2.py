"""
Mixture-of-Experts AttentiveFP -- Multi-Task Molecular Property Prediction
=========================================================================
9 datasets: ESOL, FreeSolv, Lipo, BACE, BBBP, HIV, ClinTox, Tox21, SIDER

GenFeatures copied exactly from multitask_7dataset.py:
  atom features : 39  (matches existing cache)
  bond features : 10  (matches existing cache)

Run:
  clean_and_run.bat
  (deletes HIV/SIDER cache, then runs this script)
"""

import os.path as osp
from collections import defaultdict

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

HP = {
    'lr':            10**-2.5,
    'hidden_dim':    200,
    'num_layers':    2,
    'num_timesteps': 2,
    'dropout':       0.2,
    'batch_size':    200,
    'weight_decay':  1e-5,
}

MOE_HP = {
    'num_experts':         2,
    'top_k':               2,
    'expert_hidden':       200,
    'load_balance_weight': 0.01,
}

SEEDS  = [42, 123, 7]
EPOCHS = 200

DATASETS = {
    'ESOL':     {'task': 'regression',     'num_tasks': 1,  'weight': 0.5, 'metric': 'RMSE'},
    'FreeSolv': {'task': 'regression',     'num_tasks': 1,  'weight': 0.5, 'metric': 'RMSE'},
    'Lipo':     {'task': 'regression',     'num_tasks': 1,  'weight': 0.5, 'metric': 'RMSE'},
    'BACE':     {'task': 'classification', 'num_tasks': 1,  'weight': 1.0, 'metric': 'AUC'},
    'BBBP':     {'task': 'classification', 'num_tasks': 1,  'weight': 1.0, 'metric': 'AUC'},
    'HIV':      {'task': 'classification', 'num_tasks': 1,  'weight': 1.0, 'metric': 'AUC'},
    'ClinTox':  {'task': 'classification', 'num_tasks': 2,  'weight': 1.0, 'metric': 'AUC'},
    'Tox21':    {'task': 'classification', 'num_tasks': 12, 'weight': 1.0, 'metric': 'AUC'},
    'SIDER':    {'task': 'classification', 'num_tasks': 27, 'weight': 1.0, 'metric': 'AUC'},
}

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING
# Copied exactly from multitask_7dataset.py -- produces x=39, edge_attr=10
# atom:  16(symbol)+6(degree)+2(charge/radical)+6(hybridization)+1(aromatic)
#        +5(hydrogens)+1(chirality_possible)+2(chirality_type) = 39
# bond:  4(type_onehot)+4(stereo)+1(conjugated)+1(ring) = 10
# -----------------------------------------------------------------------------

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
            attr = bond_type_onehot + stereo + [is_conjugated, is_in_ring]
            edge_attrs += [attr, attr]

        data.edge_attr = (torch.zeros((0, 10), dtype=torch.float) if not edge_attrs
                         else torch.tensor(edge_attrs, dtype=torch.float))
        return data


# -----------------------------------------------------------------------------
# SCAFFOLD SPLIT -- copied from multitask_7dataset.py
# -----------------------------------------------------------------------------

def scaffold_split(dataset, seed, frac_train=0.8, frac_val=0.1):
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
        if   len(train_idx) + len(sset) <= train_cutoff:
            train_idx.extend(sset)
        elif len(val_idx)   + len(sset) <= (val_cutoff - train_cutoff):
            val_idx.extend(sset)
        else:
            test_idx.extend(sset)

    if not test_idx:
        cut = max(1, int(len(train_idx) * 0.9))
        test_idx, train_idx = train_idx[cut:], train_idx[:cut]
    if not val_idx:
        cut = max(1, int(len(train_idx) * 0.9))
        val_idx, train_idx = train_idx[cut:], train_idx[:cut]

    return (dataset[torch.tensor(train_idx, dtype=torch.long)],
            dataset[torch.tensor(val_idx,   dtype=torch.long)],
            dataset[torch.tensor(test_idx,  dtype=torch.long)])


# -----------------------------------------------------------------------------
# MIXTURE-OF-EXPERTS MODULE
# -----------------------------------------------------------------------------

class MixtureOfExperts(nn.Module):
    """
    Sparse top-k MoE routing.
    Each molecule is routed to its top-k experts based on learned gate scores.
    Outputs a weighted sum of expert representations.
    Auxiliary load-balancing loss prevents all molecules routing to one expert.
    """
    def __init__(self, input_dim, expert_hidden, num_experts, top_k, dropout=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.gate        = nn.Linear(input_dim, num_experts, bias=False)
        self.experts     = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, input_dim),
            )
            for _ in range(num_experts)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B = x.size(0)

        # Gate scores + load-balancing loss
        gate_logits  = self.gate(x)                          # [B, K]
        gate_soft    = F.softmax(gate_logits, dim=-1)        # [B, K]
        expert_usage = gate_soft.mean(dim=0)                 # [K]
        ideal        = torch.ones_like(expert_usage) / self.num_experts
        aux_loss     = ((expert_usage - ideal) ** 2).mean()  # scalar

        # Sparse top-k selection
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        routing_weights     = F.softmax(topk_vals, dim=-1)   # [B, top_k]

        # Weighted sum of selected expert outputs
        expert_out = torch.zeros(B, x.size(-1), device=x.device, dtype=x.dtype)
        for ei in range(self.num_experts):
            mask = (topk_idx == ei).any(dim=-1)
            if not mask.any():
                continue
            pos_mask = (topk_idx == ei)
            w        = (routing_weights * pos_mask.float()).sum(dim=-1)
            out      = self.experts[ei](self.dropout(x[mask]))
            expert_out[mask] += out * w[mask].unsqueeze(-1)

        return expert_out, aux_loss, gate_soft


# -----------------------------------------------------------------------------
# MAIN MODEL: MoE AttentiveFP
# -----------------------------------------------------------------------------

class AttentiveFPMoE(nn.Module):
    """
    AttentiveFP encoder (unchanged) + MoE routing + 9 task-specific heads.

    Flow:
      x, edge_index, edge_attr, batch
        -> AttentiveFP encoder  -> mol_repr  [B, 200]
        -> MoE                  -> expert_repr [B, 200]
        -> cat([mol_repr, expert_repr])  -> fused [B, 400]
        -> per-dataset Linear heads      -> predictions
    """
    def __init__(self, in_channels, edge_dim):
        super().__init__()
        h  = HP['hidden_dim']   # 200
        dp = HP['dropout']

        # Shared encoder -- identical config to multitask_7dataset.py
        self.encoder = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=h,
            out_channels=h,          # outputs mol embedding, not prediction
            edge_dim=edge_dim,
            num_layers=HP['num_layers'],
            num_timesteps=HP['num_timesteps'],
            dropout=dp,
        )

        # MoE module
        self.moe = MixtureOfExperts(
            input_dim=h,
            expert_hidden=MOE_HP['expert_hidden'],
            num_experts=MOE_HP['num_experts'],
            top_k=MOE_HP['top_k'],
            dropout=dp,
        )

        # Task heads: one per dataset, input is fused 400-dim
        fused_dim  = h * 2   # 400
        self.heads = nn.ModuleDict({
            name: nn.Linear(fused_dim, cfg['num_tasks'])
            for name, cfg in DATASETS.items()
        })
        self.dropout = nn.Dropout(dp)

    def forward(self, x, edge_index, edge_attr, batch):
        mol_repr              = self.dropout(self.encoder(x, edge_index, edge_attr, batch))
        expert_repr, aux, g   = self.moe(mol_repr)
        fused                 = self.dropout(torch.cat([mol_repr, expert_repr], dim=-1))
        preds                 = {name: head(fused) for name, head in self.heads.items()}
        return preds, aux, g


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

def load_dataset(name, seed):
    base_path = osp.dirname(osp.realpath(__file__))
    path      = osp.join(base_path, 'data', name)
    dataset   = MoleculeNet(root=path, name=name, pre_transform=GenFeatures())
    train_ds, val_ds, test_ds = scaffold_split(dataset, seed)
    kw = dict(batch_size=HP['batch_size'], num_workers=0)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(val_ds,   shuffle=False, **kw),
            DataLoader(test_ds,  shuffle=False, **kw))


# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------

def train_epoch(model, loaders, optimizer):
    model.train()
    total_loss, n_batches = 0.0, 0

    for ds_name, (train_loader, _, _) in loaders.items():
        cfg = DATASETS[ds_name]
        for batch in train_loader:
            batch = batch.to(device)
            preds, aux_loss, _ = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            y    = batch.y.float()
            pred = preds[ds_name]
            # squeeze only if single-task to keep shapes consistent
            if cfg['num_tasks'] == 1:
                pred = pred.squeeze(-1)

            if cfg['task'] == 'regression':
                mask = ~torch.isnan(y[:, 0])
                if mask.sum() == 0: continue
                task_loss = F.mse_loss(pred[mask], y[mask, 0])
            else:
                nt = cfg['num_tasks']
                if nt == 1:
                    mask = ~torch.isnan(y[:, 0])
                    if mask.sum() == 0: continue
                    task_loss = F.binary_cross_entropy_with_logits(
                        pred[mask], y[mask, 0])
                else:
                    task_loss, valid = 0.0, 0
                    for t in range(nt):
                        mask = ~torch.isnan(y[:, t])
                        if mask.sum() == 0: continue
                        task_loss += F.binary_cross_entropy_with_logits(
                            pred[:, t][mask], y[:, t][mask])
                        valid += 1
                    if valid == 0: continue
                    task_loss = task_loss / valid

            loss = cfg['weight'] * task_loss + MOE_HP['load_balance_weight'] * aux_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, ds_name):
    model.eval()
    cfg = DATASETS[ds_name]
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        preds, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_preds.append(preds[ds_name].cpu())
        all_labels.append(batch.y.float().cpu())

    preds  = torch.cat(all_preds,  dim=0)
    labels = torch.cat(all_labels, dim=0)

    if cfg['task'] == 'regression':
        mask = ~torch.isnan(labels[:, 0])
        return ((preds[mask, 0] - labels[mask, 0]) ** 2).mean().sqrt().item()

    nt, aucs = cfg['num_tasks'], []
    for t in range(nt):
        mask = ~torch.isnan(labels[:, t])
        if mask.sum() < 2: continue
        y_true  = labels[mask, t].numpy()
        y_score = torch.sigmoid(
            preds[mask, t] if nt > 1 else preds[mask, 0]
        ).numpy()
        if len(np.unique(y_true)) < 2: continue
        aucs.append(roc_auc_score(y_true, y_score))
    return float(np.mean(aucs)) if aucs else 0.0


# -----------------------------------------------------------------------------
# ONE SEED
# -----------------------------------------------------------------------------

def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    loaders, sample_batch = {}, None
    for name in DATASETS:
        tl, vl, tel = load_dataset(name, seed)
        loaders[name] = (tl, vl, tel)
        if sample_batch is None:
            sample_batch = next(iter(tl))

    in_ch    = sample_batch.x.size(-1)
    edge_dim = sample_batch.edge_attr.size(-1)
    print(f"\n  Seed {seed} | in_channels={in_ch} | edge_dim={edge_dim}")
    assert in_ch == 39,    f"Expected 39 atom features, got {in_ch}"
    assert edge_dim == 10, f"Expected 10 edge features, got {edge_dim}"

    model = AttentiveFPMoE(in_ch, edge_dim).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=HP['lr'], weight_decay=HP['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-5)

    best_val  = {n: float('inf') if DATASETS[n]['task'] == 'regression'
                 else 0.0 for n in DATASETS}
    best_test = {n: None for n in DATASETS}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, loaders, optimizer)

        val_scores = {}
        for name, (_, vl, tel) in loaders.items():
            vs = evaluate(model, vl,  name)
            ts = evaluate(model, tel, name)
            val_scores[name] = vs
            is_reg   = DATASETS[name]['task'] == 'regression'
            improved = (vs < best_val[name]) if is_reg else (vs > best_val[name])
            if improved:
                best_val[name]  = vs
                best_test[name] = ts

        sched_val = sum(s if DATASETS[n]['task'] == 'regression' else -s
                        for n, s in val_scores.items())
        scheduler.step(sched_val)

        if epoch % 20 == 0 or epoch == 1:
            reg = "  ".join(f"{n}:{v:.4f}" for n, v in val_scores.items()
                            if DATASETS[n]['task'] == 'regression')
            cls = "  ".join(f"{n}:{v:.4f}" for n, v in val_scores.items()
                            if DATASETS[n]['task'] == 'classification')
            print(f"  Ep {epoch:3d} | loss {train_loss:.4f}")
            print(f"    REG  {reg}")
            print(f"    CLS  {cls}")

    return best_test


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 70)
    print("  MoE AttentiveFP -- Multi-Task Molecular Property Prediction")
    print(f"  {len(DATASETS)} datasets | K={MOE_HP['num_experts']} experts "
          f"| top-{MOE_HP['top_k']} routing")
    print("=" * 70)

    all_results = defaultdict(list)
    for seed in SEEDS:
        print(f"\n{'='*70}\n  SEED {seed}\n{'='*70}")
        for name, score in run_seed(seed).items():
            if score is not None:
                all_results[name].append(score)

    print("\n" + "=" * 70)
    print("  FINAL RESULTS -- scaffold split -- 3 seeds")
    print("=" * 70)
    print(f"  {'Dataset':<12} {'Task':<16} {'Metric':<8} {'Mean':>8}  {'Std':>7}")
    print(f"  {'-'*55}")
    for name, scores in all_results.items():
        cfg = DATASETS[name]
        print(f"  {name:<12} {cfg['task']:<16} {cfg['metric']:<8} "
              f"{np.mean(scores):>8.4f}  {np.std(scores):>7.4f}")

    print("\n  Seed breakdown:")
    for name, scores in all_results.items():
        print(f"    {name:<12} " + " | ".join(f"{s:.4f}" for s in scores))

    print("\n  MoE config:")
    for k, v in MOE_HP.items():
        print(f"    {k}: {v}")
    print("=" * 70)
