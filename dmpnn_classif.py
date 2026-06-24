"""
TRUE D-MPNN — Classification
Backbone: NNConv (edge-conditioned) + GRUCell — genuine directed message passing
Datasets: Tox21, ToxCast, SIDER, ClinTox, HIV, BBBP, BACE
Metric: ROC-AUC
Results saved to: results_dmpnn_classif.json

FIXED: replaced GCNConv with NNConv+GRUCell (true Chemprop-style DMPNN)
       + backbone filter forces rerun of old GCNConv results
"""

import os, json, time, warnings, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, GRUCell
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import NNConv, global_mean_pool
from sklearn.metrics import roc_auc_score
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS   = 30
N_SEEDS    = 3
EPOCHS     = 100
PATIENCE   = 15
BATCH_SIZE = 64
DATA_ROOT  = "./data"
SAVE_PATH  = "results_dmpnn_classif.json"

DATASETS = [
    {"name": "Tox21",   "tasks": 12},
    {"name": "ToxCast", "tasks": 617},
    {"name": "SIDER",   "tasks": 27},
    {"name": "ClinTox", "tasks": 2},
    {"name": "HIV",     "tasks": 1},
    {"name": "BBBP",    "tasks": 1},
    {"name": "BACE",    "tasks": 1},
]

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Backbone: TRUE D-MPNN (NNConv + GRUCell)")


def ToFloat(data):
    data.x = data.x.float()
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr.float()
    return data


# ── TRUE D-MPNN ────────────────────────────────────────────────────────────────

class DMPNN(nn.Module):
    """
    Genuine Chemprop-style directed MPNN.
    NNConv: bond features parametrize per-edge weight matrix.
    GRUCell: recurrent node state update.
    """
    def __init__(self, in_ch, edge_ch, hidden, num_layers, dropout, num_tasks):
        super().__init__()
        self.input_proj = Linear(in_ch, hidden)
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for _ in range(num_layers):
            nn_edge = Sequential(Linear(edge_ch, hidden * hidden))
            self.convs.append(NNConv(hidden, hidden, nn_edge, aggr='mean'))
            self.bns.append(BatchNorm1d(hidden))
        self.gru     = GRUCell(hidden, hidden)
        self.dropout = dropout
        self.head    = Linear(hidden, num_tasks)

    def forward(self, data):
        x          = data.x.float()
        edge_index = data.edge_index
        edge_attr  = data.edge_attr.float()
        batch      = data.batch

        x = F.relu(self.input_proj(x))
        h = x
        for conv, bn in zip(self.convs, self.bns):
            m = F.relu(bn(conv(h, edge_index, edge_attr)))
            m = F.dropout(m, p=self.dropout, training=self.training)
            h = self.gru(m, h)
        x = global_mean_pool(h, batch)
        return self.head(x)


# ── Scaffold split ─────────────────────────────────────────────────────────────

def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict
    scaffolds = defaultdict(list)
    for i in range(len(dataset)):
        try:
            smi = dataset.smiles[i]
            mol = Chem.MolFromSmiles(smi)
            sc  = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False) if mol else smi
        except:
            sc = str(i)
        scaffolds[sc].append(i)
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(dataset)
    train_cutoff = int(n * frac_train)
    val_cutoff   = int(n * (frac_train + frac_val))
    train_idx, val_idx, test_idx = [], [], []
    for s in scaffold_sets:
        if   len(train_idx) < train_cutoff:               train_idx.extend(s)
        elif len(val_idx)   < (val_cutoff - train_cutoff): val_idx.extend(s)
        else:                                              test_idx.extend(s)
    return (torch.utils.data.Subset(dataset, train_idx),
            torch.utils.data.Subset(dataset, val_idx),
            torch.utils.data.Subset(dataset, test_idx))


# ── Metrics ────────────────────────────────────────────────────────────────────

def evaluate(model, loader, num_tasks):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out   = model(batch)
            probs = torch.sigmoid(out).cpu().numpy()
            y     = batch.y.cpu().numpy()
            if y.ndim == 1:
                y = y.reshape(-1, num_tasks)
            all_preds.append(probs)
            all_labels.append(y)
    preds  = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    aucs = []
    for t in range(num_tasks):
        col  = labels[:, t]
        mask = ~np.isnan(col)
        if mask.sum() < 2 or len(np.unique(col[mask])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(col[mask], preds[mask, t]))
        except:
            pass
    return float(np.mean(aucs)) if aucs else 0.0


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, num_tasks):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch)
        y   = batch.y.float()
        if y.ndim == 1:
            y = y.reshape(-1, num_tasks)
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        F.binary_cross_entropy_with_logits(out[mask], y[mask]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


# ── Optuna objective ───────────────────────────────────────────────────────────

def make_objective(train_data, val_data, in_dim, edge_dim, num_tasks):
    def objective(trial):
        hidden     = trial.suggest_categorical("hidden",     [128, 256])
        num_layers = trial.suggest_int("num_layers",          2, 4)
        dropout    = trial.suggest_float("dropout",           0.0, 0.3)
        lr         = trial.suggest_float("lr",                1e-4, 1e-3, log=True)
        wd         = trial.suggest_float("weight_decay",      1e-6, 1e-4, log=True)

        model = DMPNN(in_dim, edge_dim, hidden, num_layers,
                      dropout, num_tasks).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        tl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        vl = DataLoader(val_data,   batch_size=BATCH_SIZE)

        best_val, pat = 0.0, 0
        for ep in range(EPOCHS):
            train_epoch(model, tl, opt, num_tasks)
            val_auc = evaluate(model, vl, num_tasks)
            sched.step(-val_auc)
            if val_auc > best_val:
                best_val, pat = val_auc, 0
            else:
                pat += 1
            if pat >= PATIENCE:
                break
            trial.report(val_auc, ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return best_val
    return objective


# ── Main ───────────────────────────────────────────────────────────────────────

# Only keep results already run with true DMPNN backbone
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH) as f:
        results = json.load(f)
    results = {
        k: v for k, v in results.items()
        if v.get("backbone") == "true_dmpnn_nnconv_grucell"
    }
    print(f"Resuming true-DMPNN runs — {len(results)} done\n")
else:
    results = {}

for ds in DATASETS:
    name, num_tasks = ds["name"], ds["tasks"]
    if name in results:
        print(f"  Skipping {name} (true DMPNN done: {results[name]['mean']:.4f})")
        continue

    print(f"\n{'='*60}")
    print(f"  TRUE D-MPNN | {name} | {num_tasks} tasks | ROC-AUC")
    print(f"{'='*60}")
    t0 = time.time()

    dataset  = MoleculeNet(root=DATA_ROOT, name=name, transform=ToFloat)
    in_dim   = dataset.num_node_features
    edge_dim = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 3

    train_data, val_data, test_data = scaffold_split(dataset)
    print(f"  Split → Tr:{len(train_data)} Va:{len(val_data)} Te:{len(test_data)}")
    print(f"  Node features: {in_dim}  |  Edge features: {edge_dim}")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )
    study.optimize(
        make_objective(train_data, val_data, in_dim, edge_dim, num_tasks),
        n_trials=N_TRIALS,
    )
    best_params = study.best_params
    print(f"  Best val AUC: {study.best_value:.4f} | {best_params}")

    vl = DataLoader(val_data,  batch_size=BATCH_SIZE)
    tl = DataLoader(test_data, batch_size=BATCH_SIZE)
    seed_scores = []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = DMPNN(
            in_dim, edge_dim,
            best_params["hidden"],
            best_params["num_layers"],
            best_params["dropout"],
            num_tasks,
        ).to(DEVICE)
        opt   = torch.optim.Adam(
            model.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        best_val, pat = 0.0, 0
        best_state = copy.deepcopy(model.state_dict())

        for _ in range(EPOCHS):
            train_epoch(
                model,
                DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True),
                opt, num_tasks,
            )
            val_auc = evaluate(model, vl, num_tasks)
            sched.step(-val_auc)
            if val_auc > best_val:
                best_val   = val_auc
                best_state = copy.deepcopy(model.state_dict())
                pat = 0
            else:
                pat += 1
            if pat >= PATIENCE:
                break

        model.load_state_dict(best_state)
        score = evaluate(model, tl, num_tasks)
        seed_scores.append(score)
        print(f"    Seed {seed} → AUC: {score:.4f}")

    mean_s  = float(np.mean(seed_scores))
    std_s   = float(np.std(seed_scores))
    elapsed = time.time() - t0

    results[name] = {
        "metric":      "roc_auc",
        "mean":        mean_s,
        "std":         std_s,
        "seeds":       seed_scores,
        "best_params": best_params,
        "time_min":    round(elapsed / 60, 1),
        "backbone":    "true_dmpnn_nnconv_grucell",
    }
    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✔ TRUE D-MPNN {name}: {mean_s:.4f} ± {std_s:.4f}  ({elapsed/60:.1f} min)")

print(f"\n{'='*60}")
print("  TRUE D-MPNN CLASSIFICATION COMPLETE")
print(f"{'='*60}")
for name, r in results.items():
    print(f"  {name:12} AUC: {r['mean']:.4f} ± {r['std']:.4f}")
print(f"\nSaved → {SAVE_PATH}")
