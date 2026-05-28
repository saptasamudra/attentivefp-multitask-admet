"""
MoE-DMPNN (GCN + MoE) — Classification
Datasets: Tox21, ToxCast, SIDER, ClinTox, HIV
Metric: ROC-AUC
Results saved to: results_moedmpnn_classif.json
"""

import os, json, time, warnings, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool
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
SAVE_PATH  = "results_moedmpnn_classif.json"

DATASETS = [
    {"name": "Tox21",   "tasks": 12},

    {"name": "SIDER",   "tasks": 27},
    {"name": "ClinTox", "tasks": 2},
    {"name": "HIV",     "tasks": 1},
]

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def ToFloat(data):
    data.x = data.x.float()
    return data

class MoELayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(in_dim, num_experts)

    def forward(self, x):
        gate_logits = self.gate(x)
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = torch.zeros_like(gate_logits).scatter_(1, topk_idx, F.softmax(topk_vals, dim=-1))
        load = weights.mean(0)
        balance_loss = self.num_experts * (load * load).sum()
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        out = (weights.unsqueeze(-1) * expert_out).sum(dim=1)
        return out, balance_loss

class MoEDMPNN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_experts, top_k, num_tasks):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe  = MoELayer(hidden, hidden, num_experts, top_k)
        self.head = nn.Linear(hidden, num_tasks)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x, bal_loss = self.moe(x)
        return self.head(x), bal_loss

def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict
    scaffolds = defaultdict(list)
    for i in range(len(dataset)):
        try:
            smi = dataset.smiles[i]
            mol = Chem.MolFromSmiles(smi)
            sc  = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else smi
        except:
            sc = str(i)
        scaffolds[sc].append(i)
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(dataset)
    train_cutoff = int(n * frac_train)
    val_cutoff   = int(n * (frac_train + frac_val))
    train_idx, val_idx, test_idx = [], [], []
    for s in scaffold_sets:
        if len(train_idx) < train_cutoff:
            train_idx.extend(s)
        elif len(val_idx) < (val_cutoff - train_cutoff):
            val_idx.extend(s)
        else:
            test_idx.extend(s)
    return (torch.utils.data.Subset(dataset, train_idx),
            torch.utils.data.Subset(dataset, val_idx),
            torch.utils.data.Subset(dataset, test_idx))

def evaluate(model, loader, num_tasks):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(DEVICE)
            out, _ = model(batch)
            probs  = torch.sigmoid(out).cpu().numpy()
            y      = batch.y.cpu().numpy()
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

def train_epoch(model, loader, optimizer, num_tasks):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out, bal_loss = model(batch)
        y = batch.y.float()
        if y.ndim == 1:
            y = y.reshape(-1, num_tasks)
        mask = ~torch.isnan(y)
        if mask.sum() == 0: continue
        loss = F.binary_cross_entropy_with_logits(out[mask], y[mask]) + 0.01 * bal_loss
        loss.backward()
        optimizer.step()

def make_objective(train_data, val_data, in_dim, num_tasks):
    def objective(trial):
        hidden      = trial.suggest_categorical("hidden", [128, 256])
        num_layers  = trial.suggest_int("num_layers", 2, 4)
        dropout     = trial.suggest_float("dropout", 0.0, 0.3)
        num_experts = trial.suggest_categorical("num_experts", [4, 8, 16])
        top_k       = trial.suggest_int("top_k", 1, min(4, num_experts))
        lr          = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        wd          = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        model = MoEDMPNN(in_dim, hidden, num_layers, dropout, num_experts, top_k, num_tasks).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        tl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        vl = DataLoader(val_data,   batch_size=BATCH_SIZE)
        best_val, pat = 0.0, 0
        for _ in range(EPOCHS):
            train_epoch(model, tl, opt, num_tasks)
            val_auc = evaluate(model, vl, num_tasks)
            sched.step(-val_auc)
            if val_auc > best_val:
                best_val, pat = val_auc, 0
            else:
                pat += 1
            if pat >= PATIENCE: break
        return best_val
    return objective

if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH) as f:
        results = json.load(f)
    print(f"Resuming — {len(results)} done")
else:
    results = {}

for ds in DATASETS:
    name, num_tasks = ds["name"], ds["tasks"]
    if name in results:
        print(f"  Skipping {name}")
        continue
    print(f"\n{'='*55}\n  MoE-DMPNN | {name} | {num_tasks} tasks | ROC-AUC\n{'='*55}")
    t0 = time.time()
    dataset = MoleculeNet(root=DATA_ROOT, name=name, transform=ToFloat)
    in_dim  = dataset.num_node_features
    train_data, val_data, test_data = scaffold_split(dataset)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(make_objective(train_data, val_data, in_dim, num_tasks), n_trials=N_TRIALS)
    best_params = study.best_params
    print(f"  Best val AUC: {study.best_value:.4f} | {best_params}")

    vl = DataLoader(val_data,  batch_size=BATCH_SIZE)
    tl = DataLoader(test_data, batch_size=BATCH_SIZE)
    seed_scores = []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        model = MoEDMPNN(in_dim, best_params["hidden"], best_params["num_layers"],
                         best_params["dropout"], best_params["num_experts"],
                         best_params["top_k"], num_tasks).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        best_val, pat = 0.0, 0
        best_state = copy.deepcopy(model.state_dict())
        for _ in range(EPOCHS):
            train_epoch(model, DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True), opt, num_tasks)
            val_auc = evaluate(model, vl, num_tasks)
            sched.step(-val_auc)
            if val_auc > best_val:
                best_val = val_auc; best_state = copy.deepcopy(model.state_dict()); pat = 0
            else:
                pat += 1
            if pat >= PATIENCE: break
        model.load_state_dict(best_state)
        seed_scores.append(evaluate(model, tl, num_tasks))
        print(f"    Seed {seed} → AUC: {seed_scores[-1]:.4f}")

    mean_s, std_s = float(np.mean(seed_scores)), float(np.std(seed_scores))
    elapsed = time.time() - t0
    results[name] = {"mean": mean_s, "std": std_s, "metric": "roc_auc",
                     "seeds": seed_scores, "best_params": best_params, "time_min": round(elapsed/60,1)}
    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ MoE-DMPNN {name}: {mean_s:.4f} ± {std_s:.4f}  ({elapsed/60:.1f} min)")

print(f"\n{'='*55}\n  MoE-DMPNN CLASSIFICATION SUMMARY\n{'='*55}")
for name, r in results.items():
    print(f"  {name:12} AUC: {r['mean']:.4f} ± {r['std']:.4f}")
print(f"\nSaved → {SAVE_PATH}")
