"""
DMPNN (GCN-based) — Regression
Datasets: Lipo only (ESOL + FreeSolv already done)
Metric: RMSE
Results saved to: results_dmpnn_regr.json (resumes existing)

Run: python dmpnn_regr.py
"""

import os, json, time, warnings, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool
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
SAVE_PATH  = "results_dmpnn_regr.json"

DATASETS = [{"name": "Lipo"}]  # ESOL + FreeSolv already saved

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def ToFloat(data):
    data.x = data.x.float()
    return data

class DMPNN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.head = nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x)

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

def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out   = model(batch).squeeze().cpu().numpy()
            y     = batch.y.squeeze().cpu().numpy()
            preds.extend(out.flatten())
            labels.extend(y.flatten())
    preds, labels = np.array(preds), np.array(labels)
    mask = ~np.isnan(labels)
    return float(np.sqrt(np.mean((preds[mask] - labels[mask]) ** 2)))

def train_epoch(model, loader, optimizer):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out  = model(batch).squeeze()
        y    = batch.y.float().squeeze()
        mask = ~torch.isnan(y)
        if mask.sum() == 0: continue
        F.mse_loss(out[mask], y[mask]).backward()
        optimizer.step()

def make_objective(train_data, val_data, in_dim):
    def objective(trial):
        hidden     = trial.suggest_categorical("hidden", [128, 256])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        dropout    = trial.suggest_float("dropout", 0.0, 0.3)
        lr         = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        wd         = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        model = DMPNN(in_dim, hidden, num_layers, dropout).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        tl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        vl = DataLoader(val_data,   batch_size=BATCH_SIZE)
        best_val, pat = float("inf"), 0
        for _ in range(EPOCHS):
            train_epoch(model, tl, opt)
            val_rmse = evaluate(model, vl)
            sched.step(val_rmse)
            if val_rmse < best_val:
                best_val, pat = val_rmse, 0
            else:
                pat += 1
            if pat >= PATIENCE: break
        return best_val
    return objective

# Load existing results (keeps ESOL + FreeSolv)
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH) as f:
        results = json.load(f)
    print(f"Resuming — {len(results)} datasets already saved")
else:
    results = {}

for ds in DATASETS:
    name = ds["name"]
    if name in results:
        print(f"  Skipping {name}")
        continue
    print(f"\n{'='*50}\n  DMPNN | {name} | RMSE\n{'='*50}")
    t0 = time.time()
    dataset = MoleculeNet(root=DATA_ROOT, name=name, transform=ToFloat)
    in_dim  = dataset.num_node_features
    train_data, val_data, test_data = scaffold_split(dataset)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(make_objective(train_data, val_data, in_dim), n_trials=N_TRIALS)
    best_params = study.best_params
    print(f"  Best val RMSE: {study.best_value:.4f} | {best_params}")

    vl = DataLoader(val_data,  batch_size=BATCH_SIZE)
    tl = DataLoader(test_data, batch_size=BATCH_SIZE)
    seed_scores = []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        model = DMPNN(in_dim, best_params["hidden"], best_params["num_layers"],
                      best_params["dropout"]).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        best_val, pat = float("inf"), 0
        best_state = copy.deepcopy(model.state_dict())
        for _ in range(EPOCHS):
            train_epoch(model, DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True), opt)
            val_rmse = evaluate(model, vl)
            sched.step(val_rmse)
            if val_rmse < best_val:
                best_val = val_rmse; best_state = copy.deepcopy(model.state_dict()); pat = 0
            else:
                pat += 1
            if pat >= PATIENCE: break
        model.load_state_dict(best_state)
        seed_scores.append(evaluate(model, tl))
        print(f"    Seed {seed} → RMSE: {seed_scores[-1]:.4f}")

    mean_s, std_s = float(np.mean(seed_scores)), float(np.std(seed_scores))
    elapsed = time.time() - t0
    results[name] = {"mean": mean_s, "std": std_s, "metric": "rmse",
                     "seeds": seed_scores, "best_params": best_params, "time_min": round(elapsed/60,1)}
    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ DMPNN {name}: {mean_s:.4f} ± {std_s:.4f}  ({elapsed/60:.1f} min)")

print(f"\n{'='*50}\n  DMPNN REGRESSION SUMMARY\n{'='*50}")
for name, r in results.items():
    print(f"  {name:12} RMSE: {r['mean']:.4f} ± {r['std']:.4f}")
print(f"\nSaved → {SAVE_PATH}")
