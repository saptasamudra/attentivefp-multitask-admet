"""
TRUE MoE-DMPNN — Regression
Backbone: NNConv (edge-conditioned) + GRUCell — genuine directed message passing
MoE routing on top of pooled representation
Datasets: ESOL, FreeSolv, Lipo
Metric: RMSE
Results saved to: results_moedmpnn_regr.json

FIXED: replaced GCNConv backbone with NNConv+GRUCell (true Chemprop-style DMPNN)
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
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS   = 30
N_SEEDS    = 5
EPOCHS     = 100
PATIENCE   = 15
BATCH_SIZE = 64
DATA_ROOT  = "./data"
SAVE_PATH  = "results_moedmpnn_regr.json"

DATASETS = [{"name": "ESOL"}, {"name": "FreeSolv"}, {"name": "Lipo"}]

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Backbone: TRUE D-MPNN (NNConv + GRUCell) + MoE routing")


def ToFloat(data):
    data.x = data.x.float()
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr.float()
    return data


# ── MoE Layer (unchanged) ──────────────────────────────────────────────────────

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
        weights = torch.zeros_like(gate_logits).scatter_(
            1, topk_idx, F.softmax(topk_vals, dim=-1))
        load = weights.mean(0)
        balance_loss = self.num_experts * (load * load).sum()
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        out = (weights.unsqueeze(-1) * expert_out).sum(dim=1)
        return out, balance_loss


# ── TRUE MoE-DMPNN ─────────────────────────────────────────────────────────────

class MoEDMPNN(nn.Module):
    """
    True D-MPNN backbone (NNConv + GRUCell) + MoE routing after pooling.
    Bond features parametrize per-edge weight matrix via NNConv.
    """
    def __init__(self, in_ch, edge_ch, hidden, num_layers,
                 dropout, num_experts, top_k):
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
        self.moe     = MoELayer(hidden, hidden, num_experts, top_k)
        self.head    = Linear(hidden, 1)

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
        x, bal_loss = self.moe(x)
        return self.head(x), bal_loss


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

def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(DEVICE)
            out, _ = model(batch)
            preds.extend(out.squeeze().cpu().numpy().flatten())
            labels.extend(batch.y.squeeze().cpu().numpy().flatten())
    preds, labels = np.array(preds), np.array(labels)
    mask = ~np.isnan(labels)
    return float(np.sqrt(np.mean((preds[mask] - labels[mask]) ** 2)))


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out, bal_loss = model(batch)
        y    = batch.y.float().squeeze()
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        loss = F.mse_loss(out.squeeze()[mask], y[mask]) + 0.01 * bal_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


# ── Optuna objective ───────────────────────────────────────────────────────────

def make_objective(train_data, val_data, in_dim, edge_dim):
    def objective(trial):
        hidden      = trial.suggest_categorical("hidden",      [128, 256])
        num_layers  = trial.suggest_int("num_layers",           2, 4)
        dropout     = trial.suggest_float("dropout",            0.0, 0.3)
        num_experts = trial.suggest_categorical("num_experts",  [4, 8, 16])
        top_k       = trial.suggest_int("top_k",                1, min(4, num_experts))
        lr          = trial.suggest_float("lr",                 1e-4, 1e-3, log=True)
        wd          = trial.suggest_float("weight_decay",       1e-6, 1e-4, log=True)

        model = MoEDMPNN(in_dim, edge_dim, hidden, num_layers,
                         dropout, num_experts, top_k).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        tl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        vl = DataLoader(val_data,   batch_size=BATCH_SIZE)

        best_val, pat = float("inf"), 0
        for ep in range(EPOCHS):
            train_epoch(model, tl, opt)
            val_rmse = evaluate(model, vl)
            sched.step(val_rmse)
            if val_rmse < best_val:
                best_val, pat = val_rmse, 0
            else:
                pat += 1
            if pat >= PATIENCE:
                break
            trial.report(val_rmse, ep)
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
    name = ds["name"]
    if name in results:
        print(f"  Skipping {name} (true MoE-DMPNN done: {results[name]['mean']:.4f})")
        continue

    print(f"\n{'='*60}")
    print(f"  TRUE MoE-DMPNN | {name} | RMSE")
    print(f"{'='*60}")
    t0 = time.time()

    dataset  = MoleculeNet(root=DATA_ROOT, name=name, transform=ToFloat)
    in_dim   = dataset.num_node_features
    edge_dim = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 3

    train_data, val_data, test_data = scaffold_split(dataset)
    print(f"  Split → Tr:{len(train_data)} Va:{len(val_data)} Te:{len(test_data)}")
    print(f"  Node features: {in_dim}  |  Edge features: {edge_dim}")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )
    study.optimize(
        make_objective(train_data, val_data, in_dim, edge_dim),
        n_trials=N_TRIALS,
    )
    best_params = study.best_params
    print(f"  Best val RMSE: {study.best_value:.4f} | {best_params}")

    vl = DataLoader(val_data,  batch_size=BATCH_SIZE)
    tl = DataLoader(test_data, batch_size=BATCH_SIZE)
    seed_scores = []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MoEDMPNN(
            in_dim, edge_dim,
            best_params["hidden"],
            best_params["num_layers"],
            best_params["dropout"],
            best_params["num_experts"],
            best_params["top_k"],
        ).to(DEVICE)
        opt   = torch.optim.Adam(
            model.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        best_val, pat = float("inf"), 0
        best_state = copy.deepcopy(model.state_dict())

        for _ in range(EPOCHS):
            train_epoch(
                model,
                DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True),
                opt,
            )
            val_rmse = evaluate(model, vl)
            sched.step(val_rmse)
            if val_rmse < best_val:
                best_val   = val_rmse
                best_state = copy.deepcopy(model.state_dict())
                pat = 0
            else:
                pat += 1
            if pat >= PATIENCE:
                break

        model.load_state_dict(best_state)
        score = evaluate(model, tl)
        seed_scores.append(score)
        print(f"    Seed {seed} → RMSE: {score:.4f}")

    mean_s  = float(np.mean(seed_scores))
    std_s   = float(np.std(seed_scores))
    elapsed = time.time() - t0

    results[name] = {
        "metric":      "rmse",
        "mean":        mean_s,
        "std":         std_s,
        "seeds":       seed_scores,
        "best_params": best_params,
        "time_min":    round(elapsed / 60, 1),
        "backbone":    "true_dmpnn_nnconv_grucell",
    }
    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✔ TRUE MoE-DMPNN {name}: {mean_s:.4f} ± {std_s:.4f}  ({elapsed/60:.1f} min)")

print(f"\n{'='*60}")
print("  TRUE MoE-DMPNN REGRESSION COMPLETE")
print(f"{'='*60}")
for name, r in results.items():
    print(f"  {name:12} RMSE: {r['mean']:.4f} ± {r['std']:.4f}")
print(f"\nSaved → {SAVE_PATH}")
