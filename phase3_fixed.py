"""
Phase 3 — MoE as Universal Plug-in Module
Models: MoE-GIN, MoE-GCN, MoE-DMPNN
Datasets: All 9 MoleculeNet (classification + regression)

Run: python phase3_moe_backbones.py
Results saved to: results_phase3_moe_backbones.json
"""

import os, json, time, warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ReLU, Sequential, GRUCell
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.nn import NNConv
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
SAVE_PATH  = "results_phase3_fixed.json"
LB_WEIGHT  = 0.01

CLASSIFICATION_DATASETS = [
    {"name": "BBBP",    "tasks": 1,  "metric": "roc_auc"},
    {"name": "BACE",    "tasks": 1,  "metric": "roc_auc"},
    {"name": "Tox21",   "tasks": 12, "metric": "roc_auc"},
    {"name": "ClinTox", "tasks": 2,  "metric": "roc_auc"},
    {"name": "SIDER",   "tasks": 27, "metric": "roc_auc"},
    {"name": "HIV",     "tasks": 1,  "metric": "roc_auc"},
]
REGRESSION_DATASETS = [
    {"name": "ESOL",      "tasks": 1, "metric": "rmse"},
    {"name": "FreeSolv",  "tasks": 1, "metric": "rmse"},
    {"name": "Lipo",      "tasks": 1, "metric": "rmse"},
]
BACKBONES = ["MoE-GIN", "MoE-GCN", "MoE-DMPNN"]

print(f"Device : {DEVICE}")
print(f"Backbones : {BACKBONES}")
print(f"Datasets  : {[d['name'] for d in CLASSIFICATION_DATASETS + REGRESSION_DATASETS]}\n")

if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH) as f:
        results = json.load(f)
    # Remove any 0.0 results so they rerun
    for bb in list(results.keys()):
        for ds in list(results[bb].keys()):
            if results[bb][ds].get("mean", 1) == 0.0:
                del results[bb][ds]
                print(f"Removed bad result: {bb}/{ds}")
    print(f"Loaded existing: {[(bb, list(ds.keys())) for bb, ds in results.items()]}\n")
else:
    results = {}


# ══════════════════════════════════════════════════════════════════════════════
# SPARSE MOE ROUTER
# ══════════════════════════════════════════════════════════════════════════════

class SparseMoE(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.experts     = torch.nn.ModuleList([
            Sequential(Linear(in_dim, out_dim), torch.nn.GELU(), Linear(out_dim, out_dim))
            for _ in range(num_experts)
        ])
        self.gate    = Linear(in_dim, num_experts, bias=False)
        self._lb     = torch.tensor(0.0)

    def forward(self, x):
        B       = x.size(0)
        out_dim = self.experts[0][-1].out_features
        logits  = self.gate(x)
        topk_v, topk_i = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(topk_v, dim=-1)

        out = torch.zeros(B, out_dim, device=x.device)
        for k in range(self.top_k):
            idx = topk_i[:, k]
            w   = weights[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    out[mask] = out[mask] + w[mask] * self.experts[e](x[mask])

        probs    = F.softmax(logits, dim=-1)
        self._lb = (probs.mean(0) * probs.mean(0)).sum() * self.num_experts
        return out

    def lb_loss(self):
        return self._lb


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class MoEGIN(torch.nn.Module):
    def __init__(self, in_ch, hidden, out_ch, num_layers, dropout, num_experts, top_k):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        for i in range(num_layers):
            inc = in_ch if i == 0 else hidden
            mlp = Sequential(Linear(inc, hidden), BatchNorm1d(hidden), ReLU(), Linear(hidden, hidden))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(BatchNorm1d(hidden))
        self.moe     = SparseMoE(hidden, hidden, num_experts, top_k)
        self.dropout = dropout
        self.lin     = Linear(hidden, out_ch)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        x = self.moe(x)
        return self.lin(x)

    def lb_loss(self):
        return self.moe.lb_loss()


class MoEGCN(torch.nn.Module):
    def __init__(self, in_ch, hidden, out_ch, num_layers, dropout, num_experts, top_k):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        for i in range(num_layers):
            inc = in_ch if i == 0 else hidden
            self.convs.append(GCNConv(inc, hidden))
            self.bns.append(BatchNorm1d(hidden))
        self.moe     = SparseMoE(hidden, hidden, num_experts, top_k)
        self.dropout = dropout
        self.lin     = Linear(hidden, out_ch)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.moe(x)
        return self.lin(x)

    def lb_loss(self):
        return self.moe.lb_loss()


class MoEDMPNN(torch.nn.Module):
    def __init__(self, in_ch, edge_ch, hidden, out_ch, num_layers, dropout, num_experts, top_k):
        super().__init__()
        self.input_proj = Linear(in_ch, hidden)
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        for _ in range(num_layers):
            nn_edge = Sequential(Linear(edge_ch, hidden * hidden))
            self.convs.append(NNConv(hidden, hidden, nn_edge, aggr='mean'))
            self.bns.append(BatchNorm1d(hidden))
        self.gru     = GRUCell(hidden, hidden)
        self.moe     = SparseMoE(hidden, hidden, num_experts, top_k)
        self.dropout = dropout
        self.lin     = Linear(hidden, out_ch)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.input_proj(x))
        h = x
        for conv, bn in zip(self.convs, self.bns):
            m = F.relu(bn(conv(h, edge_index, edge_attr)))
            m = F.dropout(m, p=self.dropout, training=self.training)
            h = self.gru(m, h)
        x = global_mean_pool(h, batch)
        x = self.moe(x)
        return self.lin(x)

    def lb_loss(self):
        return self.moe.lb_loss()


def build_model(backbone, in_ch, edge_ch, hidden, out_ch, num_layers, dropout, num_experts, top_k):
    if backbone == "MoE-GIN":
        return MoEGIN(in_ch, hidden, out_ch, num_layers, dropout, num_experts, top_k).to(DEVICE)
    elif backbone == "MoE-GCN":
        return MoEGCN(in_ch, hidden, out_ch, num_layers, dropout, num_experts, top_k).to(DEVICE)
    elif backbone == "MoE-DMPNN":
        return MoEDMPNN(in_ch, edge_ch, hidden, out_ch, num_layers, dropout, num_experts, top_k).to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# SCAFFOLD SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict
    scaffolds = defaultdict(list)
    for i, data in enumerate(dataset):
        try:
            mol = Chem.MolFromSmiles(data.smiles)
            sc  = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            sc = ""
        scaffolds[sc].append(i)
    sets    = sorted(scaffolds.values(), key=len, reverse=True)
    n       = len(dataset)
    n_train = int(n * frac_train)
    n_val   = int(n * frac_val)
    train_idx, val_idx, test_idx = [], [], []
    for s in sets:
        if   len(train_idx) + len(s) <= n_train: train_idx.extend(s)
        elif len(val_idx)   + len(s) <= n_val:   val_idx.extend(s)
        else:                                      test_idx.extend(s)
    return train_idx, val_idx, test_idx


# ══════════════════════════════════════════════════════════════════════════════
# LOSS & METRICS — shape-safe
# ══════════════════════════════════════════════════════════════════════════════

def align(out, tgt):
    """Flatten both to (N,) for single-task or (N,T) for multi-task."""
    # Flatten tgt: (N,1,1) or (N,1) -> (N,)
    while tgt.dim() > 1 and tgt.shape[-1] == 1:
        tgt = tgt.squeeze(-1)
    # Match out to tgt
    while out.dim() > tgt.dim() and out.shape[-1] == 1:
        out = out.squeeze(-1)
    return out, tgt

def masked_bce_loss(out, tgt):
    out, tgt = align(out, tgt)
    mask = ~torch.isnan(tgt)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=out.device)
    return F.binary_cross_entropy_with_logits(out[mask], tgt[mask])

def masked_mse_loss(out, tgt):
    out, tgt = align(out, tgt)
    mask = ~torch.isnan(tgt)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=out.device)
    return F.mse_loss(out[mask], tgt[mask])

def compute_roc_auc(preds, targets):
    preds, targets = np.array(preds), np.array(targets)
    if preds.ndim == 1: preds, targets = preds[:, None], targets[:, None]
    aucs = []
    for t in range(targets.shape[1]):
        mask = ~np.isnan(targets[:, t])
        if mask.sum() < 2 or len(np.unique(targets[mask, t])) < 2: continue
        try: aucs.append(roc_auc_score(targets[mask, t], preds[mask, t]))
        except: pass
    return float(np.mean(aucs)) if aucs else 0.0

def compute_rmse(preds, targets):
    p, t = np.array(preds).flatten(), np.array(targets).flatten()
    mask = ~np.isnan(t)
    return float(np.sqrt(np.mean((p[mask] - t[mask])**2)))


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, task_type):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out  = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
        tgt  = batch.y.float()
        out, tgt = align(out, tgt)
        loss = masked_bce_loss(out, tgt) if task_type == "cls" else masked_mse_loss(out, tgt)
        loss = loss + LB_WEIGHT * model.lb_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

@torch.no_grad()
def eval_epoch(model, loader, task_type):
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out   = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
        tgt   = batch.y
        out, tgt = align(out, tgt)
        preds = torch.sigmoid(out).cpu().numpy() if task_type == "cls" else out.cpu().numpy()
        all_preds.append(preds[:, None] if preds.ndim == 1 else preds)
        tgt_np = tgt.cpu().numpy()
        all_targets.append(tgt_np[:, None] if tgt_np.ndim == 1 else tgt_np)
    p = np.concatenate(all_preds,   axis=0)
    t = np.concatenate(all_targets, axis=0)
    score = compute_roc_auc(p, t) if task_type == "cls" else -compute_rmse(p, t)
    return score


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_training(train_loader, val_loader, test_loader, params,
                 backbone, in_ch, edge_ch, n_tasks, task_type, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(backbone, in_ch, edge_ch,
                        params["hidden"], n_tasks,
                        params["num_layers"], params["dropout"],
                        params["num_experts"], params["top_k"])
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7, min_lr=1e-6)
    best_val, best_state, patience_ctr = -1e9, None, 0
    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, task_type)
        val_score = eval_epoch(model, val_loader, task_type)
        scheduler.step(val_score)
        if val_score > best_val:
            best_val     = val_score
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE: break
    model.load_state_dict(best_state)
    test_score = eval_epoch(model, test_loader, task_type)
    return test_score if task_type == "cls" else -test_score


# ══════════════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(train_loader, val_loader, backbone, in_ch, edge_ch, n_tasks, task_type):
    def objective(trial):
        params = {
            "hidden"      : trial.suggest_categorical("hidden",      [64, 128, 256]),
            "num_layers"  : trial.suggest_int("num_layers",          2, 5),
            "dropout"     : trial.suggest_float("dropout",           0.0, 0.5),
            "num_experts" : trial.suggest_categorical("num_experts", [4, 8, 16]),
            "top_k"       : trial.suggest_int("top_k",               1, 4),
            "lr"          : trial.suggest_float("lr",                1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay",      1e-6, 1e-3, log=True),
        }
        params["top_k"] = min(params["top_k"], params["num_experts"])
        torch.manual_seed(42)
        model = build_model(backbone, in_ch, edge_ch,
                            params["hidden"], n_tasks,
                            params["num_layers"], params["dropout"],
                            params["num_experts"], params["top_k"])
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=params["lr"], weight_decay=params["weight_decay"])
        best_val, patience_ctr = -1e9, 0
        for epoch in range(1, EPOCHS + 1):
            train_epoch(model, train_loader, optimizer, task_type)
            val_score = eval_epoch(model, val_loader, task_type)
            if val_score > best_val:
                best_val, patience_ctr = val_score, 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE: break
            trial.report(val_score, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return best_val
    return objective


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

all_datasets = CLASSIFICATION_DATASETS + REGRESSION_DATASETS

for backbone in BACKBONES:
    if backbone not in results:
        results[backbone] = {}

    for ds_cfg in all_datasets:
        name      = ds_cfg["name"]
        n_tasks   = ds_cfg["tasks"]
        metric    = ds_cfg["metric"]
        task_type = "cls" if metric == "roc_auc" else "reg"

        if name in results[backbone]:
            print(f"  [{backbone}] {name} already done, skipping.")
            continue

        print(f"\n{'='*65}")
        print(f"  Backbone : {backbone}  |  Dataset : {name}  |  Tasks : {n_tasks}")
        print(f"{'='*65}")

        t0      = time.time()
        dataset = MoleculeNet(root=DATA_ROOT, name=name)
        in_ch   = dataset[0].x.shape[1]
        edge_ch = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 3

        train_idx, val_idx, test_idx = scaffold_split(dataset)
        train_data = [dataset[i] for i in train_idx]
        val_data   = [dataset[i] for i in val_idx]
        test_data  = [dataset[i] for i in test_idx]

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

        print(f"  Split → Train:{len(train_data)}  Val:{len(val_data)}  Test:{len(test_data)}")
        print(f"  in_ch={in_ch}  edge_ch={edge_ch}")

        print(f"  Optuna ({N_TRIALS} trials)...")
        study = optuna.create_study(
            study_name = f"{backbone}_{name}_{int(time.time())}",
            direction  = "maximize",
            sampler    = TPESampler(seed=42),
            pruner     = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
        )
        study.optimize(
            make_objective(train_loader, val_loader, backbone, in_ch, edge_ch, n_tasks, task_type),
            n_trials=N_TRIALS, timeout=3600, show_progress_bar=False,
        )
        best_params = study.best_params
        best_params["top_k"] = min(best_params["top_k"], best_params["num_experts"])
        print(f"  Best val: {study.best_value:.4f}  Params: {best_params}")

        print(f"  Final eval ({N_SEEDS} seeds)...")
        seed_scores = []
        for seed in [0, 1, 2]:
            score = run_training(train_loader, val_loader, test_loader,
                                 best_params, backbone, in_ch, edge_ch,
                                 n_tasks, task_type, seed)
            seed_scores.append(score)
            label = f"AUC: {score:.4f}" if task_type == "cls" else f"RMSE: {score:.4f}"
            print(f"    Seed {seed} → {label}")

        mean_s  = float(np.mean(seed_scores))
        std_s   = float(np.std(seed_scores))
        elapsed = time.time() - t0

        results[backbone][name] = {
            "metric"      : metric,
            "mean"        : mean_s,
            "std"         : std_s,
            "seeds"       : seed_scores,
            "best_params" : best_params,
            "time_min"    : round(elapsed / 60, 1),
        }
        print(f"  ✓ [{backbone}] {name}: {mean_s:.4f} ± {std_s:.4f}  ({elapsed/60:.1f} min)")

        with open(SAVE_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {SAVE_PATH}")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  PHASE 3 SUMMARY")
print(f"{'='*65}")
print(f"{'Backbone':<14} {'Dataset':<14} {'Score':>10} {'±Std':>8}  Metric")
print("-" * 55)
for backbone, datasets in results.items():
    for ds_name, r in datasets.items():
        print(f"{backbone:<14} {ds_name:<14} {r['mean']:>10.4f} {r['std']:>8.4f}  {r['metric']}")
print(f"\nAll results saved to: {SAVE_PATH}")
