"""
Plain D-MPNN Baseline (Chemprop-style)
No MoE — just the directed message passing backbone
All 9 MoleculeNet datasets
Results saved to: results_dmpnn_baseline.json
"""

import os, json, time, warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ReLU, Sequential, GRUCell
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
SAVE_PATH  = "results_dmpnn_baseline.json"

CLASSIFICATION_DATASETS = [
    {"name": "BBBP",    "tasks": 1,  "metric": "roc_auc"},
    {"name": "BACE",    "tasks": 1,  "metric": "roc_auc"},
    {"name": "Tox21",   "tasks": 12, "metric": "roc_auc"},
    {"name": "ClinTox", "tasks": 2,  "metric": "roc_auc"},
    {"name": "SIDER",   "tasks": 27, "metric": "roc_auc"},
    {"name": "HIV",     "tasks": 1,  "metric": "roc_auc"},
]
REGRESSION_DATASETS = [
    {"name": "ESOL",     "tasks": 1, "metric": "rmse"},
    {"name": "FreeSolv", "tasks": 1, "metric": "rmse"},
    {"name": "Lipo",     "tasks": 1, "metric": "rmse"},
]

print(f"Device : {DEVICE}")
print(f"Model  : Plain D-MPNN (no MoE)")
print(f"Datasets: {[d['name'] for d in CLASSIFICATION_DATASETS + REGRESSION_DATASETS]}\n")

results = {}
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH) as f:
        results = json.load(f)
    print(f"Loaded existing: {list(results.keys())}\n")


# ── D-MPNN Model ─────────────────────────────────────────────────────────────

class DMPNN(torch.nn.Module):
    """
    Directed Message Passing Neural Network (Chemprop-style).
    Uses NNConv for edge-conditioned message passing + GRU updates.
    """
    def __init__(self, in_ch, edge_ch, hidden, out_ch, num_layers, dropout):
        super().__init__()
        self.input_proj = Linear(in_ch, hidden)
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        for _ in range(num_layers):
            nn_edge = Sequential(Linear(edge_ch, hidden * hidden))
            self.convs.append(NNConv(hidden, hidden, nn_edge, aggr='mean'))
            self.bns.append(BatchNorm1d(hidden))
        self.gru     = GRUCell(hidden, hidden)
        self.dropout = dropout
        self.lin1    = Linear(hidden, hidden // 2)
        self.lin2    = Linear(hidden // 2, out_ch)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.input_proj(x))
        h = x
        for conv, bn in zip(self.convs, self.bns):
            m = F.relu(bn(conv(h, edge_index, edge_attr)))
            m = F.dropout(m, p=self.dropout, training=self.training)
            h = self.gru(m, h)
        x = global_mean_pool(h, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x).reshape(-1) if x.shape[0] > 0 else self.lin2(x)


# ── Scaffold split ────────────────────────────────────────────────────────────

def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict
    scaffolds = defaultdict(list)
    for i, data in enumerate(dataset):
        try:
            mol = Chem.MolFromSmiles(data.smiles)
            sc  = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except: sc = ""
        scaffolds[sc].append(i)
    sets = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(dataset); nt = int(n*0.8); nv = int(n*0.1)
    tr, va, te = [], [], []
    for s in sets:
        if   len(tr)+len(s) <= nt: tr.extend(s)
        elif len(va)+len(s) <= nv: va.extend(s)
        else:                       te.extend(s)
    return tr, va, te


# ── Loss & metrics ────────────────────────────────────────────────────────────

def to1d(t): return t.reshape(-1)

def masked_bce(out, tgt):
    out, tgt = out.view(tgt.shape) if out.numel()==tgt.numel() and out.shape!=tgt.shape else (out, tgt), tgt
    mask = ~torch.isnan(tgt)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=out.device)
    return F.binary_cross_entropy_with_logits(out[mask], tgt[mask])

def masked_mse(out, tgt):
    mask = ~torch.isnan(tgt)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=out.device)
    return F.mse_loss(out[mask], tgt[mask])

def compute_roc_auc(preds, targets):
    preds, targets = np.array(preds), np.array(targets)
    if preds.ndim == 1: preds, targets = preds[:,None], targets[:,None]
    aucs = []
    for t in range(targets.shape[1]):
        mask = ~np.isnan(targets[:,t])
        if mask.sum() < 2 or len(np.unique(targets[mask,t])) < 2: continue
        try: aucs.append(roc_auc_score(targets[mask,t], preds[mask,t]))
        except: pass
    return float(np.mean(aucs)) if aucs else 0.0

def compute_rmse(preds, targets):
    p, t = np.array(preds).flatten(), np.array(targets).flatten()
    mask = ~np.isnan(t)
    return float(np.sqrt(np.mean((p[mask]-t[mask])**2)))


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_one(model, loader, opt, task_type):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        opt.zero_grad()
        out = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
        tgt = batch.y.float()
        # align shapes
        if out.shape != tgt.shape:
            if tgt.dim() == 1:
                out = out.reshape(-1)
            else:
                out = out.reshape(tgt.shape) if out.numel()==tgt.numel() else out
        loss = masked_bce(out, tgt) if task_type=="cls" else masked_mse(out.reshape(-1), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

@torch.no_grad()
def eval_one(model, loader, task_type):
    model.eval()
    ps, ts = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
        tgt = batch.y
        ps.append(torch.sigmoid(out).cpu().numpy() if task_type=="cls" else out.cpu().numpy())
        ts.append(tgt.cpu().numpy())
    p = np.concatenate(ps, axis=0)
    t = np.concatenate(ts, axis=0)
    if task_type == "cls":
        if p.ndim == 1: p, t = p[:,None], t.reshape(-1,1) if t.ndim==1 else t
        return compute_roc_auc(p, t)
    else:
        return -compute_rmse(p.flatten(), t.flatten())


# ── Full training run ─────────────────────────────────────────────────────────

def full_run(trl, vl, tel, p, in_ch, ec, n_tasks, task_type, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = DMPNN(in_ch, ec, p["hidden"], n_tasks, p["num_layers"], p["dropout"]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=7, min_lr=1e-6)
    best_val, best_state, pat = -1e9, None, 0
    for ep in range(1, EPOCHS+1):
        train_one(model, trl, opt, task_type)
        v = eval_one(model, vl, task_type)
        sch.step(v)
        if v > best_val:
            best_val = v
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE: break
    if best_state is None: return 0.0 if task_type=="cls" else 999.0
    model.load_state_dict(best_state)
    score = eval_one(model, tel, task_type)
    return score if task_type=="cls" else -score


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_obj(trl, vl, in_ch, ec, n_tasks, task_type):
    def obj(trial):
        p = {
            "hidden"      : trial.suggest_categorical("hidden",      [64, 128, 256, 300]),
            "num_layers"  : trial.suggest_int("num_layers",          2, 5),
            "dropout"     : trial.suggest_float("dropout",           0.0, 0.5),
            "lr"          : trial.suggest_float("lr",                1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay",      1e-6, 1e-3, log=True),
        }
        torch.manual_seed(trial.number * 7 + 13)
        model = DMPNN(in_ch, ec, p["hidden"], n_tasks, p["num_layers"], p["dropout"]).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        best, pat = -1e9, 0
        for ep in range(1, EPOCHS+1):
            train_one(model, trl, opt, task_type)
            v = eval_one(model, vl, task_type)
            if v > best: best, pat = v, 0
            else:
                pat += 1
                if pat >= PATIENCE: break
            trial.report(v, ep)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        return best
    return obj


# ── Main ─────────────────────────────────────────────────────────────────────

all_datasets = CLASSIFICATION_DATASETS + REGRESSION_DATASETS

for ds_cfg in all_datasets:
    name      = ds_cfg["name"]
    n_tasks   = ds_cfg["tasks"]
    metric    = ds_cfg["metric"]
    task_type = "cls" if metric == "roc_auc" else "reg"

    if name in results:
        print(f"  {name} already done ({results[name]['mean']:.4f}), skipping.")
        continue

    print(f"\n{'='*60}")
    print(f"  D-MPNN | {name} | Tasks:{n_tasks} | {metric.upper()}")
    print(f"{'='*60}")

    t0      = time.time()
    dataset = MoleculeNet(root=DATA_ROOT, name=name)
    in_ch   = dataset[0].x.shape[1]
    ec      = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 3

    tri, vai, tei = scaffold_split(dataset)
    trl = DataLoader([dataset[i] for i in tri], batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    vl  = DataLoader([dataset[i] for i in vai], batch_size=BATCH_SIZE, shuffle=False)
    tel = DataLoader([dataset[i] for i in tei], batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Split → Tr:{len(tri)} Va:{len(vai)} Te:{len(tei)}")

    study = optuna.create_study(
        study_name = f"DMPNN_{name}_{int(time.time())}",
        direction  = "maximize",
        sampler    = TPESampler(seed=int(time.time()) % 99991),
        pruner     = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )
    study.optimize(make_obj(trl, vl, in_ch, ec, n_tasks, task_type),
                   n_trials=N_TRIALS, timeout=3600, show_progress_bar=False)

    bp = study.best_params
    print(f"  Best val: {study.best_value:.4f} | {bp}")

    scores = []
    for seed in [0, 1, 2]:
        s = full_run(trl, vl, tel, bp, in_ch, ec, n_tasks, task_type, seed)
        scores.append(s)
        label = f"AUC: {s:.4f}" if task_type=="cls" else f"RMSE: {s:.4f}"
        print(f"    Seed {seed} → {label}")

    mean_s = float(np.mean(scores))
    std_s  = float(np.std(scores))
    elapsed = time.time() - t0

    results[name] = {
        "metric": metric, "mean": mean_s, "std": std_s,
        "seeds": scores, "best_params": bp,
        "time_min": round(elapsed/60, 1),
    }
    print(f"  ✓ D-MPNN {name}: {mean_s:.4f} ± {std_s:.4f}  ({elapsed/60:.1f} min)")

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {SAVE_PATH}")

print(f"\n{'='*60}")
print("D-MPNN BASELINE COMPLETE")
print(f"{'='*60}")
print(f"{'Dataset':<14} {'Score':>10} {'±Std':>8}  Metric")
print("-"*40)
for name, r in results.items():
    print(f"{name:<14} {r['mean']:>10.4f} {r['std']:>8.4f}  {r['metric']}")
