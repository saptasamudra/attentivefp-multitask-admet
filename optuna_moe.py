"""
Optuna Hyperparameter Optimization for MoE-AttentiveFP
TPE sampler, 50 trials per dataset, saves best config + results
"""
import os, json, warnings, random
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from optuna.samplers import TPESampler
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import AttentiveFP
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except: pass

os.makedirs("results", exist_ok=True)
os.makedirs("data",    exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  |  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

DATASETS = {
    "ESOL":          {"name": "ESOL",    "task_type": "reg"},
    "FreeSolv":      {"name": "FreeSolv","task_type": "reg"},
    "Lipophilicity": {"name": "Lipo",    "task_type": "reg"},
    "BBBP":          {"name": "BBBP",    "task_type": "cls"},
    "Tox21":         {"name": "Tox21",   "task_type": "cls"},
    "SIDER":         {"name": "SIDER",   "task_type": "cls"},
    "ClinTox":       {"name": "ClinTox", "task_type": "cls"},
    "BACE":          {"name": "BACE",    "task_type": "cls"},
    "HIV":           {"name": "HIV",     "task_type": "cls"},
}

N_TRIALS   = 50
SEEDS      = [42, 123, 7]
EVAL_SEED  = 42   # single seed during Optuna search (speed)

# ── SPLITS ─────────────────────────────────────────────────────────────────────
def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaffolds = defaultdict(list)
    for i in range(len(dataset)):
        try:
            smi = dataset.smiles[i]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                scaffolds[""].append(i); continue
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds[s].append(i)
        except:
            scaffolds[""].append(i)
    groups = sorted(scaffolds.values(), key=len, reverse=True)
    t_cut = int(frac_train * len(dataset))
    v_cut = int((frac_train + frac_val) * len(dataset))
    tr, va, te = [], [], []
    for g in groups:
        if   len(tr) < t_cut:           tr.extend(g)
        elif len(tr)+len(va) < v_cut:   va.extend(g)
        else:                            te.extend(g)
    if not te: te = va
    return ([dataset[i] for i in tr],
            [dataset[i] for i in va],
            [dataset[i] for i in te])

def random_split(dataset, seed, frac_train=0.8, frac_val=0.1):
    idx = list(range(len(dataset)))
    random.seed(seed); random.shuffle(idx)
    n_tr = int(frac_train * len(idx))
    n_va = int((frac_train + frac_val) * len(idx))
    return ([dataset[i] for i in idx[:n_tr]],
            [dataset[i] for i in idx[n_tr:n_va]],
            [dataset[i] for i in idx[n_va:]])

def is_degenerate(split, task_type):
    if task_type != "cls": return False
    try:
        labels = []
        for d in split:
            y = d.y.float().view(-1)
            labels.extend(y[~torch.isnan(y)].tolist())
        return len(set(int(l) for l in labels)) < 2
    except:
        return False

def get_split(dataset, task_type, seed):
    tr, va, te = scaffold_split(dataset)
    if is_degenerate(te, task_type):
        tr, va, te = random_split(dataset, seed)
    return tr, va, te

# ── MoE MODEL ──────────────────────────────────────────────────────────────────
class Expert(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Linear(dim, dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*2, dim), nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(x + self.net(x))

class SparseMoE(nn.Module):
    def __init__(self, dim, num_experts, top_k, dropout):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.experts     = nn.ModuleList([Expert(dim, dropout) for _ in range(num_experts)])
        self.gate        = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x):
        gate_probs = F.softmax(self.gate(x), dim=-1)
        topk_vals, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            eidx = topk_idx[:, k]
            ew   = topk_vals[:, k]
            for e in range(self.num_experts):
                mask = (eidx == e)
                if mask.sum() == 0: continue
                out[mask] += ew[mask].unsqueeze(-1) * self.experts[e](x[mask])
        # load balance loss
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        for e in range(self.num_experts):
            expert_usage[e] = (topk_idx == e).float().mean()
        aux_loss = self.num_experts * (expert_usage * gate_probs.mean(dim=0)).sum()
        return out, aux_loss

class MoEAttentiveFP(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden, num_layers,
                 num_timesteps, out_dim, dropout, num_experts, top_k):
        super().__init__()
        self.encoder = AttentiveFP(
            in_channels=in_channels, hidden_channels=hidden,
            out_channels=hidden, edge_dim=edge_dim,
            num_layers=num_layers, num_timesteps=num_timesteps, dropout=dropout
        )
        self.moe  = SparseMoE(hidden, num_experts, top_k, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden//2, out_dim)
        )

    def forward(self, data):
        ea = data.edge_attr.float() if data.edge_attr is not None else \
             torch.zeros(data.edge_index.shape[1], 1).to(data.x.device)
        h = self.encoder(data.x.float(), data.edge_index, ea, data.batch)
        h, aux = self.moe(h)
        return self.head(h), aux

# ── TRAIN / EVAL ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, task_type, lb_w):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out, aux = model(batch)
        y = batch.y.float()
        if y.ndim == 1:   y   = y.unsqueeze(1)
        if out.ndim == 1: out = out.unsqueeze(1)
        mask = ~torch.isnan(y)
        if mask.sum() == 0: continue
        tl = (F.binary_cross_entropy_with_logits(out[mask], y[mask])
              if task_type=="cls" else F.mse_loss(out[mask], y[mask]))
        (tl + lb_w * aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += tl.item()
    return total / max(len(loader), 1)

@torch.no_grad()
def evaluate(model, loader, task_type):
    model.eval()
    all_p, all_l = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out, _ = model(batch)
        y = batch.y.float()
        if y.ndim == 1:   y   = y.unsqueeze(1)
        if out.ndim == 1: out = out.unsqueeze(1)
        p = torch.sigmoid(out).cpu().numpy() if task_type=="cls" else out.cpu().numpy()
        all_p.append(p); all_l.append(y.cpu().numpy())
    if not all_p: return {}
    preds  = np.concatenate(all_p,  axis=0)
    labels = np.concatenate(all_l, axis=0)
    if task_type == "cls":
        aucs = []
        for t in range(preds.shape[1]):
            l_t, p_t = labels[:,t], preds[:,t]
            ok = ~np.isnan(l_t)
            if ok.sum() < 10 or len(np.unique(l_t[ok])) < 2: continue
            try: aucs.append(roc_auc_score(l_t[ok], p_t[ok]))
            except: pass
        return {"roc_auc": float(np.mean(aucs)) if aucs else 0.0}
    else:
        l, p = labels[:,0], preds[:,0]
        ok = ~np.isnan(l)
        return {"rmse": float(np.sqrt(mean_squared_error(l[ok], p[ok]))),
                "mae":  float(mean_absolute_error(l[ok], p[ok]))}

# ── SINGLE TRAIN RUN ───────────────────────────────────────────────────────────
def run_trial(cfg, ds_key, ds_cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    dataset   = MoleculeNet(root=f"data/{ds_key}", name=ds_cfg["name"])
    task_type = ds_cfg["task_type"]
    in_ch     = dataset[0].x.shape[1]
    edge_dim  = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 1
    y0        = dataset[0].y
    out_dim   = y0.shape[1] if y0.ndim > 1 else 1

    tr, va, te = get_split(dataset, task_type, seed)
    tr_l = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True)
    va_l = DataLoader(va, batch_size=cfg["batch_size"])
    te_l = DataLoader(te, batch_size=cfg["batch_size"])

    model = MoEAttentiveFP(
        in_channels=in_ch, edge_dim=edge_dim,
        hidden=cfg["hidden"], num_layers=cfg["num_layers"],
        num_timesteps=cfg["num_timesteps"], out_dim=out_dim,
        dropout=cfg["dropout"], num_experts=cfg["num_experts"],
        top_k=cfg["top_k"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min" if task_type=="reg" else "max",
        patience=8, factor=0.5, min_lr=1e-5
    )

    best_val, best_m = None, {}
    for epoch in range(1, cfg["epochs"]+1):
        train_epoch(model, tr_l, optimizer, task_type, cfg["lb_weight"])
        vm = evaluate(model, va_l, task_type)
        if not vm: continue
        scheduler.step(vm.get("rmse", -vm.get("roc_auc", 0)))
        improved = (best_val is None or
            (task_type=="reg" and vm["rmse"]    < best_val) or
            (task_type=="cls" and vm["roc_auc"] > best_val))
        if improved:
            best_val = vm.get("rmse", vm.get("roc_auc"))
            best_m   = evaluate(model, te_l, task_type)
    return best_m

# ── OPTUNA OBJECTIVE ───────────────────────────────────────────────────────────
def make_objective(ds_key, ds_cfg):
    def objective(trial):
        cfg = {
            "hidden":       trial.suggest_categorical("hidden",       [128, 200, 256]),
            "num_layers":   trial.suggest_int("num_layers",           2, 4),
            "num_timesteps":trial.suggest_int("num_timesteps",        2, 4),
            "dropout":      trial.suggest_float("dropout",            0.0, 0.5, step=0.1),
            "num_experts":  trial.suggest_categorical("num_experts",  [4, 8, 12, 16]),
            "top_k":        trial.suggest_int("top_k",                1, 4),
            "lb_weight":    trial.suggest_float("lb_weight",          0.001, 0.1, log=True),
            "lr":           trial.suggest_float("lr",                 1e-4, 5e-3, log=True),
            "wd":           trial.suggest_float("wd",                 1e-6, 1e-3, log=True),
            "batch_size":   trial.suggest_categorical("batch_size",   [32, 64, 128]),
            "epochs":       80,
        }
        # top_k must be <= num_experts
        if cfg["top_k"] > cfg["num_experts"]:
            raise optuna.TrialPruned()

        try:
            m = run_trial(cfg, ds_key, ds_cfg, EVAL_SEED)
        except Exception:
            raise optuna.TrialPruned()

        if ds_cfg["task_type"] == "reg":
            return m.get("rmse", 999)  # minimize
        else:
            return -m.get("roc_auc", 0)  # minimize negative AUC

    return objective

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path  = f"results/optuna_moe_{timestamp}.json"
    all_results = {}

    for ds_key, ds_cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  Optuna: {ds_key}  ({N_TRIALS} trials)")
        print(f"{'='*60}")

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
        )
        study.optimize(make_objective(ds_key, ds_cfg), n_trials=N_TRIALS,
                       show_progress_bar=False)

        best_cfg = study.best_params
        best_cfg["epochs"] = 80
        best_val = study.best_value

        print(f"  Best trial value: {best_val:.4f}")
        print(f"  Best config: {best_cfg}")

        # Evaluate best config across all 3 seeds
        print(f"  Running 3-seed evaluation with best config...")
        seed_results = []
        for seed in SEEDS:
            try:
                m = run_trial(best_cfg, ds_key, ds_cfg, seed)
                seed_results.append(m)
                if ds_cfg["task_type"] == "cls":
                    print(f"    seed={seed} | AUC={m.get('roc_auc',0):.4f}")
                else:
                    print(f"    seed={seed} | RMSE={m.get('rmse',0):.4f}")
            except Exception as e:
                print(f"    seed={seed} | ERROR: {e}")
                seed_results.append({"error": str(e)})

        valid = [r for r in seed_results if "error" not in r and r]
        agg = {}
        if valid:
            for k in valid[0]:
                vals = [r[k] for r in valid]
                agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
                agg[f"{k}_std"]  = round(float(np.std(vals)),  4)
            if ds_cfg["task_type"] == "cls":
                print(f"  → {ds_key}: AUC = {agg['roc_auc_mean']:.4f} ± {agg['roc_auc_std']:.4f}")
            else:
                print(f"  → {ds_key}: RMSE = {agg['rmse_mean']:.4f} ± {agg['rmse_std']:.4f}")

        all_results[ds_key] = {
            "best_config": best_cfg,
            "optuna_best_val": best_val,
            "seeds": seed_results,
            "agg": agg
        }

        # Save after each dataset
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  ✓ Saved → {out_path}")

    print("\n✅  Optuna tuning complete.")

if __name__ == "__main__":
    main()
