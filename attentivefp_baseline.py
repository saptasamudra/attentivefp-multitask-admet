"""
AttentiveFP Baseline
Original Xiong et al. 2020 architecture on 9 MoleculeNet datasets
Uses PyG's built-in AttentiveFP implementation
Same scaffold split, seeds, eval as GNN baselines for fair comparison
"""
import os, json, warnings, random
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import AttentiveFP
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except: pass

os.makedirs("results", exist_ok=True)
os.makedirs("data",    exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  |  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
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

# AttentiveFP hyperparameters (original paper defaults)
HIDDEN     = 200
NUM_LAYERS = 2      # graph conv layers
NUM_TIMESTEPS = 2   # readout timesteps
DROPOUT    = 0.2
EPOCHS     = 80     # more epochs than GNN baselines — AFP needs more
BATCH_SIZE = 64
LR         = 1e-3
SEEDS      = [42, 123, 7]

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
    t_cut  = int(frac_train * len(dataset))
    v_cut  = int((frac_train + frac_val) * len(dataset))
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
        print(f"    [INFO] Degenerate scaffold split → random split")
        tr, va, te = random_split(dataset, seed)
    return tr, va, te

# ── MODEL ──────────────────────────────────────────────────────────────────────
class AttentiveFPModel(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden, num_layers,
                 num_timesteps, out_dim, dropout):
        super().__init__()
        self.afp = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden,
            out_channels=hidden,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim)
        )

    def forward(self, data):
        # AttentiveFP needs edge_attr
        edge_attr = data.edge_attr.float() if data.edge_attr is not None else \
                    torch.zeros(data.edge_index.shape[1], 1).to(data.x.device)
        x = self.afp(
            data.x.float(),
            data.edge_index,
            edge_attr,
            data.batch
        )
        return self.head(x)

# ── TRAIN / EVAL ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, task_type):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch)
        y   = batch.y.float()
        if y.ndim == 1:   y   = y.unsqueeze(1)
        if out.ndim == 1: out = out.unsqueeze(1)
        mask = ~torch.isnan(y)
        if mask.sum() == 0: continue
        loss = (F.binary_cross_entropy_with_logits(out[mask], y[mask])
                if task_type == "cls" else F.mse_loss(out[mask], y[mask]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)

@torch.no_grad()
def evaluate(model, loader, task_type):
    model.eval()
    all_p, all_l = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out = model(batch)
        y   = batch.y.float()
        if y.ndim == 1:   y   = y.unsqueeze(1)
        if out.ndim == 1: out = out.unsqueeze(1)
        p = torch.sigmoid(out).cpu().numpy() if task_type=="cls" else out.cpu().numpy()
        all_p.append(p)
        all_l.append(y.cpu().numpy())
    if not all_p: return {}
    preds  = np.concatenate(all_p,  axis=0)
    labels = np.concatenate(all_l, axis=0)
    if task_type == "cls":
        aucs = []
        for t in range(preds.shape[1]):
            l_t, p_t = labels[:, t], preds[:, t]
            ok = ~np.isnan(l_t)
            if ok.sum() < 10 or len(np.unique(l_t[ok])) < 2: continue
            try: aucs.append(roc_auc_score(l_t[ok], p_t[ok]))
            except: pass
        return {"roc_auc": float(np.mean(aucs)) if aucs else 0.0}
    else:
        l, p = labels[:, 0], preds[:, 0]
        ok   = ~np.isnan(l)
        return {
            "rmse": float(np.sqrt(mean_squared_error(l[ok], p[ok]))),
            "mae":  float(mean_absolute_error(l[ok], p[ok]))
        }

# ── SINGLE RUN ─────────────────────────────────────────────────────────────────
def run_one(ds_key, ds_cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset   = MoleculeNet(root=f"data/{ds_key}", name=ds_cfg["name"])
    task_type = ds_cfg["task_type"]

    in_channels = dataset[0].x.shape[1]
    edge_dim    = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 1
    y0          = dataset[0].y
    out_dim     = y0.shape[1] if y0.ndim > 1 else 1

    tr, va, te = get_split(dataset, task_type, seed)
    tr_l = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True)
    va_l = DataLoader(va, batch_size=BATCH_SIZE)
    te_l = DataLoader(te, batch_size=BATCH_SIZE)

    model = AttentiveFPModel(
        in_channels=in_channels,
        edge_dim=edge_dim,
        hidden=HIDDEN,
        num_layers=NUM_LAYERS,
        num_timesteps=NUM_TIMESTEPS,
        out_dim=out_dim,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if task_type=="reg" else "max",
        patience=8, factor=0.5, min_lr=1e-5
    )

    best_val, best_m = None, {}

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, tr_l, optimizer, task_type)
        vm = evaluate(model, va_l, task_type)
        if not vm: continue
        scheduler.step(vm.get("rmse", -vm.get("roc_auc", 0)))
        improved = (
            best_val is None or
            (task_type=="reg" and vm["rmse"]    < best_val) or
            (task_type=="cls" and vm["roc_auc"] > best_val)
        )
        if improved:
            best_val = vm.get("rmse", vm.get("roc_auc"))
            best_m   = evaluate(model, te_l, task_type)

    return best_m

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path  = f"results/attentivefp_{timestamp}.json"
    results   = {}

    print(f"{'='*65}\n  AttentiveFP (Xiong et al. 2020)\n{'='*65}")

    for ds_key, ds_cfg in DATASETS.items():
        seed_res = []

        for seed in SEEDS:
            try:
                m = run_one(ds_key, ds_cfg, seed)
                seed_res.append(m)
                if ds_cfg["task_type"] == "cls":
                    print(f"  AFP | {ds_key:<15} | seed={seed} | AUC={m.get('roc_auc',0):.4f}")
                else:
                    print(f"  AFP | {ds_key:<15} | seed={seed} | RMSE={m.get('rmse',0):.4f}  MAE={m.get('mae',0):.4f}")
            except Exception as e:
                print(f"  AFP | {ds_key:<15} | seed={seed} | ERROR: {e}")
                seed_res.append({"error": str(e)})

        valid = [r for r in seed_res if "error" not in r and r]
        agg   = {}
        if valid:
            for k in valid[0]:
                vals = [r[k] for r in valid]
                agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
                agg[f"{k}_std"]  = round(float(np.std(vals)),  4)
            if ds_cfg["task_type"] == "cls":
                print(f"  → {ds_key}: AUC  = {agg['roc_auc_mean']:.4f} ± {agg['roc_auc_std']:.4f}")
            else:
                print(f"  → {ds_key}: RMSE = {agg['rmse_mean']:.4f} ± {agg['rmse_std']:.4f}")

        results[ds_key] = {"seeds": seed_res, "agg": agg}

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Saved → {out_path}")
    print("\n✅  All done.")

if __name__ == "__main__":
    main()
