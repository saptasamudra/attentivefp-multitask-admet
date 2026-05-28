"""
regression_fix.py — Improved Regression with Uncertainty Weighting
Addresses poor FreeSolv performance using learned task uncertainty weighting.
Run: python regression_fix.py
"""

import warnings, copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

warnings.filterwarnings("ignore")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS    = 100
PATIENCE  = 15
DATA_ROOT = "./data"
N_SEEDS   = 5
print(f"Device: {DEVICE}")

def ToFloat(data):
    data.x = data.x.float()
    return data

class MoELayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts; self.top_k = top_k
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()) for _ in range(num_experts)])
        self.gate = nn.Linear(in_dim, num_experts)
    def forward(self, x):
        gl = self.gate(x); tv, ti = torch.topk(gl, self.top_k, dim=-1)
        w = torch.zeros_like(gl).scatter_(1, ti, F.softmax(tv, dim=-1))
        bal = self.num_experts * (w.mean(0)**2).sum()
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        return (w.unsqueeze(-1) * eo).sum(dim=1), bal

class MoEGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_experts, top_k, num_tasks):
        super().__init__()
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList(); self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i==0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe = MoELayer(hidden, hidden, num_experts, top_k)
        self.head = nn.Linear(hidden, num_tasks)
        # Uncertainty weights (log sigma^2) — one per task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei))); x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, b); x, bal = self.moe(x)
        return self.head(x), bal

def uncertainty_loss(pred, target, log_vars):
    """Kendall & Gal uncertainty weighting: L = sum_i (exp(-s_i)*||y-f||^2 + s_i) where s_i = log_var_i"""
    mask = ~torch.isnan(target)
    if mask.sum() == 0: return torch.tensor(0.0, requires_grad=True)
    loss = 0
    for t in range(pred.shape[1] if pred.ndim > 1 else 1):
        p = pred.squeeze() if pred.ndim == 1 else pred[:,t]
        y = target.squeeze() if target.ndim == 1 else target[:,t]
        m = ~torch.isnan(y)
        if m.sum() == 0: continue
        s = log_vars[t] if log_vars.ndim > 0 else log_vars
        mse = F.mse_loss(p[m], y[m])
        loss = loss + torch.exp(-s) * mse + s
    return loss

def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaffolds = defaultdict(list)
    for i in range(len(dataset)):
        try:
            smi = dataset.smiles[i]; mol = Chem.MolFromSmiles(smi)
            sc  = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else smi
        except: sc = str(i)
        scaffolds[sc].append(i)
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(dataset); train_cutoff = int(n*frac_train); val_cutoff = int(n*(frac_train+frac_val))
    train_idx, val_idx, test_idx = [], [], []
    for s in scaffold_sets:
        if len(train_idx) < train_cutoff: train_idx.extend(s)
        elif len(val_idx) < (val_cutoff-train_cutoff): val_idx.extend(s)
        else: test_idx.extend(s)
    return (torch.utils.data.Subset(dataset,train_idx),
            torch.utils.data.Subset(dataset,val_idx),
            torch.utils.data.Subset(dataset,test_idx))

def evaluate_rmse(model, loader):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE); out, _ = model(batch)
            preds.extend(out.squeeze().cpu().numpy().flatten())
            labels.extend(batch.y.squeeze().cpu().numpy().flatten())
    preds, labels = np.array(preds), np.array(labels)
    mask = ~np.isnan(labels)
    return float(np.sqrt(np.mean((preds[mask]-labels[mask])**2)))

DATASETS = [
    {"name":"ESOL",     "tasks":1},
    {"name":"FreeSolv", "tasks":1},
    {"name":"Lipo",     "tasks":1},
]
# Best params from existing results
PARAMS = {
    "ESOL":     {"hidden":256,"num_layers":3,"dropout":0.002,"num_experts":16,"top_k":2,"lr":6.71e-4,"weight_decay":1.99e-5},
    "FreeSolv": {"hidden":256,"num_layers":4,"dropout":0.086,"num_experts":16,"top_k":2,"lr":6.40e-4,"weight_decay":1.39e-5},
    "Lipo":     {"hidden":256,"num_layers":4,"dropout":0.001,"num_experts":16,"top_k":2,"lr":1.38e-4,"weight_decay":3.18e-5},
}

results_new = {}
for ds_cfg in DATASETS:
    name, num_tasks = ds_cfg["name"], ds_cfg["tasks"]
    p = PARAMS[name]
    print(f"\n{'='*50}\n  {name} with uncertainty weighting\n{'='*50}")
    dataset = MoleculeNet(root=DATA_ROOT, name=name, transform=ToFloat)
    in_dim  = dataset.num_node_features
    train_data, val_data, test_data = scaffold_split(dataset)
    vl = DataLoader(val_data, batch_size=BATCH_SIZE)
    tl_test = DataLoader(test_data, batch_size=BATCH_SIZE)
    seed_scores = []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        model = MoEGCN(in_dim, p["hidden"], p["num_layers"], p["dropout"],
                       p["num_experts"], p["top_k"], num_tasks).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        best_val, pat = float("inf"), 0
        best_state = copy.deepcopy(model.state_dict())

        for epoch in range(EPOCHS):
            model.train()
            for batch in DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True):
                batch = batch.to(DEVICE); opt.zero_grad()
                out, bal = model(batch)
                y = batch.y.float()
                loss = uncertainty_loss(out, y, model.log_vars) + 0.01*bal
                loss.backward(); opt.step()
            val_rmse = evaluate_rmse(model, vl)
            sched.step(val_rmse)
            if val_rmse < best_val: best_val=val_rmse; best_state=copy.deepcopy(model.state_dict()); pat=0
            else: pat+=1
            if pat >= PATIENCE: break

        model.load_state_dict(best_state)
        test_rmse = evaluate_rmse(model, tl_test)
        seed_scores.append(test_rmse)
        print(f"    Seed {seed} → RMSE: {test_rmse:.4f}")

    mean_s, std_s = float(np.mean(seed_scores)), float(np.std(seed_scores))
    results_new[name] = {"mean": mean_s, "std": std_s, "seeds": seed_scores}
    print(f"  ✓ {name} (uncertainty): {mean_s:.4f} ± {std_s:.4f}")

# Compare with original
orig = {}
import os
if os.path.exists("results_moegcn_regr.json"):
    with open("results_moegcn_regr.json") as f: orig = json.load(f)

print(f"\n{'='*50}")
print("  REGRESSION IMPROVEMENT COMPARISON")
print(f"{'='*50}")
print(f"{'Dataset':12} {'Original':>12} {'Uncertainty':>14} {'Improvement':>13}")
print("-"*55)
for name in ["ESOL","FreeSolv","Lipo"]:
    orig_m = orig.get(name,{}).get("mean", float("nan"))
    new_m  = results_new.get(name,{}).get("mean", float("nan"))
    imp = (orig_m - new_m) / orig_m * 100 if orig_m > 0 else 0
    print(f"  {name:10} {orig_m:>12.4f} {new_m:>14.4f} {imp:>11.1f}%")

# Save updated regression results
with open("results_moegcn_regr_uncertainty.json","w") as f:
    json.dump(results_new, f, indent=2)
print("\nSaved → results_moegcn_regr_uncertainty.json")

# Plot comparison
fig, ax = plt.subplots(figsize=(9,5))
datasets = ["ESOL","FreeSolv","Lipo"]
x = np.arange(len(datasets)); w = 0.35
orig_means = [orig.get(d,{}).get("mean",0) for d in datasets]
orig_stds  = [orig.get(d,{}).get("std",0)  for d in datasets]
new_means  = [results_new.get(d,{}).get("mean",0) for d in datasets]
new_stds   = [results_new.get(d,{}).get("std",0)  for d in datasets]
ax.bar(x-w/2, orig_means, w, yerr=orig_stds, label="MoE-GCN (baseline)",
       color="#2196F3", alpha=0.8, capsize=4)
ax.bar(x+w/2, new_means,  w, yerr=new_stds,  label="MoE-GCN (uncertainty weighting)",
       color="#E91E63", alpha=0.8, capsize=4)
ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylabel("RMSE (lower is better)"); ax.set_title("Regression RMSE: Baseline vs Uncertainty Weighting")
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("regression_fix.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved → regression_fix.png")
