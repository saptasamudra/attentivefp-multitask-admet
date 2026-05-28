"""
topk_comparison.py — Different K Value Comparison
Compares MoE-GCN with K=1,2,4 on Tox21 and ESOL.
Run: python topk_comparison.py
"""

import warnings, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

warnings.filterwarnings("ignore")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS    = 80
PATIENCE  = 15
DATA_ROOT = "./data"
print(f"Device: {DEVICE}")

def ToFloat(data):
    data.x = data.x.float()
    return data

class MoELayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()) for _ in range(num_experts)])
        self.gate = nn.Linear(in_dim, num_experts)
    def forward(self, x):
        gl = self.gate(x)
        tv, ti = torch.topk(gl, self.top_k, dim=-1)
        w = torch.zeros_like(gl).scatter_(1, ti, F.softmax(tv, dim=-1))
        bal = self.num_experts * (w.mean(0)**2).sum()
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        return (w.unsqueeze(-1) * eo).sum(dim=1), bal

class MoEGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_experts, top_k, num_tasks):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i==0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe = MoELayer(hidden, hidden, num_experts, top_k)
        self.head = nn.Linear(hidden, num_tasks)
    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, b)
        x, bal = self.moe(x)
        return self.head(x), bal

def scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
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
        if len(train_idx) < train_cutoff: train_idx.extend(s)
        elif len(val_idx) < (val_cutoff - train_cutoff): val_idx.extend(s)
        else: test_idx.extend(s)
    return (torch.utils.data.Subset(dataset, train_idx),
            torch.utils.data.Subset(dataset, val_idx),
            torch.utils.data.Subset(dataset, test_idx))

def run_topk(dataset, train_data, val_data, test_data, top_k, is_classif, num_tasks, n_seeds=3):
    in_dim = dataset.num_node_features
    seed_scores = []
    vl = DataLoader(val_data,  batch_size=BATCH_SIZE)
    tl_test = DataLoader(test_data, batch_size=BATCH_SIZE)

    for seed in range(n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        model = MoEGCN(in_dim, 256, 4, 0.1, 8, top_k, num_tasks).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        best_val = 0.0 if is_classif else float("inf")
        pat, best_state = 0, copy.deepcopy(model.state_dict())

        for epoch in range(EPOCHS):
            model.train()
            for batch in DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True):
                batch = batch.to(DEVICE); opt.zero_grad()
                out, bal = model(batch)
                if is_classif:
                    y = batch.y.float()
                    if y.ndim == 1: y = y.reshape(-1, num_tasks)
                    mask = ~torch.isnan(y)
                    if mask.sum() == 0: continue
                    loss = F.binary_cross_entropy_with_logits(out[mask], y[mask]) + 0.01*bal
                else:
                    y = batch.y.float().squeeze()
                    mask = ~torch.isnan(y)
                    if mask.sum() == 0: continue
                    loss = F.mse_loss(out.squeeze()[mask], y[mask]) + 0.01*bal
                loss.backward(); opt.step()

            model.eval()
            all_out, all_y = [], []
            with torch.no_grad():
                for batch in vl:
                    batch = batch.to(DEVICE); out, _ = model(batch)
                    all_out.append(out.cpu()); all_y.append(batch.y.cpu())
            if is_classif:
                preds = torch.sigmoid(torch.cat(all_out)).numpy()
                labels = torch.cat(all_y).numpy()
                if labels.ndim == 1: labels = labels.reshape(-1, num_tasks)
                aucs = []
                for t in range(num_tasks):
                    col = labels[:,t]; mask2 = ~np.isnan(col)
                    if mask2.sum()>=2 and len(np.unique(col[mask2]))>=2:
                        try: aucs.append(roc_auc_score(col[mask2], preds[mask2,t]))
                        except: pass
                val_score = float(np.mean(aucs)) if aucs else 0.0
                improved = val_score > best_val
                sched.step(-val_score)
            else:
                preds = torch.cat(all_out).squeeze().numpy()
                labels = torch.cat(all_y).squeeze().numpy()
                mask2 = ~np.isnan(labels)
                val_score = float(np.sqrt(np.mean((preds[mask2]-labels[mask2])**2)))
                improved = val_score < best_val
                sched.step(val_score)

            if improved: best_val = val_score; best_state = copy.deepcopy(model.state_dict()); pat = 0
            else: pat += 1
            if pat >= PATIENCE: break

        model.load_state_dict(best_state)
        model.eval()
        all_out, all_y = [], []
        with torch.no_grad():
            for batch in tl_test:
                batch = batch.to(DEVICE); out, _ = model(batch)
                all_out.append(out.cpu()); all_y.append(batch.y.cpu())
        if is_classif:
            preds = torch.sigmoid(torch.cat(all_out)).numpy()
            labels = torch.cat(all_y).numpy()
            if labels.ndim == 1: labels = labels.reshape(-1, num_tasks)
            aucs = []
            for t in range(num_tasks):
                col = labels[:,t]; mask2 = ~np.isnan(col)
                if mask2.sum()>=2 and len(np.unique(col[mask2]))>=2:
                    try: aucs.append(roc_auc_score(col[mask2], preds[mask2,t]))
                    except: pass
            score = float(np.mean(aucs)) if aucs else 0.0
        else:
            preds = torch.cat(all_out).squeeze().numpy()
            labels = torch.cat(all_y).squeeze().numpy()
            mask2 = ~np.isnan(labels)
            score = float(np.sqrt(np.mean((preds[mask2]-labels[mask2])**2)))
        seed_scores.append(score)
    return float(np.mean(seed_scores)), float(np.std(seed_scores))

DATASETS = [
    {"name":"Tox21","tasks":12,"is_classif":True},
    {"name":"ESOL", "tasks":1, "is_classif":False},
]
TOP_KS = [1, 2, 4, 8]
all_results = {}

for ds in DATASETS:
    name = ds["name"]
    print(f"\n{'='*50}\n  {name} — K comparison\n{'='*50}")
    dataset = MoleculeNet(root=DATA_ROOT, name=name, transform=ToFloat)
    train_data, val_data, test_data = scaffold_split(dataset)
    ds_results = {}
    for k in TOP_KS:
        if k > 8: continue
        print(f"  K={k}...")
        mean, std = run_topk(dataset, train_data, val_data, test_data,
                             k, ds["is_classif"], ds["tasks"])
        ds_results[k] = {"mean": mean, "std": std}
        metric = "AUC" if ds["is_classif"] else "RMSE"
        print(f"    K={k}: {metric}={mean:.4f} ± {std:.4f}")
    all_results[name] = ds_results

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Effect of Top-K on MoE-GCN Performance (8 Experts)", fontsize=12, fontweight='bold')

for i, ds in enumerate(DATASETS):
    name = ds["name"]
    ax = axes[i]
    ks     = sorted(all_results[name].keys())
    means  = [all_results[name][k]["mean"] for k in ks]
    stds   = [all_results[name][k]["std"]  for k in ks]
    metric = "ROC-AUC (↑)" if ds["is_classif"] else "RMSE (↓)"
    ax.errorbar(ks, means, yerr=stds, fmt='o-', color="#E91E63",
                linewidth=2, markersize=8, capsize=5, capthick=2)
    ax.set_xlabel("Top-K Value"); ax.set_ylabel(metric)
    ax.set_title(f"{name} — {metric}")
    ax.set_xticks(ks); ax.grid(alpha=0.3)
    for k, m, s in zip(ks, means, stds):
        ax.annotate(f"{m:.4f}", (k, m), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("topk_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved → topk_comparison.png")
