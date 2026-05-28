"""
training_curves.py — Training/Validation Loss Curves
Trains MoE-GCN vs plain DMPNN on Tox21 (classif) and ESOL (regr)
Records loss per epoch and plots comparison curves.

Run: python training_curves.py
Saves plots to: training_curves/
"""

import os, json, warnings
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
import copy

warnings.filterwarnings("ignore")

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS   = 80
PATIENCE = 15
DATA_ROOT = "./data"
OUT_DIR  = "training_curves"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")

def ToFloat(data):
    data.x = data.x.float()
    return data

# ── Models ────────────────────────────────────────────────────────────────────
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
        bal  = self.num_experts * (load * load).sum()
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        return (weights.unsqueeze(-1) * expert_out).sum(dim=1), bal

class PlainGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_tasks):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i==0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.head = nn.Linear(hidden, num_tasks)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(global_mean_pool(x, batch)), None

class MoEGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_experts, top_k, num_tasks):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i==0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe  = MoELayer(hidden, hidden, num_experts, top_k)
        self.head = nn.Linear(hidden, num_tasks)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x, bal = self.moe(x)
        return self.head(x), bal

# ── Scaffold split ────────────────────────────────────────────────────────────
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

# ── Training loop with curve recording ────────────────────────────────────────
def train_with_curves(model, train_data, val_data, is_classif, num_tasks, lr, wd):
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    tl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(val_data,   batch_size=BATCH_SIZE)

    train_losses, val_scores = [], []
    best_val = 0.0 if is_classif else float("inf")
    pat = 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        epoch_loss = 0; n_batches = 0
        for batch in tl:
            batch = batch.to(DEVICE); opt.zero_grad()
            out, bal = model(batch)
            if is_classif:
                y = batch.y.float()
                if y.ndim == 1: y = y.reshape(-1, num_tasks)
                mask = ~torch.isnan(y)
                if mask.sum() == 0: continue
                loss = F.binary_cross_entropy_with_logits(out[mask], y[mask])
            else:
                y = batch.y.float().squeeze()
                mask = ~torch.isnan(y)
                if mask.sum() == 0: continue
                loss = F.mse_loss(out.squeeze()[mask], y[mask])
            if bal is not None: loss = loss + 0.01 * bal
            loss.backward(); opt.step()
            epoch_loss += loss.item(); n_batches += 1
        train_losses.append(epoch_loss / max(n_batches, 1))

        # Val
        model.eval()
        all_out, all_y = [], []
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(DEVICE)
                out, _ = model(batch)
                all_out.append(out.cpu()); all_y.append(batch.y.cpu())

        if is_classif:
            preds  = torch.sigmoid(torch.cat(all_out)).numpy()
            labels = torch.cat(all_y).numpy()
            if labels.ndim == 1: labels = labels.reshape(-1, num_tasks)
            aucs = []
            for t in range(num_tasks):
                col = labels[:,t]; mask = ~np.isnan(col)
                if mask.sum()>=2 and len(np.unique(col[mask]))>=2:
                    try: aucs.append(roc_auc_score(col[mask], preds[mask,t]))
                    except: pass
            val_score = float(np.mean(aucs)) if aucs else 0.0
            improved  = val_score > best_val
            sched.step(-val_score)
        else:
            preds  = torch.cat(all_out).squeeze().numpy()
            labels = torch.cat(all_y).squeeze().numpy()
            mask   = ~np.isnan(labels)
            val_score = float(np.sqrt(np.mean((preds[mask]-labels[mask])**2)))
            improved  = val_score < best_val
            sched.step(val_score)

        val_scores.append(val_score)

        if improved:
            best_val = val_score; pat = 0
        else:
            pat += 1
        if pat >= PATIENCE:
            print(f"    Early stop at epoch {epoch+1}")
            break

    return train_losses, val_scores

# ── Plot curves ───────────────────────────────────────────────────────────────
def plot_curves(ds_name, is_classif,
                plain_train, plain_val,
                moe_train,   moe_val):

    val_label = "Val ROC-AUC" if is_classif else "Val RMSE"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Training Curves — {ds_name}", fontsize=13, fontweight='bold')

    # Training loss
    ax = axes[0]
    ax.plot(plain_train, color="#2196F3", linewidth=1.8, label="DMPNN (plain)")
    ax.plot(moe_train,   color="#E91E63", linewidth=1.8, label="MoE-GCN")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss"); ax.legend(); ax.grid(alpha=0.3)

    # Val metric
    ax2 = axes[1]
    ax2.plot(plain_val, color="#2196F3", linewidth=1.8, label="DMPNN (plain)")
    ax2.plot(moe_val,   color="#E91E63", linewidth=1.8, label="MoE-GCN")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel(val_label)
    ax2.set_title(val_label)
    ax2.legend(); ax2.grid(alpha=0.3)
    if not is_classif:
        ax2.invert_yaxis()  # lower RMSE is better

    plt.tight_layout()
    out_path = f"{OUT_DIR}/curves_{ds_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")

# ── Config ────────────────────────────────────────────────────────────────────
RUNS = [
    {"name": "Tox21", "tasks": 12, "is_classif": True,
     "plain_params": {"hidden":128,"num_layers":4,"dropout":0.0047,"lr":7.64e-4,"weight_decay":5.21e-5},
     "moe_params":   {"hidden":256,"num_layers":4,"dropout":0.242, "lr":5.07e-4,"weight_decay":1.59e-5,
                      "num_experts":4,"top_k":1}},
    {"name": "ESOL",  "tasks": 1,  "is_classif": False,
     "plain_params": {"hidden":128,"num_layers":2,"dropout":0.285,"lr":9.24e-4,"weight_decay":4.14e-5},
     "moe_params":   {"hidden":256,"num_layers":3,"dropout":0.002,"lr":6.71e-4,"weight_decay":1.99e-5,
                      "num_experts":16,"top_k":2}},
]

torch.manual_seed(42); np.random.seed(42)

for cfg in RUNS:
    name       = cfg["name"]
    num_tasks  = cfg["tasks"]
    is_classif = cfg["is_classif"]
    pp         = cfg["plain_params"]
    mp         = cfg["moe_params"]

    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    dataset = MoleculeNet(root=DATA_ROOT, name=name, transform=ToFloat)
    in_dim  = dataset.num_node_features
    train_data, val_data, _ = scaffold_split(dataset)

    # Plain GCN
    print("  Training plain DMPNN...")
    plain_model = PlainGCN(in_dim, pp["hidden"], pp["num_layers"],
                           pp["dropout"], num_tasks).to(DEVICE)
    plain_train, plain_val = train_with_curves(
        plain_model, train_data, val_data, is_classif, num_tasks,
        pp["lr"], pp["weight_decay"])

    # MoE-GCN
    print("  Training MoE-GCN...")
    moe_model = MoEGCN(in_dim, mp["hidden"], mp["num_layers"],
                       mp["dropout"], mp["num_experts"],
                       mp["top_k"], num_tasks).to(DEVICE)
    moe_train, moe_val = train_with_curves(
        moe_model, train_data, val_data, is_classif, num_tasks,
        mp["lr"], mp["weight_decay"])

    plot_curves(name, is_classif, plain_train, plain_val, moe_train, moe_val)

    best_plain = max(plain_val) if is_classif else min(plain_val)
    best_moe   = max(moe_val)   if is_classif else min(moe_val)
    metric = "AUC" if is_classif else "RMSE"
    print(f"  Best plain {metric}: {best_plain:.4f}")
    print(f"  Best MoE   {metric}: {best_moe:.4f}")

print(f"\nAll curves saved to ./{OUT_DIR}/")
