"""
expert_load_balance.py — Expert Load Balance Over Training
Tracks how expert utilization evolves across epochs.
Run: python expert_load_balance.py
"""

import warnings, os
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
EPOCHS    = 60
DATA_ROOT = "./data"
print(f"Device: {DEVICE}")

def ToFloat(data):
    data.x = data.x.float()
    return data

class MoELayerTracked(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()) for _ in range(num_experts)])
        self.gate = nn.Linear(in_dim, num_experts)
        self.epoch_load = []  # track per epoch

    def forward(self, x):
        gl = self.gate(x)
        tv, ti = torch.topk(gl, self.top_k, dim=-1)
        w = torch.zeros_like(gl).scatter_(1, ti, F.softmax(tv, dim=-1))
        load = w.mean(0).detach().cpu().numpy()
        self.epoch_load.append(load)
        bal = self.num_experts * (w.mean(0)**2).sum()
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        return (w.unsqueeze(-1) * eo).sum(dim=1), bal

class MoEGCNTracked(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_experts, top_k, num_tasks):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i==0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe = MoELayerTracked(hidden, hidden, num_experts, top_k)
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

print("Loading Tox21...")
dataset = MoleculeNet(root=DATA_ROOT, name="Tox21", transform=ToFloat)
in_dim  = dataset.num_node_features
train_data, val_data, _ = scaffold_split(dataset)
tl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Best params from saved results
model = MoEGCNTracked(in_dim, 256, 4, 0.242, 4, 1, 12).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=5.07e-4, weight_decay=1.59e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
vl    = DataLoader(val_data, batch_size=BATCH_SIZE)

# Per-epoch average load
epoch_loads = []  # shape: [epochs, num_experts]

print("Training and tracking expert load...")
for epoch in range(EPOCHS):
    model.train()
    model.moe.epoch_load = []  # reset
    for batch in tl:
        batch = batch.to(DEVICE); opt.zero_grad()
        out, bal = model(batch)
        y = batch.y.float()
        if y.ndim == 1: y = y.reshape(-1, 12)
        mask = ~torch.isnan(y)
        if mask.sum() == 0: continue
        loss = F.binary_cross_entropy_with_logits(out[mask], y[mask]) + 0.01*bal
        loss.backward(); opt.step()

    # Average load this epoch
    if model.moe.epoch_load:
        avg_load = np.mean(model.moe.epoch_load, axis=0)
        epoch_loads.append(avg_load)

    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS} | Expert loads: {[f'{l:.3f}' for l in epoch_loads[-1]]}")

epoch_loads = np.array(epoch_loads)  # [epochs, 4]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Expert Load Balance Over Training — Tox21 (MoE-GCN, 4 Experts)", fontsize=12, fontweight='bold')

colors = ["#2196F3","#E91E63","#4CAF50","#FF9800"]
epochs_range = range(1, len(epoch_loads)+1)

ax = axes[0]
for e in range(4):
    ax.plot(epochs_range, epoch_loads[:, e], color=colors[e],
            linewidth=1.8, label=f"Expert {e}")
ax.set_xlabel("Epoch"); ax.set_ylabel("Average Routing Weight")
ax.set_title("Expert Load per Epoch")
ax.legend(); ax.grid(alpha=0.3)
ax.set_ylim(0, 1)

# Stacked area chart
ax2 = axes[1]
ax2.stackplot(epochs_range,
              [epoch_loads[:,e] for e in range(4)],
              labels=[f"Expert {e}" for e in range(4)],
              colors=colors, alpha=0.75)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Routing Weight (stacked)")
ax2.set_title("Expert Utilization Distribution Over Time")
ax2.legend(loc='upper right'); ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("expert_load_balance.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved → expert_load_balance.png")
