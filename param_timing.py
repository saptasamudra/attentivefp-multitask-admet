"""
param_timing.py — Parameter Count + Inference Time Comparison
Compares PlainGCN vs MoE-GCN across different configs.
Run: python param_timing.py
"""

import time, json, warnings
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

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        return (w.unsqueeze(-1) * eo).sum(dim=1), (self.num_experts * (w.mean(0)**2).sum())

class PlainGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_tasks):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i==0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.head = nn.Linear(hidden, num_tasks)
    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(global_mean_pool(x, b)), None

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

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, loader, n_runs=3):
    model.eval()
    # Warmup
    for batch in loader:
        batch = batch.to(DEVICE)
        with torch.no_grad(): model(batch)
        break
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                model(batch)
        times.append(time.time() - t0)
    return float(np.mean(times)) * 1000  # ms

# Load Tox21 for benchmarking
print("Loading Tox21...")
dataset = MoleculeNet(root=DATA_ROOT, name="Tox21", transform=ToFloat)
loader  = DataLoader(dataset, batch_size=64, shuffle=False)
in_dim  = dataset.num_node_features
NUM_TASKS = 12

configs = [
    {"label": "Plain GCN\n(hidden=128, L=2)", "type": "plain", "hidden": 128, "num_layers": 2, "dropout": 0.0, "num_tasks": NUM_TASKS},
    {"label": "Plain GCN\n(hidden=256, L=4)", "type": "plain", "hidden": 256, "num_layers": 4, "dropout": 0.0, "num_tasks": NUM_TASKS},
    {"label": "MoE-GCN\n(E=4, K=1)",          "type": "moe",   "hidden": 256, "num_layers": 4, "dropout": 0.0, "num_experts": 4,  "top_k": 1, "num_tasks": NUM_TASKS},
    {"label": "MoE-GCN\n(E=8, K=2)",          "type": "moe",   "hidden": 256, "num_layers": 4, "dropout": 0.0, "num_experts": 8,  "top_k": 2, "num_tasks": NUM_TASKS},
    {"label": "MoE-GCN\n(E=16, K=4)",         "type": "moe",   "hidden": 256, "num_layers": 4, "dropout": 0.0, "num_experts": 16, "top_k": 4, "num_tasks": NUM_TASKS},
]

results = []
print(f"\n{'Config':35} {'Params':>12} {'Inference(ms)':>15}")
print("-" * 65)
for cfg in configs:
    if cfg["type"] == "plain":
        model = PlainGCN(in_dim, cfg["hidden"], cfg["num_layers"], cfg["dropout"], cfg["num_tasks"]).to(DEVICE)
    else:
        model = MoEGCN(in_dim, cfg["hidden"], cfg["num_layers"], cfg["dropout"],
                       cfg["num_experts"], cfg["top_k"], cfg["num_tasks"]).to(DEVICE)
    params = count_params(model)
    inf_ms = measure_inference_time(model, loader)
    results.append({"label": cfg["label"].replace("\n"," "), "params": params, "inference_ms": inf_ms})
    print(f"  {cfg['label'].replace(chr(10),' '):33} {params:>12,} {inf_ms:>13.1f} ms")

# ── Plot ──────────────────────────────────────────────────────────────────────
labels     = [r["label"] for r in results]
params     = [r["params"]/1e6 for r in results]
inf_times  = [r["inference_ms"] for r in results]
colors     = ["#2196F3","#2196F3","#E91E63","#E91E63","#E91E63"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Computational Efficiency: Plain GCN vs MoE-GCN", fontsize=13, fontweight='bold')

ax = axes[0]
bars = ax.bar(range(len(labels)), params, color=colors, alpha=0.85, edgecolor='white')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Parameters (millions)")
ax.set_title("Model Parameter Count")
for bar, p in zip(bars, params):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{p:.2f}M", ha='center', va='bottom', fontsize=8)
ax.set_ylim(0, max(params)*1.2)

ax2 = axes[1]
bars2 = ax2.bar(range(len(labels)), inf_times, color=colors, alpha=0.85, edgecolor='white')
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("Inference Time (ms)")
ax2.set_title("Full Dataset Inference Time")
for bar, t in zip(bars2, inf_times):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{t:.0f}ms", ha='center', va='bottom', fontsize=8)
ax2.set_ylim(0, max(inf_times)*1.2)

from matplotlib.patches import Patch
legend = [Patch(color="#2196F3", label="Plain GCN"), Patch(color="#E91E63", label="MoE-GCN")]
axes[0].legend(handles=legend, loc='upper left')
axes[1].legend(handles=legend, loc='upper left')

plt.tight_layout()
plt.savefig("param_timing.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved → param_timing.png")

# Save JSON
with open("param_timing.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved → param_timing.json")
