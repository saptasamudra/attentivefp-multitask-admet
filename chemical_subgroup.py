"""
chemical_subgroup.py — Chemical Subgroup Analysis of Expert Routing
Analyzes what chemical properties each expert specializes in.
Run: python chemical_subgroup.py
"""

import warnings, os
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
import matplotlib.gridspec as gridspec
from collections import defaultdict

warnings.filterwarnings("ignore")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS    = 60
PATIENCE  = 15
DATA_ROOT = "./data"
print(f"Device: {DEVICE}")

def ToFloat(data):
    data.x = data.x.float()
    return data

class MoELayerViz(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()) for _ in range(num_experts)])
        self.gate = nn.Linear(in_dim, num_experts)
        self.last_weights = None
    def forward(self, x):
        gl = self.gate(x)
        tv, ti = torch.topk(gl, self.top_k, dim=-1)
        w = torch.zeros_like(gl).scatter_(1, ti, F.softmax(tv, dim=-1))
        self.last_weights = w.detach().cpu()
        bal = self.num_experts * (w.mean(0)**2).sum()
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        return (w.unsqueeze(-1) * eo).sum(dim=1), bal

class MoEGCNViz(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_experts, top_k, num_tasks):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i==0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe = MoELayerViz(hidden, hidden, num_experts, top_k)
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

def get_mol_properties(smiles_list):
    """Compute MW, LogP, HBA, HBD, TPSA, ring count for each molecule."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    props = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None: raise ValueError
            props.append({
                "MW":    Descriptors.MolWt(mol),
                "LogP":  Descriptors.MolLogP(mol),
                "HBA":   rdMolDescriptors.CalcNumHBA(mol),
                "HBD":   rdMolDescriptors.CalcNumHBD(mol),
                "TPSA":  Descriptors.TPSA(mol),
                "Rings": rdMolDescriptors.CalcNumRings(mol),
            })
        except:
            props.append({"MW":0,"LogP":0,"HBA":0,"HBD":0,"TPSA":0,"Rings":0})
    return props

print("Loading ESOL (best t-SNE clustering)...")
dataset = MoleculeNet(root=DATA_ROOT, name="ESOL", transform=ToFloat)
in_dim  = dataset.num_node_features
train_data, val_data, _ = scaffold_split(dataset)

# Train model
print("Training MoE-GCN...")
import copy
model = MoEGCNViz(in_dim, 256, 3, 0.002, 16, 2, 1).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=6.71e-4, weight_decay=1.99e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
tl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
vl = DataLoader(val_data,   batch_size=BATCH_SIZE)
best_val, pat, best_state = float("inf"), 0, copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    model.train()
    for batch in tl:
        batch = batch.to(DEVICE); opt.zero_grad()
        out, bal = model(batch)
        y = batch.y.float().squeeze()
        mask = ~torch.isnan(y)
        if mask.sum() == 0: continue
        loss = F.mse_loss(out.squeeze()[mask], y[mask]) + 0.01*bal
        loss.backward(); opt.step()
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for batch in vl:
            batch = batch.to(DEVICE); out, _ = model(batch)
            preds.extend(out.squeeze().cpu().numpy().flatten())
            labs.extend(batch.y.squeeze().cpu().numpy().flatten())
    preds, labs = np.array(preds), np.array(labs)
    mask = ~np.isnan(labs)
    val_rmse = float(np.sqrt(np.mean((preds[mask]-labs[mask])**2)))
    sched.step(val_rmse)
    if val_rmse < best_val: best_val = val_rmse; best_state = copy.deepcopy(model.state_dict()); pat = 0
    else: pat += 1
    if pat >= PATIENCE: break
    if (epoch+1) % 10 == 0: print(f"  Epoch {epoch+1}: val RMSE={val_rmse:.4f}")

model.load_state_dict(best_state)

# Extract routing weights for full dataset
print("Extracting routing weights...")
full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
all_weights = []
model.eval()
with torch.no_grad():
    for batch in full_loader:
        batch = batch.to(DEVICE); model(batch)
        all_weights.append(model.moe.last_weights)
all_weights = torch.cat(all_weights).numpy()  # [N, 16]
dominant    = all_weights.argmax(axis=1)       # [N]

# Get molecular properties
print("Computing molecular properties...")
smiles_list = [dataset.smiles[i] for i in range(len(dataset))]
props = get_mol_properties(smiles_list)
prop_names = ["MW", "LogP", "HBA", "HBD", "TPSA", "Rings"]

# Focus on top 4 most-used experts
expert_counts = [(dominant == e).sum() for e in range(16)]
top4_experts  = sorted(range(16), key=lambda e: expert_counts[e], reverse=True)[:4]
print(f"Top 4 experts by usage: {top4_experts}")

# Plot: for each property, show distribution per expert
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Chemical Property Distribution by Expert — ESOL (MoE-GCN, Top-4 Experts)", fontsize=12, fontweight='bold')
colors = ["#2196F3","#E91E63","#4CAF50","#FF9800"]

for idx, prop in enumerate(prop_names):
    ax = axes[idx//3][idx%3]
    vals = np.array([p[prop] for p in props])
    data_by_expert = [vals[dominant == e] for e in top4_experts]
    bp = ax.boxplot(data_by_expert, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticklabels([f"Expert {e}\n(n={expert_counts[e]})" for e in top4_experts], fontsize=8)
    ax.set_ylabel(prop); ax.set_title(f"{prop} by Expert")
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("chemical_subgroup.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved → chemical_subgroup.png")

# Print summary stats
print("\n=== Expert Chemical Property Summary (ESOL) ===")
print(f"{'Expert':8}", end="")
for p in prop_names:
    print(f" {p:>10}", end="")
print()
print("-" * (8 + 10*len(prop_names)))
for e in top4_experts:
    mask = dominant == e
    vals = [np.array([props[i][p] for i in range(len(props)) if mask[i]]) for p in prop_names]
    print(f"  E{e:2d}   ", end="")
    for v in vals:
        print(f" {np.mean(v):>10.2f}", end="")
    print(f"  (n={mask.sum()})")
