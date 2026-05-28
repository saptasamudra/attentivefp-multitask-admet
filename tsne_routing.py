"""
tsne_routing.py — t-SNE Expert Routing Visualization
Trains final MoE-GCN model on each dataset, extracts:
  - Molecule hidden representations (after pooling)
  - Expert routing weights (which expert each molecule uses)
Then plots t-SNE colored by dominant expert.

Run: python tsne_routing.py
Saves plots to: tsne_plots/
"""

import os, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

warnings.filterwarnings("ignore")

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS    = 100
PATIENCE  = 15
DATA_ROOT = "./data"
OUT_DIR   = "tsne_plots"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")

# ── Datasets to visualize (pick most interesting ones) ──────────────────────
DATASETS = [
    {"name": "Tox21",   "tasks": 12,  "metric": "classif"},
    {"name": "ESOL",    "tasks": 1,   "metric": "regr"},
]

def ToFloat(data):
    data.x = data.x.float()
    return data

# ── MoE Layer with routing capture ──────────────────────────────────────────
class MoELayerViz(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(in_dim, num_experts)
        self.last_weights = None  # store routing weights

    def forward(self, x):
        gate_logits = self.gate(x)
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = torch.zeros_like(gate_logits).scatter_(
            1, topk_idx, F.softmax(topk_vals, dim=-1))
        self.last_weights = weights.detach().cpu()
        load = weights.mean(0)
        bal  = self.num_experts * (load * load).sum()
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        return (weights.unsqueeze(-1) * expert_out).sum(dim=1), bal

class MoEGCNViz(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_experts, top_k, num_tasks):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i==0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe  = MoELayerViz(hidden, hidden, num_experts, top_k)
        self.head = nn.Linear(hidden, num_tasks)
        self.last_pooled = None  # store pooled representations

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = global_mean_pool(x, batch)
        self.last_pooled = pooled.detach().cpu()
        out, bal = self.moe(pooled)
        return self.head(out), bal

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

# ── Train model ───────────────────────────────────────────────────────────────
def train_model(dataset, ds_cfg, params):
    in_dim = dataset.num_node_features
    train_data, val_data, _ = scaffold_split(dataset)
    model = MoEGCNViz(in_dim, params["hidden"], params["num_layers"],
                      params["dropout"], params["num_experts"],
                      params["top_k"], ds_cfg["tasks"]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=params["lr"],
                              weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    tl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(val_data,   batch_size=BATCH_SIZE)

    best_val = float("inf") if ds_cfg["metric"] == "regr" else 0.0
    pat, best_state = 0, None
    import copy

    for _ in range(EPOCHS):
        model.train()
        for batch in tl:
            batch = batch.to(DEVICE); opt.zero_grad()
            out, bal = model(batch)
            if ds_cfg["metric"] == "classif":
                y = batch.y.float()
                if y.ndim == 1: y = y.reshape(-1, ds_cfg["tasks"])
                mask = ~torch.isnan(y)
                if mask.sum() == 0: continue
                loss = F.binary_cross_entropy_with_logits(out[mask], y[mask]) + 0.01*bal
            else:
                y = batch.y.float().squeeze()
                mask = ~torch.isnan(y)
                if mask.sum() == 0: continue
                loss = F.mse_loss(out.squeeze()[mask], y[mask]) + 0.01*bal
            loss.backward(); opt.step()

        # Val
        model.eval()
        all_out, all_y = [], []
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(DEVICE)
                out, _ = model(batch)
                all_out.append(out.cpu()); all_y.append(batch.y.cpu())
        if ds_cfg["metric"] == "regr":
            preds  = torch.cat(all_out).squeeze().numpy()
            labels = torch.cat(all_y).squeeze().numpy()
            mask   = ~np.isnan(labels)
            val_score = float(np.sqrt(np.mean((preds[mask]-labels[mask])**2)))
            improved = val_score < best_val
            sched.step(val_score)
        else:
            from sklearn.metrics import roc_auc_score
            preds  = torch.sigmoid(torch.cat(all_out)).numpy()
            labels = torch.cat(all_y).numpy()
            if labels.ndim == 1: labels = labels.reshape(-1, ds_cfg["tasks"])
            aucs = []
            for t in range(ds_cfg["tasks"]):
                col = labels[:,t]; mask = ~np.isnan(col)
                if mask.sum()>=2 and len(np.unique(col[mask]))>=2:
                    try: aucs.append(roc_auc_score(col[mask], preds[mask,t]))
                    except: pass
            val_score = float(np.mean(aucs)) if aucs else 0.0
            improved = val_score > best_val
            sched.step(-val_score)

        if improved:
            best_val = val_score
            best_state = copy.deepcopy(model.state_dict())
            pat = 0
        else:
            pat += 1
        if pat >= PATIENCE: break

    if best_state:
        model.load_state_dict(best_state)
    return model

# ── Extract representations ───────────────────────────────────────────────────
def extract_representations(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_pooled  = []
    all_weights = []
    all_labels  = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            model(batch)
            all_pooled.append(model.last_pooled)
            all_weights.append(model.moe.last_weights)
            all_labels.append(batch.y.cpu())
    return (torch.cat(all_pooled).numpy(),
            torch.cat(all_weights).numpy(),
            torch.cat(all_labels).numpy())

# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_tsne(ds_name, pooled, weights, labels, num_experts):
    print(f"  Running t-SNE for {ds_name} ({len(pooled)} molecules)...")

    # Subsample if too large
    MAX_POINTS = 2000
    if len(pooled) > MAX_POINTS:
        idx = np.random.choice(len(pooled), MAX_POINTS, replace=False)
        pooled  = pooled[idx]
        weights = weights[idx]
        labels  = labels[idx]

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30,
                max_iter=1000, verbose=0)
    coords = tsne.fit_transform(pooled)

    # Dominant expert per molecule
    dominant_expert = weights.argmax(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"t-SNE Expert Routing — {ds_name}", fontsize=14, fontweight='bold')

    colors = cm.tab10(np.linspace(0, 1, num_experts))

    # Plot 1: colored by dominant expert
    ax = axes[0]
    for e in range(num_experts):
        mask = dominant_expert == e
        if mask.sum() == 0: continue
        ax.scatter(coords[mask,0], coords[mask,1],
                   c=[colors[e]], label=f"Expert {e}",
                   alpha=0.6, s=8, rasterized=True)
    ax.set_title("Colored by Dominant Expert", fontsize=11)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=2, fontsize=8, loc='best',
              ncol=2 if num_experts > 4 else 1)
    ax.set_xticks([]); ax.set_yticks([])

    # Plot 2: expert utilization bar chart
    ax2 = axes[1]
    expert_counts = [(dominant_expert == e).sum() for e in range(num_experts)]
    expert_pct    = [c / len(dominant_expert) * 100 for c in expert_counts]
    bars = ax2.bar(range(num_experts), expert_pct,
                   color=colors, alpha=0.8, edgecolor='white')
    ax2.set_xlabel("Expert Index"); ax2.set_ylabel("% of Molecules Routed")
    ax2.set_title("Expert Utilization Distribution", fontsize=11)
    ax2.set_xticks(range(num_experts))
    for bar, pct in zip(bars, expert_pct):
        if pct > 2:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{pct:.1f}%", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/tsne_{ds_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")
    return expert_pct

# ── Main ──────────────────────────────────────────────────────────────────────
# Load best params from saved results
classif_results = json.load(open("results_moegcn_classif.json")) if os.path.exists("results_moegcn_classif.json") else {}
regr_results    = json.load(open("results_moegcn_regr.json"))    if os.path.exists("results_moegcn_regr.json")    else {}

torch.manual_seed(0)
np.random.seed(0)

summary = {}
for ds_cfg in DATASETS:
    name = ds_cfg["name"]
    print(f"\n{'='*50}")
    print(f"  Processing: {name}")
    print(f"{'='*50}")

    # Get best params
    result_src = classif_results if ds_cfg["metric"] == "classif" else regr_results
    if name not in result_src:
        print(f"  No saved params for {name}, skipping")
        continue
    params = result_src[name]["best_params"]
    num_experts = params.get("num_experts", 4)

    # Load dataset
    dataset = MoleculeNet(root=DATA_ROOT, name=name, transform=ToFloat)
    print(f"  Dataset size: {len(dataset)}, experts: {num_experts}")

    # Train
    print(f"  Training MoE-GCN with best params...")
    model = train_model(dataset, ds_cfg, params)

    # Extract
    print(f"  Extracting representations...")
    pooled, weights, labels = extract_representations(model, dataset)
    print(f"  pooled shape: {pooled.shape}, weights shape: {weights.shape}")

    # Plot
    expert_pct = plot_tsne(name, pooled, weights, labels, num_experts)
    summary[name] = {
        "num_experts": num_experts,
        "expert_utilization_pct": [round(p,2) for p in expert_pct]
    }

# Summary
print(f"\n{'='*50}")
print("  EXPERT UTILIZATION SUMMARY")
print(f"{'='*50}")
for ds, info in summary.items():
    pcts = info["expert_utilization_pct"]
    entropy = -sum(p/100 * np.log(p/100+1e-9) for p in pcts)
    max_util = max(pcts)
    print(f"  {ds:10}: max={max_util:.1f}%, entropy={entropy:.2f} (higher=more balanced)")

print(f"\nAll plots saved to ./{OUT_DIR}/")
# Install note added to top
