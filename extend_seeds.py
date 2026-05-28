"""
extend_seeds.py — Add seeds 3 & 4 to all existing results
Loads each result JSON, reruns final eval with 2 new seeds,
merges with existing 3 seeds → 5 seeds total.

Run AFTER all main scripts finish:
    python extend_seeds.py

Saves updated results back to the same JSON files.
"""

import os, json, copy, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS     = 100
PATIENCE   = 15
DATA_ROOT  = "./data"
NEW_SEEDS  = [3, 4]

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Transforms ───────────────────────────────────────────────────────────────
def ToFloat(data):
    data.x = data.x.float()
    return data

# ── MoE Layer ────────────────────────────────────────────────────────────────
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
        weights = torch.zeros_like(gate_logits).scatter_(1, topk_idx, F.softmax(topk_vals, dim=-1))
        load = weights.mean(0)
        bal  = self.num_experts * (load * load).sum()
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        return (weights.unsqueeze(-1) * expert_out).sum(dim=1), bal

# ── Models ────────────────────────────────────────────────────────────────────
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
    from collections import defaultdict
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
        if len(train_idx) < train_cutoff:
            train_idx.extend(s)
        elif len(val_idx) < (val_cutoff - train_cutoff):
            val_idx.extend(s)
        else:
            test_idx.extend(s)
    return (torch.utils.data.Subset(dataset, train_idx),
            torch.utils.data.Subset(dataset, val_idx),
            torch.utils.data.Subset(dataset, test_idx))

# ── Eval ──────────────────────────────────────────────────────────────────────
def evaluate_classif(model, loader, num_tasks):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out, _ = model(batch)
            probs  = torch.sigmoid(out).cpu().numpy()
            y      = batch.y.cpu().numpy()
            if y.ndim == 1: y = y.reshape(-1, num_tasks)
            all_preds.append(probs); all_labels.append(y)
    preds  = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    aucs = []
    for t in range(num_tasks):
        col  = labels[:, t]
        mask = ~np.isnan(col)
        if mask.sum() < 2 or len(np.unique(col[mask])) < 2: continue
        try: aucs.append(roc_auc_score(col[mask], preds[mask, t]))
        except: pass
    return float(np.mean(aucs)) if aucs else 0.0

def evaluate_regr(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out, _ = model(batch)
            preds.extend(out.squeeze().cpu().numpy().flatten())
            labels.extend(batch.y.squeeze().cpu().numpy().flatten())
    preds, labels = np.array(preds), np.array(labels)
    mask = ~np.isnan(labels)
    return float(np.sqrt(np.mean((preds[mask]-labels[mask])**2)))

# ── Train epoch ───────────────────────────────────────────────────────────────
def train_classif(model, loader, opt, num_tasks):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE); opt.zero_grad()
        out, bal = model(batch)
        y = batch.y.float()
        if y.ndim == 1: y = y.reshape(-1, num_tasks)
        mask = ~torch.isnan(y)
        if mask.sum() == 0: continue
        loss = F.binary_cross_entropy_with_logits(out[mask], y[mask])
        if bal is not None: loss = loss + 0.01 * bal
        loss.backward(); opt.step()

def train_regr(model, loader, opt):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE); opt.zero_grad()
        out, bal = model(batch)
        y = batch.y.float().squeeze()
        mask = ~torch.isnan(y)
        if mask.sum() == 0: continue
        loss = F.mse_loss(out.squeeze()[mask], y[mask])
        if bal is not None: loss = loss + 0.01 * bal
        loss.backward(); opt.step()

# ── Build model from saved best_params ───────────────────────────────────────
def build_model(params, in_dim, num_tasks, is_moe):
    hidden     = params["hidden"]
    num_layers = params["num_layers"]
    dropout    = params["dropout"]
    if is_moe:
        return MoEGCN(in_dim, hidden, num_layers, dropout,
                      params["num_experts"], params["top_k"], num_tasks)
    else:
        return PlainGCN(in_dim, hidden, num_layers, dropout, num_tasks)

# ── Run 2 extra seeds for one dataset entry ───────────────────────────────────
def extend_one(entry, dataset_name, is_classif, is_moe, num_tasks):
    existing_seeds = entry["seeds"]
    if len(existing_seeds) >= 5:
        print(f"    Already has {len(existing_seeds)} seeds, skipping.")
        return entry

    params = entry["best_params"]
    dataset = MoleculeNet(root=DATA_ROOT, name=dataset_name, transform=ToFloat)
    in_dim  = dataset.num_node_features
    train_data, val_data, test_data = scaffold_split(dataset)

    vl = DataLoader(val_data,  batch_size=BATCH_SIZE)
    tl = DataLoader(test_data, batch_size=BATCH_SIZE)

    new_scores = []
    for seed in NEW_SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = build_model(params, in_dim, num_tasks, is_moe).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

        best_val   = 0.0 if is_classif else float("inf")
        pat        = 0
        best_state = copy.deepcopy(model.state_dict())

        for _ in range(EPOCHS):
            if is_classif:
                train_classif(model, DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True), opt, num_tasks)
                val_score = evaluate_classif(model, vl, num_tasks)
                sched.step(-val_score)
                improved = val_score > best_val
            else:
                train_regr(model, DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True), opt)
                val_score = evaluate_regr(model, vl)
                sched.step(val_score)
                improved = val_score < best_val

            if improved:
                best_val   = val_score
                best_state = copy.deepcopy(model.state_dict())
                pat = 0
            else:
                pat += 1
            if pat >= PATIENCE: break

        model.load_state_dict(best_state)
        score = evaluate_classif(model, tl, num_tasks) if is_classif else evaluate_regr(model, tl)
        new_scores.append(score)
        metric = "AUC" if is_classif else "RMSE"
        print(f"      Seed {seed} → {metric}: {score:.4f}")

    all_seeds = existing_seeds + new_scores
    entry["seeds"] = all_seeds
    entry["mean"]  = float(np.mean(all_seeds))
    entry["std"]   = float(np.std(all_seeds))
    return entry

# ── Config: which files, which datasets, is_moe, num_tasks ───────────────────
CLASSIF_TASKS = {
    "Tox21": 12, "ToxCast": 617, "SIDER": 27, "ClinTox": 2, "HIV": 1
}
REGR_TASKS = {"ESOL": 1, "FreeSolv": 1, "Lipo": 1}

FILES = [
    ("results_dmpnn_classif.json",    False, True),
    ("results_dmpnn_regr.json",       False, False),
    ("results_moegcn_classif.json",   True,  True),
    ("results_moegcn_regr.json",      True,  False),
    ("results_moedmpnn_classif.json", True,  True),
    ("results_moedmpnn_regr.json",    True,  False),
]

# ── Main ──────────────────────────────────────────────────────────────────────
for fpath, is_moe, is_classif in FILES:
    if not os.path.exists(fpath):
        print(f"\nSkipping {fpath} (not found)")
        continue

    with open(fpath) as f:
        results = json.load(f)

    task_map = CLASSIF_TASKS if is_classif else REGR_TASKS
    model_label = ("MoE-GCN" if is_moe else "DMPNN")
    print(f"\n{'='*55}")
    print(f"  Extending seeds: {fpath}")
    print(f"{'='*55}")

    changed = False
    for ds_name, num_tasks in task_map.items():
        if ds_name not in results:
            print(f"  {ds_name}: not in file, skipping")
            continue
        entry = results[ds_name]
        if len(entry["seeds"]) >= 5:
            print(f"  {ds_name}: already 5 seeds ({entry['mean']:.4f} ± {entry['std']:.4f})")
            continue

        print(f"  {ds_name}: extending {len(entry['seeds'])} → 5 seeds")
        t0 = time.time()
        results[ds_name] = extend_one(entry, ds_name, is_classif, is_moe, num_tasks)
        elapsed = time.time() - t0
        r = results[ds_name]
        metric = "AUC" if is_classif else "RMSE"
        print(f"    {metric}: {r['mean']:.4f} ± {r['std']:.4f}  ({elapsed/60:.1f} min)")
        changed = True

        # Save after each dataset
        with open(fpath, "w") as f:
            json.dump(results, f, indent=2)

    if not changed:
        print(f"  All datasets already at 5 seeds.")

print("\n\nAll done! All results updated to 5 seeds.")
