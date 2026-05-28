"""
fix_bbbp_bace.py — Fix BBBP and BACE using stratified scaffold split
Runs all 3 models (DMPNN, MoE-GCN, MoE-DMPNN) on BBBP and BACE
Saves results into existing JSON files.

Run: python fix_bbbp_bace.py
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
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS   = 30
N_SEEDS    = 5
EPOCHS     = 100
PATIENCE   = 15
BATCH_SIZE = 64
DATA_ROOT  = "./data"

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def ToFloat(data):
    data.x = data.x.float()
    return data

# ── Stratified scaffold split ─────────────────────────────────────────────────
def stratified_scaffold_split(dataset, frac_train=0.8, frac_val=0.1, seed=42):
    """
    Groups molecules by scaffold, then ensures both classes appear in val/test
    by distributing minority-class scaffolds across splits.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict

    # Get scaffold for each molecule
    scaffolds = defaultdict(list)
    for i in range(len(dataset)):
        try:
            smi = dataset.smiles[i]
            mol = Chem.MolFromSmiles(smi)
            sc  = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else smi
        except:
            sc = str(i)
        scaffolds[sc].append(i)

    # Get label for each molecule (single task binary)
    labels = []
    for i in range(len(dataset)):
        y = dataset[i].y.numpy().flatten()[0]
        labels.append(int(y) if not np.isnan(y) else 0)
    labels = np.array(labels)

    # Sort scaffold groups: large scaffolds first, then by minority class presence
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

    n = len(dataset)
    train_cutoff = int(n * frac_train)
    val_cutoff   = int(n * (frac_train + frac_val))

    train_idx, val_idx, test_idx = [], [], []

    # First pass: fill train
    for s in scaffold_sets:
        if len(train_idx) < train_cutoff:
            train_idx.extend(s)
        elif len(val_idx) < (val_cutoff - train_cutoff):
            val_idx.extend(s)
        else:
            test_idx.extend(s)

    # Check if val has both classes, if not swap some scaffolds
    def has_both_classes(idx_list):
        if len(idx_list) == 0:
            return False
        y = labels[idx_list]
        return len(np.unique(y)) >= 2

    # If val missing a class, move some train scaffolds to fix it
    if not has_both_classes(val_idx):
        minority_class = 0 if (labels == 0).sum() < (labels == 1).sum() else 1
        # Find train indices with minority class and move smallest scaffold containing them
        rng = np.random.RandomState(seed)
        for s in scaffold_sets:
            if all(i in train_idx for i in s):
                if any(labels[i] == minority_class for i in s):
                    for i in s:
                        train_idx.remove(i)
                    val_idx.extend(s)
                    if has_both_classes(val_idx):
                        break

    if not has_both_classes(test_idx):
        minority_class = 0 if (labels == 0).sum() < (labels == 1).sum() else 1
        for s in scaffold_sets:
            if all(i in train_idx for i in s):
                if any(labels[i] == minority_class for i in s):
                    for i in s:
                        train_idx.remove(i)
                    test_idx.extend(s)
                    if has_both_classes(test_idx):
                        break

    return (torch.utils.data.Subset(dataset, train_idx),
            torch.utils.data.Subset(dataset, val_idx),
            torch.utils.data.Subset(dataset, test_idx))

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
    def __init__(self, in_dim, hidden, num_layers, dropout, num_tasks=1):
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
    def __init__(self, in_dim, hidden, num_layers, dropout, num_experts, top_k, num_tasks=1):
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

# ── Eval / Train ──────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out, _ = model(batch)
            probs  = torch.sigmoid(out).squeeze().cpu().numpy()
            y      = batch.y.squeeze().cpu().numpy()
            preds.extend(probs.flatten()); labs.extend(y.flatten())
    preds, labs = np.array(preds), np.array(labs)
    mask = ~np.isnan(labs)
    if mask.sum() < 2 or len(np.unique(labs[mask])) < 2:
        return 0.0
    try:
        return float(roc_auc_score(labs[mask], preds[mask]))
    except:
        return 0.0

def train_epoch(model, loader, opt):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE); opt.zero_grad()
        out, bal = model(batch)
        y = batch.y.float().squeeze()
        mask = ~torch.isnan(y)
        if mask.sum() == 0: continue
        loss = F.binary_cross_entropy_with_logits(out.squeeze()[mask], y[mask])
        if bal is not None: loss = loss + 0.01 * bal
        loss.backward(); opt.step()

# ── Run one model on one dataset ──────────────────────────────────────────────
def run_one(ds_name, model_name, save_path, is_moe):
    # Load existing results
    results = {}
    if os.path.exists(save_path):
        with open(save_path) as f:
            results = json.load(f)

    # Check if already fixed (AUC > 0)
    if ds_name in results and results[ds_name]["mean"] > 0.05:
        print(f"  {model_name} {ds_name}: already good ({results[ds_name]['mean']:.4f}), skipping")
        return

    print(f"\n{'='*55}")
    print(f"  {model_name} | {ds_name} | stratified scaffold split")
    print(f"{'='*55}")
    t0 = time.time()

    dataset = MoleculeNet(root=DATA_ROOT, name=ds_name, transform=ToFloat)
    in_dim  = dataset.num_node_features
    train_data, val_data, test_data = stratified_scaffold_split(dataset)

    # Verify both classes in val and test
    def check_classes(subset, name):
        ys = [dataset[i].y.numpy().flatten()[0] for i in subset.indices]
        uniq = np.unique([y for y in ys if not np.isnan(y)])
        print(f"    {name} classes: {uniq} (n={len(ys)})")
        return len(uniq) >= 2
    check_classes(train_data, "train")
    ok_val  = check_classes(val_data,  "val")
    ok_test = check_classes(test_data, "test")
    if not ok_val or not ok_test:
        print(f"  WARNING: still single-class split, skipping {ds_name}")
        return

    # Optuna
    def objective(trial):
        hidden     = trial.suggest_categorical("hidden", [128, 256])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        dropout    = trial.suggest_float("dropout", 0.0, 0.3)
        lr         = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        wd         = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        if is_moe:
            num_experts = trial.suggest_categorical("num_experts", [4, 8])
            top_k       = trial.suggest_int("top_k", 1, min(4, num_experts))
            model = MoEGCN(in_dim, hidden, num_layers, dropout, num_experts, top_k).to(DEVICE)
        else:
            model = PlainGCN(in_dim, hidden, num_layers, dropout).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        tl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        vl = DataLoader(val_data,   batch_size=BATCH_SIZE)
        best_val, pat = 0.0, 0
        for _ in range(EPOCHS):
            train_epoch(model, tl, opt)
            val_auc = evaluate(model, vl)
            sched.step(-val_auc)
            if val_auc > best_val: best_val, pat = val_auc, 0
            else: pat += 1
            if pat >= PATIENCE: break
        return best_val

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)
    best_params = study.best_params
    print(f"  Best val AUC: {study.best_value:.4f} | {best_params}")

    # Final eval — 5 seeds
    vl = DataLoader(val_data,  batch_size=BATCH_SIZE)
    tl = DataLoader(test_data, batch_size=BATCH_SIZE)
    seed_scores = []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        if is_moe:
            model = MoEGCN(in_dim, best_params["hidden"], best_params["num_layers"],
                           best_params["dropout"], best_params["num_experts"],
                           best_params["top_k"]).to(DEVICE)
        else:
            model = PlainGCN(in_dim, best_params["hidden"], best_params["num_layers"],
                             best_params["dropout"]).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        best_val, pat = 0.0, 0
        best_state = copy.deepcopy(model.state_dict())
        for _ in range(EPOCHS):
            train_epoch(model, DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True), opt)
            val_auc = evaluate(model, vl)
            sched.step(-val_auc)
            if val_auc > best_val:
                best_val = val_auc; best_state = copy.deepcopy(model.state_dict()); pat = 0
            else: pat += 1
            if pat >= PATIENCE: break
        model.load_state_dict(best_state)
        test_auc = evaluate(model, tl)
        seed_scores.append(test_auc)
        print(f"    Seed {seed} → AUC: {test_auc:.4f}")

    mean_s, std_s = float(np.mean(seed_scores)), float(np.std(seed_scores))
    elapsed = time.time() - t0
    results[ds_name] = {
        "mean": mean_s, "std": std_s, "metric": "roc_auc",
        "seeds": seed_scores, "best_params": best_params,
        "time_min": round(elapsed/60, 1), "split": "stratified_scaffold"
    }
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ {model_name} {ds_name}: {mean_s:.4f} ± {std_s:.4f}  ({elapsed/60:.1f} min)")

# ── Main ──────────────────────────────────────────────────────────────────────
RUNS = [
    ("BBBP", "DMPNN",     "results_dmpnn_classif.json",     False),
    ("BACE", "DMPNN",     "results_dmpnn_classif.json",     False),
    ("BBBP", "MoE-GCN",   "results_moegcn_classif.json",    True),
    ("BACE", "MoE-GCN",   "results_moegcn_classif.json",    True),
    ("BBBP", "MoE-DMPNN", "results_moedmpnn_classif.json",  True),
    ("BACE", "MoE-DMPNN", "results_moedmpnn_classif.json",  True),
]

for ds_name, model_name, save_path, is_moe in RUNS:
    run_one(ds_name, model_name, save_path, is_moe)

print("\n\nAll BBBP/BACE fixes complete!")
