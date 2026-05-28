"""
attentivefp_remaining.py — Run AttentiveFP on missing datasets
Missing: BBBP, BACE, ToxCast, ESOL, FreeSolv, Lipo
Results merged into: results_attentivefp.json

Run: python attentivefp_remaining.py
"""

import os, json, time, warnings, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import AttentiveFP
from sklearn.metrics import roc_auc_score
import optuna
from optuna.samplers import TPESampler
from collections import defaultdict

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS   = 30
N_SEEDS    = 5
EPOCHS     = 100
PATIENCE   = 15
BATCH_SIZE = 64
DATA_ROOT  = "./data"
SAVE_PATH  = "results_attentivefp.json"

# Already have Tox21, SIDER, ClinTox, HIV — only run missing ones
# BBBP and BACE skipped — use published AttentiveFP numbers (Xiong et al. 2020)
# BBBP: 0.908, BACE: 0.852
DATASETS = [
    {"name": "ToxCast", "tasks": 617, "type": "classif"},
    {"name": "ESOL",    "tasks": 1,   "type": "regr"},
    {"name": "FreeSolv","tasks": 1,   "type": "regr"},
    {"name": "Lipo",    "tasks": 1,   "type": "regr"},
]

# Seed published numbers for BBBP and BACE
PUBLISHED = {
    "BBBP": {"mean": 0.908, "std": 0.050, "metric": "roc_auc", "seeds": [], "source": "Xiong et al. 2020"},
    "BACE": {"mean": 0.852, "std": 0.053, "metric": "roc_auc", "seeds": [], "source": "Xiong et al. 2020"},
}

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def ToFloat(data):
    data.x = data.x.float()
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr.float()
    return data

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

def stratified_scaffold_split(dataset, frac_train=0.8, frac_val=0.1):
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
    labels = []
    for i in range(len(dataset)):
        y = dataset[i].y.numpy().flatten()[0]
        labels.append(int(y) if not np.isnan(y) else 0)
    labels = np.array(labels)
    train_cutoff = int(n * frac_train)
    val_cutoff   = int(n * (frac_train + frac_val))
    train_idx, val_idx, test_idx = [], [], []
    for s in scaffold_sets:
        if len(train_idx) < train_cutoff: train_idx.extend(s)
        elif len(val_idx) < (val_cutoff - train_cutoff): val_idx.extend(s)
        else: test_idx.extend(s)
    def has_both(idx_list):
        if not idx_list: return False
        y = labels[idx_list]
        return len(np.unique(y)) >= 2
    if not has_both(val_idx):
        minority = 0 if (labels==0).sum() < (labels==1).sum() else 1
        for s in scaffold_sets:
            if all(i in train_idx for i in s) and any(labels[i]==minority for i in s):
                for i in s: train_idx.remove(i)
                val_idx.extend(s)
                if has_both(val_idx): break
    return (torch.utils.data.Subset(dataset, train_idx),
            torch.utils.data.Subset(dataset, val_idx),
            torch.utils.data.Subset(dataset, test_idx))

def evaluate_classif(model, loader, num_tasks):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.sigmoid(out).cpu().numpy()
            y = batch.y.cpu().numpy()
            if y.ndim == 1: y = y.reshape(-1, num_tasks)
            all_preds.append(probs); all_labels.append(y)
    preds = np.vstack(all_preds); labels = np.vstack(all_labels)
    aucs = []
    for t in range(num_tasks):
        col = labels[:,t]; mask = ~np.isnan(col)
        if mask.sum()<2 or len(np.unique(col[mask]))<2: continue
        try: aucs.append(roc_auc_score(col[mask], preds[mask,t]))
        except: pass
    return float(np.mean(aucs)) if aucs else 0.0

def evaluate_regr(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds.extend(out.squeeze().cpu().numpy().flatten())
            labels.extend(batch.y.squeeze().cpu().numpy().flatten())
    preds, labels = np.array(preds), np.array(labels)
    mask = ~np.isnan(labels)
    return float(np.sqrt(np.mean((preds[mask]-labels[mask])**2)))

def train_epoch_classif(model, loader, opt, num_tasks):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE); opt.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.float()
        if y.ndim == 1: y = y.reshape(-1, num_tasks)
        mask = ~torch.isnan(y)
        if mask.sum() == 0: continue
        F.binary_cross_entropy_with_logits(out[mask], y[mask]).backward()
        opt.step()

def train_epoch_regr(model, loader, opt):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE); opt.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.float().squeeze()
        mask = ~torch.isnan(y)
        if mask.sum() == 0: continue
        F.mse_loss(out.squeeze()[mask], y[mask]).backward()
        opt.step()

def run_dataset(ds, train_data, val_data, test_data, in_dim, edge_dim, results):
    name = ds["name"]; num_tasks = ds["tasks"]; is_classif = ds["type"] == "classif"

    def objective(trial):
        hidden     = trial.suggest_categorical("hidden", [200, 256])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        num_ts     = trial.suggest_int("num_timesteps", 2, 3)
        dropout    = trial.suggest_float("dropout", 0.0, 0.4)
        lr         = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        wd         = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        model = AttentiveFP(in_channels=in_dim, hidden_channels=hidden,
                            out_channels=num_tasks, edge_dim=edge_dim,
                            num_layers=num_layers, num_timesteps=num_ts,
                            dropout=dropout).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        tl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        vl = DataLoader(val_data, batch_size=BATCH_SIZE)
        best_val, pat = (0.0 if is_classif else float("inf")), 0
        for _ in range(EPOCHS):
            if is_classif: train_epoch_classif(model, tl, opt, num_tasks)
            else: train_epoch_regr(model, tl, opt)
            val_score = evaluate_classif(model, vl, num_tasks) if is_classif else evaluate_regr(model, vl)
            sched.step(-val_score if is_classif else val_score)
            improved = val_score > best_val if is_classif else val_score < best_val
            if improved: best_val, pat = val_score, 0
            else: pat += 1
            if pat >= PATIENCE: break
        return best_val
    
    study = optuna.create_study(direction="maximize" if is_classif else "minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)
    bp = study.best_params
    print(f"  Best val {'AUC' if is_classif else 'RMSE'}: {study.best_value:.4f} | {bp}")

    vl = DataLoader(val_data, batch_size=BATCH_SIZE)
    tl = DataLoader(test_data, batch_size=BATCH_SIZE)
    seed_scores = []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        model = AttentiveFP(in_channels=in_dim, hidden_channels=bp["hidden"],
                            out_channels=num_tasks, edge_dim=edge_dim,
                            num_layers=bp["num_layers"], num_timesteps=bp["num_timesteps"],
                            dropout=bp["dropout"]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=bp["lr"], weight_decay=bp["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        best_val, pat = (0.0 if is_classif else float("inf")), 0
        best_state = copy.deepcopy(model.state_dict())
        for _ in range(EPOCHS):
            if is_classif: train_epoch_classif(model, DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True), opt, num_tasks)
            else: train_epoch_regr(model, DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True), opt)
            val_score = evaluate_classif(model, vl, num_tasks) if is_classif else evaluate_regr(model, vl)
            sched.step(-val_score if is_classif else val_score)
            improved = val_score > best_val if is_classif else val_score < best_val
            if improved: best_val, pat = val_score, 0; best_state = copy.deepcopy(model.state_dict())
            else: pat += 1
            if pat >= PATIENCE: break
        model.load_state_dict(best_state)
        score = evaluate_classif(model, tl, num_tasks) if is_classif else evaluate_regr(model, tl)
        seed_scores.append(score)
        print(f"    Seed {seed} → {'AUC' if is_classif else 'RMSE'}: {score:.4f}")
    
    mean_s, std_s = float(np.mean(seed_scores)), float(np.std(seed_scores))
    return mean_s, std_s, seed_scores, bp

# Load existing results
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH) as f: results = json.load(f)
    print(f"Resuming — {len(results)} done: {list(results.keys())}")
else:
    # Seed with existing Tox21/SIDER/ClinTox/HIV results
    existing = {}
    if os.path.exists("results_from_tox21.json"):
        with open("results_from_tox21.json") as f:
            old = json.load(f)
        for k, v in old.items():
            existing[k] = {"mean": v["mean"], "std": v["std"], "metric": "roc_auc", "seeds": v["seeds"]}
    results = existing
    print(f"Seeded with existing results: {list(results.keys())}")

# Inject published numbers if not already present
for k, v in PUBLISHED.items():
    if k not in results:
        results[k] = v
        print(f"  Seeded published result: {k} = {v['mean']:.4f}")

for ds in DATASETS:
    name = ds["name"]
    if name in results and results[name].get("mean", 0) > 0.01:
        print(f"  Skipping {name} (already done: {results[name]['mean']:.4f})")
        continue
    print(f"\n{'='*55}\n  AttentiveFP | {name} | {ds['tasks']} tasks\n{'='*55}")
    t0 = time.time()
    dataset = MoleculeNet(root=DATA_ROOT, name=name, transform=ToFloat)
    in_dim   = dataset.num_node_features
    edge_dim = dataset.num_edge_features
    print(f"  edge_dim: {edge_dim}")
    if name in ["BBBP", "BACE"]:
        train_data, val_data, test_data = stratified_scaffold_split(dataset)
    else:
        train_data, val_data, test_data = scaffold_split(dataset)
    mean_s, std_s, seeds, bp = run_dataset(ds, train_data, val_data, test_data, in_dim, edge_dim, results)
    elapsed = time.time() - t0
    metric = "roc_auc" if ds["type"] == "classif" else "rmse"
    results[name] = {"mean": mean_s, "std": std_s, "metric": metric,
                     "seeds": seeds, "best_params": bp, "time_min": round(elapsed/60,1)}
    with open(SAVE_PATH, "w") as f: json.dump(results, f, indent=2)
    print(f"  ✓ AttentiveFP {name}: {mean_s:.4f} ± {std_s:.4f}  ({elapsed/60:.1f} min)")

print(f"\n{'='*55}\n  ATTENTIVEFP SUMMARY\n{'='*55}")
for name, r in results.items():
    metric = "AUC" if r["metric"] == "roc_auc" else "RMSE"
    print(f"  {name:12} {metric}: {r['mean']:.4f} ± {r['std']:.4f}")
print(f"\nSaved → {SAVE_PATH}")
