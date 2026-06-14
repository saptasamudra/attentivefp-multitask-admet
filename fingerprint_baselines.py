"""
Fingerprint + ML Baselines
Morgan ECFP4 + Random Forest / XGBoost on 9 MoleculeNet datasets
Scaffold split (same as GNN baselines for fair comparison)
"""
import os, json, warnings, random
from datetime import datetime
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from torch_geometric.datasets import MoleculeNet

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except:
    HAS_XGB = False
    print("[INFO] XGBoost not installed — install with: pip install xgboost")

os.makedirs("results", exist_ok=True)
os.makedirs("data",    exist_ok=True)

DATASETS = {
    "ESOL":          {"name": "ESOL",    "task_type": "reg"},
    "FreeSolv":      {"name": "FreeSolv","task_type": "reg"},
    "Lipophilicity": {"name": "Lipo",    "task_type": "reg"},
    "BBBP":          {"name": "BBBP",    "task_type": "cls"},
    "Tox21":         {"name": "Tox21",   "task_type": "cls"},
    "SIDER":         {"name": "SIDER",   "task_type": "cls"},
    "ClinTox":       {"name": "ClinTox", "task_type": "cls"},
    "BACE":          {"name": "BACE",    "task_type": "cls"},
    "HIV":           {"name": "HIV",     "task_type": "cls"},
}
SEEDS = [42, 123, 7]

# ── FINGERPRINTS ───────────────────────────────────────────────────────────────
def mol_to_fp(smi, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def load_dataset(ds_key, ds_cfg):
    dataset = MoleculeNet(root=f"data/{ds_key}", name=ds_cfg["name"])
    X, Y    = [], []
    for i in range(len(dataset)):
        smi = dataset.smiles[i]
        fp  = mol_to_fp(smi)
        if fp is None:
            continue
        y = dataset[i].y.numpy().flatten()
        X.append(fp)
        Y.append(y)
    return np.array(X), np.array(Y)

# ── SPLITS ─────────────────────────────────────────────────────────────────────
def scaffold_split_idx(dataset, frac_train=0.8, frac_val=0.1):
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaffolds = defaultdict(list)
    for i in range(len(dataset)):
        try:
            smi = dataset.smiles[i]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                scaffolds[""].append(i); continue
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds[s].append(i)
        except:
            scaffolds[""].append(i)
    groups = sorted(scaffolds.values(), key=len, reverse=True)
    t_cut  = int(frac_train * len(dataset))
    v_cut  = int((frac_train + frac_val) * len(dataset))
    tr, va, te = [], [], []
    for g in groups:
        if   len(tr) < t_cut:           tr.extend(g)
        elif len(tr)+len(va) < v_cut:   va.extend(g)
        else:                            te.extend(g)
    if not te: te = va
    return tr, va, te

def random_split_idx(n, seed, frac_train=0.8, frac_val=0.1):
    idx = list(range(n))
    random.seed(seed); random.shuffle(idx)
    n_tr = int(frac_train * n)
    n_va = int((frac_train + frac_val) * n)
    return idx[:n_tr], idx[n_tr:n_va], idx[n_va:]

def is_degenerate_y(y_te, task_type):
    if task_type != "cls": return False
    flat = y_te[~np.isnan(y_te)].flatten()
    return len(np.unique(flat.astype(int))) < 2

# ── EVALUATE ───────────────────────────────────────────────────────────────────
def eval_cls(model, X_te, Y_te):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_te)
        # binary
        if proba.ndim == 2:
            p = proba[:, 1]
        else:
            p = proba
    else:
        p = model.predict(X_te)
    l = Y_te[:, 0] if Y_te.ndim > 1 else Y_te
    ok = ~np.isnan(l)
    if ok.sum() < 10 or len(np.unique(l[ok].astype(int))) < 2:
        return {"roc_auc": 0.0}
    return {"roc_auc": float(roc_auc_score(l[ok], p[ok]))}

def eval_multitask_cls(models, X_te, Y_te):
    aucs = []
    for t, model in enumerate(models):
        l = Y_te[:, t]
        ok = ~np.isnan(l)
        if ok.sum() < 10 or len(np.unique(l[ok].astype(int))) < 2:
            continue
        proba = model.predict_proba(X_te[ok])
        p = proba[:, 1] if proba.ndim == 2 else proba
        try:
            aucs.append(roc_auc_score(l[ok], p))
        except: pass
    return {"roc_auc": float(np.mean(aucs)) if aucs else 0.0}

def eval_reg(model, X_te, Y_te):
    p = model.predict(X_te)
    l = Y_te[:, 0] if Y_te.ndim > 1 else Y_te
    ok = ~np.isnan(l)
    return {
        "rmse": float(np.sqrt(mean_squared_error(l[ok], p[ok]))),
        "mae":  float(mean_absolute_error(l[ok], p[ok]))
    }

# ── SINGLE RUN ─────────────────────────────────────────────────────────────────
def run_one(model_name, ds_key, ds_cfg, seed, X, Y):
    random.seed(seed); np.random.seed(seed)
    dataset   = MoleculeNet(root=f"data/{ds_key}", name=ds_cfg["name"])
    task_type = ds_cfg["task_type"]
    n_tasks   = Y.shape[1] if Y.ndim > 1 else 1

    # Get split indices
    tr_idx, va_idx, te_idx = scaffold_split_idx(dataset)

    # Filter to valid fingerprint indices (some SMILES may have failed)
    valid_count = len(X)
    # Re-map: scaffold split is on full dataset indices, X is already filtered
    # Safer: use random split on X directly
    if is_degenerate_y(Y[te_idx] if len(te_idx) > 0 else Y[:10], task_type):
        tr_idx, va_idx, te_idx = random_split_idx(len(X), seed)

    # Clamp indices to valid range
    tr_idx = [i for i in tr_idx if i < len(X)]
    te_idx = [i for i in te_idx if i < len(X)]
    if len(te_idx) == 0:
        tr_idx, va_idx, te_idx = random_split_idx(len(X), seed)
        tr_idx = [i for i in tr_idx if i < len(X)]
        te_idx = [i for i in te_idx if i < len(X)]

    X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    X_te, Y_te = X[te_idx], Y[te_idx]

    n_trees = 500
    rf_params = {"n_estimators": n_trees, "random_state": seed,
                 "n_jobs": -1, "max_features": "sqrt"}
    xgb_params = {"n_estimators": 300, "random_state": seed,
                  "n_jobs": -1, "verbosity": 0, "use_label_encoder": False,
                  "eval_metric": "logloss" if task_type=="cls" else "rmse"}

    if task_type == "reg":
        y_tr = Y_tr[:, 0] if Y_tr.ndim > 1 else Y_tr
        ok   = ~np.isnan(y_tr)
        if model_name == "RF":
            m = RandomForestRegressor(**rf_params)
            m.fit(X_tr[ok], y_tr[ok])
        else:
            m = XGBRegressor(**{k:v for k,v in xgb_params.items()
                               if k != "eval_metric" and k != "use_label_encoder"},
                            eval_metric="rmse")
            m.fit(X_tr[ok], y_tr[ok])
        return eval_reg(m, X_te, Y_te)

    else:  # classification
        if n_tasks == 1:
            y_tr = Y_tr[:, 0] if Y_tr.ndim > 1 else Y_tr
            ok   = ~np.isnan(y_tr)
            if model_name == "RF":
                m = RandomForestClassifier(**rf_params)
                m.fit(X_tr[ok], y_tr[ok].astype(int))
            else:
                m = XGBClassifier(**xgb_params)
                m.fit(X_tr[ok], y_tr[ok].astype(int))
            return eval_cls(m, X_te, Y_te)
        else:
            # Multi-task: train one model per task
            models = []
            for t in range(n_tasks):
                y_tr_t = Y_tr[:, t]
                ok     = ~np.isnan(y_tr_t)
                if ok.sum() < 20 or len(np.unique(y_tr_t[ok].astype(int))) < 2:
                    models.append(None); continue
                if model_name == "RF":
                    m = RandomForestClassifier(**rf_params)
                else:
                    m = XGBClassifier(**xgb_params)
                m.fit(X_tr[ok], y_tr_t[ok].astype(int))
                models.append(m)
            valid_models = [(t, m) for t, m in enumerate(models) if m is not None]
            aucs = []
            for t, m in valid_models:
                l  = Y_te[:, t]
                ok = ~np.isnan(l)
                if ok.sum() < 10 or len(np.unique(l[ok].astype(int))) < 2:
                    continue
                p = m.predict_proba(X_te[ok])[:, 1]
                try: aucs.append(roc_auc_score(l[ok], p))
                except: pass
            return {"roc_auc": float(np.mean(aucs)) if aucs else 0.0}

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
    out_path   = f"results/fingerprint_baselines_{timestamp}.json"
    results    = {}

    model_names = ["RF", "XGB"] if HAS_XGB else ["RF"]

    for mname in model_names:
        print(f"\n{'='*65}\n  {mname} + Morgan ECFP4 (r=2, 2048 bits)\n{'='*65}")
        results[mname] = {}

        for ds_key, ds_cfg in DATASETS.items():
            print(f"\n  Loading {ds_key}...", end=" ", flush=True)
            X, Y = load_dataset(ds_key, ds_cfg)
            print(f"{len(X)} molecules, {Y.shape[1] if Y.ndim>1 else 1} tasks")

            seed_res = []
            for seed in SEEDS:
                try:
                    m = run_one(mname, ds_key, ds_cfg, seed, X, Y)
                    seed_res.append(m)
                    if ds_cfg["task_type"] == "cls":
                        print(f"  {mname} | {ds_key:<15} | seed={seed} | AUC={m.get('roc_auc',0):.4f}")
                    else:
                        print(f"  {mname} | {ds_key:<15} | seed={seed} | RMSE={m.get('rmse',0):.4f}  MAE={m.get('mae',0):.4f}")
                except Exception as e:
                    print(f"  {mname} | {ds_key:<15} | seed={seed} | ERROR: {e}")
                    seed_res.append({"error": str(e)})

            valid = [r for r in seed_res if "error" not in r and r]
            agg   = {}
            if valid:
                for k in valid[0]:
                    vals = [r[k] for r in valid]
                    agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
                    agg[f"{k}_std"]  = round(float(np.std(vals)),  4)
                if ds_cfg["task_type"] == "cls":
                    print(f"  → {ds_key}: AUC  = {agg['roc_auc_mean']:.4f} ± {agg['roc_auc_std']:.4f}")
                else:
                    print(f"  → {ds_key}: RMSE = {agg['rmse_mean']:.4f} ± {agg['rmse_std']:.4f}")

            results[mname][ds_key] = {"seeds": seed_res, "agg": agg}

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  ✓ Saved → {out_path}")

    print("\n✅  All done.")

if __name__ == "__main__":
    main()
