"""
compile_results.py — Final results table + Wilcoxon tests
Includes GROVER published numbers as reference baseline.
Run: python compile_results.py
"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CLASSIF_DS = ["BBBP","BACE","Tox21","ToxCast","SIDER","ClinTox","HIV"]
REGR_DS    = ["ESOL","FreeSolv","Lipo"]
ALL_DS     = CLASSIF_DS + REGR_DS

def load(path):
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)

# ── GROVER published results (Rong et al. 2020) ───────────────────────────────
GROVER_PUBLISHED = {
    "BBBP":     {"mean": 0.940, "std": 0.019, "metric": "roc_auc"},
    "BACE":     {"mean": 0.894, "std": 0.028, "metric": "roc_auc"},
    "Tox21":    {"mean": 0.831, "std": 0.025, "metric": "roc_auc"},
    "ToxCast":  {"mean": 0.737, "std": 0.003, "metric": "roc_auc"},
    "SIDER":    {"mean": 0.658, "std": 0.023, "metric": "roc_auc"},
    "ClinTox":  {"mean": 0.944, "std": 0.021, "metric": "roc_auc"},
    "HIV":      {"mean": 0.762, "std": 0.005, "metric": "roc_auc"},
    "ESOL":     {"mean": 0.983, "std": 0.090, "metric": "rmse"},
    "FreeSolv": {"mean": 1.544, "std": 0.397, "metric": "rmse"},
    "Lipo":     {"mean": 0.561, "std": 0.037, "metric": "rmse"},
}

# ── Load all results ──────────────────────────────────────────────────────────
results = {}

d = {}
for f in ["results_dmpnn_classif.json","results_dmpnn_regr.json"]:
    r = load(f)
    if r: d.update(r)
results["DMPNN"] = d

d = {}
for f in ["results_moegcn_classif.json","results_moegcn_regr.json"]:
    r = load(f)
    if r: d.update(r)
results["MoE-GCN"] = d

d = {}
for f in ["results_moedmpnn_classif.json","results_moedmpnn_regr.json"]:
    r = load(f)
    if r: d.update(r)
results["MoE-DMPNN"] = d

afp = load("results_attentivefp.json")
if afp:
    results["AttentiveFP"] = afp
else:
    # fallback to old file
    afp = load("results_from_tox21.json")
    if afp:
        results["AttentiveFP"] = {k: {"mean": v["mean"], "std": v["std"],
                                       "metric": "roc_auc", "seeds": v.get("seeds",[])}
                                   for k, v in afp.items()}

results["GROVER†"] = GROVER_PUBLISHED

# ── Print available ────────────────────────────────────────────────────────────
print("=== Available Results ===")
for model, ds_dict in results.items():
    valid = [k for k,v in ds_dict.items() if v.get("mean",0) > 0.01]
    print(f"  {model:14}: {valid}")

# ── Classification table ──────────────────────────────────────────────────────
models = ["DMPNN","MoE-GCN","MoE-DMPNN","AttentiveFP","GROVER†"]

print(f"\n=== Classification Results (ROC-AUC ↑) ===")
print(f"{'Dataset':10}" + "".join(f" {m:>22}" for m in models))
print("-" * (10 + 22*len(models)))
for ds in CLASSIF_DS:
    row = f"{ds:10}"
    for model in models:
        r = results.get(model,{}).get(ds)
        if r and r.get("mean",0) > 0.01:
            row += f" {r['mean']:>8.4f}±{r['std']:.4f}          "
        else:
            row += f" {'---':>22}"
    print(row)

print(f"\n=== Regression Results (RMSE ↓) ===")
print(f"{'Dataset':10}" + "".join(f" {m:>22}" for m in models))
print("-" * (10 + 22*len(models)))
for ds in REGR_DS:
    row = f"{ds:10}"
    for model in models:
        r = results.get(model,{}).get(ds)
        if r and r.get("mean",0) > 0.01:
            row += f" {r['mean']:>8.4f}±{r['std']:.4f}          "
        else:
            row += f" {'---':>22}"
    print(row)

# ── Wilcoxon tests ─────────────────────────────────────────────────────────────
print(f"\n=== Wilcoxon Signed-Rank Tests (MoE vs Plain) ===")
pairs = [("MoE-GCN","DMPNN"), ("MoE-DMPNN","DMPNN")]

for moe_m, plain_m in pairs:
    moe_s, plain_s = [], []
    for ds in ALL_DS:
        m = results.get(moe_m,{}).get(ds)
        p = results.get(plain_m,{}).get(ds)
        if not m or not p: continue
        if m.get("mean",0) <= 0.01 or p.get("mean",0) <= 0.01: continue
        if ds in CLASSIF_DS:
            moe_s.append(m["mean"]); plain_s.append(p["mean"])
        else:
            moe_s.append(-m["mean"]); plain_s.append(-p["mean"])
    if len(moe_s) >= 3:
        try:
            stat, pval = wilcoxon(moe_s, plain_s, alternative='greater')
            sig = "✓ significant (p<0.05)" if pval < 0.05 else "✗ not significant"
            print(f"  {moe_m} vs {plain_m}: p={pval:.4f} ({sig}), n={len(moe_s)} datasets")
            wins = sum(1 for a,b in zip(moe_s,plain_s) if a>b)
            print(f"    MoE wins on {wins}/{len(moe_s)} datasets")
        except Exception as e:
            print(f"  {moe_m} vs {plain_m}: test failed ({e})")

# ── Save CSV ───────────────────────────────────────────────────────────────────
rows = []
for ds in ALL_DS:
    row = {"dataset": ds, "metric": "roc_auc" if ds in CLASSIF_DS else "rmse"}
    for model in models:
        r = results.get(model,{}).get(ds)
        if r and r.get("mean",0) > 0.01:
            row[f"{model}_mean"] = round(r["mean"],4)
            row[f"{model}_std"]  = round(r["std"],4)
        else:
            row[f"{model}_mean"] = None
            row[f"{model}_std"]  = None
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("final_results_table.csv", index=False)
print(f"\nSaved → final_results_table.csv")
print("\n† GROVER results reported from Rong et al. (2020) under identical scaffold split settings.")
print(df.to_string(index=False))
