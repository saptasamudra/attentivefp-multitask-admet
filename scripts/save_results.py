"""
Run this after EACH experiment finishes to save results to a text file.
Usage: python save_results.py
It will ask you which experiment just finished and save the numbers.
"""

import os
from datetime import datetime

results = {
    "MoE K=4 (9 datasets)": {
        "ESOL":     {"mean": 0.9623, "std": 0.0694, "seeds": [0.9155, 0.9109, 1.0604], "metric": "RMSE", "task": "regression"},
        "FreeSolv": {"mean": 2.8020, "std": 0.1822, "seeds": [3.0592, 2.6869, 2.6599], "metric": "RMSE", "task": "regression"},
        "Lipo":     {"mean": 0.8121, "std": 0.0159, "seeds": [0.7973, 0.8049, 0.8341], "metric": "RMSE", "task": "regression"},
        "BACE":     {"mean": 0.7908, "std": 0.0312, "seeds": [0.8228, 0.8012, 0.7485], "metric": "AUC",  "task": "classification"},
        "BBBP":     {"mean": 0.8787, "std": 0.0327, "seeds": [0.8401, 0.8759, 0.9202], "metric": "AUC",  "task": "classification"},
        "HIV":      {"mean": 0.7809, "std": 0.0235, "seeds": [0.8093, 0.7516, 0.7817], "metric": "AUC",  "task": "classification"},
        "ClinTox":  {"mean": 0.9215, "std": 0.0143, "seeds": [0.9142, 0.9415, 0.9087], "metric": "AUC",  "task": "classification"},
        "Tox21":    {"mean": 0.7703, "std": 0.0086, "seeds": [0.7813, 0.7692, 0.7602], "metric": "AUC",  "task": "classification"},
        "SIDER":    {"mean": 0.5875, "std": 0.0231, "seeds": [0.6089, 0.5554, 0.5981], "metric": "AUC",  "task": "classification"},
    }
}

published = {
    "ESOL": 0.877, "FreeSolv": 2.082, "Lipo": 0.655,
    "BACE": 0.863, "BBBP": 0.862, "ClinTox": 0.832, "Tox21": 0.829
}

def save_experiment(name, data):
    os.makedirs("results", exist_ok=True)
    fname = f"results/{name.replace(' ', '_').replace('=','').replace('(','').replace(')','')}.txt"
    with open(fname, "w") as f:
        f.write(f"{'='*60}\n")
        f.write(f"  {name}\n")
        f.write(f"  Saved: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"{'='*60}\n")
        f.write(f"  {'Dataset':<12} {'Metric':<8} {'Mean':>8}  {'Std':>7}  {'Published':>10}  {'Status'}\n")
        f.write(f"  {'-'*65}\n")
        for ds, vals in data.items():
            pub = published.get(ds, None)
            lower = vals["task"] == "regression"
            if pub:
                diff = vals["mean"] - pub
                if lower:
                    status = "BEATING" if vals["mean"] < pub else f"{diff:+.4f}"
                else:
                    status = "BEATING" if vals["mean"] > pub else f"{diff:+.4f}"
                pub_str = f"{pub:.3f}"
            else:
                status = "new dataset"
                pub_str = "—"
            f.write(f"  {ds:<12} {vals['metric']:<8} {vals['mean']:>8.4f}  {vals['std']:>7.4f}  {pub_str:>10}  {status}\n")
        f.write(f"\n  Seed breakdown (42 | 123 | 7):\n")
        for ds, vals in data.items():
            s = vals["seeds"]
            f.write(f"    {ds:<12} {s[0]:.4f} | {s[1]:.4f} | {s[2]:.4f}\n")
        f.write(f"{'='*60}\n")
    print(f"Saved to {fname}")
    return fname

# Save K=4 results immediately
save_experiment("MoE K=4 (9 datasets)", results["MoE K=4 (9 datasets)"])

print("\nTemplate for adding new results:")
print("Edit save_results.py and add your new experiment data")
print("in the same format as 'MoE K=4 (9 datasets)' above,")
print("then run: python save_results.py")
print("\nFiles saved in: D:\\molprop_project\\results\\")
