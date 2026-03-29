"""
Run this after each new experiment finishes.
Paste the FINAL RESULTS block from the terminal when prompted.
It parses the output automatically and saves to the results folder.

Usage: python add_results.py
"""

import os
import re
from datetime import datetime

published = {
    "ESOL": 0.877, "FreeSolv": 2.082, "Lipo": 0.655,
    "BACE": 0.863, "BBBP": 0.862, "ClinTox": 0.832, "Tox21": 0.829
}

reg_datasets = ["ESOL", "FreeSolv", "Lipo"]

print("="*60)
print("  Result Saver -- MoE AttentiveFP Project")
print("="*60)
print()
print("Which experiment just finished?")
print("  1. Multitask 9-dataset (no MoE)")
print("  2. MoE K=2")
print("  3. MoE K=8")
choice = input("Enter 1, 2, or 3: ").strip()

names = {
    "1": "Multitask_9dataset_noMoE",
    "2": "MoE_K2_9datasets",
    "3": "MoE_K8_9datasets",
}
name = names.get(choice, "Unknown_experiment")

print()
print("Paste the FINAL RESULTS block below.")
print("(Paste everything from '===FINAL RESULTS===' to the last '===')")
print("Then press Enter twice when done:")
print()

lines = []
blank_count = 0
while blank_count < 2:
    line = input()
    if line == "":
        blank_count += 1
    else:
        blank_count = 0
        lines.append(line)

text = "\n".join(lines)

# Parse mean/std lines
dataset_pattern = re.compile(
    r'(\w+)\s+(regression|classification)\s+\w+\s+([\d.]+)\s+([\d.]+)'
)
seed_pattern = re.compile(
    r'(\w+)\s+([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)'
)

datasets = {}
for m in dataset_pattern.finditer(text):
    ds, task, mean, std = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
    datasets[ds] = {"task": task, "mean": mean, "std": std, "seeds": [], "metric": "RMSE" if task=="regression" else "AUC"}

in_seeds = False
for line in lines:
    if "Seed breakdown" in line:
        in_seeds = True
        continue
    if in_seeds:
        m = seed_pattern.search(line)
        if m:
            ds = m.group(1)
            if ds in datasets:
                datasets[ds]["seeds"] = [float(m.group(2)), float(m.group(3)), float(m.group(4))]

if not datasets:
    print("Could not parse results. Please check the pasted text.")
    exit()

# Save
os.makedirs("results", exist_ok=True)
fname = f"results/{name}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"

with open(fname, "w") as f:
    f.write(f"{'='*60}\n")
    f.write(f"  {name.replace('_',' ')}\n")
    f.write(f"  Saved: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"{'='*60}\n")
    f.write(f"  {'Dataset':<12} {'Metric':<8} {'Mean':>8}  {'Std':>7}  {'Published':>10}  Status\n")
    f.write(f"  {'-'*65}\n")
    for ds, vals in datasets.items():
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
    for ds, vals in datasets.items():
        if vals["seeds"]:
            s = vals["seeds"]
            f.write(f"    {ds:<12} {s[0]:.4f} | {s[1]:.4f} | {s[2]:.4f}\n")
    f.write(f"{'='*60}\n")

print(f"\nSaved to: {fname}")
print("Results folder: D:\\molprop_project\\results\\")

# Print summary
print(f"\n{'='*60}")
print(f"  Quick summary:")
print(f"{'='*60}")
for ds, vals in datasets.items():
    pub = published.get(ds)
    lower = vals["task"] == "regression"
    if pub:
        beating = vals["mean"] < pub if lower else vals["mean"] > pub
        flag = "✓ BEATING" if beating else ""
        print(f"  {ds:<12} {vals['mean']:.4f} ± {vals['std']:.4f}   pub:{pub}  {flag}")
    else:
        print(f"  {ds:<12} {vals['mean']:.4f} ± {vals['std']:.4f}   (new dataset)")
