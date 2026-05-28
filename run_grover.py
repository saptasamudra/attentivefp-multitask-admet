"""
run_grover.py — Run GROVER finetuning on all datasets
Run from: D:\molprop_project\grover (grover_env activated)
Run: python run_grover.py

Results saved to: ../grover_results.json
"""

import os, json, subprocess, sys
import numpy as np

GROVER_DIR   = "../grover_data"
MODEL_PATH   = "models/grover_base.pt"
RESULTS_PATH = "../grover_results.json"
SAVE_DIR     = "grover_finetune"

DATASETS = [
    {"name": "BBBP",    "type": "classification", "metric": "auc",  "tasks": 1},
    {"name": "BACE",    "type": "classification", "metric": "auc",  "tasks": 1},
    {"name": "Tox21",   "type": "classification", "metric": "auc",  "tasks": 12},
    {"name": "SIDER",   "type": "classification", "metric": "auc",  "tasks": 27},
    {"name": "ClinTox", "type": "classification", "metric": "auc",  "tasks": 2},
    {"name": "HIV",     "type": "classification", "metric": "auc",  "tasks": 1},
    {"name": "ESOL",    "type": "regression",     "metric": "rmse", "tasks": 1},
    {"name": "FreeSolv","type": "regression",     "metric": "rmse", "tasks": 1},
    {"name": "Lipo",    "type": "regression",     "metric": "rmse", "tasks": 1},
]

# Load existing results
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    print(f"Resuming — {len(results)} done")
else:
    results = {}

os.makedirs(SAVE_DIR, exist_ok=True)

for ds in DATASETS:
    name    = ds["name"]
    ds_type = ds["type"]
    metric  = ds["metric"]

    if name in results:
        print(f"  Skipping {name} (already done)")
        continue

    print(f"\n{'='*55}")
    print(f"  GROVER | {name} | {ds_type} | {metric}")
    print(f"{'='*55}")

    data_path  = f"{GROVER_DIR}/{name}/train.csv"
    val_path   = f"{GROVER_DIR}/{name}/val.csv"
    test_path  = f"{GROVER_DIR}/{name}/test.csv"
    save_path  = f"{SAVE_DIR}/{name}"
    feat_path  = f"{GROVER_DIR}/{name}/train_features.npz"
    feat_val   = f"{GROVER_DIR}/{name}/val_features.npz"
    feat_test  = f"{GROVER_DIR}/{name}/test_features.npz"

    os.makedirs(save_path, exist_ok=True)

    # Step 1: Generate features
    print("  Generating features...")
    for csv_path, feat_out in [
        (data_path, feat_path),
        (val_path,  feat_val),
        (test_path, feat_test),
    ]:
        if not os.path.exists(feat_out):
            cmd = [
                sys.executable, "main.py", "fingerprint",
                "--data_path", csv_path,
                "--checkpoint_path", MODEL_PATH,
                "--output", feat_out,
                "--fingerprint_source", "both",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  Feature gen failed: {result.stderr[-500:]}")
                break
            print(f"    Features saved: {feat_out}")

    # Step 2: Finetune
    print("  Finetuning...")
    cmd = [
        sys.executable, "main.py", "finetune",
        "--data_path",          data_path,
        "--separate_val_path",  val_path,
        "--separate_test_path", test_path,
        "--features_path",      feat_path,
        "--val_features_path",  feat_val,
        "--test_features_path", feat_test,
        "--checkpoint_path",    MODEL_PATH,
        "--save_dir",           save_path,
        "--dataset_type",       ds_type,
        "--metric",             metric,
        "--epochs",             "10",
        "--batch_size",         "32",
        "--init_lr",            "0.0001",
        "--max_lr",             "0.0003",
        "--final_lr",           "0.00001",
        "--no_features_scaling",
        "--ffn_hidden_size",    "200",
        "--ffn_num_layers",     "2",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-1000:] if result.stdout else "")

    if result.returncode != 0:
        print(f"  Finetune failed: {result.stderr[-500:]}")
        continue

    # Parse result from output
    score = None
    for line in result.stdout.split('\n'):
        if metric.upper() in line and '=' in line:
            try:
                score = float(line.split('=')[-1].strip())
            except:
                pass

    if score is None:
        # Try reading from saved results file
        res_file = f"{save_path}/verbose.log"
        if os.path.exists(res_file):
            with open(res_file) as f:
                for line in f:
                    if metric in line.lower():
                        try:
                            score = float(line.split('=')[-1].strip())
                        except:
                            pass

    results[name] = {
        "mean": score if score else 0.0,
        "std":  0.0,
        "metric": "roc_auc" if ds_type == "classification" else "rmse",
        "model": "GROVER_base"
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ GROVER {name}: {score}")

print("\n=== GROVER RESULTS ===")
for name, r in results.items():
    print(f"  {name:12}: {r['mean']:.4f}")
print(f"\nSaved → {RESULTS_PATH}")
