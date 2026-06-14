"""
run_gcn_tdc_baseline.py
-----------------------
Runs plain GCN (no MoE) on all 22 TDC ADMET datasets.
Saves results to: results_gcn_tdc.json

Mirrors the exact setup used for MoE-GCN TDC runs:
  - Same scaffold splits (TDC default)
  - Same Optuna HPO (30 trials, TPE + MedianPruner)
  - 3 seeds (matching results_tdc.json seed count)
  - Same features and training protocol

Usage:
    python run_gcn_tdc_baseline.py
    python run_gcn_tdc_baseline.py --datasets hia_hou bbb_martins dili
    python run_gcn_tdc_baseline.py --skip_existing   # resume if interrupted
"""

import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import BatchNorm1d

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

# ── TDC dataset config ────────────────────────────────────────────────────────
TDC_DATASETS = {
    # Classification (AUROC)
    "hia_hou":                      {"task": "clf", "tdc_name": "HIA_Hou"},
    "pgp_broccatelli":              {"task": "clf", "tdc_name": "Pgp_Broccatelli"},
    "bioavailability_ma":           {"task": "clf", "tdc_name": "Bioavailability_Ma"},
    "bbb_martins":                  {"task": "clf", "tdc_name": "BBB_Martins"},
    "cyp2d6_veith":                 {"task": "clf", "tdc_name": "CYP2D6_Veith"},
    "cyp3a4_veith":                 {"task": "clf", "tdc_name": "CYP3A4_Veith"},
    "cyp2c9_veith":                 {"task": "clf", "tdc_name": "CYP2C9_Veith"},
    "cyp2d6_substrate_carbonmangels": {"task": "clf", "tdc_name": "CYP2D6_Substrate_CarbonMangels"},
    "cyp3a4_substrate_carbonmangels": {"task": "clf", "tdc_name": "CYP3A4_Substrate_CarbonMangels"},
    "cyp2c9_substrate_carbonmangels": {"task": "clf", "tdc_name": "CYP2C9_Substrate_CarbonMangels"},
    "herg":                         {"task": "clf", "tdc_name": "hERG"},
    "ames":                         {"task": "clf", "tdc_name": "AMES"},
    "dili":                         {"task": "clf", "tdc_name": "DILI"},
    # Regression
    "caco2_wang":                   {"task": "reg", "tdc_name": "Caco2_Wang",         "metric": "mae"},
    "lipophilicity_astrazeneca":    {"task": "reg", "tdc_name": "Lipophilicity_AstraZeneca", "metric": "mae"},
    "solubility_aqsoldb":           {"task": "reg", "tdc_name": "Solubility_AqSolDB", "metric": "mae"},
    "ppbr_az":                      {"task": "reg", "tdc_name": "PPBR_AZ",             "metric": "mae"},
    "vdss_lombardo":                {"task": "reg", "tdc_name": "VDss_Lombardo",       "metric": "spearman"},
    "half_life_obach":              {"task": "reg", "tdc_name": "Half_Life_Obach",     "metric": "spearman"},
    "clearance_microsome_az":       {"task": "reg", "tdc_name": "Clearance_Microsome_AZ", "metric": "spearman"},
    "clearance_hepatocyte_az":      {"task": "reg", "tdc_name": "Clearance_Hepatocyte_AZ", "metric": "spearman"},
    "ld50_zhu":                     {"task": "reg", "tdc_name": "LD50_Zhu",            "metric": "mae"},
}

# ── Molecular featurization ───────────────────────────────────────────────────
def mol_to_graph(smiles):
    """Convert SMILES to PyG Data object. Returns None if invalid."""
    try:
        from rdkit import Chem
        from torch_geometric.data import Data

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Node features (9-dim, same as MoE-GCN)
        atom_features = []
        for atom in mol.GetAtoms():
            feat = [
                atom.GetAtomicNum(),
                int(atom.GetChiralTag()),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
            ]
            atom_features.append(feat)

        x = torch.tensor(atom_features, dtype=torch.float)

        # Edge index + edge features (3-dim)
        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feat = [
                int(bond.GetBondTypeAsDouble()),
                int(bond.GetStereo()),
                int(bond.GetIsConjugated()),
            ]
            edge_index += [[i, j], [j, i]]
            edge_attr  += [bond_feat, bond_feat]

        if not edge_index:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, 3), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception:
        return None


def smiles_to_dataset(smiles_list, labels, task):
    from torch_geometric.data import Data
    data_list = []
    for smi, lab in zip(smiles_list, labels):
        g = mol_to_graph(smi)
        if g is None:
            continue
        y = torch.tensor([float(lab)], dtype=torch.float)
        g.y = y
        data_list.append(g)
    return data_list


# ── Plain GCN model ───────────────────────────────────────────────────────────
class PlainGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, task):
        super().__init__()
        self.task = task
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        self.drop  = dropout

        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm1d(hidden_dim))

        self.head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)


# ── Training / evaluation ─────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, task, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        y   = batch.y.squeeze()
        if task == "clf":
            loss = F.binary_cross_entropy_with_logits(out, y)
        else:
            loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(model, loader, task, metric_name, device):
    model.eval()
    preds, truths = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        preds.extend(out.cpu().numpy().tolist())
        truths.extend(batch.y.cpu().numpy().flatten().tolist())

    preds, truths = np.array(preds), np.array(truths)

    if task == "clf":
        if len(np.unique(truths)) < 2:
            return 0.5
        if metric_name == "AUROC":
            return roc_auc_score(truths, preds)
    else:
        if metric_name == "mae":
            return float(np.mean(np.abs(preds - truths)))
        elif metric_name == "spearman":
            r, _ = spearmanr(preds, truths)
            return float(r) if not np.isnan(r) else 0.0
        else:  # rmse
            return float(np.sqrt(np.mean((preds - truths)**2)))
    return 0.0


def run_single(train_data, val_data, test_data, params, task, metric_name,
               max_epochs, device, seed):
    torch.manual_seed(seed)
    model = PlainGCN(
        in_dim=9,
        hidden_dim=params["hidden"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        task=task,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=256)
    test_loader  = DataLoader(test_data,  batch_size=256)

    higher = metric_name in ("AUROC", "spearman")
    best_val  = -np.inf if higher else np.inf
    best_test = None
    patience_counter = 0

    for epoch in range(max_epochs):
        train_epoch(model, train_loader, optimizer, task, device)
        val_score = evaluate(model, val_loader, task, metric_name, device)
        scheduler.step(-val_score if higher else val_score)

        improved = (val_score > best_val) if higher else (val_score < best_val)
        if improved:
            best_val = val_score
            best_test = evaluate(model, test_loader, task, metric_name, device)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break

    return best_test if best_test is not None else 0.0


def objective_factory(train_data, val_data, task, metric_name, device):
    def objective(trial):
        params = {
            "hidden":       trial.suggest_categorical("hidden", [128, 256]),
            "num_layers":   trial.suggest_int("num_layers", 2, 4),
            "dropout":      trial.suggest_float("dropout", 0.0, 0.3),
            "lr":           trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        }
        higher = metric_name in ("AUROC", "spearman")
        seed_scores = []
        for s in [0, 1]:
            score = run_single(train_data, val_data, val_data,
                               params, task, metric_name, 60, device, s)
            seed_scores.append(score)
            trial.report(np.mean(seed_scores), s)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(seed_scores)
    return objective


# ── TDC data loader ───────────────────────────────────────────────────────────
def load_tdc_dataset(tdc_name, task):
    if task == "clf":
        from tdc.single_pred import ADME, Tox
        loaders = [ADME, Tox]
    else:
        from tdc.single_pred import ADME, Tox
        loaders = [ADME, Tox]

    data_obj = None
    for Loader in loaders:
        try:
            data_obj = Loader(name=tdc_name)
            break
        except Exception:
            continue

    if data_obj is None:
        raise ValueError(f"Could not load TDC dataset: {tdc_name}")

    split = data_obj.get_split(method="scaffold", seed=42)
    return split["train"], split["valid"], split["test"]


# ── Main runner ───────────────────────────────────────────────────────────────
def run_dataset(key, config, args, device, existing_results):
    tdc_name    = config["tdc_name"]
    task        = config["task"]
    metric_name = "AUROC" if task == "clf" else config.get("metric", "mae")
    higher      = metric_name in ("AUROC", "spearman")

    print(f"\n{'─'*55}")
    print(f"  {key}  [{tdc_name}]  metric={metric_name}")
    print(f"{'─'*55}")

    t0 = time.time()

    try:
        train_split, val_split, test_split = load_tdc_dataset(tdc_name, task)
    except Exception as e:
        print(f"  ❌ Load error: {e}")
        return None

    train_data = smiles_to_dataset(train_split["Drug"], train_split["Y"], task)
    val_data   = smiles_to_dataset(val_split["Drug"],   val_split["Y"],   task)
    test_data  = smiles_to_dataset(test_split["Drug"],  test_split["Y"],  task)

    if len(train_data) < 10:
        print(f"  ❌ Too few valid molecules ({len(train_data)})")
        return None

    print(f"  Molecules: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    print(f"  Running Optuna HPO (30 trials)...")
    direction = "maximize" if higher else "minimize"
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(
        objective_factory(train_data, val_data, task, metric_name, device),
        n_trials=30,
        show_progress_bar=False,
    )
    best_params = study.best_params
    print(f"  Best params: {best_params}")

    seed_scores = []
    for seed in [0, 1, 2]:
        score = run_single(train_data, val_data, test_data,
                           best_params, task, metric_name, 100, device, seed)
        seed_scores.append(score)
        print(f"  Seed {seed}: {metric_name}={score:.4f}")

    elapsed = (time.time() - t0) / 60
    result = {
        "mean":        float(np.mean(seed_scores)),
        "std":         float(np.std(seed_scores)),
        "seeds":       seed_scores,
        "metric":      metric_name,
        "best_params": best_params,
        "time_min":    round(elapsed, 1),
        "model":       "plain_GCN",
    }
    print(f"  ✅ {metric_name}: {result['mean']:.4f} ± {result['std']:.4f}  ({elapsed:.1f} min)")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Subset of datasets to run (default: all 22)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip datasets already in output file")
    parser.add_argument("--output", default="results_gcn_tdc.json",
                        help="Output file path")
    parser.add_argument("--device", default=None,
                        help="cuda or cpu (auto-detected if not set)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    out_path = Path(args.output)

    # ── FIX: always load existing results into all_results first ──────────────
    # Original bug: existing was loaded but all_results started empty,
    # so saving after each dataset overwrote all previous entries.
    all_results = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results from {out_path}")
    # ─────────────────────────────────────────────────────────────────────────

    to_run = args.datasets if args.datasets else list(TDC_DATASETS.keys())

    for key in to_run:
        if key not in TDC_DATASETS:
            print(f"⚠️  Unknown dataset: {key}, skipping")
            continue
        if args.skip_existing and key in all_results:
            print(f"  ⏭️  Skipping {key} (already done)")
            continue

        config = TDC_DATASETS[key]
        result = run_dataset(key, config, args, device, all_results)
        if result is not None:
            all_results[key] = result
            # Save after each dataset — all_results always has full history
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
            print(f"  💾 Saved to {out_path}")

    print(f"\n{'='*55}")
    print(f"  DONE: {len(all_results)}/22 datasets complete")
    print(f"  Results: {out_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
