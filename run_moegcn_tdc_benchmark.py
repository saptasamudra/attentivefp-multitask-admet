"""
run_moegcn_tdc_benchmark.py
---------------------------
Re-runs MoE-GCN on all 22 TDC ADMET datasets with IDENTICAL protocol
to run_gcn_tdc_baseline.py so comparison is apples-to-apples.

Key differences from old results_tdc.json:
  - 30-trial Optuna HPO (TPE + MedianPruner) — same as GCN baseline
  - 5 seeds (0-4) instead of 3, enabling Wilcoxon signed-rank n=22*5=110
  - Same mol_to_graph featurization (9-dim node features)
  - Same TDC scaffold split (seed=42)
  - Same 100 epoch max, patience=15, ReduceLROnPlateau

Output: results_moegcn_tdc_v2.json

Usage:
    python run_moegcn_tdc_benchmark.py
    python run_moegcn_tdc_benchmark.py --skip_existing
    python run_moegcn_tdc_benchmark.py --datasets hia_hou dili bbb_martins
    python run_moegcn_tdc_benchmark.py --seeds 5   # default
"""

import json, time, argparse, copy
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

# ── TDC dataset config ────────────────────────────────────────────────────────
TDC_DATASETS = {
    # Classification (AUROC)
    "hia_hou":                        {"task": "clf", "tdc_name": "HIA_Hou"},
    "pgp_broccatelli":                {"task": "clf", "tdc_name": "Pgp_Broccatelli"},
    "bioavailability_ma":             {"task": "clf", "tdc_name": "Bioavailability_Ma"},
    "bbb_martins":                    {"task": "clf", "tdc_name": "BBB_Martins"},
    "cyp2d6_veith":                   {"task": "clf", "tdc_name": "CYP2D6_Veith"},
    "cyp3a4_veith":                   {"task": "clf", "tdc_name": "CYP3A4_Veith"},
    "cyp2c9_veith":                   {"task": "clf", "tdc_name": "CYP2C9_Veith"},
    "cyp2d6_substrate_carbonmangels": {"task": "clf", "tdc_name": "CYP2D6_Substrate_CarbonMangels"},
    "cyp3a4_substrate_carbonmangels": {"task": "clf", "tdc_name": "CYP3A4_Substrate_CarbonMangels"},
    "cyp2c9_substrate_carbonmangels": {"task": "clf", "tdc_name": "CYP2C9_Substrate_CarbonMangels"},
    "herg":                           {"task": "clf", "tdc_name": "hERG"},
    "ames":                           {"task": "clf", "tdc_name": "AMES"},
    "dili":                           {"task": "clf", "tdc_name": "DILI"},
    # Regression
    "caco2_wang":                     {"task": "reg", "tdc_name": "Caco2_Wang",              "metric": "mae"},
    "lipophilicity_astrazeneca":      {"task": "reg", "tdc_name": "Lipophilicity_AstraZeneca","metric": "mae"},
    "solubility_aqsoldb":             {"task": "reg", "tdc_name": "Solubility_AqSolDB",       "metric": "mae"},
    "ppbr_az":                        {"task": "reg", "tdc_name": "PPBR_AZ",                  "metric": "mae"},
    "vdss_lombardo":                  {"task": "reg", "tdc_name": "VDss_Lombardo",             "metric": "spearman"},
    "half_life_obach":                {"task": "reg", "tdc_name": "Half_Life_Obach",           "metric": "spearman"},
    "clearance_microsome_az":         {"task": "reg", "tdc_name": "Clearance_Microsome_AZ",    "metric": "spearman"},
    "clearance_hepatocyte_az":        {"task": "reg", "tdc_name": "Clearance_Hepatocyte_AZ",   "metric": "spearman"},
    "ld50_zhu":                       {"task": "reg", "tdc_name": "LD50_Zhu",                  "metric": "mae"},
}

# ── Featurization (identical to run_gcn_tdc_baseline.py) ─────────────────────
def mol_to_graph(smiles):
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
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


def smiles_to_dataset(smiles_list, labels):
    data_list = []
    for smi, lab in zip(smiles_list, labels):
        g = mol_to_graph(smi)
        if g is None:
            continue
        g.y = torch.tensor([float(lab)], dtype=torch.float)
        data_list.append(g)
    return data_list


# ── MoE Layer (identical to moegcn_classif.py / moegcn_regr.py) ──────────────
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
        weights = torch.zeros_like(gate_logits).scatter_(
            1, topk_idx, F.softmax(topk_vals, dim=-1)
        )
        load = weights.mean(0)
        balance_loss = self.num_experts * (load * load).sum()
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        out = (weights.unsqueeze(-1) * expert_out).sum(dim=1)
        return out, balance_loss


# ── MoE-GCN Model ─────────────────────────────────────────────────────────────
class MoEGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_experts, top_k):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe  = MoELayer(hidden, hidden, num_experts, top_k)
        self.head = nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x, bal_loss = self.moe(x)
        return self.head(x).squeeze(-1), bal_loss


# ── Training / Evaluation ─────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, task, device):
    model.train()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out, bal_loss = model(batch)
        y = batch.y.squeeze()
        if task == "clf":
            loss = F.binary_cross_entropy_with_logits(out, y)
        else:
            loss = F.mse_loss(out, y)
        loss = loss + 0.01 * bal_loss
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, task, metric_name, device):
    model.eval()
    preds, truths = [], []
    for batch in loader:
        batch = batch.to(device)
        out, _ = model(batch)
        preds.extend(out.cpu().numpy().tolist())
        truths.extend(batch.y.cpu().numpy().flatten().tolist())
    preds, truths = np.array(preds), np.array(truths)
    if task == "clf":
        if len(np.unique(truths)) < 2:
            return 0.5
        return roc_auc_score(truths, preds)
    else:
        if metric_name == "mae":
            return float(np.mean(np.abs(preds - truths)))
        elif metric_name == "spearman":
            r, _ = spearmanr(preds, truths)
            return float(r) if not np.isnan(r) else 0.0
        else:
            return float(np.sqrt(np.mean((preds - truths) ** 2)))


def run_single(train_data, val_data, test_data, params, task, metric_name,
               max_epochs, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MoEGCN(
        in_dim=9,
        hidden=params["hidden"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        num_experts=params["num_experts"],
        top_k=params["top_k"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # drop_last=True on small datasets to avoid BatchNorm crash on batch of 1
    drop_last = len(train_data) % 64 == 1
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=drop_last)
    val_loader   = DataLoader(val_data,   batch_size=256)
    test_loader  = DataLoader(test_data,  batch_size=256)

    higher = metric_name in ("AUROC", "spearman")
    best_val  = -np.inf if higher else np.inf
    best_test = None
    best_state = None
    patience_counter = 0

    for _ in range(max_epochs):
        train_epoch(model, train_loader, optimizer, task, device)
        val_score = evaluate(model, val_loader, task, metric_name, device)
        scheduler.step(-val_score if higher else val_score)

        improved = (val_score > best_val) if higher else (val_score < best_val)
        if improved:
            best_val   = val_score
            best_state = copy.deepcopy(model.state_dict())
            best_test  = evaluate(model, test_loader, task, metric_name, device)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        best_test = evaluate(model, test_loader, task, metric_name, device)

    return best_test if best_test is not None else 0.0


# ── Optuna objective ──────────────────────────────────────────────────────────
def objective_factory(train_data, val_data, task, metric_name, device):
    def objective(trial):
        params = {
            "hidden":       trial.suggest_categorical("hidden", [128, 256]),
            "num_layers":   trial.suggest_int("num_layers", 2, 4),
            "dropout":      trial.suggest_float("dropout", 0.0, 0.3),
            "num_experts":  trial.suggest_categorical("num_experts", [4, 8, 16]),
            "top_k":        trial.suggest_int("top_k", 1, 4),
            "lr":           trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        }
        # clamp top_k to num_experts
        params["top_k"] = min(params["top_k"], params["num_experts"])

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
    from tdc.single_pred import ADME, Tox
    data_obj = None
    for Loader in [ADME, Tox]:
        try:
            data_obj = Loader(name=tdc_name)
            break
        except Exception:
            continue
    if data_obj is None:
        raise ValueError(f"Could not load TDC dataset: {tdc_name}")
    split = data_obj.get_split(method="scaffold", seed=42)
    return split["train"], split["valid"], split["test"]


# ── Per-dataset runner ────────────────────────────────────────────────────────
def run_dataset(key, config, device, n_seeds):
    tdc_name    = config["tdc_name"]
    task        = config["task"]
    metric_name = "AUROC" if task == "clf" else config.get("metric", "mae")
    higher      = metric_name in ("AUROC", "spearman")

    print(f"\n{'─'*57}")
    print(f"  {key}  [{tdc_name}]  metric={metric_name}")
    print(f"{'─'*57}")
    t0 = time.time()

    train_split, val_split, test_split = load_tdc_dataset(tdc_name, task)
    train_data = smiles_to_dataset(train_split["Drug"], train_split["Y"])
    val_data   = smiles_to_dataset(val_split["Drug"],   val_split["Y"])
    test_data  = smiles_to_dataset(test_split["Drug"],  test_split["Y"])

    if len(train_data) < 10:
        print(f"  ✗ Too few valid molecules ({len(train_data)})")
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
    best_params["top_k"] = min(best_params["top_k"], best_params["num_experts"])
    print(f"  Best params: {best_params}")

    seed_scores = []
    for seed in range(n_seeds):
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
        "model":       "MoE-GCN",
        "n_seeds":     n_seeds,
    }
    print(f"  ✅ {metric_name}: {result['mean']:.4f} ± {result['std']:.4f}  ({elapsed:.1f} min)")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",      nargs="+", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--output",        default="results_moegcn_tdc_v2.json")
    parser.add_argument("--device",        default=None)
    parser.add_argument("--seeds",         type=int, default=5,
                        help="Number of seeds per dataset (default=5 for Wilcoxon)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Seeds per dataset: {args.seeds}")

    out_path = Path(args.output)
    all_results = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results from {out_path}")

    to_run = args.datasets if args.datasets else list(TDC_DATASETS.keys())

    for key in to_run:
        if key not in TDC_DATASETS:
            print(f"⚠️  Unknown dataset: {key}, skipping")
            continue
        if args.skip_existing and key in all_results:
            print(f"  ⏭️  Skipping {key} (already done)")
            continue

        result = run_dataset(key, TDC_DATASETS[key], device, args.seeds)
        if result is not None:
            all_results[key] = result
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
            print(f"  💾 Saved to {out_path}")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*57}")
    print(f"  DONE: {len(all_results)}/22 datasets complete")
    print(f"  Results: {out_path}")
    print(f"{'='*57}")

    if len(all_results) > 0:
        print(f"\n{'Dataset':<40} {'Metric':<10} {'Mean':>8} {'±Std':>8} {'N_seeds':>8}")
        print("─" * 78)
        for k, v in all_results.items():
            print(f"  {k:<38} {v['metric']:<10} {v['mean']:>8.4f} {v['std']:>8.4f} {v.get('n_seeds',3):>8}")


if __name__ == "__main__":
    main()
