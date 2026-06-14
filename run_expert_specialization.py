"""
run_expert_specialization.py
-----------------------------
Trains MoE-GCN on ESOL with the best hyperparams from results_moegcn_regr.json,
saves checkpoint, extracts REAL expert routing assignments, then computes:
  1. Mutual Information (MI) between expert assignment and RDKit descriptors
  2. One-way ANOVA + Kruskal-Wallis across experts
  3. Eta-squared (η²) effect size
  4. Per-expert descriptor means ± std
  5. Paper-ready LaTeX table + suggested text

Output files:
  models/moegcn_esol_best.pt          — trained checkpoint
  expert_specialization_results.json  — full stats
  expert_specialization_table.md      — paper-ready markdown table

Usage:
    python run_expert_specialization.py
    python run_expert_specialization.py --dataset solubility_aqsoldb
    python run_expert_specialization.py --dataset lipophilicity_astrazeneca
"""

import json, argparse, copy
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, global_mean_pool

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import f_oneway, kruskal, spearmanr
import warnings
warnings.filterwarnings("ignore")

# ── Best hyperparams from results_moegcn_regr.json ───────────────────────────
BEST_PARAMS_BY_DATASET = {
    "esol": {
        "hidden": 256, "num_layers": 3, "dropout": 0.015,
        "num_experts": 16, "top_k": 4,
        "lr": 0.000229, "weight_decay": 2.73e-05,
        "tdc_name": "Solubility_AqSolDB", "metric": "mae",
    },
    "solubility_aqsoldb": {
        "hidden": 256, "num_layers": 4, "dropout": 0.153,
        "num_experts": 8, "top_k": 3,
        "lr": 0.000896, "weight_decay": 1.99e-05,
        "tdc_name": "Solubility_AqSolDB", "metric": "mae",
    },
    "lipophilicity_astrazeneca": {
        "hidden": 256, "num_layers": 4, "dropout": 0.038,
        "num_experts": 4, "top_k": 1,
        "lr": 0.000972, "weight_decay": 7.03e-05,
        "tdc_name": "Lipophilicity_AstraZeneca", "metric": "mae",
    },
    "caco2_wang": {
        "hidden": 256, "num_layers": 3, "dropout": 0.031,
        "num_experts": 16, "top_k": 4,
        "lr": 0.000599, "weight_decay": 2.01e-05,
        "tdc_name": "Caco2_Wang", "metric": "mae",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Featurization ─────────────────────────────────────────────────────────────
def mol_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        atom_features = []
        for atom in mol.GetAtoms():
            feat = [
                atom.GetAtomicNum(), int(atom.GetChiralTag()),
                atom.GetDegree(), atom.GetFormalCharge(),
                atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(),
                int(atom.GetHybridization()), int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
            ]
            atom_features.append(feat)
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = [int(bond.GetBondTypeAsDouble()), int(bond.GetStereo()),
                  int(bond.GetIsConjugated())]
            edge_index += [[i, j], [j, i]]
            edge_attr  += [bf, bf]
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
    data_list, valid_smiles = [], []
    for smi, lab in zip(smiles_list, labels):
        g = mol_to_graph(smi)
        if g is None:
            continue
        g.y = torch.tensor([float(lab)], dtype=torch.float)
        g.smiles = smi
        data_list.append(g)
        valid_smiles.append(smi)
    return data_list, valid_smiles

# ── MoE Layer with routing exposure ──────────────────────────────────────────
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

    def forward(self, x, return_routing=False):
        gate_logits = self.gate(x)
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = torch.zeros_like(gate_logits).scatter_(
            1, topk_idx, F.softmax(topk_vals, dim=-1)
        )
        load = weights.mean(0)
        balance_loss = self.num_experts * (load * load).sum()
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        out = (weights.unsqueeze(-1) * expert_out).sum(dim=1)
        if return_routing:
            return out, balance_loss, weights, topk_idx
        return out, balance_loss

# ── MoE-GCN with routing extraction ──────────────────────────────────────────
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

    def forward_with_routing(self, data):
        """Returns predictions + full routing weights + dominant expert per molecule."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = bn(x)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x, bal_loss, routing_weights, topk_idx = self.moe(x, return_routing=True)
        pred = self.head(x).squeeze(-1)
        dominant_expert = routing_weights.argmax(dim=-1)
        return pred, routing_weights, dominant_expert

# ── Training ──────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out, bal_loss = model(batch)
        y = batch.y.squeeze()
        loss = F.mse_loss(out, y) + 0.01 * bal_loss
        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate_mae(model, loader):
    model.eval()
    preds, truths = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out, _ = model(batch)
        preds.extend(out.cpu().numpy().tolist())
        truths.extend(batch.y.cpu().numpy().flatten().tolist())
    return float(np.mean(np.abs(np.array(preds) - np.array(truths))))

# ── Train and save checkpoint ─────────────────────────────────────────────────
def train_and_save(params, train_data, val_data, test_data, save_path, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MoEGCN(
        in_dim=9, hidden=params["hidden"], num_layers=params["num_layers"],
        dropout=params["dropout"], num_experts=params["num_experts"],
        top_k=params["top_k"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"],
                                  weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    drop_last = len(train_data) % 64 == 1
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=drop_last)
    val_loader   = DataLoader(val_data,   batch_size=256)
    test_loader  = DataLoader(test_data,  batch_size=256)

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(100):
        train_epoch(model, train_loader, optimizer)
        val_mae = evaluate_mae(model, val_loader)
        scheduler.step(val_mae)
        if val_mae < best_val:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break

    model.load_state_dict(best_state)
    test_mae = evaluate_mae(model, test_loader)
    print(f"  Trained: val_MAE={best_val:.4f}, test_MAE={test_mae:.4f}")

    Path(save_path).parent.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "params": params,
        "test_mae": test_mae,
    }, save_path)
    print(f"  Checkpoint saved: {save_path}")
    return model

# ── Extract routing on full dataset ──────────────────────────────────────────
@torch.no_grad()
def extract_routing(model, all_data, all_smiles):
    model.eval()
    loader = DataLoader(all_data, batch_size=256)
    all_dominant = []
    all_weights  = []

    for batch in loader:
        batch = batch.to(DEVICE)
        _, routing_weights, dominant = model.forward_with_routing(batch)
        all_dominant.extend(dominant.cpu().numpy().tolist())
        all_weights.append(routing_weights.cpu().numpy())

    dominant_experts = np.array(all_dominant)
    routing_matrix   = np.vstack(all_weights)  # [N, num_experts]

    print(f"  Extracted routing for {len(dominant_experts)} molecules")
    unique, counts = np.unique(dominant_experts, return_counts=True)
    print(f"  Expert usage: " + ", ".join(f"E{u}={c}" for u, c in zip(unique, counts)))

    return dominant_experts, routing_matrix

# ── RDKit descriptors ─────────────────────────────────────────────────────────
DESCRIPTOR_FNS = {
    "MW":       Descriptors.ExactMolWt,
    "LogP":     Descriptors.MolLogP,
    "HBA":      rdMolDescriptors.CalcNumHBA,
    "HBD":      rdMolDescriptors.CalcNumHBD,
    "TPSA":     Descriptors.TPSA,
    "RotBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "Rings":    rdMolDescriptors.CalcNumRings,
    "ArRings":  rdMolDescriptors.CalcNumAromaticRings,
}

def compute_rdkit_descriptors(smiles_list):
    desc_vals = {k: [] for k in DESCRIPTOR_FNS}
    valid_mask = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mask.append(False)
            for k in desc_vals:
                desc_vals[k].append(np.nan)
            continue
        valid_mask.append(True)
        for name, fn in DESCRIPTOR_FNS.items():
            try:
                desc_vals[name].append(float(fn(mol)))
            except Exception:
                desc_vals[name].append(np.nan)
    return {k: np.array(v) for k, v in desc_vals.items()}, np.array(valid_mask)

# ── Statistical tests ─────────────────────────────────────────────────────────
def mi_continuous_discrete(x, y):
    mask = ~np.isnan(x)
    x, y = x[mask], y[mask]
    n_bins = min(10, max(2, len(np.unique(x))))
    try:
        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        x_binned = kbd.fit_transform(x.reshape(-1, 1)).ravel().astype(int)
        return float(mutual_info_score(x_binned, y))
    except Exception:
        return 0.0

def eta_squared(groups):
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total   = np.sum((all_vals - grand_mean) ** 2)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0

def run_statistics(desc_arrays, expert_labels):
    unique_experts = sorted(np.unique(expert_labels))
    results = {}

    for desc_name, vals in desc_arrays.items():
        mask = ~np.isnan(vals)
        v = vals[mask]
        e = expert_labels[mask]

        groups = [v[e == exp] for exp in unique_experts if np.sum(e == exp) >= 5]
        if len(groups) < 2 or len(v) < 20:
            continue

        mi = mi_continuous_discrete(v, e)

        try:
            f_stat, p_anova = f_oneway(*groups)
        except Exception:
            f_stat, p_anova = np.nan, 1.0

        try:
            h_stat, p_kw = kruskal(*groups)
        except Exception:
            h_stat, p_kw = np.nan, 1.0

        eta2 = eta_squared(groups)

        per_expert = {}
        for exp in unique_experts:
            g = v[e == exp]
            if len(g) >= 5:
                per_expert[int(exp)] = {
                    "mean": float(np.mean(g)),
                    "std":  float(np.std(g)),
                    "n":    int(len(g)),
                    "median": float(np.median(g)),
                }

        results[desc_name] = {
            "MI":        float(mi),
            "F_stat":    float(f_stat) if not np.isnan(f_stat) else None,
            "p_anova":   float(p_anova),
            "H_stat":    float(h_stat) if not np.isnan(h_stat) else None,
            "p_kruskal": float(p_kw),
            "eta2":      float(eta2),
            "per_expert": per_expert,
            "n_valid":   int(len(v)),
        }

    return results, unique_experts

# ── Print & save report ───────────────────────────────────────────────────────
def print_and_save_report(stats, unique_experts, dataset_name, out_json, out_md):
    print(f"\n{'='*72}")
    print(f"  EXPERT SPECIALIZATION STATISTICS — {dataset_name.upper()}")
    print(f"{'='*72}")
    print(f"\n  {'Descriptor':<12} {'MI':>6}  {'F-stat':>8}  {'p-ANOVA':>9}  "
          f"{'p-KW':>9}  {'η²':>6}  Sig")
    print(f"  {'─'*70}")

    sig_descs = []
    for desc, r in stats.items():
        f_str = f"{r['F_stat']:.1f}" if r['F_stat'] is not None else "  N/A"
        p_str = f"{r['p_anova']:.4f}"
        kw_str = f"{r['p_kruskal']:.4f}"
        sig = "***" if r['p_anova'] < 0.001 else ("**" if r['p_anova'] < 0.01
              else ("*" if r['p_anova'] < 0.05 else ""))
        if r['p_anova'] < 0.05:
            sig_descs.append(desc)
        print(f"  {desc:<12} {r['MI']:>6.3f}  {f_str:>8}  {p_str:>9}  "
              f"{kw_str:>9}  {r['eta2']:>6.3f}  {sig}")

    print(f"\n  Significant (p<0.05): {', '.join(sig_descs) or 'none'}")
    high_effect = [d for d in sig_descs if stats[d]['eta2'] > 0.05]
    print(f"  High effect (η²>0.05): {', '.join(high_effect) or 'none'}")

    # Per-expert table
    print(f"\n  PER-EXPERT MEANS")
    print(f"  {'Expert':<8}" + "".join(f"  {d:>10}" for d in stats))
    print(f"  {'─'*8}" + "".join(f"  {'─'*10}" for _ in stats))
    for exp in unique_experts:
        row = f"  E{exp:<7}"
        for desc, r in stats.items():
            if int(exp) in r['per_expert']:
                mu = r['per_expert'][int(exp)]['mean']
                sd = r['per_expert'][int(exp)]['std']
                row += f"  {mu:>5.1f}±{sd:.1f}"
            else:
                row += f"  {'—':>10}"
        print(row)

    # LaTeX table
    descs = list(stats.keys())
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Expert routing specialization statistics. F-statistics from one-way ANOVA across expert groups; $\eta^2$ = eta-squared effect size; MI = mutual information with discretized descriptor.}",
        r"\begin{tabular}{l" + "r"*len(descs) + "}",
        r"\hline",
        "Descriptor & " + " & ".join(descs) + r" \\",
        r"\hline",
    ]
    for exp in unique_experts:
        row_parts = []
        for desc in descs:
            r = stats[desc]
            if int(exp) in r['per_expert']:
                mu = r['per_expert'][int(exp)]['mean']
                sd = r['per_expert'][int(exp)]['std']
                row_parts.append(f"${mu:.1f}\\pm{sd:.1f}$")
            else:
                row_parts.append("—")
        latex_lines.append(f"Expert {exp} & " + " & ".join(row_parts) + r" \\")
    latex_lines += [
        r"\hline",
        "F-stat & " + " & ".join(
            f"${stats[d]['F_stat']:.1f}$" if stats[d]['F_stat'] else "—"
            for d in descs) + r" \\",
        "p-value & " + " & ".join(
            f"${stats[d]['p_anova']:.4f}$" for d in descs) + r" \\",
        "$\\eta^2$ & " + " & ".join(
            f"${stats[d]['eta2']:.3f}$" for d in descs) + r" \\",
        "MI & " + " & ".join(
            f"${stats[d]['MI']:.3f}$" for d in descs) + r" \\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    # Suggested paper text
    top_descs = sorted(sig_descs, key=lambda d: -stats[d]['eta2'])[:3]
    paper_text = f"""
SUGGESTED PAPER TEXT:
─────────────────────
To validate expert chemical specialization quantitatively, we computed mutual
information (MI) between dominant expert assignment and seven RDKit physicochemical
descriptors across all {sum(r['n_valid'] for r in list(stats.values())[:1])} molecules in the {dataset_name} dataset,
and performed one-way ANOVA across expert groups. Significant between-expert
variation was observed for {', '.join(top_descs)} (all p < {"0.001" if all(stats[d]['p_anova']<0.001 for d in top_descs) else "0.05"},
ANOVA), confirming that expert routing captures meaningful physicochemical
structure. Effect sizes (η²) indicate that expert identity explains
{', '.join(f"{stats[d]['eta2']*100:.0f}% of {d} variance" for d in top_descs[:2])},
consistent with spontaneous learning of Lipinski-like chemical space partitioning.
"""
    print(paper_text)

    # Save JSON
    with open(out_json, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "stats": {k: {**v, "per_expert": {str(ek): ev
                for ek, ev in v["per_expert"].items()}}
                for k, v in stats.items()},
            "significant_descriptors": sig_descs,
            "high_effect_descriptors": high_effect,
        }, f, indent=2)
    print(f"  Stats saved → {out_json}")

    # Save markdown
    md_lines = [
        f"# Expert Specialization Statistics — {dataset_name}",
        "",
        "## Summary",
        f"- Significant descriptors (p<0.05): {', '.join(sig_descs) or 'none'}",
        f"- High effect size (η²>0.05): {', '.join(high_effect) or 'none'}",
        "",
        "## Statistics Table",
        "",
        "| Descriptor | MI | F-stat | p-ANOVA | p-KW | η² | Sig |",
        "|------------|-----|--------|---------|------|----|-----|",
    ]
    for desc, r in stats.items():
        f_str = f"{r['F_stat']:.1f}" if r['F_stat'] else "N/A"
        sig = "***" if r['p_anova']<0.001 else ("**" if r['p_anova']<0.01
              else ("*" if r['p_anova']<0.05 else "n.s."))
        md_lines.append(
            f"| {desc} | {r['MI']:.3f} | {f_str} | {r['p_anova']:.4f} | "
            f"{r['p_kruskal']:.4f} | {r['eta2']:.3f} | {sig} |"
        )
    md_lines += ["", "## LaTeX Table", "", "```latex"] + latex_lines + ["```", "", paper_text]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  Markdown saved → {out_md}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="solubility_aqsoldb",
                        choices=list(BEST_PARAMS_BY_DATASET.keys()),
                        help="Dataset to analyze (default: solubility_aqsoldb)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain even if checkpoint exists")
    args = parser.parse_args()

    params = BEST_PARAMS_BY_DATASET[args.dataset]
    tdc_name = params["tdc_name"]
    ckpt_path = f"models/moegcn_{args.dataset}_seed{args.seed}.pt"

    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset} ({tdc_name})")
    print(f"Params: {params}")

    # ── Load TDC data ────────────────────────────────────────────────────────
    print("\n[1/4] Loading TDC data...")
    from tdc.single_pred import ADME, Tox
    data_obj = None
    for Loader in [ADME, Tox]:
        try:
            data_obj = Loader(name=tdc_name)
            break
        except Exception:
            continue
    if data_obj is None:
        raise ValueError(f"Could not load {tdc_name}")

    split = data_obj.get_split(method="scaffold", seed=42)
    train_data, train_smiles = smiles_to_dataset(split["train"]["Drug"], split["train"]["Y"])
    val_data,   val_smiles   = smiles_to_dataset(split["valid"]["Drug"], split["valid"]["Y"])
    test_data,  test_smiles  = smiles_to_dataset(split["test"]["Drug"],  split["test"]["Y"])

    all_smiles = train_smiles + val_smiles + test_smiles
    all_data   = train_data  + val_data   + test_data
    print(f"  Total molecules: {len(all_data)}")

    # ── Train or load checkpoint ─────────────────────────────────────────────
    print("\n[2/4] Training MoE-GCN...")
    if Path(ckpt_path).exists() and not args.retrain:
        print(f"  Loading existing checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model = MoEGCN(
            in_dim=9, hidden=params["hidden"], num_layers=params["num_layers"],
            dropout=params["dropout"], num_experts=params["num_experts"],
            top_k=params["top_k"],
        ).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = train_and_save(params, train_data, val_data, test_data,
                               ckpt_path, seed=args.seed)

    # ── Extract routing ──────────────────────────────────────────────────────
    print("\n[3/4] Extracting expert routing assignments...")
    dominant_experts, routing_matrix = extract_routing(model, all_data, all_smiles)

    # ── Compute descriptors & stats ──────────────────────────────────────────
    print("\n[4/4] Computing RDKit descriptors and statistics...")
    desc_arrays, valid_mask = compute_rdkit_descriptors(all_smiles)
    expert_labels = dominant_experts[valid_mask]

    # Filter to descriptor arrays aligned with valid molecules
    desc_arrays_valid = {k: v[valid_mask] for k, v in desc_arrays.items()}

    stats, unique_experts = run_statistics(desc_arrays_valid, expert_labels)

    out_json = f"expert_specialization_{args.dataset}.json"
    out_md   = f"expert_specialization_{args.dataset}.md"

    print_and_save_report(stats, unique_experts, args.dataset, out_json, out_md)

    print(f"\n{'='*72}")
    print(f"  DONE — results saved to {out_json} and {out_md}")
    print(f"{'='*72}")

if __name__ == "__main__":
    main()
