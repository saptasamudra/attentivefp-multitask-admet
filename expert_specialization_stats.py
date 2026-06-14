"""
expert_specialization_stats.py
-------------------------------
Replaces the illustrative Table 6 in the report with quantitative
statistical validation of expert chemical space partitioning.

Computes:
  1. Mutual Information (MI) between expert assignment and each RDKit descriptor
  2. One-way ANOVA across experts for each descriptor
  3. Eta-squared (η²) effect size
  4. A clean LaTeX/markdown table ready for the paper

Requirements:
    pip install rdkit scikit-learn scipy numpy

Usage:
    python expert_specialization_stats.py
    python expert_specialization_stats.py --dataset ESOL --checkpoint path/to/model.pt

The script can run in two modes:
  A) CHECKPOINT MODE: loads your trained MoE-GCN, runs inference on ESOL,
     extracts actual expert assignments, computes stats. (Best for paper.)
  B) DEMO MODE (default): reconstructs from the descriptor means in Table 6
     to verify the statistical approach works, then tells you how to switch
     to checkpoint mode.
"""

import argparse
import json
import numpy as np
from pathlib import Path

# ── Try importing heavy dependencies gracefully ───────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("⚠️  RDKit not found. Install with: pip install rdkit")

try:
    from sklearn.metrics import mutual_info_score
    from sklearn.preprocessing import KBinsDiscretizer
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from scipy.stats import f_oneway, kruskal
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ── RDKit descriptor computation ──────────────────────────────────────────────
DESCRIPTORS = {
    "MW":   Descriptors.ExactMolWt       if RDKIT_OK else None,
    "LogP": Descriptors.MolLogP          if RDKIT_OK else None,
    "HBA":  rdMolDescriptors.CalcNumHBA  if RDKIT_OK else None,
    "HBD":  rdMolDescriptors.CalcNumHBD  if RDKIT_OK else None,
    "TPSA": Descriptors.TPSA             if RDKIT_OK else None,
    "RotBonds": rdMolDescriptors.CalcNumRotatableBonds if RDKIT_OK else None,
    "Rings": rdMolDescriptors.CalcNumRings if RDKIT_OK else None,
}

def compute_descriptors(smiles_list):
    """Returns dict of descriptor_name -> np.array of values, and valid indices."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid = [i for i, m in enumerate(mols) if m is not None]
    mols  = [mols[i] for i in valid]

    desc_arrays = {}
    for name, fn in DESCRIPTORS.items():
        if fn is None:
            continue
        vals = []
        for mol in mols:
            try:
                vals.append(float(fn(mol)))
            except Exception:
                vals.append(float("nan"))
        desc_arrays[name] = np.array(vals)

    return desc_arrays, valid


# ── Statistical tests ─────────────────────────────────────────────────────────
def mutual_information_continuous(x, y_labels):
    """
    Compute mutual information between continuous x and discrete expert labels y.
    Uses k-bins discretization on x.
    """
    n_bins = min(10, len(np.unique(x)))
    if n_bins < 2:
        return 0.0
    try:
        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        x_binned = kbd.fit_transform(x.reshape(-1, 1)).ravel().astype(int)
        return mutual_info_score(x_binned, y_labels)
    except Exception:
        return 0.0


def eta_squared(groups):
    """Effect size η² for one-way ANOVA. groups: list of arrays."""
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total   = np.sum((all_vals - grand_mean)**2)
    return ss_between / ss_total if ss_total > 0 else 0.0


def run_stats(desc_arrays, expert_labels):
    """
    For each descriptor, compute:
      - MI with expert assignment
      - ANOVA F-stat and p-value
      - η² effect size
      - Per-expert mean ± std
    """
    unique_experts = sorted(np.unique(expert_labels))
    results = {}

    for desc_name, vals in desc_arrays.items():
        # Remove NaNs
        mask = ~np.isnan(vals)
        v = vals[mask]
        e = expert_labels[mask]

        if len(v) < 10:
            continue

        groups = [v[e == exp] for exp in unique_experts if np.sum(e == exp) >= 3]
        if len(groups) < 2:
            continue

        # MI
        mi = mutual_information_continuous(v, e[np.isin(e, unique_experts)])

        # ANOVA
        try:
            f_stat, p_anova = f_oneway(*groups)
        except Exception:
            f_stat, p_anova = float("nan"), 1.0

        # Kruskal-Wallis (non-parametric alternative)
        try:
            h_stat, p_kruskal = kruskal(*groups)
        except Exception:
            h_stat, p_kruskal = float("nan"), 1.0

        # Effect size
        eta2 = eta_squared(groups)

        # Per-expert stats
        per_expert = {
            exp: (float(np.mean(groups[i])), float(np.std(groups[i])))
            for i, exp in enumerate(unique_experts)
            if np.sum(e == exp) >= 3
        }

        results[desc_name] = {
            "MI":        mi,
            "F_stat":    f_stat,
            "p_anova":   p_anova,
            "H_stat":    h_stat,
            "p_kruskal": p_kruskal,
            "eta2":      eta2,
            "per_expert": per_expert,
        }

    return results


# ── Checkpoint mode: load model and extract expert assignments ────────────────
def extract_expert_assignments_from_checkpoint(checkpoint_path, dataset_name, device="cuda"):
    """
    Loads your trained MoE-GCN checkpoint, runs inference on the dataset,
    and returns (smiles_list, expert_assignments_per_molecule).

    Adapt the model import to match your actual model file.
    """
    if not TORCH_OK:
        raise ImportError("PyTorch not available")

    # ── Import your model ──
    # Adjust this import to match your project structure:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from model import MoEGCN  # adjust to your actual class name
    except ImportError:
        raise ImportError(
            "Could not import MoEGCN from model.py. "
            "Edit the import at the top of extract_expert_assignments_from_checkpoint() "
            "to match your actual model file and class name."
        )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Try common checkpoint key patterns
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    # You'll need to instantiate with the right hyperparams — read from checkpoint if saved
    # Example (adjust hidden_dim, num_layers, num_experts, top_k to your best params):
    model_kwargs = ckpt.get("model_kwargs", {
        "in_dim": 9, "hidden_dim": 256, "num_layers": 3,
        "num_experts": 16, "top_k": 4, "dropout": 0.015
    })
    model = MoEGCN(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load dataset
    from tdc.single_pred import ADME
    data_obj = ADME(name=dataset_name)
    split = data_obj.get_split()
    all_smiles = (
        list(split["train"]["Drug"]) +
        list(split["valid"]["Drug"]) +
        list(split["test"]["Drug"])
    )

    # Run inference with routing hooks
    from featurize import smiles_to_graph  # adjust to your featurizer
    from torch_geometric.data import DataLoader

    graphs = []
    valid_smiles = []
    for smi in all_smiles:
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)
            valid_smiles.append(smi)

    loader = DataLoader(graphs, batch_size=256)
    all_expert_assignments = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Call forward with routing info returned
            # Adjust based on how your MoE layer exposes routing weights:
            _, routing_weights = model.forward_with_routing(batch)
            # routing_weights: [batch_size, num_experts]
            dominant_expert = routing_weights.argmax(dim=-1).cpu().numpy()
            all_expert_assignments.extend(dominant_expert.tolist())

    return valid_smiles, np.array(all_expert_assignments)


# ── Demo mode: reconstruct from Table 6 means ────────────────────────────────
def demo_mode():
    """
    Simulates expert assignments based on Table 6 descriptor means,
    to verify the statistical machinery works before running on real data.
    """
    print("\n" + "="*60)
    print("  DEMO MODE — simulating from Table 6 statistics")
    print("  (Run with --checkpoint for real expert assignments)")
    print("="*60)

    rng = np.random.default_rng(42)
    n_per_expert = {"E15": 472, "E2": 304, "E4": 159, "E10": 137}

    # Table 6 means ± assumed std (estimated from context)
    expert_params = {
        "E15": {"MW": (253.7, 60), "LogP": (2.35, 1.2), "HBA": (3.22, 1.5),
                "HBD": (1.07, 0.8), "TPSA": (57.9, 25)},
        "E2":  {"MW": (197.6, 50), "LogP": (3.54, 1.0), "HBA": (1.03, 0.8),
                "HBD": (0.35, 0.5), "TPSA": (14.0, 10)},
        "E4":  {"MW": (143.9, 40), "LogP": (1.71, 0.9), "HBA": (1.40, 0.9),
                "HBD": (0.19, 0.4), "TPSA": (20.5, 15)},
        "E10": {"MW": (114.1, 35), "LogP": (1.21, 0.8), "HBA": (1.56, 0.9),
                "HBD": (0.92, 0.7), "TPSA": (25.2, 18)},
    }

    all_desc = {d: [] for d in ["MW", "LogP", "HBA", "HBD", "TPSA"]}
    all_labels = []
    label_map = {"E15": 15, "E2": 2, "E4": 4, "E10": 10}

    for exp_name, n in n_per_expert.items():
        params = expert_params[exp_name]
        for desc_name in all_desc:
            mu, sigma = params[desc_name]
            all_desc[desc_name].extend(rng.normal(mu, sigma, n).tolist())
        all_labels.extend([label_map[exp_name]] * n)

    desc_arrays = {k: np.array(v) for k, v in all_desc.items()}
    expert_labels = np.array(all_labels)

    return desc_arrays, expert_labels


# ── Report printer ────────────────────────────────────────────────────────────
def print_report(stats, expert_labels, mode="demo"):
    unique_experts = sorted(np.unique(expert_labels))
    expert_names = {exp: f"E{exp}" for exp in unique_experts}

    print(f"\n{'='*70}")
    print(f"  EXPERT SPECIALIZATION — STATISTICAL VALIDATION  [{mode}]")
    print(f"{'='*70}")
    print(f"\n  n={len(expert_labels)} molecules, {len(unique_experts)} experts")
    print(f"  Expert counts: " + ", ".join(
        f"{expert_names[e]}={np.sum(expert_labels==e)}" for e in unique_experts))

    print(f"\n{'─'*70}")
    print(f"  {'Descriptor':<10} {'MI':>6}  {'F-stat':>8}  {'p-ANOVA':>9}  "
          f"{'p-KW':>9}  {'η²':>6}  {'Sig?':>5}")
    print(f"{'─'*70}")

    sig_descriptors = []
    for desc, r in stats.items():
        sig = r["p_anova"] < 0.05
        sig_kw = r["p_kruskal"] < 0.05
        if sig or sig_kw:
            sig_descriptors.append(desc)
        marker = "***" if r["p_anova"] < 0.001 else ("**" if r["p_anova"] < 0.01 else
                 ("*" if sig else ""))
        print(f"  {desc:<10} {r['MI']:>6.3f}  {r['F_stat']:>8.1f}  "
              f"{r['p_anova']:>9.4f}  {r['p_kruskal']:>9.4f}  "
              f"{r['eta2']:>6.3f}  {marker:>5}")

    print(f"\n  Significant descriptors (p<0.05): {', '.join(sig_descriptors) or 'none'}")

    # Per-expert table (paper-ready)
    print(f"\n{'─'*70}")
    print(f"  PER-EXPERT DESCRIPTOR MEANS (for paper Table 6 replacement)")
    print(f"{'─'*70}")
    header = f"  {'Expert':<8}" + "".join(f"  {d:>8}" for d in stats.keys())
    print(header)
    print(f"  {'─'*8}" + "".join(f"  {'─'*8}" for _ in stats))
    for exp in unique_experts:
        row = f"  {expert_names[exp]:<8}"
        for desc, r in stats.items():
            if exp in r["per_expert"]:
                mu, sd = r["per_expert"][exp]
                row += f"  {mu:>6.1f}±{sd:.0f}"
            else:
                row += f"  {'N/A':>8}"
        print(row)

    # LaTeX table snippet
    print(f"\n{'─'*70}")
    print(f"  LATEX TABLE SNIPPET (paste into paper)")
    print(f"{'─'*70}")
    descs = list(stats.keys())
    print(r"  \begin{tabular}{l" + "r"*len(descs) + r"}")
    print(r"  \hline")
    print(f"  Expert & " + " & ".join(descs) + r" \\")
    print(r"  \hline")
    for exp in unique_experts:
        row_parts = []
        for desc in descs:
            r = stats[desc]
            if exp in r["per_expert"]:
                mu, sd = r["per_expert"][exp]
                row_parts.append(f"${mu:.1f}$")
            else:
                row_parts.append("—")
        print(f"  {expert_names[exp]} & " + " & ".join(row_parts) + r" \\")
    print(r"  \hline")
    print(f"  F-stat & " + " & ".join(
        f"${stats[d]['F_stat']:.1f}$" for d in descs) + r" \\")
    print(f"  p-value & " + " & ".join(
        f"${stats[d]['p_anova']:.4f}$" for d in descs) + r" \\")
    print(f"  $\\eta^2$ & " + " & ".join(
        f"${stats[d]['eta2']:.3f}$" for d in descs) + r" \\")
    print(r"  \hline")
    print(r"  \end{tabular}")

    # Paper text suggestion
    print(f"\n{'─'*70}")
    print(f"  SUGGESTED PAPER TEXT")
    print(f"{'─'*70}")
    sig_with_large_effect = [d for d in descs
                              if stats[d]["p_anova"] < 0.05 and stats[d]["eta2"] > 0.1]
    print(f"""
  \"To validate expert chemical specialization quantitatively, we computed
  mutual information between expert assignment and seven RDKit physicochemical
  descriptors, and one-way ANOVA across expert groups. Significant between-expert
  variation was observed for {', '.join(sig_with_large_effect) or '[descriptors]'}
  (all p < 0.05, η² > 0.1), confirming that expert routing captures
  meaningful physicochemical structure beyond random assignment.
  The effect sizes (η²) indicate that expert identity explains
  {', '.join(f'{stats[d]["eta2"]*100:.0f}% of {d} variance' for d in sig_with_large_effect[:3])}.\"
    """)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained MoE-GCN checkpoint (.pt file)")
    parser.add_argument("--dataset",    default="Solubility_AqSolDB",
                        help="TDC dataset name to run inference on (checkpoint mode)")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    if not RDKIT_OK or not SKLEARN_OK or not SCIPY_OK:
        print("Missing dependencies. Install:")
        print("  pip install rdkit scikit-learn scipy")
        return

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Checkpoint mode: {args.checkpoint}")
        try:
            smiles_list, expert_labels = extract_expert_assignments_from_checkpoint(
                args.checkpoint, args.dataset, args.device
            )
            desc_arrays, valid_idx = compute_descriptors(smiles_list)
            expert_labels = expert_labels[valid_idx]
            mode = "checkpoint"
        except Exception as e:
            print(f"Checkpoint mode failed: {e}")
            print("Falling back to demo mode...")
            desc_arrays, expert_labels = demo_mode()
            mode = "demo"
    else:
        if args.checkpoint:
            print(f"⚠️  Checkpoint not found: {args.checkpoint}")
        desc_arrays, expert_labels = demo_mode()
        mode = "demo"

    stats = run_stats(desc_arrays, expert_labels)
    if not stats:
        print("No statistics computed — check input data")
        return

    print_report(stats, expert_labels, mode=mode)

    # Save JSON for reference
    out = {
        desc: {
            "MI": float(r["MI"]),
            "F_stat": float(r["F_stat"]),
            "p_anova": float(r["p_anova"]),
            "p_kruskal": float(r["p_kruskal"]),
            "eta2": float(r["eta2"]),
        }
        for desc, r in stats.items()
    }
    with open("expert_specialization_stats.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Stats saved to expert_specialization_stats.json")


if __name__ == "__main__":
    main()
