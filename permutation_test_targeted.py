"""
permutation_test_targeted.py
Permutation null test for MW, RingCount, MolRefract
Only on datasets where MoE won those descriptors

Place in D:\molprop_project\ and run:
    python permutation_test_targeted.py
"""

import os, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from torch_geometric.data import Data
from torch_geometric.data import DataLoader as GeoLoader
from torch_geometric.nn import GCNConv, global_mean_pool

print("Imports OK")

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# Only test descriptors and datasets where MoE showed wins
# ══════════════════════════════════════════════════════════════════════════

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PERMS    = 1000   # permutations per test
MOE_EPOCHS = 50

# (tdc_name, display_name, [descriptors to test])
# Only datasets where MoE won at least one target descriptor
TARGETS = [
    ("caco2_wang",               "Caco-2",    ["MW", "HBA", "RingCount", "HeavyAtoms", "MolRefract"]),
    ("solubility_aqsoldb",       "AqSolDB",   ["MW", "RingCount", "HeavyAtoms", "MolRefract"]),
    ("ld50_zhu",                 "LD50",      ["LogP", "MW", "RingCount", "HeavyAtoms", "MolRefract"]),
    ("half_life_obach",          "Half-Life", ["LogP", "HBD", "ArRings", "FracCSP3", "QED"]),
    ("lipophilicity_astrazeneca","Lipo",      ["LogP"]),
]

DEFAULT_PARAMS = {
    'hidden': 256, 'num_layers': 3, 'dropout': 0.1,
    'num_experts': 4, 'top_k': 2, 'lr': 5e-4, 'weight_decay': 1e-5
}

DESCRIPTOR_FUNCS = {
    'LogP':       lambda m: Crippen.MolLogP(m),
    'MW':         lambda m: Descriptors.MolWt(m),
    'TPSA':       lambda m: Descriptors.TPSA(m),
    'HBD':        lambda m: Descriptors.NumHDonors(m),
    'HBA':        lambda m: Descriptors.NumHAcceptors(m),
    'RotBonds':   lambda m: Descriptors.NumRotatableBonds(m),
    'ArRings':    lambda m: rdMolDescriptors.CalcNumAromaticRings(m),
    'RingCount':  lambda m: Descriptors.RingCount(m),
    'HeavyAtoms': lambda m: Descriptors.HeavyAtomCount(m),
    'FracCSP3':   lambda m: Descriptors.FractionCSP3(m),
    'QED':        lambda m: Descriptors.qed(m),
    'MolRefract': lambda m: Crippen.MolMR(m),
}

print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════
# MODEL  (same as expanded_specialization_test.py)
# ══════════════════════════════════════════════════════════════════════════

class MoELayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.experts     = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            for _ in range(num_experts)
        ])
        self.gate        = nn.Linear(in_dim, num_experts)
        self.last_weights = None

    def forward(self, x):
        gate_logits = self.gate(x)
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = torch.zeros_like(gate_logits).scatter_(
            1, topk_idx, F.softmax(topk_vals, dim=-1)
        )
        self.last_weights = weights.detach()
        load = weights.mean(0)
        balance_loss = self.num_experts * (load * load).sum()
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        out = (weights.unsqueeze(-1) * expert_out).sum(dim=1)
        return out, balance_loss


class MoEGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_experts, top_k, num_tasks):
        super().__init__()
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe  = MoELayer(hidden, hidden, num_experts, top_k)
        self.head = nn.Linear(hidden, num_tasks)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x, bal_loss = self.moe(x)
        return self.head(x), bal_loss


# ══════════════════════════════════════════════════════════════════════════
# DATA + DESCRIPTOR HELPERS
# ══════════════════════════════════════════════════════════════════════════

def load_tdc_smiles(tdc_name):
    from tdc.single_pred import ADME, Tox
    try:
        data = ADME(name=tdc_name)
    except Exception:
        data = Tox(name=tdc_name)
    df     = data.get_data()
    smiles = df['Drug'].tolist()
    labels = df['Y'].values.astype(float)
    return smiles, labels


def smiles_to_pyg(smiles_list, labels):
    dataset, valid_idx = [], []
    for i, (smi, lab) in enumerate(zip(smiles_list, labels)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        feats = []
        for atom in mol.GetAtoms():
            feats.append([
                atom.GetAtomicNum(), atom.GetDegree(),
                atom.GetFormalCharge(), int(atom.GetHybridization()),
                int(atom.GetIsAromatic()), atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
            ])
        x = torch.tensor(feats, dtype=torch.float)
        edges = []
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges += [[u, v], [v, u]]
        if not edges:
            continue
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        dataset.append(Data(x=x, edge_index=edge_index,
                            y=torch.tensor([lab], dtype=torch.float)))
        valid_idx.append(i)
    return dataset, valid_idx


def compute_descriptors(smiles_list, desc_names):
    valid_idx  = []
    desc_lists = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            row = {k: DESCRIPTOR_FUNCS[k](mol) for k in desc_names}
            if all(np.isfinite(v) for v in row.values()):
                valid_idx.append(i)
                for k, v in row.items():
                    desc_lists[k].append(v)
        except Exception:
            continue
    return valid_idx, {k: np.array(v) for k, v in desc_lists.items()}


def eta_squared(groups, values):
    groups     = np.array(groups)
    values     = np.array(values, dtype=float)
    grand_mean = np.mean(values)
    ss_total   = np.sum((values - grand_mean) ** 2)
    if ss_total < 1e-10:
        return 0.0
    ss_between = sum(
        np.sum(groups == g) * (np.mean(values[groups == g]) - grand_mean) ** 2
        for g in np.unique(groups)
    )
    return float(ss_between / ss_total)


# ══════════════════════════════════════════════════════════════════════════
# TRAIN MoE + EXTRACT ROUTING
# ══════════════════════════════════════════════════════════════════════════

def load_best_params(tdc_name):
    for rf in ["results_moegcn_tdc_benchmark.json",
               "results_moegcn_regr.json",
               "results_moegcn_classif.json"]:
        if not os.path.exists(rf):
            continue
        with open(rf) as f:
            res = json.load(f)
        for key in res:
            if tdc_name.lower() in key.lower() or key.lower() in tdc_name.lower():
                bp = res[key].get('best_params')
                if bp:
                    print(f"    Loaded params from {rf} [{key}]")
                    return bp
    print(f"    Using default params")
    return DEFAULT_PARAMS.copy()


def train_and_extract_routing(dataset, params, epochs):
    in_dim = dataset[0].x.shape[1]
    model  = MoEGCN(
        in_dim=in_dim, hidden=params['hidden'],
        num_layers=params['num_layers'], dropout=params['dropout'],
        num_experts=params['num_experts'], top_k=params['top_k'],
        num_tasks=1
    ).to(DEVICE)

    opt    = torch.optim.Adam(model.parameters(),
                               lr=params['lr'], weight_decay=params['weight_decay'])
    loader = GeoLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for ep in range(epochs):
        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            out, bal = model(batch)
            labels   = batch.y.float().squeeze()
            mask     = ~torch.isnan(labels)
            if mask.sum() == 0:
                continue
            loss = F.mse_loss(out.squeeze()[mask], labels[mask]) + 0.01 * bal
            loss.backward()
            opt.step()

    model.eval()
    all_w = []
    with torch.no_grad():
        for batch in GeoLoader(dataset, batch_size=256, shuffle=False):
            batch = batch.to(DEVICE)
            _     = model(batch)
            all_w.append(model.moe.last_weights.cpu().numpy())

    weights = np.concatenate(all_w, axis=0)
    assigns = weights.argmax(axis=1)
    return assigns, weights


# ══════════════════════════════════════════════════════════════════════════
# PERMUTATION TEST
# ══════════════════════════════════════════════════════════════════════════

def permutation_test(assignments, desc_vals, n_perms=1000, verbose=True):
    """
    H0: routing is random with respect to descriptor values
    Observed η² vs null distribution of η² under random permutation
    Returns: observed_eta2, p_value, null_distribution
    """
    observed = eta_squared(assignments, desc_vals)
    null     = np.zeros(n_perms)

    for i in range(n_perms):
        perm_assign = np.random.permutation(assignments)
        null[i]     = eta_squared(perm_assign, desc_vals)
        if verbose and (i+1) % 200 == 0:
            print(f"      {i+1}/{n_perms} permutations...", flush=True)

    p_value    = float(np.mean(null >= observed))
    percentile = float(np.mean(null < observed) * 100)

    return observed, p_value, percentile, null


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════

def plot_null_distributions(all_perm_results, save_path="permutation_null_plots.png"):
    """Plot observed η² vs null distribution for each test."""
    n = len(all_perm_results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, res in zip(axes, all_perm_results):
        null = res['null']
        obs  = res['observed_eta2']
        p    = res['p_value']

        ax.hist(null, bins=50, color='#90CAF9', edgecolor='white',
                linewidth=0.3, alpha=0.9, label='Null distribution')
        ax.axvline(obs, color='#D32F2F', linewidth=2,
                   label=f'Observed η²={obs:.3f}')
        ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1],
                         obs, max(null.max(), obs)*1.1,
                         alpha=0.15, color='#D32F2F')

        stars = sig_stars(p)
        ax.set_title(f"{res['dataset']} — {res['descriptor']}\n"
                     f"p={p:.4f} {stars}",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('η²', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for ax in axes[len(all_perm_results):]:
        ax.set_visible(False)

    plt.suptitle('Permutation Null Tests — MoE Routing vs Random\n'
                 'Red line = observed η²  |  *** p<0.001',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


def plot_summary_heatmap(all_perm_results, save_path="permutation_summary_heatmap.png"):
    """Heatmap of p-values across dataset × descriptor."""
    datasets = sorted(set(r['dataset'] for r in all_perm_results))
    descs    = sorted(set(r['descriptor'] for r in all_perm_results))

    # Build matrix
    p_matrix   = np.ones((len(descs), len(datasets)))
    eta_matrix = np.zeros((len(descs), len(datasets)))

    lookup = {(r['dataset'], r['descriptor']): r for r in all_perm_results}

    for i, desc in enumerate(descs):
        for j, ds in enumerate(datasets):
            key = (ds, desc)
            if key in lookup:
                p_matrix[i, j]   = lookup[key]['p_value']
                eta_matrix[i, j] = lookup[key]['observed_eta2']

    # Plot -log10(p) so smaller p = brighter
    log_p = -np.log10(np.clip(p_matrix, 1e-4, 1.0))

    fig, ax = plt.subplots(figsize=(len(datasets)*2 + 2, len(descs)*0.7 + 2))
    im = ax.imshow(log_p, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=4)

    ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, fontsize=10)
    ax.set_yticks(range(len(descs)));   ax.set_yticklabels(descs, fontsize=10)

    for i, desc in enumerate(descs):
        for j, ds in enumerate(datasets):
            key = (ds, desc)
            if key in lookup:
                p   = lookup[key]['p_value']
                eta = lookup[key]['observed_eta2']
                stars = sig_stars(p)
                txt   = f"{stars}\nη²={eta:.3f}"
                color = 'white' if log_p[i, j] > 2 else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=7, color=color)
            else:
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=9, color='gray')

    plt.colorbar(im, ax=ax, label='-log₁₀(p)  [brighter = more significant]')
    ax.set_title('Permutation Test Significance\n'
                 '*** p<0.001  ** p<0.01  * p<0.05  ns = not significant',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*70)
    print("  Permutation Null Tests — Targeted Descriptor Analysis")
    print(f"  N_PERMS={N_PERMS} per test")
    print("="*70)

    all_perm_results = []
    json_out         = []

    for tdc_name, display_name, desc_list in TARGETS:
        print(f"\n{'='*70}")
        print(f"  {display_name}  ({tdc_name})")
        print(f"  Testing: {desc_list}")
        print(f"{'='*70}")

        # Load data
        smiles, labels = load_tdc_smiles(tdc_name)
        dataset, valid_pyg_idx = smiles_to_pyg(smiles, labels)
        valid_smiles = [smiles[i] for i in valid_pyg_idx]

        # Compute only needed descriptors
        valid_desc_idx, desc_np = compute_descriptors(valid_smiles, desc_list)
        if len(valid_desc_idx) < 50:
            print("  Too few molecules — skipping")
            continue

        dataset_sub = [dataset[i] for i in valid_desc_idx]
        print(f"  Molecules: {len(dataset_sub)}")

        # Train MoE and get routing
        params   = load_best_params(tdc_name)
        n_experts = params.get('num_experts', 4)
        print(f"  Training MoEGCN ({MOE_EPOCHS} epochs, {n_experts} experts)...")
        assigns, weights = train_and_extract_routing(dataset_sub, params, MOE_EPOCHS)

        n_active = len(np.unique(assigns))
        print(f"  Active experts: {n_active}/{n_experts}")
        if n_active < 2:
            print("  Routing collapsed — skipping")
            continue

        # Run permutation test for each descriptor
        print(f"\n  {'Descriptor':<14} {'Obs η²':>8} {'p-value':>9} {'%ile':>7} Sig")
        print(f"  {'-'*50}")

        for desc_name in desc_list:
            if desc_name not in desc_np:
                continue
            desc_vals = desc_np[desc_name]

            obs, p, pct, null = permutation_test(
                assigns, desc_vals, n_perms=N_PERMS, verbose=False
            )

            stars = sig_stars(p)
            print(f"  {desc_name:<14} {obs:>8.4f} {p:>9.4f} {pct:>6.1f}% {stars}")

            result = {
                'dataset':        display_name,
                'tdc_name':       tdc_name,
                'descriptor':     desc_name,
                'observed_eta2':  round(obs, 4),
                'p_value':        round(p, 4),
                'percentile':     round(pct, 2),
                'significance':   stars,
                'n_molecules':    len(dataset_sub),
                'n_active_experts': int(n_active),
                'null':           null  # keep for plotting
            }
            all_perm_results.append(result)
            json_out.append({k: v for k, v in result.items() if k != 'null'})

    # ── Final table ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PERMUTATION TEST SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Dataset':<14} {'Descriptor':<14} {'η²':>7} {'p':>8} Sig")
    print(f"  {'-'*55}")

    for r in all_perm_results:
        print(f"  {r['dataset']:<14} {r['descriptor']:<14} "
              f"{r['observed_eta2']:>7.4f} {r['p_value']:>8.4f} {r['significance']}")

    # ── Paper-ready statements ─────────────────────────────────────────────
    sig_results = [r for r in all_perm_results if r['p_value'] < 0.05]
    highly_sig  = [r for r in all_perm_results if r['p_value'] < 0.001]

    print(f"\n  Significant results (p<0.05): {len(sig_results)}/{len(all_perm_results)}")
    print(f"  Highly significant (p<0.001): {len(highly_sig)}/{len(all_perm_results)}")

    if highly_sig:
        print(f"\n  PAPER-READY CLAIM:")
        desc_counts = defaultdict(list)
        for r in highly_sig:
            desc_counts[r['descriptor']].append(r['dataset'])
        for desc, datasets in desc_counts.items():
            print(f"    MoE routing shows significant {desc} alignment "
                  f"(p<0.001) in: {', '.join(datasets)}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    with open('permutation_targeted_results.json', 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f"\n[SAVED] permutation_targeted_results.json")

    # ── Plots ──────────────────────────────────────────────────────────────
    try:
        plot_null_distributions(all_perm_results)
        plot_summary_heatmap(all_perm_results)
    except Exception as e:
        print(f"[Plot error] {e}")
        import traceback; traceback.print_exc()

    # ── Interpretation ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  WHAT THIS MEANS FOR YOUR PAPER")
    print(f"{'='*70}")
    print("""
  For every descriptor where p < 0.001 (***):
    → MoE routing is HIGHLY unlikely to be random with respect to that descriptor
    → You can claim: "MoE experts spontaneously partition chemical space
      along [descriptor] axes (permutation test, p<0.001)"

  For p < 0.05 (*):
    → Significant but weaker — mention but don't lead with it

  For p > 0.05 (ns):
    → Do not claim alignment with that descriptor

  Combining with η² comparison from expanded_specialization_test.py:
    → If MoE η² > GCN η² AND p<0.001: STRONGEST claim
    → If MoE η² > GCN η² AND p<0.05:  Moderate claim
    → If MoE η² < GCN η² AND p<0.001: Routing non-random but GCN clusters
                                        more physicochemically structured
                                        (use Fix 3 reframe)
""")


if __name__ == '__main__':
    main()

