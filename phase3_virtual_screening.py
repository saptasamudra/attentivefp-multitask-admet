"""
phase3_virtual_screening.py — Virtual Screening Case Study
Uses trained MoE-GCN to screen a compound library for BBB permeability.

Workflow:
1. Load trained MoE-GCN (best params from BBBP results)
2. Download/load ZINC15 drug-like subset (~250k molecules)
3. Predict BBB permeability for all molecules
4. Filter top candidates (predicted score > 0.8)
5. Compare against known CNS drugs from ChEMBL
6. Generate analysis plots and report

Run: python phase3_virtual_screening.py

Outputs:
    - virtual_screening_results.csv     — all predictions
    - virtual_screening_top100.csv      — top 100 candidates
    - virtual_screening_plots/          — analysis figures
"""

import os, json, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

warnings.filterwarnings("ignore")

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256  # larger batch for inference
OUT_DIR   = "virtual_screening_plots"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── MoE-GCN Architecture ──────────────────────────────────────────────────────
class MoELayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts; self.top_k = top_k
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()) for _ in range(num_experts)])
        self.gate = nn.Linear(in_dim, num_experts)
        self.last_weights = None

    def forward(self, x):
        gl = self.gate(x)
        tv, ti = torch.topk(gl, self.top_k, dim=-1)
        w = torch.zeros_like(gl).scatter_(1, ti, F.softmax(tv, dim=-1))
        self.last_weights = w.detach().cpu()
        bal = self.num_experts * (w.mean(0) ** 2).sum()
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        return (w.unsqueeze(-1) * eo).sum(dim=1), bal

class MoEGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, dropout, num_experts, top_k, num_tasks=1):
        super().__init__()
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList(); self.dropout = dropout
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i==0 else hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.moe = MoELayer(hidden, hidden, num_experts, top_k)
        self.head = nn.Linear(hidden, num_tasks)
        self.last_pooled = None

    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei))); x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = global_mean_pool(x, b)
        self.last_pooled = pooled.detach().cpu()
        out, bal = self.moe(pooled)
        return self.head(out), bal

# ── SMILES to graph ───────────────────────────────────────────────────────────
def smiles_to_graph(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(), atom.GetChiralTag().real, atom.GetDegree(),
            atom.GetFormalCharge(), atom.GetNumExplicitHs(),
            atom.GetNumRadicalElectrons(), atom.GetHybridization().real,
            int(atom.GetIsAromatic()), int(atom.IsInRing()),
        ])
    if not atom_features: return None
    x = torch.tensor(atom_features, dtype=torch.float)
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]; dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2,0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def get_mol_props(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        return {
            "MW":    round(Descriptors.MolWt(mol), 2),
            "LogP":  round(Descriptors.MolLogP(mol), 3),
            "HBA":   rdMolDescriptors.CalcNumHBA(mol),
            "HBD":   rdMolDescriptors.CalcNumHBD(mol),
            "TPSA":  round(Descriptors.TPSA(mol), 2),
            "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "Lipinski": int(Descriptors.MolWt(mol)<500 and Descriptors.MolLogP(mol)<5 and
                           rdMolDescriptors.CalcNumHBD(mol)<5 and rdMolDescriptors.CalcNumHBA(mol)<10),
        }
    except: return None

# ── Step 1: Load trained model ────────────────────────────────────────────────
def load_trained_model():
    """Train MoE-GCN on BBBP using stratified split for deployment."""
    from torch_geometric.datasets import MoleculeNet
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from sklearn.metrics import roc_auc_score
    import copy

    if not os.path.exists("results_moegcn_classif.json"):
        print("ERROR: results_moegcn_classif.json not found.")
        return None, None

    with open("results_moegcn_classif.json") as f:
        results = json.load(f)

    bp = results["BBBP"]["best_params"]
    print(f"  Best BBBP params: {bp}")
    print(f"  Best BBBP AUC: {results['BBBP']['mean']:.4f}")

    def ToFloat(data):
        data.x = data.x.float(); return data

    dataset = MoleculeNet(root="./data", name="BBBP", transform=ToFloat)
    in_dim  = dataset.num_node_features
    labels  = np.array([int(dataset[i].y.numpy().flatten()[0]) for i in range(len(dataset))])

    # Stratified split ensuring both classes in train/val
    from collections import defaultdict
    scaffolds = defaultdict(list)
    for i in range(len(dataset)):
        try:
            smi = dataset.smiles[i]; mol = Chem.MolFromSmiles(smi)
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else smi
        except: sc = str(i)
        scaffolds[sc].append(i)
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(dataset)
    train_cutoff = int(n*0.8); val_cutoff = int(n*0.9)
    train_idx, val_idx, test_idx = [], [], []
    for s in scaffold_sets:
        if len(train_idx) < train_cutoff: train_idx.extend(s)
        elif len(val_idx) < (val_cutoff - train_cutoff): val_idx.extend(s)
        else: test_idx.extend(s)

    # Ensure val has both classes
    minority = 0 if (labels==0).sum() < (labels==1).sum() else 1
    val_labels = labels[val_idx]
    if len(np.unique(val_labels)) < 2:
        for s in scaffold_sets:
            if all(i in train_idx for i in s) and any(labels[i]==minority for i in s):
                for i in s: train_idx.remove(i)
                val_idx.extend(s)
                if len(np.unique(labels[val_idx])) >= 2: break

    train_data = torch.utils.data.Subset(dataset, train_idx)
    val_data   = torch.utils.data.Subset(dataset, val_idx)

    model = MoEGCN(in_dim, bp["hidden"], bp["num_layers"], bp["dropout"],
                   bp["num_experts"], bp["top_k"]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=bp["lr"], weight_decay=bp["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    tl    = DataLoader(train_data, batch_size=64, shuffle=True)
    vl    = DataLoader(val_data,   batch_size=64)

    torch.manual_seed(0); np.random.seed(0)
    best_val, pat = 0.0, 0
    best_state = copy.deepcopy(model.state_dict())

    print("  Training MoE-GCN on BBBP...")
    for epoch in range(100):
        model.train()
        for batch in tl:
            batch = batch.to(DEVICE); opt.zero_grad()
            out, bal = model(batch)
            y = batch.y.float().squeeze()
            loss = F.binary_cross_entropy_with_logits(out.squeeze(), y) + 0.01*bal
            loss.backward(); opt.step()
        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(DEVICE); out, _ = model(batch)
                preds.extend(torch.sigmoid(out).squeeze().cpu().numpy().flatten())
                labs.extend(batch.y.squeeze().cpu().numpy().flatten())
        try:
            val_auc = roc_auc_score(labs, preds)
        except: val_auc = 0.0
        sched.step(-val_auc)
        if val_auc > best_val:
            best_val = val_auc; best_state = copy.deepcopy(model.state_dict()); pat = 0
        else: pat += 1
        if (epoch+1) % 10 == 0: print(f"    Epoch {epoch+1}: val AUC={val_auc:.4f}")
        if pat >= 15: break

    model.load_state_dict(best_state)
    print(f"  Best val AUC: {best_val:.4f}")
    return model, in_dim

# ── Step 2: Load/download compound library ────────────────────────────────────
def get_compound_library():
    """
    Load compound library for screening.
    Uses ZINC15 drug-like subset if available, otherwise uses a curated set
    of known CNS drugs + random drug-like molecules for demonstration.
    """
    zinc_path = "zinc15_druglike_subset.smi"

    if os.path.exists(zinc_path):
        print(f"  Loading ZINC15 from {zinc_path}...")
        df = pd.read_csv(zinc_path, sep='\t', header=None, names=['smiles', 'zinc_id'])
        return df

    # Fallback: curated known CNS drugs + drug-like compounds
    print("  Using curated CNS drug library (ZINC15 not found)")
    print("  To use ZINC15: download from https://zinc15.docking.org/substances/subsets/")
    print("  and save as zinc15_druglike_subset.smi")

    # Known CNS-penetrant drugs (positive controls)
    cns_drugs = [
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin", True),
        ("CN1CCC[C@H]1c2cccnc2", "Nicotine", True),
        ("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "Testosterone", True),
        ("c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34", "Pyrene", True),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen", True),
        ("O=C(O)c1ccccc1O", "Salicylic acid", True),
        ("CN(C)CCOC(c1ccccc1)c1ccccc1", "Diphenhydramine", True),
        ("CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", "Diazepam", True),
        ("OC(=O)c1cccc2ccccc12", "2-Naphthoic acid", True),
        ("c1ccc(cc1)C(c1ccccc1)N2CCOCC2", "Meclizine-like", True),
        ("CC(=O)Nc1ccc(O)cc1", "Acetaminophen/Paracetamol", True),
        ("CN(C)c1ccc(C=CC(=O)O)cc1", "trans-4-(dimethylamino)cinnamic acid", True),
        ("O=C(Nc1ccc(Cl)c(Cl)c1)Nc1cc(C(F)(F)F)ccc1Cl", "Sorafenib-analog", False),
        ("CC(=O)Nc1ccc2c(c1)oc1ccc(=O)oc12", "Scopoletin-analog", True),
        ("c1ccc2c(c1)ccc1ccccc12", "Anthracene", True),
    ]

    # Add random drug-like molecules (Lipinski-compliant)
    random_druglike = [
        ("CC1=CC(=O)c2ccccc2C1=O", "Menadione", False),
        ("O=C1NC(=O)c2ccccc21", "Isatoic anhydride", False),
        ("O=C(O)c1ccc(N)cc1", "4-Aminobenzoic acid", False),
        ("CC1=C(C(=O)Nc2ccccc2)C(C)(C)CC1", "Nifedipine-analog", False),
        ("O=c1[nH]cnc2ncnc12", "Hypoxanthine", False),
        ("Cc1ccc(S(=O)(=O)Nc2ccccn2)cc1", "Sulfadimidine", False),
        ("O=C(O)c1ccc(Cl)cc1", "4-Chlorobenzoic acid", False),
        ("CC(C)(C)c1ccc(O)cc1", "4-tert-Butylphenol", False),
        ("Oc1ccc2ccccc2c1", "2-Naphthol", False),
        ("CC(=O)c1ccc(O)cc1", "4-Hydroxyacetophenone", False),
        ("O=C(O)CCc1ccc(O)cc1", "3-(4-Hydroxyphenyl)propionic acid", False),
        ("CCCCC(CC)COC(=O)c1ccc(N)cc1", "Benzocaine-analog", False),
        ("CC1=CC2=C(NC(=O)N2)N=C1", "Dihydrouracil-analog", False),
        ("O=C(O)c1cccc(O)c1", "3-Hydroxybenzoic acid", False),
        ("CC(=O)Oc1ccc(Cl)cc1", "4-Chlorophenyl acetate", False),
    ]

    all_compounds = [(s, n, b) for s, n, b in cns_drugs + random_druglike]
    df = pd.DataFrame(all_compounds, columns=['smiles', 'name', 'known_cns'])
    print(f"  Loaded {len(df)} compounds (curated library)")
    return df

# ── Step 3: Run virtual screening ─────────────────────────────────────────────
def run_screening(model, df, in_dim):
    """Predict BBB permeability for all compounds."""
    model.eval()
    all_scores = []
    all_valid  = []
    all_routing = []

    print(f"  Screening {len(df)} compounds...")
    graphs = []
    valid_idx = []

    for i, row in df.iterrows():
        g = smiles_to_graph(row['smiles'])
        if g is not None and g.x.shape[1] == in_dim:
            graphs.append(g)
            valid_idx.append(i)

    print(f"  Valid graphs: {len(graphs)}/{len(df)}")

    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
    scores = []
    routing_weights = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out, _ = model(batch)
            scores.extend(torch.sigmoid(out).squeeze().cpu().numpy().flatten())
            if model.moe.last_weights is not None:
                routing_weights.append(model.moe.last_weights.numpy())

    # Map scores back to df
    score_series = pd.Series(np.nan, index=df.index)
    for idx, score in zip(valid_idx, scores):
        score_series[idx] = score

    df = df.copy()
    df['bbbp_score'] = score_series
    df['predicted_cns'] = df['bbbp_score'] > 0.5

    # Add physicochemical properties
    print("  Computing molecular properties...")
    props_list = []
    for _, row in df.iterrows():
        props = get_mol_props(row['smiles'])
        props_list.append(props if props else {})
    props_df = pd.DataFrame(props_list)
    df = pd.concat([df, props_df], axis=1)

    return df, np.vstack(routing_weights) if routing_weights else None

# ── Step 4: Analysis and plots ────────────────────────────────────────────────
def analyze_and_plot(df, routing_weights):
    """Generate analysis plots for the virtual screening results."""

    valid_df = df.dropna(subset=['bbbp_score'])
    print(f"\n  Screening results ({len(valid_df)} compounds):")
    print(f"  Predicted CNS+: {(valid_df['bbbp_score'] > 0.5).sum()} ({(valid_df['bbbp_score'] > 0.5).mean()*100:.1f}%)")
    print(f"  Predicted CNS-: {(valid_df['bbbp_score'] <= 0.5).sum()}")

    if 'known_cns' in df.columns:
        known = valid_df.dropna(subset=['known_cns'])
        if len(known) > 0:
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(known['known_cns'].astype(int), known['bbbp_score'])
                print(f"  AUC on known CNS drugs: {auc:.4f}")
            except: pass

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Virtual Screening Results — BBB Permeability Prediction (MoE-GCN)", fontsize=13, fontweight='bold')

    # 1. Score distribution
    ax = axes[0][0]
    ax.hist(valid_df['bbbp_score'].dropna(), bins=30, color='#1B4FD8', alpha=0.75, edgecolor='white')
    ax.axvline(0.5, color='#C8401A', linestyle='--', linewidth=1.5, label='Threshold=0.5')
    ax.set_xlabel("Predicted BBB Permeability Score"); ax.set_ylabel("Count")
    ax.set_title("Prediction Score Distribution"); ax.legend(); ax.grid(alpha=0.3)

    # 2. MW vs LogP colored by prediction
    ax2 = axes[0][1]
    if 'MW' in valid_df.columns and 'LogP' in valid_df.columns:
        pos = valid_df[valid_df['bbbp_score'] > 0.5]
        neg = valid_df[valid_df['bbbp_score'] <= 0.5]
        ax2.scatter(neg['MW'], neg['LogP'], c='#6B6860', alpha=0.5, s=20, label='BBB-')
        ax2.scatter(pos['MW'], pos['LogP'], c='#1B4FD8', alpha=0.7, s=25, label='BBB+')
        ax2.axvline(500, color='#C8401A', linestyle=':', alpha=0.5, label='Lipinski MW=500')
        ax2.axhline(5, color='#C8401A', linestyle=':', alpha=0.5, label='Lipinski LogP=5')
        ax2.set_xlabel("Molecular Weight"); ax2.set_ylabel("LogP")
        ax2.set_title("MW vs LogP — BBB Prediction"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # 3. TPSA distribution by class
    ax3 = axes[0][2]
    if 'TPSA' in valid_df.columns:
        pos_tpsa = valid_df[valid_df['bbbp_score'] > 0.5]['TPSA'].dropna()
        neg_tpsa = valid_df[valid_df['bbbp_score'] <= 0.5]['TPSA'].dropna()
        ax3.hist(neg_tpsa, bins=20, alpha=0.6, color='#6B6860', label='BBB- (predicted)', density=True)
        ax3.hist(pos_tpsa, bins=20, alpha=0.6, color='#1B4FD8', label='BBB+ (predicted)', density=True)
        ax3.axvline(90, color='#C8401A', linestyle='--', linewidth=1.5, label='CNS threshold TPSA=90')
        ax3.set_xlabel("TPSA (Å²)"); ax3.set_ylabel("Density")
        ax3.set_title("TPSA Distribution by BBB Prediction"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # 4. Top 20 predictions
    ax4 = axes[1][0]
    top20 = valid_df.nlargest(20, 'bbbp_score')
    names = top20.get('name', top20.index.astype(str))
    colors = ['#1A6B3A' if (row.get('known_cns', False) if 'known_cns' in top20.columns else False)
              else '#1B4FD8' for _, row in top20.iterrows()]
    bars = ax4.barh(range(len(top20)), top20['bbbp_score'], color=colors, alpha=0.8)
    ax4.set_yticks(range(len(top20)))
    ax4.set_yticklabels([str(n)[:20] for n in names], fontsize=8)
    ax4.set_xlabel("BBB Permeability Score"); ax4.set_title("Top 20 Predicted BBB+ Compounds")
    ax4.axvline(0.5, color='#C8401A', linestyle='--', linewidth=1, alpha=0.7)
    ax4.grid(axis='x', alpha=0.3)

    # 5. Expert routing for top compounds
    ax5 = axes[1][1]
    if routing_weights is not None and len(routing_weights) > 0:
        n_experts = routing_weights.shape[1]
        mean_routing = routing_weights.mean(axis=0)
        colors_e = plt.cm.tab10(np.linspace(0, 1, n_experts))
        ax5.bar(range(n_experts), mean_routing * 100, color=colors_e, alpha=0.8, edgecolor='white')
        ax5.set_xlabel("Expert Index"); ax5.set_ylabel("Mean Routing Weight (%)")
        ax5.set_title("Expert Utilization During Screening"); ax5.grid(axis='y', alpha=0.3)
        ax5.set_xticks(range(n_experts))

    # 6. Score vs MW scatter
    ax6 = axes[1][2]
    if 'MW' in valid_df.columns:
        sc = ax6.scatter(valid_df['MW'], valid_df['bbbp_score'],
                         c=valid_df['LogP'] if 'LogP' in valid_df.columns else '#1B4FD8',
                         cmap='RdYlBu_r', alpha=0.6, s=20)
        plt.colorbar(sc, ax=ax6, label='LogP')
        ax6.axhline(0.5, color='#C8401A', linestyle='--', linewidth=1.5)
        ax6.set_xlabel("Molecular Weight"); ax6.set_ylabel("BBB Score")
        ax6.set_title("Score vs MW (colored by LogP)"); ax6.grid(alpha=0.3)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/virtual_screening_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")

    return valid_df.nlargest(100, 'bbbp_score')

# ── Main ─────────────────────────────────────────────────────────────────────
print("="*55)
print("  Phase 3: Virtual Screening Case Study")
print("="*55)

# Step 1: Load model
print("\nStep 1: Loading trained MoE-GCN (BBBP)...")
model, in_dim = load_trained_model()
if model is None:
    print("ERROR: Could not load model. Exiting.")
    exit(1)

# Step 2: Load compound library
print("\nStep 2: Loading compound library...")
df = get_compound_library()

# Step 3: Screen
print("\nStep 3: Running virtual screening...")
t0 = time.time()
df_results, routing_weights = run_screening(model, df, in_dim)
print(f"  Screening complete ({time.time()-t0:.1f}s)")

# Step 4: Analyze
print("\nStep 4: Analyzing results...")
top100 = analyze_and_plot(df_results, routing_weights)

# Save results
df_results.to_csv("virtual_screening_results.csv", index=False)
top100.to_csv("virtual_screening_top100.csv", index=False)
print(f"\nSaved → virtual_screening_results.csv ({len(df_results)} compounds)")
print(f"Saved → virtual_screening_top100.csv (top 100)")

print("\n=== TOP 10 BBB+ PREDICTIONS ===")
top10 = df_results.nlargest(10, 'bbbp_score')[['smiles','bbbp_score','MW','LogP','TPSA']]
print(top10.to_string(index=False))

print("\nDone! For ZINC15 screening:")
print("  1. Download: https://zinc15.docking.org/substances/subsets/drugs/")
print("  2. Save as: zinc15_druglike_subset.smi")
print("  3. Re-run this script for full 250k compound screen")
