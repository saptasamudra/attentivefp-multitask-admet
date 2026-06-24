"""
Expanded Descriptor η² Analysis
Tests MoE routing alignment vs GCN k-means across 12 molecular descriptors
Compares: MoE η² vs GCN η² to find axes where MoE > GCN
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
from pathlib import Path

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# scipy
from scipy.stats import f_oneway

print("Script loaded. Paste this into your D:\\molprop_project directory.")
print("=" * 70)

# ── Descriptor definitions ─────────────────────────────────────────────────
DESCRIPTOR_FUNCS = {
    'LogP':        lambda m: Crippen.MolLogP(m),
    'MW':          lambda m: Descriptors.MolWt(m),
    'TPSA':        lambda m: Descriptors.TPSA(m),
    'HBD':         lambda m: Descriptors.NumHDonors(m),
    'HBA':         lambda m: Descriptors.NumHAcceptors(m),
    'RotBonds':    lambda m: Descriptors.NumRotatableBonds(m),
    'ArRings':     lambda m: rdMolDescriptors.CalcNumAromaticRings(m),
    'RingCount':   lambda m: Descriptors.RingCount(m),
    'HeavyAtoms':  lambda m: Descriptors.HeavyAtomCount(m),
    'FracCSP3':    lambda m: Descriptors.FractionCSP3(m),
    'QED':         lambda m: Descriptors.qed(m),
    'MolRefract':  lambda m: Crippen.MolMR(m),
}

def compute_descriptors(smiles_list):
    """Compute all descriptors for a list of SMILES. Returns (valid_indices, desc_dict)."""
    valid_idx = []
    desc_arrays = {k: [] for k in DESCRIPTOR_FUNCS}
    
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            vals = {k: fn(mol) for k, fn in DESCRIPTOR_FUNCS.items()}
            # Check all finite
            if all(np.isfinite(v) for v in vals.values()):
                valid_idx.append(i)
                for k, v in vals.items():
                    desc_arrays[k].append(v)
        except Exception:
            continue
    
    desc_np = {k: np.array(v) for k, v in desc_arrays.items()}
    return valid_idx, desc_np


def eta_squared(groups, values):
    """One-way ANOVA η² = SS_between / SS_total."""
    unique = np.unique(groups)
    grand_mean = np.mean(values)
    ss_total = np.sum((values - grand_mean) ** 2)
    if ss_total < 1e-10:
        return 0.0
    ss_between = sum(
        np.sum(groups == g) * (np.mean(values[groups == g]) - grand_mean) ** 2
        for g in unique
    )
    return float(ss_between / ss_total)


def get_moe_routing(model, loader, device, n_experts):
    """Extract per-molecule expert assignments from MoE model."""
    model.eval()
    all_assignments = []
    all_weights = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Forward pass — need routing weights
            # Assumes model has a .get_routing_weights() method
            # or we hook into the MoE layer
            try:
                weights = model.get_routing_weights(batch)  # [B, n_experts]
                assignments = weights.argmax(dim=-1).cpu().numpy()
                all_assignments.append(assignments)
                all_weights.append(weights.cpu().numpy())
            except AttributeError:
                # Fallback: run forward and grab from model.last_routing
                _ = model(batch)
                weights = model.last_routing  # [B, n_experts]
                assignments = weights.argmax(dim=-1).cpu().numpy()
                all_assignments.append(assignments)
                all_weights.append(weights.cpu().numpy())
    
    return np.concatenate(all_assignments), np.concatenate(all_weights, axis=0)


def gcn_kmeans_clusters(embeddings, n_clusters):
    """K-means on GCN embeddings."""
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(emb_scaled)
    return labels


def analyze_dataset(dataset_name, smiles, moe_assignments, gcn_embs_list, 
                    n_experts, valid_moe_idx):
    """
    Full η² comparison: MoE routing vs GCN k-means for all descriptors.
    gcn_embs_list: list of np arrays (one per seed)
    valid_moe_idx: indices into smiles that are valid for MoE
    """
    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*70}")
    
    # Compute descriptors on valid MoE molecules
    smiles_subset = [smiles[i] for i in valid_moe_idx]
    valid_desc_idx, desc_np = compute_descriptors(smiles_subset)
    
    if len(valid_desc_idx) < 50:
        print("  Too few valid molecules — skipping")
        return None
    
    print(f"  Molecules with all descriptors: {len(valid_desc_idx)}")
    
    # Align MoE assignments to descriptor-valid subset
    moe_assign_sub = moe_assignments[valid_desc_idx]
    
    # Check routing collapse
    n_active = len(np.unique(moe_assign_sub))
    if n_active < 2:
        print("  Routing collapsed — skipping")
        return None
    print(f"  Active experts: {n_active}/{n_experts}")
    
    # GCN k-means (average across seeds)
    gcn_labels_list = []
    for embs in gcn_embs_list:
        embs_sub = embs[valid_desc_idx]
        labels = gcn_kmeans_clusters(embs_sub, n_clusters=n_experts)
        gcn_labels_list.append(labels)
    
    # Compute η² for each descriptor
    results = {}
    print(f"\n  {'Descriptor':<14} {'MoE η²':>8} {'GCN η²':>8} {'Delta':>8} {'Winner'}")
    print(f"  {'-'*55}")
    
    moe_wins = 0
    gcn_wins = 0
    
    for desc_name, desc_vals in desc_np.items():
        moe_eta2 = eta_squared(moe_assign_sub, desc_vals)
        
        # Average GCN η² across seeds
        gcn_eta2_list = [eta_squared(gl, desc_vals) for gl in gcn_labels_list]
        gcn_eta2 = np.mean(gcn_eta2_list)
        
        delta = moe_eta2 - gcn_eta2
        winner = "MoE ✓" if delta > 0.005 else ("GCN" if delta < -0.005 else "tie")
        
        if delta > 0.005:
            moe_wins += 1
        elif delta < -0.005:
            gcn_wins += 1
        
        results[desc_name] = {
            'moe_eta2': round(moe_eta2, 4),
            'gcn_eta2': round(gcn_eta2, 4),
            'delta': round(delta, 4)
        }
        
        print(f"  {desc_name:<14} {moe_eta2:>8.4f} {gcn_eta2:>8.4f} "
              f"{'%+.4f' % delta:>8} {winner}")
    
    print(f"\n  MoE wins: {moe_wins}/12   GCN wins: {gcn_wins}/12")
    
    # Find best MoE descriptor
    best_desc = max(results, key=lambda k: results[k]['delta'])
    best_delta = results[best_desc]['delta']
    print(f"  Best MoE axis: {best_desc} (Δ={best_delta:+.4f})")
    
    return {
        'dataset': dataset_name,
        'n_molecules': len(valid_desc_idx),
        'n_active_experts': n_active,
        'descriptors': results,
        'moe_wins': moe_wins,
        'gcn_wins': gcn_wins,
        'best_moe_descriptor': best_desc,
        'best_moe_delta': best_delta
    }


# ── Integration stub ───────────────────────────────────────────────────────
# This is what you paste into your existing benchmark pipeline.
# Replace the sections below with your actual model/data loading code.

INTEGRATION_TEMPLATE = '''
# ─── ADD THIS TO YOUR EXISTING SCRIPT ────────────────────────────────────
# Place after model training, before evaluation

from expanded_descriptor_eta2 import (
    compute_descriptors, eta_squared, gcn_kmeans_clusters, analyze_dataset,
    DESCRIPTOR_FUNCS
)

# 1. Get MoE routing assignments
#    Replace with your actual loader + model
moe_assignments, moe_weights = get_moe_routing(
    model=trained_moe_model,
    loader=full_dataloader,        # all molecules, not just test
    device=device,
    n_experts=config.n_experts
)

# 2. Get GCN embeddings (3 seeds)
gcn_embs_list = []
for seed in [0, 1, 2]:
    gcn_emb = train_vanilla_gcn_get_embeddings(
        smiles_list, labels, seed=seed
    )
    gcn_embs_list.append(gcn_emb)

# 3. Run analysis
result = analyze_dataset(
    dataset_name=dataset_name,
    smiles=smiles_list,
    moe_assignments=moe_assignments,
    gcn_embs_list=gcn_embs_list,
    n_experts=config.n_experts,
    valid_moe_idx=list(range(len(smiles_list)))  # adjust if filtered
)
'''

if __name__ == '__main__':
    print("\nThis module provides analysis functions.")
    print("Import and use analyze_dataset() in your benchmark script.")
    print("\nIntegration template:")
    print(INTEGRATION_TEMPLATE)
