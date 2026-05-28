# Sparse MoE-GNN for Multi-Task ADMET Prediction

**Saptasamudra Gogoi** · SAMLab, Guizhou University · Supervisor: Prof. Li Yuquan

---

## Overview

This repository benchmarks **Sparse Mixture-of-Experts (MoE) routing** as a universal plug-in enhancement module for graph neural networks on multi-task ADMET molecular property prediction across 10 MoleculeNet datasets.

MoE routing is applied identically to three backbone architectures — **GCN, DMPNN, and GIN** — demonstrating architecture-agnostic improvement with chemical interpretability analysis.

---

## Key Results

| Dataset | DMPNN | MoE-GCN | MoE-DMPNN | AttentiveFP | GROVER† |
|---------|-------|---------|-----------|-------------|---------|
| Tox21 (AUC↑) | 0.727 | **0.745** | 0.731 | 0.746 | 0.831 |
| ToxCast (AUC↑) | 0.633 | **0.647** | 0.643 | 0.675 | 0.737 |
| ESOL (RMSE↓) | 1.324 | **1.067** | 1.105 | 1.061 | 0.983 |
| FreeSolv (RMSE↓) | 5.103 | 3.591 | **3.432** | 2.573 | 1.544 |
| Lipo (RMSE↓) | 0.723 | 0.722 | **0.712** | 0.685 | 0.561 |

† GROVER results from Rong et al. (2020) — pretrained on 10M molecules, shown as reference only.

**MoE-GCN achieves 19.4% RMSE reduction on ESOL and MoE-DMPNN 32.7% on FreeSolv vs plain DMPNN.**

---

## Project Structure

```
molprop_project/
│
├── Core benchmark scripts
│   ├── moegcn_classif.py          # MoE-GCN classification (7 datasets)
│   ├── moegcn_regr.py             # MoE-GCN regression (3 datasets)
│   ├── moedmpnn_classif.py        # MoE-DMPNN classification
│   ├── moedmpnn_regr.py           # MoE-DMPNN regression
│   ├── dmpnn_classif.py           # Plain DMPNN classification baseline
│   ├── dmpnn_regr.py              # Plain DMPNN regression baseline
│   └── toxcast_all.py             # ToxCast multi-model benchmark
│
├── Data preparation
│   ├── fix_bbbp_bace.py           # Stratified scaffold split fix
│   ├── prepare_grover_data.py     # Convert to GROVER CSV format
│   └── attentivefp_remaining.py   # AttentiveFP on remaining datasets
│
├── Post-compute analysis
│   ├── extend_seeds.py            # Extend 3-seed results to 5 seeds
│   ├── compile_results.py         # Final results table + Wilcoxon tests
│   ├── run_grover.py              # GROVER finetuning runner
│   └── save_previous.py          # Manual result checkpointing
│
├── Interpretability & ablations
│   ├── tsne_routing.py            # t-SNE expert routing visualization
│   ├── expert_load_balance.py     # Expert load over training epochs
│   ├── chemical_subgroup.py       # Physicochemical descriptor per expert
│   ├── topk_comparison.py         # K=1,2,4,8 ablation study
│   ├── param_timing.py            # Parameter count + inference timing
│   ├── training_curves.py         # Training/validation loss curves
│   └── regression_fix.py         # Uncertainty weighting for regression
│
├── Results (JSON)
│   ├── results_dmpnn_classif.json
│   ├── results_dmpnn_regr.json
│   ├── results_moegcn_classif.json
│   ├── results_moegcn_regr.json
│   ├── results_moedmpnn_classif.json
│   ├── results_moedmpnn_regr.json
│   └── results_attentivefp.json
│
├── Figures
│   ├── tsne_plots/                # t-SNE routing visualizations
│   ├── training_curves/           # Loss curve plots
│   ├── chemical_subgroup.png      # Expert chemical profile boxplots
│   ├── expert_load_balance.png    # Expert load over epochs
│   ├── topk_comparison.png        # Top-K ablation
│   ├── param_timing.png           # Efficiency comparison
│   └── regression_fix.png        # Uncertainty weighting results
│
└── final_results_table.csv        # Paper-ready results table
```

---

## Setup

```bash
conda create -n moe_admet python=3.10
conda activate moe_admet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install rdkit scikit-learn optuna pandas matplotlib
```

---

## Running the Benchmark

```bash
# Step 1: Run MoE-GCN
python moegcn_classif.py
python moegcn_regr.py

# Step 2: Run MoE-DMPNN
python moedmpnn_classif.py
python moedmpnn_regr.py

# Step 3: Run plain DMPNN baselines
python dmpnn_classif.py
python dmpnn_regr.py

# Step 4: Fix BBBP and BACE
python fix_bbbp_bace.py

# Step 5: Extend to 5 seeds
python extend_seeds.py

# Step 6: Compile final table
python compile_results.py
```

Scripts use JSON checkpointing — safe to interrupt and resume.

---

## MoE Architecture

The sparse MoE layer is inserted between global mean pooling and the task head:

```
G(h) = TopK( W_g * h, K )
y = sum_{i in TopK} w_i * Expert_i(h)
L_bal = E * sum_i (mean_load_i)^2
L_total = L_task + 0.01 * L_bal
```

E = number of experts, K = active experts per forward pass.

---

## Expert Chemical Specialization (ESOL)

| Expert | MW | LogP | TPSA | Character | n |
|--------|-----|------|------|-----------|---|
| E15 | 253.7 | 2.35 | 57.9 | Drug-like polar | 472 |
| E2 | 197.6 | 3.54 | 14.0 | Lipophilic | 304 |
| E4 | 143.9 | 1.71 | 20.5 | Small nonpolar | 159 |
| E10 | 114.1 | 1.21 | 25.2 | Small hydrophilic | 137 |

Experts learn Lipinski-like chemical space partitioning without explicit supervision.

---

## Citation

> Gogoi, S. (2026). Sparse MoE-GNN for Multi-Task ADMET Molecular Property Prediction.
> SAMLab, Guizhou University. Supervisor: Prof. Li Yuquan.

---

## Acknowledgements

Supervised by Prof. Li Yuquan, SAMLab, College of Life Sciences, Guizhou University.
Built with PyTorch Geometric, RDKit, and Optuna.
