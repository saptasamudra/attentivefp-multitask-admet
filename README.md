# MoE-GCN: Mixture-of-Experts Graph Neural Network for ADMET Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" />
  <img src="https://img.shields.io/badge/PyTorch-1.12+-red.svg" />
  <img src="https://img.shields.io/badge/RDKit-2022+-green.svg" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  <img src="https://img.shields.io/badge/Status-Under%20Review-orange.svg" />
</p>

> **Architecture-agnostic Mixture-of-Experts plug-in for molecular graph neural networks with quantified physicochemical expert specialization.**

---

## Overview

This repository contains the full benchmark pipeline for **MoE-GCN**, a universal Mixture-of-Experts (MoE) module that enhances graph neural network backbones for ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) molecular property prediction.

Unlike prior MoE-GNN works that fix a single backbone, our MoE module is **architecture-agnostic** — it plugs into GCN, DMPNN, and GIN without architectural changes, consistently improving performance. We further demonstrate that expert routing **spontaneously learns Lipinski-like chemical space partitioning**, providing quantified interpretability via mutual information and ANOVA (LogP η²=0.33 replicated across two independent datasets).

**Supervised by:** Prof. Li Yuquan, SAMLab, Guizhou University  
**Target venue:** Journal of Cheminformatics

---

## Key Results

### MoleculeNet Benchmarks (vs. Plain DMPNN Baseline)

| Dataset | Metric | GCN | DMPNN | MoE-GCN | MoE-DMPNN | AttentiveFP | GROVER† |
|---------|--------|-----|-------|---------|-----------|-------------|---------|
| ESOL | RMSE↓ | 1.189 | 1.324 | **1.067** | 1.105 | 1.061 | 0.983 |
| FreeSolv | RMSE↓ | 4.231 | 5.103 | 3.591 | **3.432** | 2.573 | 1.544 |
| Lipophilicity | RMSE↓ | 0.741 | 0.723 | 0.722 | **0.712** | 0.685 | 0.561 |
| Tox21 | AUC↑ | 0.741 | 0.727 | **0.745** | 0.731 | 0.746 | 0.831 |
| ToxCast | AUC↑ | 0.638 | 0.633 | **0.647** | 0.643 | 0.675 | 0.737 |

† GROVER uses published numbers from Rong et al. (2020); pretrained on 10M molecules — shown as reference only.

**MoE-GCN achieves 19.4% RMSE reduction on ESOL and MoE-DMPNN achieves 32.7% on FreeSolv vs. plain DMPNN.**  
Wilcoxon signed-rank test across regression datasets: p = 0.0246 (statistically significant).

---

### TDC ADMET Benchmark Highlights (22 Datasets)

| Category | Dataset | MoE-GCN | Metric |
|----------|---------|---------|--------|
| Absorption | HIA (Hou) | **0.917** | AUROC↑ |
| Toxicity | DILI | **0.913** | AUROC↑ |
| Metabolism | CYP2C9 (Veith) | **0.875** | AUROC↑ |
| Absorption | Caco-2 (Wang) | **0.342** | MAE↓ |
| Distribution | BBB (Martini) | **0.891** | AUROC↑ |

Full 22-dataset results in [`results_moegcn_tdc_v2.json`](results_moegcn_tdc_v2.json).

---

## Novel Contribution: Quantified Expert Specialization

Expert routing gates **spontaneously learn to partition chemical space** along Lipinski-like physicochemical axes — without any explicit supervision signal.

| Expert | Dominant Profile | LogP (mean) | MW (mean) | Role |
|--------|-----------------|-------------|-----------|------|
| E2 | Lipophilic | 3.8 | 412 | Lipid-soluble drug-like |
| E4 | Small nonpolar | 1.2 | 198 | Fragment-like |
| E10 | Hydrophilic | -0.4 | 287 | Water-soluble |
| E15 | Drug-like | 2.1 | 356 | Ro5-compliant |

**Statistical validation across 8 RDKit descriptors (LogP, TPSA, MW, HBA, HBD, RotBonds, AromaticRings, FractionCSP3):**
- One-way ANOVA: p < 0.001 for all 8 descriptors on both Solubility (n=9,980) and Caco-2 (n=910)
- Kruskal-Wallis: confirmed non-parametric significance
- Effect size: LogP η² = 0.33 (large effect), replicated across two independent datasets

Scripts: [`expert_specialization_stats.py`](expert_specialization_stats.py), [`run_expert_specialization.py`](run_expert_specialization.py)

---

## Architecture

```
Molecule (SMILES)
      │
      ▼
  RDKit Featurization
  (39-dim atom, 10-dim bond)
      │
      ▼
  GNN Backbone (GCN / DMPNN / GIN)
      │
      ▼
  ┌─────────────────────────────┐
  │     MoE Routing Module      │
  │  ┌───┐ ┌───┐ ┌───┐ ┌───┐  │
  │  │E1 │ │E2 │ │...│ │En │  │
  │  └───┘ └───┘ └───┘ └───┘  │
  │     Sparse Top-K Gating     │
  └─────────────────────────────┘
      │
      ▼
  Weighted Expert Aggregation
      │
      ▼
  Property Prediction Head
  (Classification / Regression)
```

The MoE module is a **drop-in replacement** for the final graph-level representation layer. No backbone modifications required.

---

## Project Structure

```
molprop_project/
│
├── Core Model
│   ├── generic_moe_module.py          # Universal MoE plug-in module
│   ├── moe_attentivefp.py             # MoE-AttentiveFP implementation
│   └── phase3_fixed.py                # Phase 3 TDC benchmark runner
│
├── Baselines
│   ├── dmpnn_baseline.py              # Plain D-MPNN baseline
│   ├── attentivefp_baseline.py        # AttentiveFP baseline
│   ├── gnn_baselines.py               # GCN / GIN / GAT baselines
│   ├── fingerprint_baselines.py       # RF+ECFP4, XGBoost baselines
│   └── baselines/                     # Chemprop (D-MPNN) experiment outputs
│
├── TDC Benchmark
│   ├── run_gcn_tdc_baseline.py        # Plain GCN across 22 TDC datasets
│   ├── run_moegcn_tdc_benchmark.py    # MoE-GCN across 22 TDC datasets
│   ├── results_gcn_tdc.json           # GCN TDC results
│   └── results_moegcn_tdc_v2.json    # MoE-GCN TDC results
│
├── Statistical Analysis
│   ├── compute_wilcoxon.py            # Pooled Wilcoxon signed-rank test
│   ├── compute_statistics.py          # Full stats with effect sizes
│   └── statistical_tester.py         # Pairwise model comparison
│
├── Expert Specialization
│   ├── run_expert_specialization.py   # Train + extract routing assignments
│   ├── expert_specialization_stats.py # MI, ANOVA, KW, eta-squared
│   ├── routing_analyzer.py            # Routing pattern analysis
│   ├── expert_specialization_caco2_wang.json
│   ├── expert_specialization_caco2_wang.md
│   ├── expert_specialization_solubility_aqsoldb.json
│   └── expert_specialization_solubility_aqsoldb.md
│
├── HPO
│   ├── optuna_moe.py                  # Optuna HPO (30 trials)
│   └── optuna_moe_remaining.py        # Resume incomplete HPO runs
│
├── GROVER Baseline
│   ├── grover/                        # GROVER submodule
│   └── grover_results.json            # Published + finetuned results
│
└── results/                           # All experiment output JSONs
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/SpoonierElf3378/attentivefp-multitask-admet.git
cd attentivefp-multitask-admet

# Create conda environment
conda create -n moe_admet python=3.8
conda activate moe_admet

# Install dependencies
pip install torch==1.12.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
pip install torch-geometric torch-scatter torch-sparse
pip install rdkit-pypi deepchem PyTDC optuna
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

## Reproducing Results

### Phase 1 — MoleculeNet Baselines
```bash
# GCN / GIN / GAT baselines
python gnn_baselines.py

# D-MPNN baseline
python dmpnn_baseline.py

# RF+ECFP4 / XGBoost
python fingerprint_baselines.py

# AttentiveFP
python attentivefp_baseline.py
```

### Phase 2 — MoE-GCN on MoleculeNet
```bash
# Run with Optuna HPO (30 trials per dataset)
python optuna_moe.py

# Run remaining datasets if interrupted
python optuna_moe_remaining.py
```

### Phase 3 — TDC ADMET Benchmark (22 datasets)
```bash
# Plain GCN baseline on TDC
python run_gcn_tdc_baseline.py

# MoE-GCN on TDC
python run_moegcn_tdc_benchmark.py
```

### Expert Specialization Analysis
```bash
# Train and extract routing assignments
python run_expert_specialization.py

# Compute MI, ANOVA, Kruskal-Wallis, eta-squared
python expert_specialization_stats.py
```

### Statistical Tests
```bash
# Wilcoxon signed-rank test (regression datasets)
python compute_wilcoxon.py

# Full statistical analysis with effect sizes
python compute_statistics.py
```

---

## Hardware

All experiments were run on:
- **GPU:** NVIDIA GeForce GTX 1660 Ti (6GB VRAM)
- **CPU:** Intel Core i7
- **OS:** Windows 10
- **CUDA:** 11.6

Total compute time: ~120 hours across all phases.

---

## Differentiation from Prior Work

| Method | Backbone | Multi-backbone | TDC Coverage | Routing Interpretability | Regression |
|--------|----------|---------------|--------------|--------------------------|------------|
| GNN-MoCE | GCN | ✗ | ✗ | ✗ | ✓ |
| Mol-MoE | MPNN | ✗ | ✗ | ✗ | ✓ |
| ASE-Mol | GCN | ✗ | ✗ | Substructure attribution | ✗ |
| MolGraph-xLSTM | xLSTM | ✗ | Partial | ✗ | ✓ |
| MI-MoE | GIN | ✗ | ✗ | ✗ | ✓ |
| **Ours (MoE-GCN)** | **GCN/DMPNN/GIN** | **✓** | **✓ (22 datasets)** | **Physicochemical (quantified)** | **✓** |

---

## Citation

> Paper under review. Citation will be added upon acceptance.

```bibtex
@article{gogoi2026moegcn,
  title     = {MoE-GCN: An Architecture-Agnostic Mixture-of-Experts Module
               for Molecular Property Prediction with Physicochemical Interpretability},
  author    = {Gogoi, Saptasamudra and Li, Yuquan},
  journal   = {Journal of Cheminformatics},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>SAMLab · Guizhou University · 2026</b>
</p>
