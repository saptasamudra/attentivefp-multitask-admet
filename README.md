# Sparse MoE-GNN for Multi-Task ADMET Prediction

**Saptasamudra Gogoi** · SAMLab, Guizhou University · Supervisor: Prof. Li Yuquan

---

## Overview

This repository presents a systematic benchmark of **Sparse Mixture-of-Experts (MoE) routing** as a universal, architecture-agnostic plug-in enhancement for graph neural networks in multi-task ADMET molecular property prediction.

MoE routing is applied identically to three backbone architectures — **GCN, DMPNN, and GIN** — and evaluated across:
- **10 MoleculeNet datasets** (7 classification, 3 regression)
- **22 TDC ADMET datasets** (full A/D/M/E/T coverage)
- **Virtual screening case study** (BBB permeability, CNS drug validation)

---

## Key Results

### MoleculeNet Benchmark

| Dataset | DMPNN | MoE-GCN | MoE-DMPNN | AttentiveFP | GROVER† |
|---------|-------|---------|-----------|-------------|---------|
| Tox21 (AUC↑) | 0.727 | **0.745** | 0.731 | 0.746 | 0.831 |
| ToxCast (AUC↑) | 0.633 | **0.647** | 0.643 | 0.675 | 0.737 |
| ESOL (RMSE↓) | 1.324 | **1.067** | 1.105 | 1.061 | 0.983 |
| FreeSolv (RMSE↓) | 5.103 | 3.591 | **3.432** | 2.573 | 1.544 |
| Lipo (RMSE↓) | 0.723 | 0.722 | **0.712** | 0.685 | 0.561 |

**MoE-GCN: 19.4% RMSE reduction on ESOL · MoE-DMPNN: 32.7% on FreeSolv vs plain DMPNN**

† GROVER pretrained on 10M molecules — shown as upper-bound reference only.

### TDC ADMET Benchmark (Phase 3)

| Category | Dataset | Metric | Score |
|----------|---------|--------|-------|
| Absorption | HIA_Hou | AUROC | **0.917** |
| Absorption | Pgp_Broccatelli | AUROC | 0.854 |
| Distribution | BBB_Martins | AUROC | 0.855 |
| Metabolism | CYP2C9_Veith | AUROC | **0.875** |
| Metabolism | CYP3A4_Veith | AUROC | 0.873 |
| Metabolism | CYP2D6_Veith | AUROC | 0.839 |
| Toxicity | DILI | AUROC | **0.913** |
| Toxicity | AMES | AUROC | 0.825 |
| Toxicity | hERG | AUROC | 0.696 |
| Excretion | Clearance_Microsome | Spearman | 0.530 |

Full 22-dataset results in `results_tdc.json`.

### Virtual Screening (Phase 3)

MoE-GCN trained on BBBP (1,631 molecules) correctly prioritizes known CNS-penetrant drugs:

| Rank | Compound | BBB Score | Validation |
|------|----------|-----------|------------|
| 1 | Testosterone | 0.9996 | ✓ Known CNS steroid |
| 6 | Diphenhydramine | 0.9973 | ✓ CNS antihistamine |
| 8 | Nicotine | 0.9966 | ✓ CNS psychoactive |
| 9 | Meclizine-analog | 0.9963 | ✓ CNS antivertigo |

---

## Chemical Interpretability — Key Finding

Experts learn **Lipinski-like chemical space partitioning** without explicit supervision:

| Expert | MW | LogP | TPSA | Character |
|--------|-----|------|------|-----------|
| E15 | 253.7 | 2.35 | 57.9 | Drug-like polar |
| E2 | 197.6 | 3.54 | 14.0 | Lipophilic |
| E4 | 143.9 | 1.71 | 20.5 | Small nonpolar |
| E10 | 114.1 | 1.21 | 25.2 | Small hydrophilic |

This emergent specialization provides a mechanistic explanation for MoE regression improvements.

---

## Project Structure

```
molprop_project/
│
├── Core benchmark
│   ├── moegcn_classif.py          # MoE-GCN classification
│   ├── moegcn_regr.py             # MoE-GCN regression
│   ├── moedmpnn_classif.py        # MoE-DMPNN classification
│   ├── moedmpnn_regr.py           # MoE-DMPNN regression
│   ├── dmpnn_classif.py           # Plain DMPNN baseline
│   ├── dmpnn_regr.py              # Plain DMPNN baseline
│   └── toxcast_all.py             # ToxCast multi-model
│
├── Data preparation
│   ├── fix_bbbp_bace.py           # Stratified scaffold split fix
│   ├── prepare_grover_data.py     # GROVER CSV format
│   └── attentivefp_remaining.py   # AttentiveFP benchmark
│
├── Post-compute
│   ├── extend_seeds.py            # 3 → 5 seed extension
│   ├── compile_results.py         # Final table + Wilcoxon tests
│   └── run_grover.py              # GROVER finetuning
│
├── Interpretability
│   ├── tsne_routing.py            # t-SNE expert routing
│   ├── expert_load_balance.py     # Load balance over epochs
│   ├── chemical_subgroup.py       # Physicochemical per expert
│   ├── topk_comparison.py         # K=1,2,4,8 ablation
│   ├── param_timing.py            # Parameter + inference timing
│   ├── training_curves.py         # Loss curves
│   └── regression_fix.py         # Uncertainty weighting
│
├── Phase 3
│   ├── phase3_tdc_benchmark.py    # TDC classification datasets
│   ├── phase3_tdc_regression.py   # TDC regression datasets
│   └── phase3_virtual_screening.py # BBB virtual screening
│
├── Results (JSON)
│   ├── results_dmpnn_classif.json
│   ├── results_dmpnn_regr.json
│   ├── results_moegcn_classif.json
│   ├── results_moegcn_regr.json
│   ├── results_moedmpnn_classif.json
│   ├── results_moedmpnn_regr.json
│   ├── results_attentivefp.json
│   └── results_tdc.json           # 22 TDC ADMET datasets
│
├── Figures
│   ├── tsne_plots/                # t-SNE routing visualizations
│   ├── training_curves/           # Loss curve plots
│   ├── chemical_subgroup.png
│   ├── expert_load_balance.png
│   ├── topk_comparison.png
│   ├── param_timing.png
│   └── virtual_screening_plots/
│
└── final_results_table.csv
```

---

## Setup

```bash
conda create -n moe_admet python=3.10
conda activate moe_admet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install rdkit scikit-learn optuna pandas matplotlib seaborn
```

For TDC benchmark:
```bash
pip install PyTDC --no-deps
pip install setuptools==69.5.1 huggingface_hub fuzzywuzzy
```

---

## Running

```bash
# MoleculeNet
python moegcn_classif.py && python moegcn_regr.py
python moedmpnn_classif.py && python moedmpnn_regr.py
python dmpnn_classif.py && python dmpnn_regr.py
python fix_bbbp_bace.py
python extend_seeds.py
python compile_results.py

# TDC ADMET (Phase 3)
python phase3_tdc_benchmark.py
python phase3_tdc_regression.py

# Virtual screening (Phase 3)
python phase3_virtual_screening.py
```

All scripts use JSON checkpointing — safe to interrupt and resume.

---

## MoE Architecture

```
SMILES → Mol Graph → GCNConv × L → Global Mean Pool
                                          ↓
                              Sparse MoE (Top-K Gating)
                                          ↓
                                     Task Head

G(h) = TopK( W_g * h, K )
y = sum_{i in TopK} w_i * Expert_i(h)
L_bal = E * sum_i (mean_load_i)^2
L_total = L_task + 0.01 * L_bal
```

Recommended: E=4 experts, K=4 routing.

---

## Citation

> Gogoi, S. (2026). Sparse MoE-GNN for Multi-Task ADMET Molecular Property Prediction.
> SAMLab, Guizhou University. Supervisor: Prof. Li Yuquan.

---

## Acknowledgements

Supervised by Prof. Li Yuquan, SAMLab, College of Life Sciences, Guizhou University.
Built with PyTorch Geometric, RDKit, Optuna, and TDC.
