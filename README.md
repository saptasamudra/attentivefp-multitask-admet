# Multi-task AttentiveFP for ADMET Molecular Property Prediction

> A shared graph neural network that jointly predicts multiple drug-related molecular properties, benchmarked on 7 MoleculeNet datasets with scaffold split.

## Table of contents

- [About](#about)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [File structure](#file-structure)
- [Methodology](#methodology)
- [Acknowledgements](#acknowledgements)

## About

Predicting molecular properties like solubility, lipophilicity, binding affinity, and toxicity is critical in early-stage drug discovery. This project builds a multi-task learning framework on top of [AttentiveFP](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959) (Xiong et al., 2020) — a graph attention network for molecules — to predict multiple ADMET properties simultaneously using a shared encoder with task-specific output heads.

**Key contributions:**
- Comprehensive benchmark of AttentiveFP across 7 MoleculeNet datasets (3 regression + 4 classification)
- Multi-task model with shared encoder and 7 task-specific heads
- Task-weighted loss to mitigate negative transfer between regression and classification tasks
- Optuna-based automatic hyperparameter optimization

## Results

All results use Bemis-Murcko scaffold split and are averaged over 3 random seeds (mean ± std).

### Single-task baselines

| Dataset | Molecules | Task | Metric | Our result | Published |
|---------|-----------|------|--------|------------|-----------|
| ESOL | 1,128 | Solubility (regression) | RMSE ↓ | 1.0365 ± 0.0544 | 0.877 |
| FreeSolv | 642 | Hydration energy (regression) | RMSE ↓ | 2.2363 ± 0.0553 | 2.082 |
| Lipo | 4,200 | Lipophilicity (regression) | RMSE ↓ | 0.6514 ± 0.0016 | 0.655 |
| BACE | 1,513 | BACE-1 inhibition (classification) | AUC ↑ | 0.8918 ± 0.0125 | 0.863 |
| BBBP | 2,039 | Blood-brain barrier (classification) | AUC ↑ | 0.6471 ± 0.0991 | 0.862 |
| ClinTox | 1,478 | Clinical toxicity (classification) | AUC ↑ | 0.8742 ± 0.0067 | 0.832 |
| Tox21 | 7,831 | Toxicity (classification) | AUC ↑ | 0.7286 ± 0.0113 | 0.829 |

### Multi-task results

*In progress — running 7-dataset multi-task model with shared encoder.*

### Task weight ablation (ESOL + BACE, earlier experiments)

| Weights (w_esol, w_bace) | ESOL RMSE ↓ | BACE AUC ↑ |
|---------------------------|-------------|------------|
| Single-task baseline | 0.9791 ± 0.0238 | 0.9708 ± 0.0145 |
| (1.0, 1.0) equal weight | 0.9072 ± 0.0128 | 0.9446 ± 0.0078 |
| (1.0, 2.0) boost BACE | 0.8802 ± 0.0122 | 0.9028 ± 0.0223 |
| **(0.5, 1.0) reduce ESOL** | **0.8688 ± 0.0303** | **0.9612 ± 0.0275** |

## Installation

```bash
conda create -n molprop python=3.10 -y
conda activate molprop
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install rdkit scikit-learn matplotlib pandas numpy optuna
```

## Usage

```bash
conda activate molprop

# ── Single-task baselines ──
python moleculenet_baseline.py          # All 7 datasets, 3 seeds each
python bace_baseline.py                 # Standalone BACE baseline

# ── Multi-task models ──
python attentivefp_multitask.py         # 2-dataset (ESOL+BACE) with weight ablation
python multitask_7dataset.py            # 7-dataset multi-task model

# ── Hyperparameter optimization ──
python optuna_esol_simple.py            # Optuna on single-task ESOL (learning demo)
python optuna_multitask.py              # Optuna on multi-task ESOL+BACE
python optuna_final_eval.py             # Final eval with Optuna-optimized params
```

All datasets download automatically from MoleculeNet on first run.

## File structure

```
molprop_project/
├── moleculenet_baseline.py        # 7-dataset single-task baselines
├── multitask_7dataset.py          # 7-dataset multi-task model (main contribution)
├── attentivefp_baseline.py        # Original 2-dataset baselines (ESOL+BACE)
├── attentivefp_multitask.py       # 2-dataset multi-task with weight ablation
├── bace_baseline.py               # Standalone BACE classification baseline
├── optuna_esol_simple.py          # Optuna tutorial on single-task ESOL
├── optuna_multitask.py            # Optuna for multi-task hyperparameters
├── optuna_final_eval.py           # Final evaluation with optimized params
├── molprop.ipynb                  # Attention weight visualizations
├── attention_maps.png             # Figure: gradient-based atom importance
├── requirements.txt
├── .gitignore
└── README.md
```

## Methodology

**Model:** AttentiveFP with 39-dim atom features (including chirality type), 10-dim bond features, 200-dim hidden layer, 2 message passing layers, 2 GRU readout timesteps, and 0.2 dropout.

**Split:** Bemis-Murcko scaffold split (80/10/10). Classification datasets use class-aware seeding to ensure both classes appear in val and test splits.

**Multi-task design:** One shared AttentiveFP encoder with dataset-specific linear output heads. Regression tasks are downweighted (w=0.5) relative to classification tasks (w=1.0) to prevent gradient domination from MSE loss.

**Datasets:** 7 MoleculeNet benchmarks covering the full ADMET pipeline — solubility (ESOL), hydration energy (FreeSolv), lipophilicity (Lipo), enzyme inhibition (BACE), brain penetration (BBBP), clinical toxicity (ClinTox), and in-vitro toxicity (Tox21).

**Reproducibility:** All experiments across 3 random seeds (42, 123, 7) with mean ± std reported.

## Acknowledgements

Built at SAMLab, Guizhou University. Based on the AttentiveFP architecture by Xiong et al. (2020).

**Reference:**
Xiong, Z. et al. (2020). Pushing the boundaries of molecular representation for drug discovery with the graph attention mechanism. *Journal of Medicinal Chemistry*, 63(16), 8749-8760.
