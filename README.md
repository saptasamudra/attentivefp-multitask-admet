# Multi-task AttentiveFP for ADMET Molecular Property Prediction

> A shared graph neural network that jointly predicts multiple drug-related molecular properties, trained and evaluated on MoleculeNet benchmarks with scaffold split.

## Table of contents

- [About](#about)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [File structure](#file-structure)
- [Methodology](#methodology)
- [Next steps](#next-steps)
- [Acknowledgements](#acknowledgements)

## About

Predicting molecular properties like solubility, binding affinity, and toxicity is critical in early-stage drug discovery. This project builds a multi-task learning framework on top of [AttentiveFP](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959) (Xiong et al., 2020) — a graph attention network for molecules — to predict multiple ADMET properties simultaneously using a shared encoder with task-specific output heads.

**Key finding:** Task-weighted multi-task learning improves ESOL solubility prediction by 11.3% over the single-task baseline, surpassing the original AttentiveFP paper. We show that reducing the dominant task's loss weight is more effective than amplifying the weaker task's weight to mitigate negative transfer.

## Results

All results use Bemis-Murcko scaffold split and are averaged over 3 random seeds (mean ± std).

### Single-task baselines

| Dataset | Task | Molecules | Metric | Our result | Published baseline |
|---------|------|-----------|--------|------------|--------------------|
| ESOL | Solubility (regression) | 1,128 | RMSE ↓ | 0.9848 ± 0.0049 | 0.877 |
| BACE | BACE-1 inhibition (classification) | 1,513 | AUC ↑ | 0.9558 ± 0.0083 | 0.863 |
| FreeSolv | Hydration energy (regression) | 642 | RMSE ↓ | *in progress* | 2.082 |
| BBBP | Blood-brain barrier (classification) | 2,039 | AUC ↑ | *in progress* | 0.862 |

### Multi-task weight ablation (ESOL + BACE)

| Weights (w_esol, w_bace) | ESOL RMSE ↓ | BACE AUC ↑ |
|---------------------------|-------------|------------|
| Single-task baseline | 0.9791 ± 0.0238 | 0.9708 ± 0.0145 |
| (1.0, 1.0) equal weight | 0.9072 ± 0.0128 | 0.9446 ± 0.0078 |
| (1.0, 2.0) boost BACE | 0.8802 ± 0.0122 | 0.9028 ± 0.0223 |
| **(0.5, 1.0) reduce ESOL** | **0.8688 ± 0.0303** | **0.9612 ± 0.0275** |

## Installation

```bash
# Create and activate environment
conda create -n molprop python=3.10 -y
conda activate molprop

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Other dependencies
pip install rdkit scikit-learn matplotlib pandas numpy
```

## Usage

```bash
conda activate molprop

# Run single-task baselines (ESOL + BACE)
python attentivefp_baseline.py

# Run standalone BACE baseline
python bace_baseline.py

# Run multi-task model with weight ablation
python attentivefp_multitask.py
```

All datasets download automatically from MoleculeNet on first run.

**Expected runtime (GTX 1660 Ti):**
- Baseline: ~25 min (3 seeds × 2 datasets × 200 epochs)
- Multi-task: ~35 min (weight search + 3 seeds × 200 epochs)

## File structure

```
molprop_project/
├── attentivefp_baseline.py     # Single-task baselines (ESOL + BACE)
├── attentivefp_multitask.py    # Multi-task model with weight ablation
├── bace_baseline.py            # Standalone BACE classification baseline
├── molprop.ipynb               # Attention weight visualizations
├── attention_maps.png          # Figure: gradient-based atom importance
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

## Methodology

**Model:** AttentiveFP with 39-dim atom features, 10-dim bond features, 200-dim hidden layer, 2 message passing layers, 2 GRU readout timesteps, and 0.2 dropout.

**Split:** Bemis-Murcko scaffold split (80/10/10 train/val/test). Scaffold split ensures test molecules have novel core structures not seen during training, providing a realistic evaluation of generalization.

**Multi-task design:** One shared AttentiveFP encoder with task-specific linear output heads. Combined loss: `L = w_esol × MSE + w_bace × BCE`. Task weights selected on validation set only (seed=42); test set never used for hyperparameter selection.

**Reproducibility:** All experiments run across 3 random seeds (42, 123, 7) with mean ± std reported.

## Next steps

- [ ] Add FreeSolv and BBBP single-task baselines
- [ ] Expand multi-task model to 4 datasets
- [ ] Integrate Optuna for automatic hyperparameter tuning
- [ ] Update paper draft with expanded results

## Acknowledgements

Built at SAMLab, Guizhou University. Based on the AttentiveFP architecture by Xiong et al. (2020).

**Reference:**
Xiong, Z. et al. (2020). Pushing the boundaries of molecular representation for drug discovery with the graph attention mechanism. *Journal of Medicinal Chemistry*, 63(16), 8749-8760.
