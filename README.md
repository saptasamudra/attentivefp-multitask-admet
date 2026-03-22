# Multi-task AttentiveFP for ADME/T Molecular Property Prediction

A graph neural network model that jointly predicts molecular solubility (ESOL) and BACE-1 enzyme inhibition using a shared AttentiveFP encoder with task-weighted loss to mitigate negative transfer.

## Results

All results use scaffold split and are averaged across 3 random seeds.

| Model | ESOL RMSE ↓ | BACE AUC ↑ | Split |
|---|---|---|---|
| ECFP + RF | 1.074 | 0.861 | scaffold |
| MPNN | 1.167 | 0.815 | scaffold |
| AttentiveFP (Xiong et al. 2020) | 0.877 | 0.863 | scaffold |
| AFP single-task (ours) | 0.9791 ± 0.0238 | 0.9708 ± 0.0145 | scaffold |
| **AFP multi-task w=(0.5,1.0) (ours)** | **0.8688 ± 0.0303** | **0.9612 ± 0.0275** | scaffold |

**Key finding:** Task-weighted multi-task learning reduces ESOL RMSE by 11.3% over single-task baseline, beating the original AttentiveFP paper. Equal-weight multi-task causes negative transfer on BACE; reducing the dominant task weight (w_esol=0.5) restores balance.

## Architecture

```
SMILES input
     ↓
GenFeatures (RDKit)  →  node features [num_atoms × 39]
                         edge features [num_bonds × 10]
     ↓
Atom embedding  (39 → 200)
     ↓
Attentive message passing × 2 layers   ← atom-level attention
     ↓
GRU super-node readout × 2 timesteps   ← molecule-level attention
     ↓
200-dim molecule vector
     ↙              ↘
ESOL head        BACE head
Linear(200→1)    Linear(200→1)
     ↓                ↓
solubility       inhibitor prob
(regression)     (classification)

Loss = 0.5 × MSE(ESOL) + 1.0 × BCE(BACE)
```

## Installation

```bash
# Create environment
conda create -n molprop python=3.10 -y
conda activate molprop

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyG
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Install remaining dependencies
pip install rdkit scikit-learn matplotlib pandas numpy
```

## Usage

**Run single-task baseline (ESOL + BACE separately):**
```bash
python attentivefp_baseline.py
```

**Run multi-task model with weight search:**
```bash
python attentivefp_multitask.py
```

Both scripts automatically download the ESOL and BACE datasets from MoleculeNet on first run.

Expected runtime on GTX 1660 Ti:
- Baseline: ~25 minutes (3 seeds × 2 datasets × 200 epochs)
- Multi-task: ~35 minutes (weight search + 3 seeds × 200 epochs)

## File Structure

```
molprop_project/
├── attentivefp_baseline.py     # Single-task baseline (ESOL + BACE)
├── attentivefp_multitask.py    # Multi-task model with weight search
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Key Design Decisions

**Scaffold split** — all results use Bemis-Murcko scaffold split (not random split), which is the standard for MoleculeNet benchmarks. Random split inflates performance by allowing similar molecules in train and test.

**Task-weighted loss** — equal-weight multi-task loss (w=1.0, 1.0) causes negative transfer on BACE because MSE loss has higher magnitude than BCE loss (~0.8 vs ~0.3), giving ESOL disproportionate gradient signal. Setting w_esol=0.5 corrects this imbalance without requiring separate loss scaling.

**Weight selection on val set** — task weights are selected by maximising combined val score (-ESOL_val + BACE_val) on seed=42 only. Test set is never used for hyperparameter selection.

## Ablation — Effect of Task Weights

| Weights (w_esol, w_bace) | ESOL RMSE ↓ | BACE AUC ↑ |
|---|---|---|
| Single-task baseline | 0.9791 ± 0.0238 | 0.9708 ± 0.0145 |
| (1.0, 1.0) equal weight | 0.9072 ± 0.0128 | 0.9446 ± 0.0078 |
| (1.0, 2.0) boost BACE | 0.8802 ± 0.0122 | 0.9028 ± 0.0223 |
| **(0.5, 1.0) reduce ESOL** | **0.8688 ± 0.0303** | **0.9612 ± 0.0275** |

## Dependencies

- Python 3.10
- PyTorch 2.5.1 + CUDA 12.1
- PyTorch Geometric 2.7.0
- RDKit 2025.9.6
- scikit-learn 1.7.2

## Citation

If you use this code, please cite:

```
@article{xiong2020attentivefp,
  title={Pushing the boundaries of molecular representation for drug discovery
         with the graph attention mechanism},
  author={Xiong, Zhaoping and others},
  journal={Journal of Medicinal Chemistry},
  year={2020}
}
```

## License

MIT License
