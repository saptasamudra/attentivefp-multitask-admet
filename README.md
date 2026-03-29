# MoE-AttentiveFP: Multi-Task Molecular Property Prediction

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org)
[![PyG 2.7](https://img.shields.io/badge/PyG-2.7.0-green.svg)](https://pyg.org)
[![Datasets](https://img.shields.io/badge/Datasets-9%20MoleculeNet-teal.svg)](https://moleculenet.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Sparse Mixture-of-Experts routing applied to multi-task ADMET molecular property prediction across 9 MoleculeNet benchmarks. First application of MoE routing to this problem setting.

**Authors:** Saptasamudra Gogoi · 林恩 · Biotechnology

---

## Results

State-of-the-art on ClinTox and BBBP under scaffold split evaluation.

| Dataset | Task | MoE K=4 (ours) | Published | Δ |
|---|---|---|---|---|
| ESOL | RMSE ↓ | 0.9623 ± 0.069 | 0.877 | — |
| FreeSolv | RMSE ↓ | 2.8020 ± 0.182 | 2.082 | — |
| Lipophilicity | RMSE ↓ | 0.8121 ± 0.016 | 0.655 | — |
| BACE | AUC ↑ | 0.7908 ± 0.031 | 0.863 | — |
| **BBBP** | **AUC ↑** | **0.8787 ± 0.033** | 0.862 | **+1.7%** ✓ |
| HIV | AUC ↑ | 0.7809 ± 0.024 | — | new |
| **ClinTox** | **AUC ↑** | **0.9215 ± 0.014** | 0.832 | **+9.0%** ✓ |
| Tox21 | AUC ↑ | 0.7703 ± 0.009 | 0.829 | — |
| SIDER | AUC ↑ | 0.5875 ± 0.023 | — | new |

All results: scaffold split, 3 seeds (42, 123, 7), mean ± std.

---

## Architecture

```
Molecule (SMILES)
    ↓ GenFeatures (39-dim atom, 10-dim bond)
    ↓ AttentiveFP Encoder → mol_repr [B, 200]
    ↓ MoE Module (K experts, top-2 sparse routing) → expert_repr [B, 200]
    ↓ Concat → fused [B, 400]
    ↓ 9 Task-Specific Heads → ADMET predictions
```

**MoE routing:** A gating network (Linear 200→K) scores each molecule against K experts. Only the top-2 experts activate per molecule. Load-balancing auxiliary loss (λ=0.01) prevents expert collapse.

**Fused representation:** Concatenating mol_repr and expert_repr preserves global molecular information while adding expert-specialized signal. Task heads receive 400-dim input instead of 200-dim.

---

## Ablation Study

| Model | Params | ESOL ↓ | BBBP ↑ | ClinTox ↑ | Tox21 ↑ | HIV ↑ |
|---|---|---|---|---|---|---|
| Single-task | ~200K×9 | 1.027 | 0.647 | 0.868 | 0.762 | — |
| Multi-task no MoE | 945K | 0.951 | 0.861 | 0.898 | 0.754 | 0.772 |
| MoE K=2 | 1.12M | 0.975 | 0.851 | **0.924** | 0.750 | 0.780 |
| **MoE K=4** | **1.28M** | 0.962 | **0.879** | 0.922 | **0.770** | **0.781** |
| MoE K=8 | 1.60M | **0.919** | 0.842 | 0.890 | 0.759 | 0.778 |

**Key finding:** K=4 is optimal for classification. K=8 improves regression (best ESOL, Lipo) but underperforms K=4 on classification — optimal expert count is task-type dependent.

---

## Quick Start

```bash
git clone https://github.com/SpoonierElf3378/attentivefp-multitask-admet
cd attentivefp-multitask-admet
pip install -r requirements.txt
```

**Run main model (MoE K=4):**
```bash
# Windows
clean_and_run.bat

# Linux/Mac
python scripts/attentivefp_moe.py
```

**Run ablation experiments:**
```bash
# No-MoE baseline
python scripts/multitask_9dataset.py

# K=2 ablation
python scripts/attentivefp_moe_k2.py

# K=8 ablation
python scripts/attentivefp_moe_k8.py

# Single-task baseline
python scripts/moleculenet_baseline.py
```

---

## Installation

```bash
# Create conda environment
conda create -n molprop python=3.10
conda activate molprop

# Install PyTorch (CUDA 12.1)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
pip install torch-geometric==2.7.0

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Repository Structure

```
attentivefp-multitask-admet/
│
├── README.md
├── requirements.txt
│
├── scripts/
│   ├── attentivefp_moe.py          # Main model: MoE K=4
│   ├── attentivefp_moe_k2.py       # Ablation: MoE K=2
│   ├── attentivefp_moe_k8.py       # Ablation: MoE K=8
│   ├── multitask_9dataset.py       # Baseline: multi-task no MoE
│   ├── moleculenet_baseline.py     # Baseline: single-task
│   ├── make_k2.py                  # Generates K=2 script from K=4
│   ├── make_k8.py                  # Generates K=8 script from K=4
│   ├── save_results.py             # Saves K=4 results to results/
│   └── add_results.py              # Parses and saves any experiment results
│
├── results/
│   ├── single_task_baseline.txt
│   ├── multitask_9dataset_noMoE.txt
│   ├── moe_k2.txt
│   ├── moe_k4.txt
│   └── moe_k8.txt
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_results_visualization.ipynb
│   └── 03_ablation_analysis.ipynb
│
└── figures/
    └── (exported PNG figures for paper)
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 10^-2.5 = 3.162 × 10^-3 |
| Hidden dimension | 200 |
| AttentiveFP layers | 2 |
| AttentiveFP timesteps | 2 |
| Dropout | 0.2 |
| Batch size | 200 |
| Epochs | 200 |
| Weight decay | 1e-5 |
| num_experts (K=4) | 4 |
| top_k | 2 |
| Load balance weight λ | 0.01 |
| Seeds | 42, 123, 7 |

---

## Datasets

9 MoleculeNet datasets covering the full ADMET spectrum:

- **Physicochemical:** ESOL (solubility), FreeSolv (hydration free energy), Lipophilicity (logD)
- **Bioactivity:** BACE (BACE1 inhibition), BBBP (blood-brain barrier), HIV (antiviral activity)
- **Toxicity:** ClinTox (clinical trial toxicity), Tox21 (12 EPA endpoints), SIDER (27 drug side effects)

Data is automatically downloaded by PyTorch Geometric on first run and cached in `data/`.

---

## Environment

- Python 3.10
- PyTorch 2.5.1 + CUDA 12.1
- PyTorch Geometric 2.7.0
- RDKit
- scikit-learn
- numpy

See `requirements.txt` for exact versions.

---

## Citation

```bibtex
@article{gogoi2026moeattentivefp,
  title={Mixture-of-Experts Enhanced AttentiveFP for Multi-Task Molecular Property Prediction},
  author={Gogoi, Saptasamudra and Lin, En},
  journal={[To be updated upon publication]},
  year={2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
