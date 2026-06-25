# Sparse MoE Routing Spontaneously Partitions Chemical Space for Molecular Property Prediction

> **Manuscript:** Gogoi S, Li Y. *Sparse mixture-of-experts routing spontaneously partitions chemical space along physicochemical axes for molecular property prediction.* Target: Journal of Cheminformatics (2026).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20855212-blue)](https://doi.org/10.5281/zenodo.20855212)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

---

## Overview

This repository contains all code, configs, and analysis scripts for a study showing that a **sparse Mixture-of-Experts (MoE) plug-in** added to a Graph Convolutional Network (GCN):

1. Improves regression accuracy over plain GCN on **10/12 ADMET datasets** (Wilcoxon *P* = 0.025)
2. **Spontaneously partitions chemical space** along Lipinski-like physicochemical axes (LogP η² = 0.33) — *without any chemical supervision*
3. This routing specialization **replicates** across two independent datasets (Solubility + Caco-2, *P* < 0.001)

The MoE module is **architecture-agnostic** — it attaches after pooling and leaves the backbone untouched.

---

## Repository Structure

```
molprop_project/
├── data/                   # Dataset loaders (MoleculeNet, TDC ADMET)
├── models/
│   ├── gcn.py              # Plain GCN backbone
│   ├── moe_gcn.py          # MoE-GCN (primary model)
│   ├── gin.py              # GIN backbone (robustness check)
│   └── dmpnn.py            # Corrected NNConv/GRU DMPNN backbone
├── train.py                # Training loop with Optuna HPO
├── evaluate.py             # Evaluation across seeds
├── analysis/
│   ├── specialization.py   # η², MI, ANOVA routing analysis
│   └── phase_transition.py # Per-epoch expert routing tracker
├── figures/                # Reproduction scripts for Figs 1–4
├── configs/                # Per-dataset hyperparameter configs
├── environment.yml         # Conda environment
└── README.md
```

---

## Installation

```bash
git clone https://github.com/saptasamudra/attentivefp-multitask-admet.git
cd attentivefp-multitask-admet
conda env create -f environment.yml
conda activate molprop
```

**Key dependencies:** PyTorch, PyTorch Geometric, RDKit, TDC, DeepChem, Optuna, scikit-learn

---

## Quickstart

```bash
# Train MoE-GCN on ESOL
python train.py --dataset esol --model moe_gcn --seeds 5

# Run specialization analysis (η², MI, ANOVA)
python analysis/specialization.py --dataset solubility --n_experts 8

# Reproduce Figure 1 (benchmark bar charts)
python figures/fig1_benchmark.py
```

---

## Key Results

| Dataset | GCN (RMSE↓) | MoE-GCN (RMSE↓) | Δ |
|---|---|---|---|
| ESOL | 1.324 | **1.067 ± 0.046** | +19.4% |
| FreeSolv | 4.927 | 3.591 ± 0.527 | +27.1% |
| Lipophilicity | 0.812 | 0.722 ± 0.008 | +11.1% |

**Pooled Wilcoxon (regression, 12 datasets):** *P* = 0.025, rank-biserial = 0.28, 10 wins / 2 losses

**Expert specialization (η², LogP):** Solubility = 0.33, Caco-2 = 0.15 — both *P* < 0.001

> Classification tasks showed no significant improvement (*P* = 0.94). MoE-GCN is outperformed by AttentiveFP and GROVER on most tasks — these are reported honestly.

---

## Mechanistic Finding

Expert routing **self-organizes** through a training-time phase transition (~epoch 15–25) into chemically interpretable clusters:

| Expert | Chemical Character | Mean LogP | Mean TPSA |
|---|---|---|---|
| E5 | Lipophilic / aromatic | 4.47 | 44.9 |
| E1 | Hydrophilic / polar | −0.68 | 96.2 |
| E7 | H-bond donors | 0.88 | 64.8 |
| E3 | Drug-like, intermediate | 2.74 | 56.3 |
| E6 | Small fragments | 0.30 | 30.8 |

No descriptor labels or chemical rules were provided to the router.

---

## Datasets

- **MoleculeNet** (10 datasets): ESOL, FreeSolv, Lipophilicity, BBBP, BACE, Tox21, ToxCast, SIDER, ClinTox, HIV
- **TDC ADMET** (22 datasets): Full benchmark group — https://tdcommons.ai

All splits: Bemis–Murcko scaffold, 80/10/10. Stratified scaffold for BBBP and BACE.

---

## Citation

```bibtex
@article{gogoi2026moe,
  title   = {Sparse mixture-of-experts routing spontaneously partitions chemical space
             along physicochemical axes for molecular property prediction},
  author  = {Gogoi, Saptasamudra and Li, Yuquan},
  journal = {Journal of Cheminformatics},
  year    = {2026},
  note    = {Under review}
}
```

---

## Authors

- **Saptasamudra Gogoi** — SAMLab, Guizhou University · iec.GOGOI24@gzu.edu.cn
- **Yuquan Li** (Corresponding) — SAMLab, Guizhou University · yvquan.li@gzu.edu.cn

**Funding:** NSFC (32560689, 32125033, 62162008), National Key R&D (2024YFD2001100), Guizhou Provincial S&T ([2024]002), and others — see manuscript Acknowledgements.

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Archived at Zenodo: [10.5281/zenodo.20855212](https://doi.org/10.5281/zenodo.20855212)*
