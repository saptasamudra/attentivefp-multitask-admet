# Sparse Mixture-of-Experts Routing for Molecular Property Prediction

Code, experiments, and analysis for the manuscript:

**"Sparse mixture-of-experts routing spontaneously partitions chemical space along physicochemical axes for molecular property prediction"**
Saptasamudra Gogoi, Yuquan Li — submitted to *Journal of Cheminformatics*.

---

## What this is

We attach a sparse top-K mixture-of-experts (MoE) module to standard graph neural network backbones (GCN, GIN, DMPNN) for molecular property prediction. The router receives **no chemical supervision** — no descriptors, no substructure rules, no labels — yet it spontaneously organizes molecules along Lipinski-like physicochemical axes (lipophilicity, aromatic-ring count). We quantify this with mutual information and ANOVA (η² effect sizes) across four independent ADMET datasets, and test whether the routing mechanism itself — rather than simply added parameters — drives any performance difference, using a parameter-matched ablation against equal-capacity dense baselines.

**The central claim is mechanistic, not a performance claim.** MoE routing improves regression accuracy over a plain GCN backbone (Wilcoxon *P* = 0.025, pooled across 12 datasets), but a parameter-matched ablation shows no significant advantage over equal-capacity dense baselines (*P* = 0.58–0.99). The value of the architecture is the interpretable expert specialization it produces at no accuracy cost — not a benchmark win.

## Repository structure

```
molprop_project/
├── ablation_routing.py          # MoE vs. equal-parameter Dense-uniform / Dense-wide (main ablation)
├── ablation_optuna.py           # Same comparison with per-mode Optuna HPO (capacity-unconstrained; supplementary only)
├── attentivefp_moe.py           # MoE-GCN training / evaluation, MoleculeNet + TDC
├── scripts/multitask_9dataset.py  # Plain multi-task baseline (no MoE), 9 MoleculeNet datasets
├── make_fig2.py, fig5_ablation.py # Figure generation scripts
├── DEPRECATED_pharma_cross_arch_ablation.py  # Old pharmacophore-guided routing experiments — NOT part of the current manuscript; kept for archival reference only
├── results_*.json               # Raw experiment outputs (plain / MoE, per backbone, per task type)
├── ablation_routing_results.json # Parameter-matched ablation results (5 datasets × 3 modes × 5 seeds)
└── figures/                     # Publication figures (Figs. 1–6)
```

## Key results

| Question | Result |
|---|---|
| Does MoE improve regression accuracy over plain GCN? | Yes — 10/12 datasets, Wilcoxon *P* = 0.025 (pooled, one-sided) |
| Does MoE improve classification accuracy? | No significant difference (*P* = 0.94 MoleculeNet classification; *P* = 0.22 all-TDC) |
| Does the routing *mechanism* (vs. just more parameters) drive the gain? | No — parameter-matched ablation shows no significant difference vs. Dense-uniform or Dense-wide (*P* = 0.58–0.99) |
| Does routing recover chemically meaningful structure? | Yes — LogP and aromatic-ring count are the dominant axes (η² up to 0.33 and 0.445 respectively), replicated on 3/4 datasets; the 4th (narrow-diversity lipophilicity assay) shows no specialization, consistent with the mechanism requiring chemical diversity |
| Does MoE transfer to other backbones? | Mixed. DMPNN: improves on ESOL, degrades on FreeSolv and Lipophilicity. GIN transferability has not been tested. |

## Reproducing the main results

```bash
conda activate moe_admet
python attentivefp_moe.py          # main MoE-GCN results (Tables 1–2)
python ablation_routing.py         # parameter-matched ablation (Table 6)
python sig_test.py                 # significance tests on ablation_routing_results.json
```

Hyperparameter search: Optuna, tree-structured Parzen estimator, 30 trials with median pruning, per dataset. All experiments run on a single NVIDIA GTX 1660 Ti (6 GB).

## Data

All datasets are public: [MoleculeNet](https://moleculenet.org) and the [Therapeutics Data Commons](https://tdcommons.ai) ADMET benchmark group. No registration or login required.

## Honesty notes (things this README will not let you miss)

- The plug-in does **not** achieve state-of-the-art accuracy — it is outperformed by AttentiveFP on regression and by GROVER (pretrained) on most tasks.
- GIN cross-architecture transferability is claimed nowhere in the current manuscript because it has not been tested. Only GCN and DMPNN have real plain-vs-MoE comparisons.
- `DEPRECATED_pharma_cross_arch_ablation.py` reflects an earlier, abandoned pharmacophore-guided routing approach and is **not** connected to any result in the current manuscript.

## Citation

If you use this code, please cite the manuscript (citation to be added on acceptance).

## License

MIT.
