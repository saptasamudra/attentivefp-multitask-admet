# Phase 1 GNN Baseline Implementation Package

**Complete implementation of GIN, GCN, and GAT baselines for MoleculeNet benchmark**

Following Prof. Li Yuquan's Phase 1 Priority ① Requirements (Highest Priority Blocker)

---

## 📦 Package Contents

| File | Purpose | Size |
|------|---------|------|
| `PHASE1_EXECUTIVE_SUMMARY.md` | **START HERE** - Situation analysis, timeline, next steps | 13KB |
| `GNN_BASELINE_QUICKSTART.md` | Detailed execution guide with 3 pathways | 9.4KB |
| `gnn_baselines.py` | Core implementation (GIN/GCN/GAT models) | 22KB |
| `run_gnn_baselines.sh` | SLURM cluster batch script (135 parallel jobs) | 2.2KB |
| `run_gnn_baselines_local.bat` | Windows local fallback script | 3.5KB |
| `aggregate_gnn_results.py` | Results aggregation (mean ± std, statistical tests) | 6.6KB |
| `generate_baseline_table.py` | Publication-ready comparison table generator | 11KB |

**Total experiments:** 135 jobs (3 models × 9 datasets × 5 seeds)

---

## 🚀 Quick Start (30 Seconds)

**Step 1:** Read `PHASE1_EXECUTIVE_SUMMARY.md` first (critical context)

**Step 2:** Choose execution pathway:
- **SLURM cluster** → See "SLURM Execution" below
- **Baidu AI Studio** → See `GNN_BASELINE_QUICKSTART.md` section "Option B"
- **Local GTX 1660 Ti** → See "Local Execution" below (emergency only)

**Step 3:** Run experiments, aggregate results, generate table

---

## 🖥️ SLURM Execution (Recommended - Fastest)

```bash
# 1. Upload to cluster
scp -r phase1_gnn_baselines/ your_username@samlab_cluster:~/

# 2. SSH into cluster
ssh your_username@samlab_cluster

# 3. Update data path in run_gnn_baselines.sh (line 75)
cd ~/phase1_gnn_baselines
nano run_gnn_baselines.sh
# Change: --data_dir /path/to/moleculenet/data

# 4. Submit job array
chmod +x run_gnn_baselines.sh
sbatch run_gnn_baselines.sh

# 5. Monitor progress
squeue -u your_username
tail -f logs/gnn_*.out

# Expected completion: 12-24 hours
```

---

## 💻 Local Execution (Windows Fallback)

```cmd
cd D:\molprop_project\phase1_gnn_baselines

:: 1. Update data path in run_gnn_baselines_local.bat (line 30)
notepad run_gnn_baselines_local.bat

:: 2. Run batch script
run_gnn_baselines_local.bat

:: Expected completion: 2-3 days (sequential execution)
```

---

## 📊 Results Analysis

After experiments complete:

```bash
# 1. Aggregate results across seeds
python aggregate_gnn_results.py \
    --results_file ./baselines/results/gnn_results.json \
    --output_csv ./baselines/results/gnn_summary.csv \
    --stat_test

# 2. Generate publication table (integrates with D-MPNN results)
python generate_baseline_table.py \
    --gnn_results ./baselines/results/gnn_results.json \
    --output_format latex \
    --save_csv ./baselines/results/baseline_comparison.csv
```

**Output:**
- Mean ± std for each model-dataset pair
- Statistical significance tests (paired t-tests)
- LaTeX table ready for paper insertion
- Performance summary (win counts, average ranks)

---

## 📋 Prerequisites

**Data format:** CSV files with structure:
```
Column 1: smiles (SMILES strings)
Columns 2+: Task labels (1 for single-task, multiple for multi-task)
```

**Environment:**
```bash
conda activate molprop
pip install torch_geometric --break-system-packages
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html --break-system-packages
```

**Datasets required:** ESOL, FreeSolv, Lipo, BACE, BBBP, HIV, ClinTox, Tox21, SIDER

---

## ⏱️ Timeline

**Target completion:** April 7, 2026 (7 days from April 1)

**Day-by-day breakdown:**
- **Day 1 (Apr 1):** Submit SLURM jobs or start local runs
- **Days 2-3 (Apr 2-3):** Jobs running on cluster
- **Day 4 (Apr 4):** Download results, run aggregation
- **Days 5-6 (Apr 5-6):** Generate comparison tables, verify stats
- **Day 7 (Apr 7):** **Checkpoint complete** - all baselines ready

---

## 🎯 Expected Results

Based on literature benchmarks:

**Regression (RMSE - lower is better):**
- ESOL: GIN ~0.90-1.00, GCN ~0.95-1.05, GAT ~0.92-1.02
- Lipo: GIN ~0.75-0.85, GCN ~0.80-0.90, GAT ~0.78-0.88

**Classification (AUC - higher is better):**
- BBBP: GIN ~0.85-0.90, GCN ~0.83-0.88, GAT ~0.86-0.91
- ClinTox: GIN ~0.88-0.93, GCN ~0.86-0.91, GAT ~0.87-0.92

Compare against your D-MPNN baseline to identify improvements.

---

## 🔧 Troubleshooting

**Issue:** CUDA Out of Memory  
**Solution:** Reduce batch size: `--batch_size 32`

**Issue:** Invalid SMILES  
**Solution:** Script auto-skips invalid molecules. Check if >10% failing.

**Issue:** Job fails on SLURM  
**Solution:** Check logs: `cat logs/gnn_JOBID_TASKID.err`

**Issue:** NaN in metrics  
**Solution:** Expected for sparse multi-task labels (Tox21). Script handles via masking.

---

## 📞 Support

**Technical:** Lab seniors (Wu Nanwan, Luo Xixuan) for SLURM access  
**Conceptual:** Professor Li Yuquan via WeChat  
**Documentation:** See `GNN_BASELINE_QUICKSTART.md` for detailed guides

---

## ✅ Deliverable (By April 7)

- [ ] 135 experiments complete (GIN/GCN/GAT on 9 datasets, 5 seeds each)
- [ ] Results aggregated: mean ± std computed
- [ ] Statistical tests performed: p-values < 0.05 identified
- [ ] Comparison table generated: D-MPNN vs GIN vs GCN vs GAT
- [ ] LaTeX table ready for paper insertion

---

## 📄 Citation

If you use this implementation in your research, cite:

```bibtex
@article{gogoi2026moe,
  title={Mixture-of-Experts Graph Neural Networks for Multi-Task ADMET Prediction},
  author={Gogoi, Saptasamudra and Li, Yuquan},
  journal={Journal of Cheminformatics},
  year={2026},
  note={In preparation}
}
```

---

**Author:** Sapta (林恩), Second-year Biotechnology, Guizhou University  
**Lab:** SAMLab (Prof. Li Yuquan)  
**Date:** April 1, 2026  
**License:** MIT (for research purposes)

---

**Questions?** Read `PHASE1_EXECUTIVE_SUMMARY.md` and `GNN_BASELINE_QUICKSTART.md` first.

**Ready to start?** Upload to SLURM cluster or run locally. See you at the April 7 checkpoint! 🚀
