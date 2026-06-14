# GNN Baseline Experiments - Quick Start Guide

**Objective:** Supplement modern baseline comparisons (GIN, GCN, GAT) on 9 MoleculeNet datasets with 5 seeds each, following Professor Li Yuquan's Phase 1 Priority ① requirements.

**Author:** Sapta (林恩)  
**Date:** 2026-04-01  
**Lab:** SAMLab, Guizhou University  
**Deadline:** Complete by April 7, 2026 (1 week)

---

## 📋 Prerequisites

### 1. Data Preparation

Ensure your MoleculeNet datasets are in CSV format with this structure:

```
data/moleculenet/
├── ESOL.csv
├── FreeSolv.csv
├── Lipo.csv
├── BACE.csv
├── BBBP.csv
├── HIV.csv
├── ClinTox.csv
├── Tox21.csv
└── SIDER.csv
```

Each CSV should have:
- **Column 1:** `smiles` (SMILES strings)
- **Columns 2+:** Task labels (1 column for single-task, multiple for multi-task)

### 2. Environment Setup

```bash
conda activate molprop

# Verify PyTorch Geometric installation
python -c "import torch_geometric; print(torch_geometric.__version__)"

# If not installed:
pip install torch_geometric --break-system-packages
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html --break-system-packages
```

---

## 🚀 Execution Options

### Option A: SLURM Cluster (Recommended - Fastest)

**Step 1:** Verify SLURM access

```bash
# SSH into SAMLab supercomputer (see lab onboarding docs sections 7.1.1-7.1.2)
ssh your_username@samlab_cluster

# Check SLURM availability
sinfo
```

**Step 2:** Upload scripts to cluster

```bash
# From your local machine
scp gnn_baselines.py your_username@samlab_cluster:~/molprop_baselines/
scp run_gnn_baselines.sh your_username@samlab_cluster:~/molprop_baselines/
```

**Step 3:** Update data path in `run_gnn_baselines.sh`

Edit line 75:
```bash
--data_dir /path/to/moleculenet/data \
```

Change to your actual path on the cluster.

**Step 4:** Submit job array

```bash
cd ~/molprop_baselines
chmod +x run_gnn_baselines.sh

# Submit all 135 jobs (3 models × 9 datasets × 5 seeds)
sbatch run_gnn_baselines.sh
```

**Step 5:** Monitor progress

```bash
# Check job status
squeue -u your_username

# Check specific job output
tail -f logs/gnn_JOBID_TASKID.out
```

**Expected completion time:** 12-24 hours (depending on cluster load)

---

### Option B: Baidu AI Studio (Free V100 Access)

**Step 1:** Apply for free GPU credits

1. Go to https://aistudio.baidu.com/
2. Register with your `.edu.cn` email (Guizhou University email)
3. Apply for free V100 GPU hours (usually 100 hours/month for students)

**Step 2:** Create notebook environment

1. Create new notebook project
2. Select "GPU: V100" as runtime
3. Upload `gnn_baselines.py` and datasets

**Step 3:** Run experiments in batches

Since AI Studio typically limits sessions to 8-12 hours, run in batches:

```python
# Batch 1: GIN on all datasets, seed 0
!python gnn_baselines.py --model GIN --dataset ESOL --seed 0 --data_dir ./data
!python gnn_baselines.py --model GIN --dataset FreeSolv --seed 0 --data_dir ./data
# ... (repeat for all 9 datasets)

# Batch 2: GIN on all datasets, seed 1
# ... (repeat)
```

---

### Option C: Local GTX 1660 Ti (Slowest - Fallback Only)

⚠️ **WARNING:** This will take 2-3 days to complete all experiments sequentially.

**Step 1:** Update data path in `run_gnn_baselines_local.bat`

Edit line 30:
```batch
set DATA_DIR=D:\molprop_project\data\moleculenet
```

**Step 2:** Run batch script

```cmd
cd D:\molprop_project\baselines\gin_gcn_gat
run_gnn_baselines_local.bat
```

**Step 3:** Leave computer running

The script will run all 135 experiments sequentially. You can monitor progress via the console output.

---

## 📊 Results Analysis

### After All Experiments Complete

**Step 1:** Aggregate results

```bash
python aggregate_gnn_results.py \
    --results_file ./baselines/results/gnn_results.json \
    --output_csv ./baselines/results/gnn_summary.csv \
    --stat_test
```

**Step 2:** Review output

The script will print:
1. **Mean ± Std** for each model-dataset combination
2. **Regression tasks summary** (RMSE comparison table)
3. **Classification tasks summary** (AUC comparison table)
4. **Seed breakdown** (individual seed results for statistical tests)
5. **Comparison with D-MPNN** (your completed baseline)

**Step 3:** Statistical significance testing

The aggregation script automatically performs paired t-tests. Check for:
- **p < 0.05** → Statistically significant difference
- **p ≥ 0.05** → No significant difference

---

## 📈 Expected Results

Based on literature benchmarks, you should see:

### Regression Tasks (RMSE - lower is better)
| Dataset  | D-MPNN (your baseline) | GIN (expected) | GCN (expected) | GAT (expected) |
|----------|------------------------|----------------|----------------|----------------|
| ESOL     | 0.962 ± 0.069         | ~0.90-1.00     | ~0.95-1.05     | ~0.92-1.02     |
| FreeSolv | 2.802 ± 0.182         | ~2.50-2.90     | ~2.70-3.00     | ~2.60-2.95     |
| Lipo     | 0.812 ± 0.016         | ~0.75-0.85     | ~0.80-0.90     | ~0.78-0.88     |

### Classification Tasks (AUC - higher is better)
| Dataset  | D-MPNN (your baseline) | GIN (expected) | GCN (expected) | GAT (expected) |
|----------|------------------------|----------------|----------------|----------------|
| BACE     | 0.791 ± 0.031         | ~0.78-0.82     | ~0.76-0.80     | ~0.77-0.81     |
| BBBP     | 0.879 ± 0.033         | ~0.85-0.90     | ~0.83-0.88     | ~0.86-0.91     |
| HIV      | 0.781 ± 0.024         | ~0.76-0.80     | ~0.74-0.78     | ~0.75-0.79     |
| ClinTox  | 0.922 ± 0.014         | ~0.88-0.93     | ~0.86-0.91     | ~0.87-0.92     |
| Tox21    | 0.810 ± 0.024         | ~0.78-0.82     | ~0.76-0.80     | ~0.77-0.81     |
| SIDER    | 0.600 ± 0.000         | ~0.58-0.63     | ~0.56-0.61     | ~0.57-0.62     |

**Note:** These are rough estimates. Your actual results will vary depending on exact hyperparameters and random seeds.

---

## 🔧 Troubleshooting

### Issue 1: CUDA Out of Memory

**Solution:** Reduce batch size

```bash
python gnn_baselines.py ... --batch_size 32  # instead of 64
```

### Issue 2: PyG Installation Fails

**Solution:** Use CPU-only version

```bash
pip install torch_geometric --break-system-packages
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cpu.html --break-system-packages
```

Then run with `--device cpu` (note: will be very slow)

### Issue 3: Invalid SMILES

**Error:** `RuntimeError: Invalid molecule`

**Solution:** The script automatically skips invalid molecules and reports the count. If too many molecules are invalid (>10%), check your CSV format.

### Issue 4: NaN in Loss/Metrics

**Cause:** All labels are missing for a particular task

**Solution:** This is expected for multi-task datasets with sparse labels (e.g., Tox21). The script handles this automatically by using `torch.isnan()` masking.

---

## 📝 Next Steps (After GIN/GCN/GAT Complete)

Following the professor's prioritized experimental plan:

### Phase 1b: GROVER Baseline (April 8-10)
- Download pre-trained GROVER model
- Fine-tune on 9 datasets
- **Critical:** GROVER has BBBP 0.94, ClinTox 0.944 - these are strong baselines

### Phase 1c: Fingerprint + ML Baseline (April 10-11)
- Generate ECFP4/Morgan fingerprints
- Train RandomForest or XGBoost
- Industry-standard baseline

### Phase 1d: (Optional) Uni-Mol Baseline (if time permits)
- 3D pre-training, computationally expensive
- Only if you have extra time or strong compute resources

---

## 📊 Deliverable Format for Paper

After aggregating results, create this table for your technical report:

```
Table 1: Baseline Comparison on MoleculeNet Benchmark (Scaffold Split)

Dataset    Task Type     Metric   D-MPNN       GIN          GCN          GAT
------------------------------------------------------------------------
ESOL       Regression    RMSE     0.96±0.07    X.XX±X.XX    X.XX±X.XX    X.XX±X.XX
FreeSolv   Regression    RMSE     2.80±0.18    X.XX±X.XX    X.XX±X.XX    X.XX±X.XX
...
```

Bold the **best result** for each dataset (considering std error bars).

Report statistical significance:
- Use † symbol if p < 0.05 vs D-MPNN
- Use ‡ symbol if p < 0.01 vs D-MPNN

---

## ⏱️ Timeline Checkpoint

**By April 7, 2026, you should have:**
- ✅ All 135 GNN baseline experiments complete
- ✅ Results aggregated with mean ± std (5 seeds)
- ✅ Statistical significance tests performed
- ✅ Comparison table vs D-MPNN ready

**If behind schedule:**
- Reduce to 3 seeds instead of 5 (still valid, just less statistical power)
- Skip GAT if time-constrained (focus on GIN + GCN)
- Prioritize classification tasks (BBBP, ClinTox) - these are where MoE showed most improvement in your current results

---

## 📞 Support

If you encounter issues:

1. **Check logs first:** `logs/GIN_BBBP_seed0.log`
2. **Verify environment:** `conda list | grep torch`
3. **Test single run:** Run one experiment manually to debug
   ```bash
   python gnn_baselines.py --model GIN --dataset BBBP --seed 0 --data_dir ./data
   ```

**Lab resources:**
- SLURM documentation: Lab onboarding sections 7.1.1-7.1.2
- Contact senior students: Wu Nanwan, Luo Xixuan for cluster access help

---

Good luck! Remember: these baselines are **mandatory** before you can submit to a Q1 journal. The professor's document was very clear about this being the highest priority blocker.

**Target completion: April 7, 2026 (7 days from now)**
