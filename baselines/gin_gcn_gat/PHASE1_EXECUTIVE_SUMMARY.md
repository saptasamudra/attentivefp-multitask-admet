# Phase 1 GNN Baseline Implementation - Executive Summary

**Student:** Sapta (林恩) | Second-year Biotechnology, Guizhou University  
**Lab:** SAMLab (Prof. Li Yuquan)  
**Project:** MoE-AttentiveFP for Multi-Task ADMET Prediction  
**Target:** Journal of Cheminformatics submission by April 30, 2026  
**Date:** April 1, 2026

---

## 🚨 CRITICAL BLOCKER IDENTIFIED

Your professor's feedback document (uploaded images) identified **two fatal issues** preventing Q1 publication:

### ❌ Problem #1: Invalidated Novelty Claim

**Your claim:** "First application of sparse MoE routing to multi-task ADMET prediction"

**Reality:** Already done by:
- **GNN-MoCE** (arXiv 2023) — Multi-task MoE, 35 ADMET tasks
- **Mol-MoE** (IBM Research, NeurIPS 2024) — Three-modal 12-expert MoE  
- **MI-MoE** (arXiv 2026.01) — Topology-aware MoE, **ClinTox 0.927** under scaffold split

### ❌ Problem #2: Severely Outdated Baselines

**Your comparison:** Only vs. AttentiveFP (2020 paper)

**Current standards (2026):**
- D-MPNN/Chemprop (BBBP 0.90+) ✅ **You have this**
- **GROVER** (BBBP 0.94, ClinTox 0.944) ❌ **Missing - critical**
- **Uni-Mol** (ClinTox 0.919, HIV 0.808) ❌ **Missing**
- **MolFM-Lite** (BBBP 0.956, latest 2026) ❌ **Missing**
- **GIN/GCN/GAT** (classic GNN baselines) ❌ **Missing**

**Consequence:** Comparing against a 6-year-old paper in a field that improves annually = **automatic rejection from Q1 journals**.

---

## ✅ WHAT I'VE DELIVERED (This Response)

Following your professor's **ranked priority list** strictly, I've implemented **Phase 1a: Supplement Modern Baseline Comparison** (highest priority).

### 📦 Complete Implementation Package

**File 1: `gnn_baselines.py` (22KB)**
- Full implementation of GIN, GCN, and GAT models
- Identical experimental protocol as your D-MPNN runs:
  - Same 9 MoleculeNet datasets
  - Same scaffold split methodology
  - Same random seeds (0-4 per professor's requirement)
  - Same evaluation metrics (RMSE for regression, AUC for classification)
- Handles multi-task datasets (ClinTox, Tox21, SIDER) with NaN masking
- Early stopping with patience (prevents overfitting)
- Automatic model checkpointing

**File 2: `run_gnn_baselines.sh` (2.2KB)**
- SLURM batch script for SAMLab cluster execution
- Job array: 135 parallel jobs (3 models × 9 datasets × 5 seeds)
- Estimated completion: **12-24 hours** on GPU cluster
- Automatic log management

**File 3: `run_gnn_baselines_local.bat` (3.5KB)**
- Windows fallback script for your GTX 1660 Ti
- Sequential execution of all 135 experiments
- Estimated completion: **2-3 days**
- Progress monitoring and error handling

**File 4: `aggregate_gnn_results.py` (6.6KB)**
- Computes mean ± std across seeds
- Generates publication-ready summary tables
- Performs paired t-tests for statistical significance
- Outputs comparison vs. your D-MPNN baseline

**File 5: `generate_baseline_table.py` (11KB)**
- Comprehensive comparison table generator
- Integrates D-MPNN + GIN + GCN + GAT results
- Identifies best-performing model per dataset
- Generates **LaTeX table** ready for paper insertion
- Performance summary statistics (win counts, average ranks)

**File 6: `GNN_BASELINE_QUICKSTART.md` (9.4KB)**
- Step-by-step execution guide
- Three execution pathways:
  1. **SLURM cluster** (fastest, recommended)
  2. **Baidu AI Studio** (free V100, good fallback)
  3. **Local GTX 1660 Ti** (slowest, emergency only)
- Troubleshooting guide
- Expected results benchmarks
- Timeline checkpoints

---

## 📋 EXECUTION PRIORITY (Next 7 Days)

Following the professor's document **strictly**, here's your execution order:

### **URGENT: Verify SLURM Access (Today - April 1)**

```bash
# SSH into SAMLab cluster (lab onboarding sections 7.1.1-7.1.2)
ssh your_username@samlab_cluster

# Check GPU availability
sinfo
```

**If SLURM unavailable:** Immediately apply for Baidu AI Studio (free V100 with .edu.cn email)

**Critical:** Do NOT start local runs on GTX 1660 Ti unless both SLURM and Baidu AI Studio fail. You'll waste 3 days on what could be done in 12 hours on cluster.

---

### **Phase 1a: GIN/GCN/GAT Baselines (April 1-7)**

**Day 1 (April 1):**
- Upload `gnn_baselines.py` and `run_gnn_baselines.sh` to SLURM cluster
- Update data path in script (line 75 of `run_gnn_baselines.sh`)
- Submit job array: `sbatch run_gnn_baselines.sh`

**Days 2-3 (April 2-3):**
- Monitor job progress: `squeue -u your_username`
- Check logs for errors: `tail -f logs/gnn_*.out`
- Jobs should complete within 12-24 hours

**Day 4 (April 4):**
- Download results: `scp cluster:~/molprop_baselines/baselines/results/gnn_results.json ./`
- Run aggregation: `python aggregate_gnn_results.py`
- Verify 5 seeds completed per model-dataset pair

**Days 5-6 (April 5-6):**
- Generate comparison table: `python generate_baseline_table.py --output_format latex`
- Review statistical significance tests
- Document any failed experiments and re-run if needed

**Day 7 (April 7):**
- **Checkpoint:** All 135 experiments complete with mean ± std reported
- Baseline comparison table ready for paper

---

### **Phase 1b: GROVER Baseline (April 8-10)** ⚠️ **CRITICAL**

GROVER achieved:
- **BBBP: 0.94** (vs. your D-MPNN 0.879, AttentiveFP-MoE 0.861)
- **ClinTox: 0.944** (vs. your D-MPNN 0.922, AttentiveFP-MoE 0.898)

**This is the strongest baseline.** Your MoE model MUST beat GROVER to claim improvement.

**Implementation:**
1. Clone GROVER repo: `git clone https://github.com/tencent-ailab/grover`
2. Download pre-trained weights
3. Fine-tune on your 9 datasets with same scaffold splits
4. Report results with 5 seeds

**Estimated time:** 2-3 days

---

### **Phase 1c: Fingerprint + ML Baseline (April 10-11)**

Industry-standard baseline (RandomForest or XGBoost + ECFP4/Morgan fingerprints)

**Implementation:**
```python
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier

# Generate fingerprints
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

# Train RF
model = RandomForestClassifier(n_estimators=1000, random_state=seed)
```

**Estimated time:** 1-2 days (fast to implement)

---

### **Phase 1d: Uni-Mol (Optional - If Time Permits)**

3D pre-training model, computationally expensive.

**Only pursue if:**
- You have ≥3 days buffer before April 30
- SLURM cluster has V100/A100 GPUs available
- GROVER and fingerprint baselines are already complete

**Otherwise:** Skip this. GIN/GCN/GAT + D-MPNN + GROVER + RF is sufficient for Q1.

---

## 📊 REVISED NOVELTY CLAIM (Post Phase 1 Completion)

Once all baselines are complete, your paper's contribution shifts from:

❌ **Old (invalid):** "First application of sparse MoE routing to multi-task ADMET prediction"

✅ **New (defensible):**
1. **MoE as plug-and-play module** across diverse GNN architectures (AttentiveFP, GIN, GCN)
2. **Task-type-dependent optimal expert count:** K=4 for classification, K=8 for regression
3. **Interpretability via expert specialization analysis:** t-SNE visualization + chemical clustering

**Why this works:**
- MI-MoE and GNN-MoCE showed MoE works, but tied to specific architectures
- Your contribution: proving MoE is **architecture-agnostic** (works on AttentiveFP, GIN, GCN)
- Plus: empirical finding that classification vs. regression tasks need different K values

---

## 🎯 DELIVERABLE CHECKLIST (By April 30)

### **Mandatory (Cannot submit without these):**

- [ ] **GIN/GCN/GAT baselines** — 5 seeds, all 9 datasets, mean ± std reported
- [ ] **GROVER baseline** — 5 seeds, all 9 datasets (highest priority after GIN/GCN/GAT)
- [ ] **Fingerprint + RF/XGBoost** — 5 seeds, all 9 datasets
- [ ] **MoE generalization** — AttentiveFP-MoE + GIN-MoE implemented, tested
- [ ] **Statistical significance tests** — Paired t-tests, p-values < 0.05 documented
- [ ] **Interpretability analysis** — t-SNE expert routing, chemical property clustering

### **Strongly Recommended:**

- [ ] **Expert utilization statistics** — Which expert handles which molecules?
- [ ] **Regression task fixes** — FreeSolv performance improved (currently worse than Lipo)
- [ ] **Computational efficiency comparison** — Parameter count, training time, inference speed

### **Optional (Nice to Have):**

- [ ] Uni-Mol baseline (only if timeline allows)
- [ ] D-MPNN with MoE (additional architecture for generalization claim)
- [ ] Virtual screening case study (BBB-permeable molecules)

---

## ⚠️ CRITICAL TIMELINE WARNING

**Today is April 1, 2026.**  
**Submission deadline is April 30, 2026.**  
**You have 29 days.**

**Realistic breakdown:**
- GIN/GCN/GAT baselines: 7 days (April 1-7)
- GROVER baseline: 3 days (April 8-10)
- Fingerprint baseline: 2 days (April 10-11)
- MoE generalization (GIN-MoE): 5 days (April 12-16)
- Interpretability analysis: 3 days (April 17-19)
- Statistical tests + fixes: 3 days (April 20-22)
- **Writing + internal review: 8 days (April 23-30)**

**Total: 31 days** → You're **2 days over budget** even without Uni-Mol.

**What to cut if behind schedule:**
1. Reduce seeds from 5 to 3 (saves ~40% time, still statistically valid)
2. Skip GAT (focus on GIN + GCN + AttentiveFP)
3. Skip Uni-Mol entirely (defer to future work)
4. Accept FreeSolv as-is (don't chase regression fixes unless critical)

**Do NOT cut:**
- GROVER (strongest baseline, mandatory)
- MoE generalization (your new core novelty)
- Statistical tests (Q1 journals require p-values)

---

## 📧 IMMEDIATE NEXT STEPS (Today)

1. **Verify SLURM access** (lab onboarding sections 7.1.1-7.1.2)
   - If unavailable: Apply for Baidu AI Studio immediately

2. **Prepare MoleculeNet datasets** in CSV format:
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

3. **Update data paths** in scripts:
   - `run_gnn_baselines.sh` line 75
   - `run_gnn_baselines_local.bat` line 30

4. **Start Phase 1a execution**:
   - If SLURM: `sbatch run_gnn_baselines.sh`
   - If local: `run_gnn_baselines_local.bat`

5. **Parallel task:** Email professor update:
   > "Dear Professor Li,
   >
   > Following your feedback document, I've identified the two critical issues (invalidated novelty + outdated baselines). I've implemented GIN/GCN/GAT baselines and started Phase 1 experiments today (April 1). Estimated completion: April 7.
   >
   > Revised novelty claim: MoE as architecture-agnostic plug-and-play module with task-type-dependent expert counts.
   >
   > I will provide weekly updates. Phase 1 complete by April 7, GROVER by April 10, MoE generalization by April 16.
   >
   > Best regards,
   > Sapta"

---

## 📞 Support Resources

**Technical issues:**
- Lab seniors: Wu Nanwan, Luo Xixuan (SLURM access)
- SLURM docs: Lab onboarding sections 7.1.1-7.1.2

**Conceptual questions:**
- Professor Li (WeChat)
- Review uploaded images for professor's detailed requirements

**Compute resources:**
1. SAMLab SLURM cluster (first choice)
2. Baidu AI Studio (free V100, backup)
3. AutoDL (student discount, paid backup)
4. Local GTX 1660 Ti (emergency only)

---

## 🎓 Learning Reflection

This situation highlights a crucial lesson for undergraduate research:

**Publishing is not just about novelty, it's about positioning.**

Your MoE work is solid — you beat AttentiveFP on ClinTox and BBBP. But in a competitive field:
- Claiming "first" requires exhaustive literature review (GNN-MoCE, Mol-MoE, MI-MoE existed)
- Baselines define success (AttentiveFP 2020 vs. GROVER 2024 is a 4-year gap)
- Q1 journals expect SOTA comparisons (not just internal ablations)

**Your professor's feedback was harsh but constructive.** He gave you:
1. Explicit prioritized task list
2. Specific missing baselines with performance numbers
3. Concrete timeline (complete by Phase 1, Phase 2, Phase 3)
4. Journal targets (Journal of Cheminformatics, Briefings in Bioinformatics)

**This is recoverable.** You have 29 days and a clear roadmap. Execute Phase 1a this week, and you're back on track.

---

## ✅ FILES DELIVERED

All files are in: `/mnt/user-data/outputs/phase1_gnn_baselines/`

1. `gnn_baselines.py` — Core implementation
2. `run_gnn_baselines.sh` — SLURM execution
3. `run_gnn_baselines_local.bat` — Local fallback
4. `aggregate_gnn_results.py` — Results analysis
5. `generate_baseline_table.py` — Publication table
6. `GNN_BASELINE_QUICKSTART.md` — Execution guide

**Download these, read the quickstart, and start experiments TODAY.**

Good luck. You've got this. 🚀

---

**Target: April 7, 2026 — Phase 1a GNN baselines complete**  
**Next checkpoint: April 10, 2026 — GROVER baseline complete**  
**Final deadline: April 30, 2026 — Paper submission**
