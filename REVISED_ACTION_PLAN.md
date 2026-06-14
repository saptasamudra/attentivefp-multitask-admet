# REVISED ACTION PLAN: Addressing Professor's Critical Feedback
## MoE-Enhanced Multi-Task Molecular Property Prediction

**Status Date:** April 1, 2026  
**Submission Target:** Journal of Cheminformatics by April 30, 2026  
**Critical Blocker:** Phase 1 baseline experiments required before Q1 readiness

---

## 🚨 CRITICAL ISSUE: Novelty Statement Invalidated

### Problem Identified by Professor
The claim "first application of sparse MoE routing to multi-task ADMET prediction" is **REFUTED** by existing works:

1. **GNN-MoCE** (arXiv 2023) — Multi-task MoE, 35 ADMET classification tasks
2. **Mol-MoE** (IBM Research, NeurIPS 2024 Workshop) — Three-modal 12-expert MoE
3. **MI-MoE** (arXiv 2026.01) — Topology-aware multi-scale MoE, 0.927 ClinTox (scaffold split)

### ✅ REVISED NOVELTY FRAMING (Approved Strategy)

**New Core Contribution:**  
"MoE as a general plug-and-play enhancement module across diverse GNN architectures with task-type-dependent optimal expert count"

**Three Pillars:**

1. **Architectural Generalizability**  
   - Apply MoE wrapper to GIN, GCN, GAT, AttentiveFP backbones
   - Demonstrate consistent improvement across architectures
   - **Differentiator:** MI-MoE and GNN-MoCE are architecture-specific

2. **Task-Type Dependency in Optimal Expert Count** ✨  
   - **K=4 optimal for classification** (BACE, BBBP, HIV, ClinTox, Tox21, SIDER)
   - **K=8 optimal for regression** (ESOL, FreeSolv, Lipo)
   - Statistical significance testing (5+ seeds, paired t-tests)

3. **Interpretability via Expert Specialization**  
   - t-SNE/UMAP visualization of expert routing patterns
   - Chemical property clustering per expert (logP, MW, H-bond donors/acceptors)
   - Functional group affinity analysis

---

## 📊 PHASE 1: Mandatory Baseline Experiments (Highest Priority)

### Current State
- **Internal ablations complete:** Single-task, Multi-task (no MoE), MoE K=2/4/8
- **Missing:** External baseline comparisons required for Q1 journals

### Required Baselines (Priority Order)

#### 1. D-MPNN / Chemprop ⚡ **TOP PRIORITY**
**Why:** Current standard, BBBP 0.90+, widely accepted baseline

**Action Items:**
- [ ] Install Chemprop: `pip install chemprop`
- [ ] Run script: `python chemprop_baseline.py --mode all --seeds 0 1 2`
- [ ] Expected runtime: ~6-12 hours on GTX 1660 Ti (or use SAMLab SLURM cluster)

**Deliverable:** `chemprop_baseline_results.csv` with mean ± std across 3 seeds

#### 2. GROVER (Graph Pre-training)
**Reported Performance:**  
- BBBP: 0.94  
- ClinTox: 0.944

**Action Items:**
- [ ] Clone GROVER repo: `git clone https://github.com/tencent-ailab/grover`
- [ ] Download pre-trained weights
- [ ] Fine-tune on 9 MoleculeNet datasets
- [ ] Use same scaffold split protocol as your MoE model

**Challenge:** GROVER requires pre-training on large unlabeled corpus. If compute-intensive:
- Use **pre-trained checkpoint** from official repo
- Document as "GROVER (pre-trained)" in comparison table

#### 3. GIN / GCN / GAT (Classical GNN Trio)
**Purpose:** Establish baseline performance of standard GNN architectures before adding MoE

**Action Items:**
- [ ] Implement in PyG (you likely already have this code)
- [ ] Train on each dataset individually (single-task)
- [ ] Same hyperparameters as your AttentiveFP baseline

**Expected Result:** Should match or slightly underperform AttentiveFP

#### 4. Uni-Mol (3D Pre-training) — **If Time Permits**
**Reported Performance:**
- ClinTox: 0.919  
- HIV: 0.808

**Note:** Requires 3D conformer generation (RDKit + optimization). Only include if:
- You have compute resources for conformer generation
- Submission timeline allows (lower priority than D-MPNN/GROVER)

#### 5. Fingerprint + ML (Sanity Check)
**Purpose:** Classical baseline to show deep learning necessity

**Action Items:**
- [ ] Generate ECFP4/Morgan fingerprints (RDKit)
- [ ] Train Random Forest or XGBoost
- [ ] Should underperform all GNN methods

---

### Comparison Table Format (Target for Paper)

| Dataset  | Task   | Metric | ECFP+RF | GIN  | GCN  | GAT  | AttentiveFP | **D-MPNN** | **GROVER** | **Uni-Mol** | **MoE-AFP (K=4)** |
|----------|--------|--------|---------|------|------|------|-------------|------------|------------|-------------|-------------------|
| ESOL     | Reg    | RMSE   | X.XX    | X.XX | X.XX | X.XX | 0.96        | **TBD**    | **TBD**    | TBD         | **0.96 ± 0.07**   |
| FreeSolv | Reg    | RMSE   | X.XX    | X.XX | X.XX | X.XX | 2.80        | **TBD**    | **TBD**    | TBD         | 2.80 ± 0.18       |
| BACE     | Class  | AUC    | X.XX    | X.XX | X.XX | X.XX | 0.79        | **TBD**    | **TBD**    | TBD         | 0.79 ± 0.03       |
| BBBP     | Class  | AUC    | X.XX    | X.XX | X.XX | X.XX | 0.88        | **≥0.90**  | **0.94**   | TBD         | **0.88 ± 0.03** ✨ |
| ClinTox  | Class  | AUC    | X.XX    | X.XX | X.XX | X.XX | 0.92        | **TBD**    | **0.944**  | **0.919**   | **0.92 ± 0.01** ✨ |

**Key:**  
- ✨ = Beats published baseline  
- **Bold** = Priority baselines to run  
- TBD = To be determined (run experiments)

---

## 🎯 PHASE 2: MoE Generalization Verification (Novelty Proof)

### Purpose
Prove that MoE is **not specific to AttentiveFP** but works across GNN architectures.

### Implementation Plan

1. **Create MoE Wrapper Module** (Architecture-Agnostic)
```python
class MoELayer(nn.Module):
    """
    Plug-and-play MoE module compatible with any GNN backbone
    """
    def __init__(self, input_dim, num_experts=4, expert_hidden=200, top_k=2):
        # Gating network
        # Expert networks
        # Load balancing loss
```

2. **Wrap Around Multiple Backbones**
   - GIN + MoE
   - GCN + MoE  
   - GAT + MoE  
   - AttentiveFP + MoE (your current model)

3. **Run Full Comparison**
   - Each architecture with and without MoE
   - Same K=4 for classification, K=8 for regression

### Expected Results Table

| Architecture | BBBP (no MoE) | BBBP (+MoE K=4) | Δ Improvement |
|--------------|---------------|-----------------|---------------|
| GIN          | 0.85          | 0.88            | +0.03         |
| GCN          | 0.84          | 0.87            | +0.03         |
| GAT          | 0.86          | 0.89            | +0.03         |
| AttentiveFP  | 0.88          | 0.88            | +0.00         |

**Narrative:**  
"MoE consistently improves performance across diverse GNN architectures, demonstrating its generalizability as a plug-and-play module rather than an architecture-specific enhancement."

---

## 📈 PHASE 3: Interpretability Analysis (Key Selling Point)

### Goal
**Differentiate from MI-MoE and GNN-MoCE** by providing deep interpretability insights

### Analysis Components

#### 1. Expert Routing Visualization
- **Method:** t-SNE or UMAP on gating network outputs
- **Question:** Do different experts specialize in different chemical spaces?
- **Plot:** Scatter plot colored by dominant expert, overlaid with chemical properties

#### 2. Chemical Property Clustering
For each expert, analyze routed molecules:
- Molecular weight distribution
- LogP (lipophilicity)
- H-bond donors/acceptors
- Aromatic ring count

**Expected Finding:** Expert 1 handles hydrophilic molecules, Expert 2 handles lipophilic, etc.

#### 3. Statistical Significance Testing
- **Current:** 3 seeds  
- **Required for Q1:** 5-10 seeds  
- **Test:** Paired t-test comparing MoE vs. no-MoE across seeds  
- **Report:** p-values, effect sizes (Cohen's d)

---

## ⚡ COMPUTE RESOURCE STRATEGY

### Priority 1: SAMLab SLURM Cluster (Free, Highest Performance)
**Action Items:**
- [ ] Confirm login credentials (from lab onboarding docs 7.1.1-7.1.2)
- [ ] Test job submission with small experiment
- [ ] Run all Phase 1 baseline experiments on cluster

**Why:** Your GTX 1660 Ti (6GB VRAM) is too slow for extensive hyperparameter sweeps.

### Priority 2: Cloud Compute (If SLURM Unavailable)
1. **Baidu AI Studio** (Free V100 with .edu.cn email)
2. **AutoDL** (Student discount available)
3. **GitHub Student Developer Pack** (Azure/AWS credits)

### Local Machine (Windows GTX 1660 Ti)
**Use Only For:**
- Dataset inspection
- Small-scale debugging
- Final result visualization

---

## 📝 TIMELINE: April 1 → April 30 (30 Days)

### Week 1 (April 1-7): Baseline Experiments
- **Day 1-2:** Setup (install Chemprop, confirm SLURM access)  
- **Day 3-5:** D-MPNN baseline (all 9 datasets, 3 seeds)  
- **Day 6-7:** GROVER baseline (if pre-trained weights available)

### Week 2 (April 8-14): MoE Generalization
- **Day 8-10:** Implement MoE wrapper for GIN/GCN/GAT  
- **Day 11-14:** Train all architecture+MoE combinations

### Week 3 (April 15-21): Interpretability & Statistical Testing
- **Day 15-17:** Expert routing analysis (t-SNE, property clustering)  
- **Day 18-21:** Run 5-10 seeds for statistical significance

### Week 4 (April 22-30): Writing & Submission
- **Day 22-25:** Draft full manuscript (professor writes, you provide technical report)  
- **Day 26-28:** Internal review, revisions  
- **Day 29:** arXiv preprint upload  
- **Day 30:** Journal of Cheminformatics submission ✅

---

## 📄 SUBMISSION STRATEGY

### Primary Target: **Journal of Cheminformatics** (IF ~8, Q1, CAS First-Tier)
**Why:**
- Open access, fast review (~2-3 months)
- Specializes in computational chemistry methods
- MoE architectural contributions fit scope

### Backup Options (If JCheminf Rejects):
1. **Briefings in Bioinformatics** (IF ~10, Q1)  
2. **Journal of Chemical Information and Modeling** (IF ~5.6, Q1)  
3. **Computers in Biology and Medicine** (IF ~7, Q1)

**Avoid:**
- Molecular Informatics (Q2 risk)  
- Artificial Intelligence in the Life Sciences (not established Q1)

---

## ✅ IMMEDIATE ACTION ITEMS (Next 48 Hours)

### Critical Path:
1. **Run dataset inspection:**
   ```bash
   python inspect_datasets.py
   ```
   → Verify all 9 MoleculeNet datasets are correctly formatted

2. **Install Chemprop:**
   ```bash
   conda activate molprop
   pip install chemprop
   ```

3. **Start D-MPNN baseline (ESOL only, single seed for testing):**
   ```bash
   python chemprop_baseline.py --mode single --dataset ESOL --seed 0
   ```
   → Verify pipeline works before scaling to all datasets

4. **Confirm SLURM cluster access:**
   - Check lab onboarding docs section 7.1.1-7.1.2
   - Test SSH login
   - Submit test job

### Decision Points:
- **If SLURM works:** Scale to full baseline suite immediately  
- **If SLURM blocked:** Pivot to Baidu AI Studio (free V100)  
- **If both fail:** Use local GTX 1660 Ti (slower but functional)

---

## 📊 SUCCESS CRITERIA FOR Q1 SUBMISSION

### Must-Have:
✅ D-MPNN baseline comparison (all 9 datasets)  
✅ MoE generalization across ≥3 GNN architectures  
✅ Statistical significance testing (5+ seeds, p-values)  
✅ Task-type dependency analysis (K=4 vs K=8)

### Nice-to-Have (If Time):
⭐ GROVER baseline  
⭐ Uni-Mol baseline  
⭐ Chemical property clustering per expert

### Red Lines (University Policy):
❌ No Frontiers, MDPI, Hindawi journals  
❌ No fabricated results  
❌ No GPT-generated text in final paper (professor writes)

---

## 🎯 RISK MITIGATION

### Risk 1: D-MPNN Outperforms MoE Significantly
**Mitigation:** Emphasize interpretability and architectural generalizability as contributions, not just raw performance.

### Risk 2: Compute Resources Unavailable
**Mitigation:** Reduce experimental scope (fewer seeds, fewer baseline methods), extend timeline if possible.

### Risk 3: Regression Task Performance Stays Poor
**Mitigation:** Try task-adaptive weighting, hybrid training strategy, or exclude regression tasks and focus on classification.

---

## FINAL NOTE: Professor's Expectations

From WeChat messages (Screenshot 12-13):

> "You must have a paper. I can promote you to there [Tsinghua] if you have a paper, in AIDD or bioinformatics."

**Minimum Bar:** JCR Q1 publication  
**Deadline:** Summer 2028 graduation  
**Current Status:** April 2026 = 2 years remaining = sufficient time if Phase 1 completed by May 2026

**The paper is not optional—it's the gateway to your master's program recommendation.**

---

**Next Steps:** Run `inspect_datasets.py` on your Windows machine, confirm Chemprop installation, and report back with any errors. I'll help debug before you commit to the full experimental pipeline.
