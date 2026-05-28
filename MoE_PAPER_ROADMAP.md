# MoE Multi-Task ADMET — Paper Roadmap (Sapta)

## 🎯 Current Status
- ✅ MoE K=2,4,8 benchmarked (9 datasets, 3 seeds)
- ❌ **Novelty claim broken** (5+ competitors published similar work)
- ❌ Baselines outdated (only 2020 AttentiveFP)

---

## PHASE 1: Hard Requirements (2-3 weeks)

### 1.1 Modern Baselines [CRITICAL]
**On same scaffold split, same 9 datasets:**
- [ ] GIN, GCN, GAT (classical trio)
- [ ] D-MPNN (Chemprop)
- [ ] GROVER (pretrained)
- [ ] Uni-Mol (3D, if GPU available)
- [ ] ECFP4 + XGBoost (fingerprint baseline)

**Strategy:** Use existing PyG models where possible; clone Chemprop repo

### 1.2 MoE Generalization [CORE NOVELTY FIX]
**Apply MoE to multiple architectures:**
- [ ] MoE + GIN
- [ ] MoE + GCN
- [ ] MoE + D-MPNN
- [ ] Ablate MoE contribution per architecture

**Claim shift:** "MoE is plug-and-play enhancement module" (vs. narrow "MoE on AttentiveFP only")

### 1.3 Routing Interpretability [DIFFERENTIATOR vs MI-MoE]
- [ ] t-SNE/UMAP visualization of expert routing by molecule
- [ ] Analyze learned expert specialization (scaffold? molecular weight? pharmacophore?)
- [ ] Expert utilization heatmap across K={2,4,8}
- [ ] Per-dataset expert distribution statistics

### 1.4 Statistical Rigor
- [ ] Increase seeds: 3 → 5-10
- [ ] Paired t-tests / Wilcoxon signed-rank on all results
- [ ] Report 95% confidence intervals, not just mean±std

---

## PHASE 2: Competitive Edge (2-3 weeks)

### 2.1 Regression Performance Fix
- [ ] Task weighting: GradNorm / Uncertainty Weighting
- [ ] Try: Classification multi-task + Regression independent
- [ ] Investigate FreeSolv/Lipo gap vs SOTA

### 2.2 Training Dynamics
- [ ] Loss curves (train/val) by K
- [ ] Load-balance loss tracking over epochs
- [ ] Convergence speed comparison

### 2.3 Computational Analysis
- [ ] Parameters, training time, inference time per config
- [ ] Memory vs speedup trade-off

---

## PHASE 3: Polish (if targeting top journal)
- [ ] TDC ADMET Benchmark external validation
- [ ] Pretrain + MoE fine-tune pipeline
- [ ] Virtual screening case study

---

## 📊 Target Journals (by likelihood post-experiments)
| Result Quality | Target | IF |
|---|---|---|
| Strong | Journal of Cheminformatics | ~8 |
| Strong | Briefings in Bioinformatics | ~10 |
| Good | Journal of Chemical Information and Modeling | ~5.6 |
| Backup | Artificial Intelligence in the Life Sciences | ~5 |

---

## ⏱️ Timeline
- **Week 1-2:** Phase 1.1, 1.2 (baselines + MoE generalization)
- **Week 2-3:** Phase 1.3, 1.4 (interpretability + statistics)
- **Week 4-5:** Phase 2 (performance fixes)
- **Week 6+:** Phase 3 (polish) + manuscript writing

---

## 💾 Code Reuse Strategy
- Existing: GenFeatures, scaffold_split, train_epoch logic ✓
- Adapt: MoE module → generic wrapper for any encoder
- New: Baseline runners, routing analysis, visualization scripts

**Estimated new code:** ~2-3k lines (modest footprint)
