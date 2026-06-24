# PHASE 1 EXECUTION PLAN

## Status: Code Ready (no training yet)

### Files Created:
1. ✅ `baseline_runner.py` — GIN, GCN, GAT on 9 datasets (scaffold split, 3 seeds)
2. ✅ `generic_moe_module.py` — Plug-and-play MoE wrapper for any encoder
3. ✅ `routing_analyzer.py` — t-SNE, expert utilization heatmaps, chemical correlation
4. ✅ `statistical_tester.py` — Paired t-tests, Wilcoxon, 95% CI
5. ✅ `MoE_PAPER_ROADMAP.md` — Full 6-week timeline

---

## NEXT STEPS (In Order)

### Step 1: Run Baselines (Week 1-2)
```bash
python baseline_runner.py
# Outputs: results/Baseline_GIN_9datasets_[TIMESTAMP].txt
#          results/Baseline_GCN_9datasets_[TIMESTAMP].txt
#          results/Baseline_GAT_9datasets_[TIMESTAMP].txt
```
**Expected runtime:** ~8-10 hours GPU (3 seeds × 3 models)

### Step 2: Create MoE Variants for Each Baseline (Week 2)
Modify `baseline_runner.py` to wrap each model with MoE:
```python
# Pseudo-code
from generic_moe_module import GenericMoEModel, MixtureOfExperts

for model_class, model_name in [GINMultiTask, GCNMultiTask, GATMultiTask]:
    encoder = model_class.encoder  # extract encoder
    moe_model = GenericMoEModel(encoder, K=4)  # wrap with MoE
    # train & evaluate
```

### Step 3: Routing Analysis (Week 3)
```python
from routing_analyzer import RoutingAnalyzer

analyzer = RoutingAnalyzer(moe_model, test_loaders, num_experts=4)
routing_data = analyzer.collect_routing_decisions()
analyzer.print_stats(routing_data)
analyzer.visualize_routing_tsne(routing_data)
analyzer.visualize_expert_utilization_heatmap(routing_data)
```
**Output:** 
- `routing_tsne.png` — molecular clustering by expert
- `expert_util_heatmap.png` — per-dataset expert utilization
- Console stats on expert specialization

### Step 4: Statistical Significance (Week 3)
```python
from statistical_tester import StatisticalTester

# Load all results from Step 1 & 2
tester = StatisticalTester(all_seed_results_dict)
tester.print_detailed_comparison('Baseline_GIN', 'MoE_K4_GIN', datasets, test_type='ttest')
tester.print_detailed_comparison('Baseline_GIN', 'MoE_K4_GIN', datasets, test_type='wilcoxon')
tester.confidence_intervals('MoE_K4_GIN', datasets)
```

---

## DELIVERABLES (End Phase 1)

1. **Baseline Results** (3 new .txt files): GIN, GCN, GAT baseline scores
2. **MoE Generalization Results** (3 new .txt files): MoE+GIN, MoE+GCN, MoE+GAT
3. **Routing Analysis Plots** (2 PNG files): t-SNE, expert utilization heatmap
4. **Statistical Report** (1 CSV + console output): p-values, 95% CI, significance markers

---

## EXPECTED OUTCOMES

✓ **Novelty Claim Restored:** "MoE is plug-and-play module" (not just AttentiveFP-specific)
✓ **Competitive Baselines:** Compared vs GIN/GCN/GAT (20+ papers cite these)
✓ **Interpretability Edge:** Expert routing visualization (MI-MoE doesn't do this depth)
✓ **Rigorous Stats:** 95% CI + paired tests (publishable rigor)

---

## REGRESSION PROBLEM (Phase 2)

Note: FreeSolv/Lipo underperform vs published SOTA.
**Phase 2 solution:** Task weighting (GradNorm, Uncertainty Weighting, PCGrad)
Plan to address Week 4-5.

---

## Code Integration Checklist

Before running:
- [ ] GPU available (check `torch.cuda.is_available()`)
- [ ] PyG installed with GIN, GCN, GAT models
- [ ] RDKit installed (for GenFeatures)
- [ ] scikit-learn, scipy installed (for stats)
- [ ] matplotlib, seaborn installed (for plots)
- [ ] `data/` directory exists (MoleculeNet auto-downloads on first run)

---

## Commands to Execute Now

```bash
# Check setup
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Run baseline_runner.py (single longest step)
# Start in background if needed:
# nohup python baseline_runner.py > baseline_log.txt 2>&1 &

# After complete, run stats
python statistical_tester.py  # will use saved .txt files

# Generate routing analysis (only for MoE models)
# python routing_analyzer.py  # requires MoE results
```

---

## TIMELINE SUMMARY

| Week | Task | Hours GPU | Deliverable |
|------|------|-----------|-------------|
| 1-2 | Baselines (GIN/GCN/GAT) | 8-10h | 3 .txt files |
| 2 | MoE+Baselines | 8-10h | 3 .txt files |
| 3 | Routing analysis + stats | 1-2h | 2 plots + report |
| 4-5 | Task weighting (Phase 2) | 5-6h | improved FreeSolv/Lipo |

**Total Phase 1: 22-28 hours GPU + 1-2 hours analysis**
