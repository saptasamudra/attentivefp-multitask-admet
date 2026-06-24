# Table: Differentiation from Competing Methods

| Aspect | ASE-Mol | MolGraph-xLSTM | **MoE-GCN (Ours)** |
|---|---|---|---|
| **Core architecture** | Adaptive substructure extraction + GNN | xLSTM sequence model on molecular graphs | GCN/DMPNN/GIN backbone + sparse MoE routing layer |
| **MoE routing** | No | No | Yes — top-k sparse expert routing |
| **Architecture-agnostic** | No (tied to substructure extraction) | No (tied to xLSTM) | **Yes** — plug-in module for any GNN encoder |
| **Backbone flexibility** | Single architecture | Single architecture | GCN, DMPNN, GIN tested |
| **Dataset coverage** | MoleculeNet (6–8 datasets) | MoleculeNet (6–8 datasets) | **22 TDC ADMET + MoleculeNet (25 total)** |
| **Routing interpretability** | No routing mechanism | No routing mechanism | **Permutation-tested physicochemical alignment (p<0.001)** |
| **Endpoint analysis** | Not provided | Not provided | **Endpoint-dependent gain framework (Mann-Whitney p=0.0076)** |
| **Statistical validation** | Mean ± std | Mean ± std | Wilcoxon signed-rank + permutation tests |
| **Failure mode analysis** | Not provided | Not provided | **Explicit: metabolic endpoints identified** |
| **Demo / deployment** | No | No | Hugging Face Spaces (live) |
| **Parameter efficiency** | Not analyzed | Not analyzed | Ablation vs equal-parameter baseline |

## Key differentiators (narrative form for Introduction)

**vs ASE-Mol:**
ASE-Mol focuses on adaptive substructure extraction as the novelty —
it learns which molecular substructures to attend to. MoE-GCN operates
at a different level: rather than changing what features are extracted,
it learns *which expert pathway* a molecule should traverse based on
its position in chemical space. These are complementary, not competing,
contributions. Additionally, ASE-Mol is architecture-specific while
MoE-GCN is a plug-in applicable to any GNN backbone.

**vs MolGraph-xLSTM:**
MolGraph-xLSTM replaces the GNN message-passing paradigm with
sequential xLSTM processing of graph-linearized molecular representations.
MoE-GCN retains message-passing (preserving established inductive biases
for molecular graphs) and adds mixture-of-experts routing on top.
MolGraph-xLSTM has not been benchmarked on TDC ADMET datasets;
our 22-dataset TDC coverage provides a more comprehensive ADMET
evaluation than either competing method.

**The unified claim:**
Neither ASE-Mol nor MolGraph-xLSTM (1) operates as an architecture-agnostic
plug-in, (2) provides interpretable routing analysis with statistical
validation, (3) covers 22 TDC ADMET endpoints, or (4) identifies
endpoint-dependent performance conditions. MoE-GCN addresses all four.
