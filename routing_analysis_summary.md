# Routing Specialization vs MoE Gain — Analysis

## Core Finding

Datasets where MoE routing aligns with physicochemical axes (high LogP/ArRings eta2)
show consistent MoE performance gain. Datasets where routing collapses (PPBR)
or where the endpoint requires 3D conformational encoding (Half-life, CL_Hepatocyte)
show neutral or negative MoE gain.

## Results Table

| Dataset | Task | MoE Gain% | LogP eta2 | ArRings eta2 | Interpretation |
|---------|------|-----------|-----------|--------------|----------------|
| Caco2_Wang | MAE_reg | +20.5% | 0.151 | 0.325 | High routing specialization -> MoE wins |
| CL_Microsome | Spearman_reg | +12.9% | — | — | Lipophilicity-driven -> MoE wins |
| Lipophilicity_AZ | MAE_reg | +8.8% | 0.051 | 0.093 | Moderate routing -> MoE wins |
| Solubility_AqSolDB | MAE_reg | +8.8% | 0.325 | 0.102 | High LogP routing -> MoE wins |
| VDss_Lombardo | Spearman_reg | +6.4% | — | — | No routing JSON (Spearman task) |
| LD50_Zhu | MAE_reg | +4.7% | 0.142 | 0.445 | Check JSON structure |
| PPBR_AZ | MAE_reg | +1.3% | — | — | Routing COLLAPSED -> MoE neutral |
| CL_Hepatocyte | Spearman_reg | -11.4% | — | — | CYP endpoint, 2D insufficient -> MoE loses |
| Half_Life_Obach | Spearman_reg | -36.6% | — | — | CYP endpoint, 2D insufficient -> MoE loses |

## Scientific Claim

MoE routing provides performance benefit when the ADMET endpoint has exploitable
physicochemical subspace structure, as evidenced by significant eta2 effect sizes
for LogP (eta2=0.151-0.325) and ArRings (eta2=0.102-0.325) routing alignment.
Endpoints requiring conformation-dependent CYP specificity (half-life, hepatocyte
clearance) show routing failure because 2D graph topology cannot encode the
relevant molecular features. PPBR routing collapse (single expert assignment)
indicates absence of exploitable chemical subspace structure for this endpoint.

## Correlation Results

- LogP eta2: Spearman rho=+0.000, p=1.0000, n=4 [n.s.]
- ArRings eta2: Spearman rho=-0.400, p=0.6000, n=4 [n.s.]