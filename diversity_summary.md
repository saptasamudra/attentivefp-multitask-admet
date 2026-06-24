# Chemical Diversity Analysis -- Summary

## Per-Dataset Results

| Dataset | N | MoE Gain% | LogP Std | Scaffold Div | Silhouette | Tanimoto | Task |
|---------|---|-----------|----------|--------------|------------|----------|------|
| FreeSolv | - | +27.1% | - | - | - | - | MAE_reg |
| Caco2_Wang | 910 | +20.5% | 2.156 | 0.495 | 0.208 | 0.878 | MAE_reg |
| ESOL | - | +19.4% | - | - | - | - | MAE_reg |
| CL_Microsome | 1102 | +12.9% | 1.343 | 0.645 | 0.185 | 0.857 | Spearman_reg |
| Lipophilicity_MN | - | +11.1% | - | - | - | - | MAE_reg |
| Lipophilicity_AZ | 4200 | +8.8% | 1.323 | 0.573 | 0.164 | 0.871 | MAE_reg |
| Solubility_AqSolDB | 9980 | +8.8% | 3.519 | 0.195 | 0.257 | 0.920 | MAE_reg |
| VDss_Lombardo | 1130 | +6.4% | 2.863 | 0.678 | 0.219 | 0.892 | Spearman_reg |
| LD50_Zhu | 7385 | +4.7% | 1.821 | 0.227 | 0.234 | 0.916 | MAE_reg |
| Bioavailability_Ma | 640 | +4.2% | 2.164 | 0.688 | 0.202 | 0.887 | clf |
| CYP3A4_Veith | 12328 | +1.8% | 1.944 | 0.602 | 0.184 | 0.876 | clf |
| PPBR_AZ | 2790 | +1.3% | 1.408 | 0.418 | 0.177 | 0.868 | MAE_reg |
| HIA_Hou | 578 | +1.2% | 2.199 | 0.666 | 0.198 | 0.891 | clf |
| Pgp_Broccatelli | 1218 | +0.6% | 2.071 | 0.560 | 0.199 | 0.874 | clf |
| BBB_Martins | 2030 | +0.5% | 2.096 | 0.505 | 0.243 | 0.890 | clf |
| AMES | 7278 | -0.4% | 2.012 | 0.217 | 0.257 | 0.909 | clf |
| CYP2D6_Veith | 13130 | -1.8% | 1.914 | 0.592 | 0.176 | 0.881 | clf |
| CYP3A4_Substrate | 670 | -1.9% | 1.653 | 0.676 | 0.201 | 0.890 | clf |
| CYP2D6_Substrate | 667 | -2.0% | 1.651 | 0.675 | 0.204 | 0.893 | clf |
| CYP2C9_Substrate | 669 | -3.5% | 1.650 | 0.674 | 0.200 | 0.894 | clf |
| hERG | 655 | -3.6% | 2.155 | 0.600 | 0.212 | 0.887 | clf |
| DILI | 475 | -5.2% | 2.760 | 0.655 | 0.190 | 0.897 | clf |
| CL_Hepatocyte | 1213 | -11.4% | 1.600 | 0.614 | 0.190 | 0.870 | Spearman_reg |
| Half_Life_Obach | 667 | -36.6% | 2.433 | 0.667 | 0.220 | 0.889 | Spearman_reg |

## Spearman Correlations (Diversity -> MoE Gain)

| Metric | rho | p-value | n | Sig |
|--------|-----|---------|---|-----|
| logp_std | -0.021 | 0.9288 | 21 | n.s. |
| scaffold_diversity | -0.314 | 0.1653 | 21 | n.s. |
| silhouette_k5 | -0.038 | 0.8712 | 21 | n.s. |
| tanimoto_diversity | -0.166 | 0.4714 | 21 | n.s. |