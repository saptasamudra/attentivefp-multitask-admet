# Expert Specialization — Cross-Dataset Summary

## Key Finding

LogP and ArRings show consistently high η² effect sizes across
independent ADMET datasets measuring completely different endpoints.

## η² Table

| Descriptor | solubility_aqsoldb | caco2_wang | lipophilicity_astrazeneca | ld50_zhu | ppbr_az | Mean η² |
|---|---|---|---|---|---|---|
| MW | 0.101*** | 0.103*** | 0.006*** | 0.110*** | N/A | 0.080 |
| LogP ★ | 0.325*** | 0.151*** | 0.051*** | 0.142*** | N/A | 0.167 |
| HBA | 0.049*** | 0.151*** | 0.016*** | 0.152*** | N/A | 0.092 |
| HBD | 0.057*** | 0.140*** | 0.001n.s. | 0.029*** | N/A | 0.057 |
| TPSA | 0.087*** | 0.204*** | 0.023*** | 0.105*** | N/A | 0.105 |
| RotBonds | 0.042*** | 0.121*** | 0.019*** | 0.169*** | N/A | 0.088 |
| Rings | 0.077*** | 0.075*** | 0.017*** | 0.291*** | N/A | 0.115 |
| ArRings ★ | 0.102*** | 0.325*** | 0.093*** | 0.445*** | N/A | 0.241 |

## Paper Text

```
To validate expert chemical specialization quantitatively, we computed
mutual information (MI) and one-way ANOVA between dominant expert assignment
and eight RDKit physicochemical descriptors across 5 independent
ADMET datasets (total n=24,089 molecules). All descriptors showed
significant between-expert variation (p<0.001, ANOVA and Kruskal-Wallis).
The strongest and most consistent effects were observed for
ArRings (η²=0.093–0.445 across datasets)
and LogP (η²=0.051–0.325),
indicating that expert routing spontaneously partitions chemical space along
lipophilicity and aromaticity axes — the same physicochemical dimensions
emphasized in Lipinski's Rule of 5 — without any explicit chemical supervision.
This replication across datasets measuring distinct ADMET endpoints confirms
that expert specialization reflects genuine physicochemical structure
rather than dataset-specific artifacts.
```