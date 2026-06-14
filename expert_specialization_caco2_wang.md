# Expert Specialization Statistics — caco2_wang

## Summary
- Significant descriptors (p<0.05): MW, LogP, HBA, HBD, TPSA, RotBonds, Rings, ArRings
- High effect size (η²>0.05): MW, LogP, HBA, HBD, TPSA, RotBonds, Rings, ArRings

## Statistics Table

| Descriptor | MI | F-stat | p-ANOVA | p-KW | η² | Sig |
|------------|-----|--------|---------|------|----|-----|
| MW | 0.109 | 25.9 | 0.0000 | 0.0000 | 0.103 | *** |
| LogP | 0.115 | 40.2 | 0.0000 | 0.0000 | 0.151 | *** |
| HBA | 0.124 | 40.3 | 0.0000 | 0.0000 | 0.151 | *** |
| HBD | 0.111 | 36.9 | 0.0000 | 0.0000 | 0.140 | *** |
| TPSA | 0.200 | 58.1 | 0.0000 | 0.0000 | 0.204 | *** |
| RotBonds | 0.151 | 31.2 | 0.0000 | 0.0000 | 0.121 | *** |
| Rings | 0.113 | 18.4 | 0.0000 | 0.0000 | 0.075 | *** |
| ArRings | 0.240 | 108.8 | 0.0000 | 0.0000 | 0.325 | *** |

## LaTeX Table

```latex
\begin{table}[h]
\centering
\caption{Expert routing specialization statistics. F-statistics from one-way ANOVA across expert groups; $\eta^2$ = eta-squared effect size; MI = mutual information with discretized descriptor.}
\begin{tabular}{lrrrrrrrr}
\hline
Descriptor & MW & LogP & HBA & HBD & TPSA & RotBonds & Rings & ArRings \\
\hline
Expert 0 & $344.0\pm131.5$ & $2.8\pm1.7$ & $4.4\pm2.4$ & $1.6\pm1.6$ & $69.3\pm42.0$ & $3.9\pm2.9$ & $3.5\pm1.4$ & $2.4\pm0.9$ \\
Expert 6 & $515.1\pm191.6$ & $1.6\pm2.4$ & $8.6\pm5.1$ & $4.3\pm3.1$ & $140.4\pm78.2$ & $4.9\pm3.1$ & $4.4\pm1.5$ & $0.7\pm1.0$ \\
Expert 12 & $432.0\pm140.8$ & $2.6\pm1.5$ & $5.2\pm1.7$ & $2.6\pm1.7$ & $104.0\pm39.7$ & $7.5\pm3.7$ & $3.2\pm1.5$ & $2.2\pm0.9$ \\
Expert 13 & $445.7\pm166.7$ & $1.0\pm2.5$ & $6.4\pm2.9$ & $2.9\pm2.1$ & $125.4\pm53.3$ & $6.3\pm4.7$ & $2.7\pm1.9$ & $1.1\pm1.1$ \\
Expert 15 & $401.3\pm94.5$ & $3.1\pm1.4$ & $5.3\pm1.6$ & $1.7\pm0.9$ & $95.3\pm26.1$ & $5.5\pm2.8$ & $3.1\pm0.9$ & $2.6\pm0.9$ \\
\hline
F-stat & $25.9$ & $40.2$ & $40.3$ & $36.9$ & $58.1$ & $31.2$ & $18.4$ & $108.8$ \\
p-value & $0.0000$ & $0.0000$ & $0.0000$ & $0.0000$ & $0.0000$ & $0.0000$ & $0.0000$ & $0.0000$ \\
$\eta^2$ & $0.103$ & $0.151$ & $0.151$ & $0.140$ & $0.204$ & $0.121$ & $0.075$ & $0.325$ \\
MI & $0.109$ & $0.115$ & $0.124$ & $0.111$ & $0.200$ & $0.151$ & $0.113$ & $0.240$ \\
\hline
\end{tabular}
\end{table}
```


SUGGESTED PAPER TEXT:
─────────────────────
To validate expert chemical specialization quantitatively, we computed mutual
information (MI) between dominant expert assignment and seven RDKit physicochemical
descriptors across all 910 molecules in the caco2_wang dataset,
and performed one-way ANOVA across expert groups. Significant between-expert
variation was observed for ArRings, TPSA, HBA (all p < 0.001,
ANOVA), confirming that expert routing captures meaningful physicochemical
structure. Effect sizes (η²) indicate that expert identity explains
32% of ArRings variance, 20% of TPSA variance,
consistent with spontaneous learning of Lipinski-like chemical space partitioning.
