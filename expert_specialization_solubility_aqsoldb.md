# Expert Specialization Statistics — solubility_aqsoldb

## Summary
- Significant descriptors (p<0.05): MW, LogP, HBA, HBD, TPSA, RotBonds, Rings, ArRings
- High effect size (η²>0.05): MW, LogP, HBD, TPSA, Rings, ArRings

## Statistics Table

| Descriptor | MI | F-stat | p-ANOVA | p-KW | η² | Sig |
|------------|-----|--------|---------|------|----|-----|
| MW | 0.144 | 279.5 | 0.0000 | 0.0000 | 0.101 | *** |
| LogP | 0.390 | 1203.0 | 0.0000 | 0.0000 | 0.325 | *** |
| HBA | 0.071 | 129.6 | 0.0000 | 0.0000 | 0.049 | *** |
| HBD | 0.042 | 150.4 | 0.0000 | 0.0000 | 0.057 | *** |
| TPSA | 0.089 | 237.1 | 0.0000 | 0.0000 | 0.087 | *** |
| RotBonds | 0.050 | 108.3 | 0.0000 | 0.0000 | 0.042 | *** |
| Rings | 0.129 | 208.1 | 0.0000 | 0.0000 | 0.077 | *** |
| ArRings | 0.110 | 284.6 | 0.0000 | 0.0000 | 0.102 | *** |

## LaTeX Table

```latex
\begin{table}[h]
\centering
\caption{Expert routing specialization statistics. F-statistics from one-way ANOVA across expert groups; $\eta^2$ = eta-squared effect size; MI = mutual information with discretized descriptor.}
\begin{tabular}{lrrrrrrrr}
\hline
Descriptor & MW & LogP & HBA & HBD & TPSA & RotBonds & Rings & ArRings \\
\hline
Expert 1 & $314.8\pm252.3$ & $-0.7\pm4.3$ & $4.8\pm5.6$ & $1.4\pm1.9$ & $96.2\pm104.9$ & $4.1\pm5.4$ & $1.5\pm2.1$ & $1.0\pm1.6$ \\
Expert 3 & $281.9\pm122.9$ & $2.7\pm2.2$ & $3.5\pm2.6$ & $0.7\pm0.9$ & $56.3\pm43.8$ & $3.2\pm3.6$ & $1.4\pm1.1$ & $1.0\pm0.7$ \\
Expert 5 & $313.2\pm196.1$ & $4.5\pm3.2$ & $2.8\pm2.9$ & $0.7\pm1.1$ & $44.9\pm46.1$ & $5.5\pm7.8$ & $2.1\pm1.7$ & $1.6\pm1.5$ \\
Expert 6 & $88.5\pm30.6$ & $0.3\pm1.0$ & $1.5\pm0.9$ & $1.1\pm1.1$ & $30.8\pm26.3$ & $1.4\pm1.5$ & $0.2\pm0.4$ & $0.0\pm0.2$ \\
Expert 7 & $199.5\pm101.0$ & $0.9\pm1.6$ & $3.3\pm2.3$ & $1.5\pm1.6$ & $64.8\pm42.7$ & $3.0\pm2.7$ & $1.1\pm1.3$ & $0.7\pm0.7$ \\
\hline
F-stat & $279.5$ & $1203.0$ & $129.6$ & $150.4$ & $237.1$ & $108.3$ & $208.1$ & $284.6$ \\
p-value & $0.0000$ & $0.0000$ & $0.0000$ & $0.0000$ & $0.0000$ & $0.0000$ & $0.0000$ & $0.0000$ \\
$\eta^2$ & $0.101$ & $0.325$ & $0.049$ & $0.057$ & $0.087$ & $0.042$ & $0.077$ & $0.102$ \\
MI & $0.144$ & $0.390$ & $0.071$ & $0.042$ & $0.089$ & $0.050$ & $0.129$ & $0.110$ \\
\hline
\end{tabular}
\end{table}
```


SUGGESTED PAPER TEXT:
─────────────────────
To validate expert chemical specialization quantitatively, we computed mutual
information (MI) between dominant expert assignment and seven RDKit physicochemical
descriptors across all 9980 molecules in the solubility_aqsoldb dataset,
and performed one-way ANOVA across expert groups. Significant between-expert
variation was observed for LogP, ArRings, MW (all p < 0.001,
ANOVA), confirming that expert routing captures meaningful physicochemical
structure. Effect sizes (η²) indicate that expert identity explains
33% of LogP variance, 10% of ArRings variance,
consistent with spontaneous learning of Lipinski-like chemical space partitioning.
