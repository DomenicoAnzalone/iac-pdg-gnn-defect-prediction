# E1 Feature-Family Analysis

Questa analisi confronta la baseline E1 usando Random Forest e una sola famiglia di metriche alla volta:

- `process`
- `product`
- `iac_oriented`
- `delta`

Le run usano lo stesso dataset finale, gli stessi split walk-forward, lo stesso seed, lo stesso bilanciamento e la stessa feature selection della baseline E1. Cambia solo il pool di feature candidate.

Il dry-run crea manifest e split, ma non addestra i modelli.

## Esecuzione completa

```bash
.venv/Scripts/python.exe -m experiments.sensitivity.run_e1_feature_family_analysis --quiet --no-progress
```

`--quiet --no-progress` mantiene la console leggibile: lo script stampa solo una riga per famiglia e il riepilogo finale. Di default questa analisi usa `--n-jobs 1` per evitare lo spam di warning prodotto dal parallelismo interno di scikit-learn in alcune versioni. I dettagli completi restano nei file `logs/run.log` delle singole run.

Se serve privilegiare la velocità rispetto alla pulizia della console, si può passare `--n-jobs -1`, ma in alcune installazioni questo può far ricomparire warning ripetuti di scikit-learn/joblib.

Output principali:

```text
experiments/results/exploratory/e1_feature_family_analysis/e1_feature_family_summary.csv
experiments/results/exploratory/e1_feature_family_analysis/e1_feature_family_summary.md
experiments/results/exploratory/e1_feature_family_<family>_random_forest/
```

