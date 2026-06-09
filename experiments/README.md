# Training Pipelines

Questa cartella contiene le pipeline sperimentali riproducibili per il confronto della tesi:

- `e1_tabular_baseline`: classificatori classici addestrati su metriche tabellari non-PDG.
- `e2_tabular_pdg`: gli stessi classificatori classici addestrati su metriche non-PDG più metriche PDG configurabili.
- `e3_gnn`: GNN graph-level addestrate direttamente sui PDG file-level.
- `common`: utility condivise per loading, splitting, preprocessing, balancing, evaluation, reporting e riproducibilità.
- `sensitivity`: utility per la small-graph sensitivity analysis.
- `configs`: configurazioni YAML compatibili con JSON.

## Dataset

Input di default:

```bash
datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv
```

La pipeline si aspetta `repository`, `commit`, `filepath`, `committed_at`, `failure_prone`, `graphml_local_path` oppure `graphml_path`, le colonne di dimensione del grafo `nodes` ed `edges`, e le 11 metriche PDG:

```text
maxPdgVertices, lackOfCohesion, verticesCount, edgesCount, edgesToVerticesRatio,
globalInput, globalOutput, directFanIn, indirectFanIn, directFanOut, indirectFanOut
```

Il confronto principale usa solo campioni con metriche PDG valide e GraphML caricabile, così E1, E2 ed E3 lavorano sulle stesse righe.

## Protocollo

Tutti gli esperimenti usano within-project walk-forward validation:

1. raggruppa le righe per `repository`;
2. ordina i commit usando `committed_at`;
3. usa i commit precedenti come sorgente di train e validation;
4. riserva il commit successivo come test;
5. crea la validation dalla parte più recente della storia di training;
6. salta esplicitamente gli split non validi quando train/test sono vuoti o single-class;
7. salva un `split_manifest.csv` condiviso.

Validation e test non vengono mai bilanciati. Imputation, scaling e feature filtering vengono appresi solo sulla partizione di training di ogni split.

## Preprocessing

E1/E2 usano median imputation, rimozione opzionale delle feature costanti, scaling configurabile (`none`, `min-max`, `standard`) e uno slot configurabile per feature selection (`none`, `variance_threshold`). E1 esclude identificativi, label, path dei grafi, colonne ausiliarie sulla dimensione del grafo e metriche PDG. E2 aggiunge di default tutte le 11 metriche PDG, oppure un sottoinsieme separato da virgole tramite `--pdg-metrics`.

E3 carica GraphML con parsing deterministico tramite NetworkX e crea feature nodali per indicatore di task, gradi, tipo di nodo, semplici feature testuali della label, numero di attributi e presenza di informazioni di posizione. Crea `edge_type` per R-GCN e ignora gli edge type nei modelli che non li usano. Lo scaling delle node feature viene appreso solo sui grafi di training.

## Balancing

Usa:

```text
none | random_undersampling | random_oversampling
```

Il balancing viene applicato solo ai campioni di training dopo la creazione di validation e test. In E3 duplica o rimuove campioni/grafi interi, senza modificare la struttura interna dei grafi.

## Modelli

Modelli classici:

```text
decision_tree, logistic_regression, naive_bayes, random_forest, svm
```

Modelli GNN:

```text
gcn, graphsage, gat, gin, rgcn
```

Random Forest e GraphSAGE sono le scelte di default più comode per gli smoke test.

## Metriche E Output

Ogni run scrive in:

```text
experiments/results/<run_name>/
```

File principali:

```text
config.yaml
metadata.json
split_manifest.csv
feature_manifest.csv
excluded_samples.csv
predictions/*_predictions.csv
metrics/per_split_metrics.csv
metrics/per_repository_metrics.csv
metrics/aggregated_metrics.csv
logs/skipped_splits.csv
reports/run_summary.md
```

Le metriche sono AUC-PR, AUC-ROC, MCC, precision, recall, F1, accuracy e conteggi della confusion matrix. I casi in cui AUC/MCC non sono definiti vengono salvati come `NaN` con warning.

## Smoke Test

Le pipeline stampano log leggibili in console e salvano sempre una copia completa in `experiments/results/<run_name>/logs/run.log`. Le progress bar aggiornano la stessa riga quando il terminale lo supporta. Usa `--no-progress` per disattivarle, `--quiet` per scrivere solo su file e `--log-every-epochs N` per ridurre i log per epoca nelle GNN.

Per le run GNN lunghe, usa `--compact-progress`: la console mostra solo i log essenziali e una singola riga di avanzamento che riporta split globale, epoca corrente, metriche sintetiche ed ETA. I dettagli completi restano nel file `logs/run.log`. La riga compatta si adatta alla larghezza corrente della console per ridurre i problemi quando si ridimensiona PowerShell.

E1 rapido:

```bash
python -m experiments.e1_tabular_baseline.run --config experiments/configs/e1_default.yaml --run-name smoke_e1 --models random_forest --max-repositories 2 --max-splits 2 --max-samples 2000 --balance random_oversampling --scaler standard --seed 42
```

E2 rapido:

```bash
python -m experiments.e2_tabular_pdg.run --config experiments/configs/e2_default.yaml --run-name smoke_e2 --pdg-metrics all --models random_forest --max-repositories 2 --max-splits 2 --max-samples 2000 --balance random_oversampling --scaler standard --seed 42
```

E3 rapido:

```bash
python -m experiments.e3_gnn.run --config experiments/configs/e3_default.yaml --run-name smoke_e3 --model graphsage --min-nodes 3 --min-edges 2 --max-repositories 1 --max-splits 1 --max-samples 1500 --balance random_oversampling --epochs 3 --batch-size 8 --seed 42 --compact-progress
```

Usa `--dry-run` per validare loading, filtering e creazione degli split senza eseguire training.

## Run Complete

E1:

```bash
python -m experiments.e1_tabular_baseline.run --config experiments/configs/e1_default.yaml --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --run-name e1_baseline_rf --models random_forest --balance random_oversampling --scaler standard --seed 42
```

E2:

```bash
python -m experiments.e2_tabular_pdg.run --config experiments/configs/e2_default.yaml --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --run-name e2_pdg_all_metrics --pdg-metrics all --models random_forest,svm,logistic_regression --balance random_oversampling --scaler standard --seed 42
```

E3:

```bash
python -m experiments.e3_gnn.run --config experiments/configs/e3_default.yaml --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --run-name e3_graphsage --model graphsage --min-nodes 3 --min-edges 2 --balance random_oversampling --epochs 100 --batch-size 32 --seed 42
```

## Confronto

Dopo che una cartella risultati contiene metriche per split da uno o più esperimenti:

```bash
python -m experiments.compare_results --results-dir experiments/results/<run_name> --output experiments/results/<run_name>/reports/comparison_summary.md
```

Produce `comparison_summary.csv` e `.md`, includendo test Wilcoxon paired quando sono disponibili coppie a livello di split.

## Small-Graph Sensitivity

Report descrittivo delle soglie:

```bash
python -m experiments.sensitivity.small_graph_analysis --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --thresholds 3:2,5:4,8:6,10:6 --output-dir experiments/results/small_graph_sensitivity
```

La sensitivity sul training si esegue ripetendo E1/E2/E3 con valori diversi di `--min-nodes` e `--min-edges`. Ogni run salva config, esclusioni e performance, quindi l'effetto delle soglie può essere confrontato con `compare_results.py`.

## Limiti Noti

- Il protocollo derivato dai PDF è codificato a partire dal contesto del progetto e dalle note implementative locali disponibili; durante l'implementazione l'ambiente non aveva una libreria per estrarre testo dai PDF.
- RFECV/RFE è presente come slot di configurazione, ma non è abilitato nella pipeline veloce di default.
- Il confronto statistico include al momento sintesi descrittive e Wilcoxon paired; Friedman/Nemenyi può essere aggiunto usando la tabella per split già salvata.
- La pipeline finale per E1/E2/E3 è sotto `experiments/`; non dipende da una cartella legacy esterna.
