# Training Pipelines

Questa cartella contiene le pipeline sperimentali riproducibili per il confronto della tesi:

- `e1_tabular_baseline`: classificatori classici addestrati su metriche tabellari non-PDG.
- `e2_tabular_pdg`: gli stessi classificatori classici addestrati su metriche non-PDG più metriche PDG configurabili.
- `e3_gnn`: GNN graph-level addestrate direttamente sui PDG file-level.
- `common`: utility condivise per loading, splitting, preprocessing, balancing, evaluation, reporting e riproducibilità.
- `sensitivity`: utility per analisi esplorative e sensitivity analysis.
- `configs`: configurazioni YAML compatibili con JSON.
- `results/exploratory`: risultati dei test usati per fissare il setup sperimentale.
- `results/benchmark`: risultati del benchmark finale.

## Dataset

Input di default:

```bash
datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv
```

La pipeline si aspetta `repository`, `commit`, `filepath`, `committed_at`, `failure_prone`, `graphml_local_path` oppure `graphml_path`, le colonne di dimensione del grafo `nodes` ed `edges`, e le 11 metriche PDG disponibili nel dataset:

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

E1/E2 usano median imputation, rimozione opzionale delle feature costanti, scaling configurabile (`none`, `min-max`, `standard`) e feature selection configurabile (`none`, `variance_threshold`, `rfe`, `rfecv`). E1 esclude identificativi, label, path dei grafi, colonne ausiliarie sulla dimensione del grafo e metriche PDG. E2 aggiunge di default tutte le 11 metriche PDG come candidate e usa RFECV train-only per selezionare le feature più utili in ogni split. Gli alias `top4` e `top5` restano disponibili per analisi secondarie.

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

Per E3 l'early stopping usa di default MCC sulla validation. Se in uno split la metrica non è definibile perché la validation è troppo piccola o non contiene entrambe le classi, la pipeline usa la validation loss come fallback solo per scegliere il checkpoint migliore. Le metriche finali restano calcolate esclusivamente sul test set.

Random Forest e GraphSAGE sono le scelte di default più comode per gli smoke test.

## Metriche E Output

Ogni run scrive in:

```text
experiments/results/<run_name>/
```

Le run esplorative e di sensitivity sono raccolte in:

```text
experiments/results/exploratory/
```

Il benchmark finale viene scritto separatamente in:

```text
experiments/results/benchmark/<benchmark_name>/
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
metrics/pooled_metrics.csv
metrics/per_repository_pooled_metrics.csv
metrics/aggregated_metrics.csv
logs/skipped_splits.csv
reports/run_summary.md
```

Le metriche sono AUC-PR, AUC-ROC, MCC, precision, recall, F1, accuracy e conteggi della confusion matrix. I risultati principali sono in `pooled_metrics.csv`: la pipeline aggrega prima tutte le predizioni dei test walk-forward e poi calcola le metriche sul totale. Le medie per split restano in `aggregated_metrics.csv` come diagnostica. I casi in cui AUC/MCC non sono definiti vengono salvati come `NaN` con warning.

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
python -m experiments.e2_tabular_pdg.run --config experiments/configs/e2_default.yaml --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --run-name e2_pdg_all_rfecv --pdg-metrics all --models random_forest,svm,logistic_regression --balance random_oversampling --scaler standard --seed 42
```

E3:

```bash
python -m experiments.e3_gnn.run --config experiments/configs/e3_default.yaml --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --run-name e3_graphsage --model graphsage --min-nodes 3 --min-edges 2 --balance random_oversampling --epochs 100 --batch-size 32 --seed 42
```

## Benchmark Finale

Il benchmark completo esegue in sequenza:

- E1 con `decision_tree`, `logistic_regression`, `naive_bayes`, `random_forest`, `svm`;
- E2 con gli stessi cinque classificatori, tutte le 11 metriche PDG candidate e RFECV train-only;
- E3 con `gcn`, `graphsage`, `gat`, `gin`, `rgcn`.

Comando principale:

```bash
python -m experiments.run_full_benchmark --benchmark-name final_benchmark_v1 --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv
```

Output:

```text
experiments/results/benchmark/final_benchmark_v1/
```

Ogni modello ha una cartella separata, ad esempio:

```text
experiments/results/benchmark/final_benchmark_v1/e1_random_forest/
experiments/results/benchmark/final_benchmark_v1/e2_random_forest/
experiments/results/benchmark/final_benchmark_v1/e3_graphsage/
```

Il riepilogo progressivo è in:

```text
experiments/results/benchmark/final_benchmark_v1/_summary/benchmark_plan.csv
experiments/results/benchmark/final_benchmark_v1/_summary/benchmark_summary.csv
experiments/results/benchmark/final_benchmark_v1/_summary/benchmark_summary.md
```

La pipeline è riavviabile: se viene interrotta, rilancia lo stesso comando. Le run che hanno già `metrics/pooled_metrics.csv` vengono saltate automaticamente. Se una run si interrompe a metà, quella specifica run viene rieseguita dall'inizio, ma le run già completate non vengono ripetute.

Per forzare il ricalcolo anche delle run già complete:

```bash
python -m experiments.run_full_benchmark --benchmark-name final_benchmark_v1 --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --rerun-existing
```

Smoke test del benchmark:

```bash
python -m experiments.run_full_benchmark --benchmark-name smoke_benchmark --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --max-repositories 1 --max-splits 1 --max-samples 1500 --epochs 1 --batch-size 8
```

## Confronto

Dopo che una cartella risultati contiene metriche per split da uno o più esperimenti:

```bash
python -m experiments.compare_results --results-dir experiments/results/<run_name> --output experiments/results/<run_name>/reports/comparison_summary.md
```

Per il benchmark finale si può passare direttamente la root del benchmark:

```bash
python -m experiments.compare_results --results-dir experiments/results/benchmark/final_benchmark_v1
```

Produce `comparison_summary.csv` e `.md`, includendo test Wilcoxon paired quando sono disponibili coppie a livello di split. Su una root benchmark gli output vengono scritti in `_summary/`.

## Small-Graph Sensitivity

Report descrittivo delle soglie:

```bash
python -m experiments.sensitivity.small_graph_analysis --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --thresholds 3:2,5:4,8:6,10:6 --output-dir experiments/results/exploratory/small_graph_sensitivity
```

La sensitivity sul training si esegue ripetendo E1/E2/E3 con valori diversi di `--min-nodes` e `--min-edges`. Ogni run salva config, esclusioni e performance, quindi l'effetto delle soglie può essere confrontato con `compare_results.py`.

Per E3 si può lanciare una sweep sequenziale con un solo comando:

```bash
python -m experiments.sensitivity.run_threshold_sweep --config experiments/configs/e3_default.yaml --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --thresholds 3:2,5:4,8:6,10:6 --run-prefix e3_graphsage_threshold --model graphsage --balance random_oversampling --epochs 100 --batch-size 32 --seed 42 --log-every-epochs 5
```

La sweep crea una run separata per ogni soglia e un riepilogo in `experiments/results/exploratory/small_graph_threshold_sweep/threshold_run_summary.csv`.

## Split Reliability

Dopo aver scelto la soglia sui grafi, si può analizzare quanto gli split con test set molto piccoli influenzano le metriche finali. Questa analisi non riaddestra i modelli: legge le predizioni salvate, filtra gli split in base a `test_size` e numero di positivi nel test, poi ricalcola le metriche pooled.

```bash
python -m experiments.sensitivity.split_reliability_analysis --results-dir experiments/results/exploratory/e3_graphsage_threshold_n3_e2 --min-test-sizes 2,5,10,20,30,50 --min-test-positives 1,2,3,5 --min-test-negatives 1
```

Output principali:

```text
experiments/results/exploratory/e3_graphsage_threshold_n3_e2/reports/split_reliability/split_reliability_summary.csv
experiments/results/exploratory/e3_graphsage_threshold_n3_e2/reports/split_reliability/split_reliability_summary.md
experiments/results/exploratory/e3_graphsage_threshold_n3_e2/reports/split_reliability/split_test_size_distribution.csv
```

La decisione finale deve bilanciare stabilità delle metriche e copertura: una soglia che migliora leggermente MCC ma elimina troppi campioni di test non è necessariamente migliore.

## Balance Strategy Sweep

Dopo aver fissato soglia dei grafi e gestione degli split, si può confrontare la strategia di bilanciamento del training set. Lo sweep lancia E3 GraphSAGE più volte con la stessa configurazione e cambia solo:

```text
none, random_oversampling, random_undersampling
```

Comando completo:

```bash
python -m experiments.sensitivity.run_balance_sweep --config experiments/configs/e3_default.yaml --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --strategies none,random_oversampling,random_undersampling --run-prefix e3_graphsage_balance --model graphsage --min-nodes 3 --min-edges 2 --epochs 100 --batch-size 32 --seed 42 --log-every-epochs 5 --compact-progress
```

Output principali:

```text
experiments/results/exploratory/e3_balance_sweep/balance_run_summary.csv
experiments/results/exploratory/e3_balance_sweep/balance_run_summary.md
```

Ogni singola run viene salvata in:

```text
experiments/results/exploratory/e3_graphsage_balance_none/
experiments/results/exploratory/e3_graphsage_balance_random_oversampling/
experiments/results/exploratory/e3_graphsage_balance_random_undersampling/
```

Se una run esiste già e vuoi ricostruire solo il riepilogo senza riaddestrare, aggiungi:

```bash
--skip-existing
```

Il confronto va letto soprattutto da `mcc`, `auc_pr`, `precision`, `recall`, `f1` e `predicted_positive_rate`. MCC resta la metrica guida; AUC-PR è importante perché il dataset è sbilanciato; precision/recall servono a capire se una strategia sta semplicemente predicendo troppi positivi.

## Seed Stability Sweep

Dopo aver fissato soglia dei grafi, gestione degli split e bilanciamento, si può verificare se la configurazione E3 è stabile rispetto al seed. Lo sweep lancia GraphSAGE più volte cambiando solo il seed.

Comando completo:

```bash
python -m experiments.sensitivity.run_seed_sweep --config experiments/configs/e3_default.yaml --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --seeds 42,7,123 --run-prefix e3_graphsage_seed --model graphsage --min-nodes 3 --min-edges 2 --balance random_oversampling --epochs 100 --batch-size 32 --log-every-epochs 5 --compact-progress
```

Output principali:

```text
experiments/results/exploratory/e3_seed_sweep/seed_run_summary.csv
experiments/results/exploratory/e3_seed_sweep/seed_run_summary.md
```

Ogni singola run viene salvata in:

```text
experiments/results/exploratory/e3_graphsage_seed_42/
experiments/results/exploratory/e3_graphsage_seed_7/
experiments/results/exploratory/e3_graphsage_seed_123/
```

Se una run esiste già e vuoi ricostruire solo il riepilogo senza riaddestrare, aggiungi:

```bash
--skip-existing
```

La configurazione è considerata stabile se MCC, AUC-PR e F1 cambiano poco tra seed e se il positive rate predetto resta coerente.

## E1/E2 Tabular Common Setup

Dopo aver fissato la configurazione comune dalla fase esplorativa E3, si può lanciare il primo confronto tabellare E1 vs E2. Questo test usa Random Forest come modello iniziale:

- E1: feature tabellari non-PDG;
- E2: stesse feature di E1 più tutte le 11 metriche PDG candidate;
- E2 usa RFECV train-only di default;
- il bilanciamento è `random_oversampling`, solo sul training set;
- soglia grafi `3/2`, così E1/E2 restano sul dataset comune usato da E3.

Comando completo:

```bash
python -m experiments.sensitivity.run_tabular_e1_e2 --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --run-prefix tabular_rf_common --models random_forest --pdg-metrics all --balance random_oversampling --scaler standard --e1-feature-selection none --e2-feature-selection rfecv --seed 42
```

Output principali:

```text
experiments/results/exploratory/tabular_e1_e2_common/tabular_e1_e2_summary.csv
experiments/results/exploratory/tabular_e1_e2_common/tabular_e1_e2_summary.md
```

Run singole:

```text
experiments/results/exploratory/tabular_rf_common_e1/
experiments/results/exploratory/tabular_rf_common_e2/
```

Smoke test rapido:

```bash
python -m experiments.sensitivity.run_tabular_e1_e2 --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --run-prefix smoke_tabular_rf_common --summary-dir experiments/results/exploratory/smoke_tabular_e1_e2_common --models random_forest --pdg-metrics all --balance random_oversampling --scaler standard --e1-feature-selection none --e2-feature-selection rfecv --seed 42 --max-repositories 1 --max-splits 2 --max-samples 1500
```

Se una run esiste già e vuoi rigenerare solo il riepilogo, aggiungi:

```bash
--skip-existing
```

## Limiti Noti

- E2 usa tutte le 11 metriche PDG come candidate e RFECV come default; questo può rendere le run tabellari più lente rispetto alla modalità senza feature selection.
- Il confronto statistico include al momento sintesi descrittive e Wilcoxon paired; Friedman/Nemenyi può essere aggiunto usando la tabella per split già salvata.
- La pipeline finale per E1/E2/E3 è sotto `experiments/`; non dipende da una cartella legacy esterna.
