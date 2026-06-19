# IaC PDG GNN Defect Prediction

Repository di tesi per costruire un dataset di file Ansible etichettati e
confrontare approcci tabellari e graph-based per la defect/failure-proneness
prediction in Infrastructure as Code.

Il progetto integra quattro macro-fasi indipendenti ma collegate:

1. discovery ed estrazione di metriche RADON da repository Ansible;
2. estrazione di Program Dependence Graph file-level con Scansible;
3. costruzione di un dataset finale versionato con metriche RADON e PDG;
4. benchmark sperimentale tra modelli tabellari e Graph Neural Network.

L'obiettivo è ottenere un confronto riproducibile tra:

- **E1 - Tabular baseline**: classificatori tradizionali su metriche statiche,
  process e delta;
- **E2 - Tabular + PDG metrics**: gli stessi classificatori arricchiti con
  metriche derivate dai Program Dependence Graph;
- **E3 - GNN on PDG**: Graph Neural Network addestrate direttamente sui grafi
  file-level.

## Struttura Del Progetto

```text
.
├── docs/                       # contesto metodologico, note e log di tesi
├── radon_dataset_extraction/   # discovery repository + dataset RADON
├── pdg_file_level_extraction/  # estrazione PDG file-level con Scansible
├── dataset_preparation/        # qualità, filtering e dataset finale
├── datasets/                   # artefatti dataset versionati
├── experiments/                # training, benchmark e sensitivity analysis
├── tests/                      # test della pipeline sperimentale
├── output/                     # output intermedi e run operative locali
├── requirements.txt            # dipendenze principali per esperimenti/test
└── README.md
```

Ogni macro-fase ha un README dedicato con comandi, parametri e note operative.
Il README root serve invece come mappa generale della repository e del flusso
end-to-end.

## Pipeline End-To-End

Il flusso logico della repository è:

```text
radon_dataset_extraction
  -> pdg_file_level_extraction
  -> dataset_preparation
  -> experiments
```

La pipeline è pensata per essere modulare: ogni fase può essere rilanciata,
testata o aggiornata separatamente, mantenendo snapshot, metadati e report
riproducibili. Gli output operativi finiscono in `output/`, mentre gli artefatti
curati e versionati sono conservati in `datasets/`.

## Dataset Finale Corrente

Il dataset finale usato dagli esperimenti si trova in:

```text
datasets/ansible-pdg-defect-dataset/final/v2026-06-06/
```

Artefatti principali:

```text
ansible-pdg-defect-dataset_v2026-06-06_final.csv
ansible-pdg-defect-dataset_v2026-06-06_exclusions.csv
ansible-pdg-defect-dataset_v2026-06-06_pdg_metrics.csv
DATASET_REPORT.md
manifest.json
reports/
```

Sintesi della versione `v2026-06-06`:

| Misura | Valore |
|---|---:|
| Righe dopo filtering RADON | 61.542 |
| Righe con PDG estratto con successo | 37.834 |
| Righe finali dopo graph-quality filtering | 37.796 |
| Repository finali | 123 |
| Colonne finali | 160 |
| Campioni non failure-prone | 23.880 |
| Campioni failure-prone | 13.916 |

La policy di qualità dei grafi mantiene solo PDG con almeno 3 nodi e 2 archi.
Questa soglia rimuove grafi vuoti, edgeless o placeholder, senza escludere a
priori file Ansible piccoli ma validi. La motivazione completa è documentata in
`DATASET_REPORT.md`.

## Fasi Principali

### 1. RADON Dataset Extraction

Cartella: `radon_dataset_extraction/`

Questa fase costruisce il dataset tabellare di partenza. Include:

- discovery automatica di repository GitHub legate ad Ansible;
- filtering delle repository candidate;
- estrazione di fixing commit e file Ansible coinvolti;
- calcolo di metriche statiche, process e delta;
- generazione di dataset per repository;
- merge e filtering finale delle repository con dati sufficienti.

La pipeline è eseguita tramite Docker per fissare l'ambiente Python e isolare le
dipendenze RADON/repository-miner.

Output tipici:

```text
output/runs/<run-name>/batch_summary.csv
output/runs/<run-name>/merged_dataset.csv
output/runs/<run-name>/merged_dataset_filtered.csv
```

### 2. File-Level PDG Extraction

Cartella: `pdg_file_level_extraction/`

Questa fase prende il dataset RADON e prova a estrarre un PDG file-level per
ogni riga, usando la tupla:

```text
repository, commit, filepath
```

La pipeline gestisce checkout dei commit, wrapper temporanei per file `tasks/`,
invocazione di Scansible, conversione DOT -> GraphML, validazione minima del
grafo, parallelismo per repository e resume sicuro.

Output tipici:

```text
output/<run-name>/extraction_status.csv
output/<run-name>/pdg_file_level/.../pdg.graphml
output/<run-name>/<input>_rows_successful_extracted.csv
```

Gli stati di estrazione distinguono successi, file non supportati, errori reali,
grafi vuoti, grafi troppo piccoli, timeout e problemi di checkout/clone.

### 3. Dataset Preparation

Cartella: `dataset_preparation/`

Questa fase trasforma gli output RADON e PDG in un dataset finale adatto agli
esperimenti. In particolare:

- unisce dataset RADON e `extraction_status.csv`;
- mantiene solo righe con PDG valido;
- calcola metriche PDG file-level;
- produce esclusioni motivate;
- genera manifest JSON, report Markdown e tabelle di controllo;
- salva una versione immutabile sotto `datasets/<dataset-id>/final/<version>/`.

Metriche PDG incluse:

```text
maxPdgVertices, lackOfCohesion, verticesCount, edgesCount,
edgesToVerticesRatio, globalInput, globalOutput,
directFanIn, indirectFanIn, directFanOut, indirectFanOut
```

Le metriche sono marcate con semantica `file_level_proxy_v1`, perché il GraphML
finale rappresenta un grafo aggregato a livello di file.

### 4. Experiments

Cartella: `experiments/`

Contiene le pipeline sperimentali finali:

- `e1_tabular_baseline/`: modelli classici su feature non-PDG;
- `e2_tabular_pdg/`: modelli classici con aggiunta di metriche PDG;
- `e3_gnn/`: GNN graph-level sui PDG file-level;
- `common/`: loading, splitting, preprocessing, balancing, evaluation e report;
- `sensitivity/`: analisi su soglie, split, balancing, seed e setup tabellare;
- `configs/`: configurazioni YAML per E1, E2 ed E3.

Il protocollo principale usa **within-project walk-forward validation**:

1. raggruppa i campioni per repository;
2. ordina la storia tramite `committed_at`;
3. usa il passato per train/validation;
4. valuta sul commit successivo;
5. evita leakage facendo apprendere preprocessing e feature selection solo sul
   training set.

Metriche prodotte:

```text
AUC-PR, AUC-ROC, MCC, precision, recall, F1, accuracy,
confusion matrix counts
```

Il benchmark finale confronta cinque classificatori classici e cinque modelli
GNN:

```text
decision_tree, logistic_regression, naive_bayes, random_forest, svm
gcn, graphsage, gat, gin, rgcn
```

## Quick Start

Installare le dipendenze principali:

```powershell
python -m pip install -r requirements.txt
```

Eseguire i test disponibili:

```powershell
python -m pytest
```

Eseguire uno smoke test E1:

```powershell
python -m experiments.e1_tabular_baseline.run `
  --config experiments/configs/e1_default.yaml `
  --run-name smoke_e1 `
  --models random_forest `
  --max-repositories 2 `
  --max-splits 2 `
  --max-samples 2000 `
  --balance random_oversampling `
  --scaler standard `
  --seed 42
```

Eseguire uno smoke test E3:

```powershell
python -m experiments.e3_gnn.run `
  --config experiments/configs/e3_default.yaml `
  --run-name smoke_e3 `
  --model graphsage `
  --min-nodes 3 `
  --min-edges 2 `
  --max-repositories 1 `
  --max-splits 1 `
  --max-samples 1500 `
  --balance random_oversampling `
  --epochs 3 `
  --batch-size 8 `
  --seed 42 `
  --compact-progress
```

Eseguire il benchmark completo:

```powershell
python -m experiments.run_full_benchmark `
  --benchmark-name final_benchmark_v1 `
  --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv
```

Gli output del benchmark vengono scritti in:

```text
experiments/results/benchmark/final_benchmark_v1/
```

## Convenzioni Di Output

La repository distingue tra output operativi e artefatti consolidati:

- `output/`: run temporanee, log, cloni intermedi, estrazioni operative,
  report esplorativi;
- `datasets/`: dataset curati, versionati e documentati;
- `experiments/results/`: run sperimentali, metriche, predizioni, manifest e
  report di confronto.

Questa separazione permette di ripetere o scartare run operative senza perdere
gli artefatti finali usati nella tesi.

## Documentazione Utile

- `docs/THESIS_PROJECT_CONTEXT.md`: contesto metodologico generale;
- `docs/EXPERIMENT_LOG.md`: log e note sugli esperimenti;
- `radon_dataset_extraction/README.md`: discovery e pipeline RADON;
- `pdg_file_level_extraction/README.md`: estrazione PDG file-level;
- `dataset_preparation/README.md`: qualità, filtering e dataset versionato;
- `datasets/ansible-pdg-defect-dataset/README.md`: descrizione del dataset;
- `experiments/README.md`: training, benchmark e sensitivity analysis.

## Note Di Riproducibilità

- Le fasi RADON e PDG usano Docker per rendere stabile l'ambiente operativo.
- Le run principali salvano snapshot degli input e metadati di esecuzione.
- Le pipeline supportano resume quando possibile.
- Il dataset finale include manifest, report umano e report CSV di controllo.
- Gli esperimenti salvano configurazioni, split manifest, predizioni, metriche
  aggregate e log.

In sintesi, la repository è organizzata per mantenere separati tre livelli:
costruzione dei dati, consolidamento del dataset e valutazione sperimentale. In
questo modo ogni risultato finale può essere ricondotto alla fase e agli
artefatti che lo hanno prodotto.
