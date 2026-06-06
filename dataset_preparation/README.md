# Dataset Quality Analysis and Filtering

Questa cartella contiene la terza pipeline operativa del progetto di tesi,
dedicata sia all'analisi della qualità sia alla costruzione del dataset finale
versionato usato per defect/failure-proneness prediction su file Ansible.

Gli obiettivi sono:

- produrre report chiari per capire se il dataset è abbastanza grande,
  bilanciato, pulito e adatto agli esperimenti;
- filtrare le righe senza PDG file-level valido;
- filtrare grafi vuoti, edgeless o troppo piccoli per rappresentare dipendenze
  utili;
- arricchire il dataset con metriche PDG file-level;
- salvare dataset, esclusioni, manifest e report in una versione sotto
  `datasets/<dataset-id>/final/<version>/`.

Il dataset finale è il punto di ingresso comune per il confronto tra:

1. classificatori tradizionali su metriche statiche/process/delta;
2. classificatori tradizionali su metriche statiche + metriche PDG;
3. GNN addestrate direttamente sui PDG file-level.

## Contenuto

```text
dataset_quality_analysis/
├── README.md
├── requirements.txt
└── scripts/
    ├── analyze_dataset_quality.py
    └── build_versioned_dataset.py
```

## Pipeline finale versionata

`build_versioned_dataset.py` prende il CSV RADON completo e lo status della PDG
extraction, li unisce sulla chiave logica:

```text
repository, commit, filepath
```

poi produce:

- dataset finale arricchito con metriche RADON, path GraphML e metriche PDG;
- report delle esclusioni con una motivazione primaria per ogni riga rimossa;
- manifest JSON con input, parametri, output e conteggi di ogni fase;
- report Markdown unico con storia del dataset, soglie e note metodologiche;
- tabelle di supporto su label, repository, status, esclusioni e metriche PDG.

### Comando usato per `ansible-2 v2026-06-06`

```powershell
python dataset_quality_analysis\scripts\build_versioned_dataset.py `
  --radon-input output\second_pdg_extraction\input_dataset.csv `
  --extraction-status datasets\ansible-2\pdg_extraction\extraction_status.csv `
  --output-root datasets\ansible-2\final `
  --dataset-id ansible-2 `
  --version v2026-06-06 `
  --min-pdg-nodes 3 `
  --min-pdg-edges 2 `
  --graph-base-dir output `
  --force
```

Output principale:

```text
datasets/ansible-2/final/v2026-06-06/
├── ansible-2_v2026-06-06_final.csv
├── ansible-2_v2026-06-06_exclusions.csv
├── ansible-2_v2026-06-06_pdg_metrics.csv
├── DATASET_REPORT.md
├── manifest.json
└── reports/
```

### Soglie PDG

La soglia predefinita è:

```text
min_pdg_nodes = 3
min_pdg_edges = 2
```

La scelta è conservativa e coincide con la configurazione della run di
estrazione PDG. Serve a rimuovere grafi vuoti, edgeless o placeholder, non a
scartare file Ansible piccoli ma validi. La motivazione è documentata in
`DATASET_REPORT.md`, insieme ai riferimenti online controllati:

- PyTorch Geometric: rappresentazione tramite `edge_index` e gestione dei nodi
  isolati;
- DGL: message passing su nodi e archi;
- NetworkX: distinzione tra grafi nulli, vuoti e senza archi.

### Metriche PDG calcolate

Il dataset finale include le 11 metriche della linea Iuliano/Pontillo:

```text
maxPdgVertices, lackOfCohesion, verticesCount, edgesCount,
edgesToVerticesRatio, globalInput, globalOutput,
directFanIn, indirectFanIn, directFanOut, indirectFanOut
```

Poiché l'artefatto attuale è un GraphML file-level già aggregato, alcune
metriche sono proxy file-level documentate con
`pdg_metric_semantics=file_level_proxy_v1`. In particolare, la sovrapposizione
esatta tra slice task-level non è ricostruibile dal solo GraphML finale; per
questo `lackOfCohesion` viene calcolata come proxy normalizzata di connettività
tra task nel grafo file-level.

## Input atteso

Lo script di sola analisi `analyze_dataset_quality.py` accetta un CSV e richiede
quattro colonne obbligatorie:

- label binaria;
- repository;
- commit;
- filepath.

Le altre colonne sono opzionali. Se una sezione opzionale non può essere
calcolata, viene saltata senza interrompere l'analisi.

## Output prodotti

La cartella indicata da `--output-dir` viene creata automaticamente e contiene:

- `report_summary.txt`;
- `report_summary.json`;
- `missing_values_by_column.csv`;
- `label_distribution.csv`;
- `repository_distribution.csv`;
- `duplicate_keys.csv`;
- `numeric_metrics_summary.csv`;
- `suspicious_features.csv`;
- `highly_correlated_features.csv`;
- `pdg_coverage_report.csv`;
- `gnn_coverage_report.csv`;

Quando applicabile vengono prodotti anche:

- `label_distribution_by_repository.csv`;
- `instances_by_year.csv`;
- `instances_by_month.csv`;
- `label_distribution_by_month.csv`.

## Uso sul CSV della PDG extraction

Per analizzare la run `output/second_pdg_extraction`, usare il CSV completo degli
stati. Questo è il file più utile per valutare copertura PDG, errori di
estrazione e dataset GNN utilizzabile.

```powershell
cd "C:\Users\dosoa\Documents\Tesi Magistrale\iac-pdg-gnn-defect-prediction"

python dataset_quality_analysis\scripts\analyze_dataset_quality.py `
  --input output\second_pdg_extraction\extraction_status.csv `
  --label-column failure_prone `
  --repo-column repository `
  --commit-column commit `
  --file-column filepath `
  --status-column status `
  --graph-path-column graphml_path `
  --graph-base-dir output `
  --pdg-prefixes nodes,edges,pdg_,graph_ `
  --output-dir output\dataset_quality_report\second_pdg_extraction_status
```

Nota: `graphml_path` contiene percorsi Docker del tipo `/app/output/...`. Lo
script li riconcilia con `--graph-base-dir output` quando verifica l'esistenza
dei file sul disco locale.

## Uso sul dataset RADON di input

Per analizzare metriche statiche/process/delta prima della PDG extraction:

```powershell
python dataset_quality_analysis\scripts\analyze_dataset_quality.py `
  --input output\second_pdg_extraction\input_dataset.csv `
  --label-column failure_prone `
  --repo-column repo_url `
  --commit-column commit `
  --file-column filepath `
  --date-column committed_at `
  --static-prefixes additions,deletions,highest_contributor_experience,hunks_median,minor_contributors_count,delta_,num_,lines_,avg_,change_set_,code_churn_,commits_,contributors_,text_entropy `
  --pdg-prefixes pdg_,graph_ `
  --output-dir output\dataset_quality_report\second_pdg_extraction_input_dataset
```

## Parametri principali

| Parametro | Descrizione |
|---|---|
| `--input` | CSV da analizzare |
| `--label-column` | Colonna della label binaria |
| `--repo-column` | Colonna repository o URL repository |
| `--commit-column` | Colonna commit |
| `--file-column` | Colonna filepath |
| `--date-column` | Colonna data opzionale |
| `--status-column` | Colonna opzionale con esito PDG |
| `--graph-path-column` | Colonna opzionale con path del grafo |
| `--graph-base-dir` | Base locale per verificare l'esistenza dei grafi |
| `--static-prefixes` | Prefissi metriche statiche/process/delta |
| `--pdg-prefixes` | Prefissi metriche PDG/graph |
| `--output-dir` | Directory dei report |

## Interpretazione

`report_summary.txt` è il punto di partenza umano. I CSV servono per analisi più
mirate, ad esempio:

- feature costanti o quasi costanti;
- coppie di feature altamente correlate;
- repository con poche istanze;
- chiavi duplicate `repository, commit, filepath`;
- copertura PDG per label e repository;
- grafi realmente utilizzabili per GNN.

Il report include anche warning sul rischio di data leakage: uno split random
row-level può essere troppo ottimistico se lo stesso file o la stessa repository
compare sia in training sia in test. Per il confronto finale restano preferibili
split per repository, split temporali o within-project walk-forward.
