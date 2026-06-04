# Dataset Quality Analysis

Questa cartella contiene la terza pipeline operativa del progetto di tesi,
dedicata all'analisi della qualità dei dataset usati per defect/failure-proneness
prediction su file Ansible.

L'obiettivo è produrre report chiari per capire se il dataset è abbastanza
grande, bilanciato, pulito e adatto al confronto tra:

1. classificatori tradizionali su metriche statiche/process/delta;
2. classificatori tradizionali su metriche statiche + metriche PDG;
3. GNN addestrate direttamente sui PDG file-level.

## Contenuto

```text
dataset_quality_analysis/
├── README.md
├── requirements.txt
└── scripts/
    └── analyze_dataset_quality.py
```

## Input atteso

Lo script accetta un CSV e richiede quattro colonne obbligatorie:

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
