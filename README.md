# IaC PDG GNN Defect Prediction

Questa repository contiene la pipeline di tesi per costruire un dataset di file
Ansible etichettati e confrontare tre strategie di defect prediction:

1. modelli tabellari con metriche RADON;
2. modelli tabellari con metriche RADON + metriche PDG;
3. Graph Neural Network addestrate sui PDG file-level.

## Idea della pipeline

Ogni cartella principale rappresenta una macro-fase separata della tesi.

Ogni macro-fase funziona in modo autonomo: ha i propri script, README,
dipendenze e output. Allo stesso tempo, le fasi sono collegate in una pipeline:
l'output di una fase diventa l'input della fase successiva.

```text
radon_dataset_extraction
  -> pdg_file_level_extraction
  -> dataset_preparation
  -> gnn
```

`docs`, `datasets` e `output` supportano questa pipeline: conservano contesto,
dataset versionati e risultati intermedi.

## Cartelle principali

### `docs/`

Contiene il contesto metodologico della tesi e le note operative.

### `radon_dataset_extraction/`

Macro-fase 1. Cerca repository Ansible candidate, estrae metriche RADON e produce
un dataset tabellare con label `failure_prone`.

Output principale atteso: un CSV RADON con colonne come repository, commit,
filepath, label e metriche statiche/process/delta.

### `pdg_file_level_extraction/`

Macro-fase 2. Prende il dataset RADON, fa checkout dei commit, invoca Scansible e
produce PDG file-level per ogni riga quando l'estrazione riesce.

Output principale atteso: `extraction_status.csv`, grafi `pdg.graphml` e metadati
della run.

### `dataset_preparation/`

Macro-fase 3. Analizza qualità, filtra righe senza PDG valido, calcola metriche
PDG, unisce tutto al dataset RADON e produce un dataset finale versionato.

Output principale attuale:

```text
datasets/ansible-pdg-defect-dataset/final/v2026-06-06/
```

Questa fase conserva anche il report delle esclusioni e il manifest che spiega
come è stata generata ogni versione del dataset.

### `datasets/`

Contiene dataset selezionati e versionati. Il dataset finale corrente è:

```text
datasets/ansible-pdg-defect-dataset/final/v2026-06-06/
```

Qui si trovano il CSV finale, le metriche PDG calcolate, le esclusioni, il report
umano e il manifest JSON.

### `gnn/`

Macro-fase 4. Contiene preprocessing dei grafi, split, bilanciamento, modelli,
training ed evaluation per gli esperimenti GNN e le baseline.

Questa fase dovrà usare il dataset finale prodotto da `dataset_preparation`.
Nella fase sperimentale finale dovrà includere anche la sensitivity analysis sui
grafi piccoli descritta nei documenti di tesi.

### `output/`

Contiene output intermedi e artefatti di run operative, ad esempio run RADON,
estrazioni PDG e report temporanei. Non è il punto di ingresso finale per gli
esperimenti: per quello usare `datasets/`.

## Flusso di lavoro consigliato

1. Leggere `docs/THESIS_PROJECT_CONTEXT.md`.
2. Usare `radon_dataset_extraction/` per generare o aggiornare il dataset RADON.
3. Usare `pdg_file_level_extraction/` per estrarre i PDG file-level.
4. Usare `dataset_preparation/` per creare una versione finale del dataset.
5. Usare `gnn/` per training, valutazione e confronto sperimentale.

Ogni cartella ha un README dedicato con dettagli specifici della fase.
