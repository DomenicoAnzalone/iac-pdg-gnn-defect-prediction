# GNN Pipeline

This folder implements model selection, training, evaluation and baseline comparisons for PDG-based defect prediction using Graph Neural Networks.

Quick usage (after installing dependencies):

Run a preview of preprocessing and splits (no training):

```
python -m gnn.run_pipeline --label-csv output/ansible_rows_successfull_extracted.csv --preview-per-split 3 --max-splits 2
```

Run training for one or more models (example):

```
python -m gnn.run_pipeline --train --models gcn --epochs 10 --batch-size 8 --output-root output/gnn_runs
```

All results for each run are written under the `--output-root` folder in a timestamped subfolder.
# GNN Pipeline

Questo folder riunisce la pipeline di preprocessing e sampling per il problema di defect prediction sui PDG file-level.

## Scopo

Il codice qui disponibile è progettato per:
- caricare i PDG estratti come grafi `graphml` o `dot`
- costruire feature nodali e edge feature per modelli GNN
- generare split cronologici walk-forward per repository
- bilanciare le classi `failure_prone` solo sul training set
- fornire un runner `gnn/run_pipeline.py` che unisce tutte le fasi create finora

---

## Contenuto principale

### `run_pipeline.py`

Esegue la pipeline completa con i seguenti passaggi:
1. legge il CSV delle label e dei percorsi ai grafi
2. risolve i percorsi dei grafi con una rimappatura locale
3. crea split walk-forward per repository usando commit e date
4. estrae validation set dal training set in modo cronologico quando possibile
5. bilancia il training set con oversampling o undersampling
6. costruisce i dati dei grafi in preview per train/validation/test

Opzioni principali:
- `--label-csv`: CSV delle label (default: `output/ansible_rows_successfull_extracted.csv`)
- `--repositories-root`: radice delle repository locali per il walk-forward split
- `--balance-strategy`: `oversample` o `undersample`
- `--validation-ratio`: percentuale di validation set
- `--preview-per-split`: numero massimo di grafi buildati per split
- `--max-splits`: numero massimo di split da elaborare

### Output ideale

Consulta `gnn/ideal_output_example.md` per un esempio di output formato e i campi previsti dal runner.

### `preprocessing/`

Questo modulo gestisce il caricamento e la normalizzazione dei grafi:
- `graph_loader.py`: parser deterministico per GraphML/DOT
- `graph_inspector.py`: sommario della struttura dei grafi
- `feature_engineering.py`: estrazione di feature nodali e edge feature
- `dataset.py`: costruzione dei campioni e conversione in dati compatibili con GNN
- `normalization.py`: scaling delle feature numeriche

### `sampling/`

Questo modulo gestisce la creazione dei sottoinsiemi di training/validation/test:
- `balance.py`: oversampling/undersampling del training set
- `splitter.py`: walk-forward split basato sul repository e sul commit

---

## Come usare il runner

### Esecuzione base

```bash
python -m gnn.run_pipeline --label-csv output/ansible_rows_successfull_extracted.csv --repositories-root input/repositories --balance-strategy oversample --validation-ratio 0.2 --preview-per-split 5 --max-splits 10
```

### Cosa produce

- stampa riepiloghi dei split walk-forward
- mostra le dimensioni di train, validation e test
- verifica che il caricamento dei grafi funzioni per una preview di nodi/archi
- non esegue training automatico di modelli, ma costruisce i dati base della pipeline

---

## Pipeline completa descritta

1. `GraphDatasetBuilder` legge il CSV e risolve i percorsi ai file PDG.
2. `WalkForwardSplitter` crea split in stile walk-forward usando commit ordinati per data quando possibile.
3. `train_validation_split` estrae un validation set dal training set prima del bilanciamento.
4. `GraphBalancer` bilancia il training set senza toccare il test set.
5. `GraphDatasetBuilder.build_graph_data` costruisce feature nodali, edge index, edge attributes e label.
6. Lavoriamo su subset preview per validare il flusso senza processare tutti i grafi in un colpo solo.

---

## Nota operativa

- Il bilanciamento va applicato solo sul training set.
- Il validation set viene creato prima del bilanciamento per mantenere la correttezza temporale.
- `run_pipeline.py` è pensato come punto d’ingresso per verificare che il preprocessing e il sampling funzionino insieme.
- Per un esperimento completo con un modello GNN, il prossimo passo è aggiungere un trainer che consumi i dizionari generati da `GraphDatasetBuilder`.
