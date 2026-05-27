# GNN Sampling and Walk-Forward Splits

Questo folder contiene il codice per la generazione dei sottoinsiemi di training, validation e test e per il bilanciamento delle classi nel problema di defect prediction su grafi file-level.

## Obiettivo

- verificare lo squilibrio tra le classi `failure_prone`
- applicare tecniche di bilanciamento sul training set solo dopo aver creato la divisione initiale
- creare split cronologici walk-forward per ogni repository
- mantenere separate le fasi di bilanciamento e di validazione per evitare data leakage

---

## File principali

### `balance.py`

Responsabilità:
- analizzare il bilanciamento delle classi
- generare casi bilanciati con undersampling o oversampling
- supportare sia liste di campioni che DataFrame

Strategie implementate:
- `undersample`: riduce la classe maggioritaria fino al numero della minoritaria
- `oversample`: duplica la classe minoritaria fino al numero della maggioritaria
- `balance`: wrapper che sceglie la tecnica desiderata

Nota: per il problema grafico, SMOTE/ADASYN richiedono generazione di grafi sintetici sofisticata. Questa implementazione mantiene il dataset bilanciato con tecniche non distruttive che operano sui record dei grafi esistenti.

### `splitter.py`

Responsabilità:
- costruire split cronologici walk-forward per ogni repository
- usare le date dei commit quando disponibili nel clone locale di repository
- offrire una suddivisione training/validation/test compatibile con la metodologia di Iuliano

Come funziona:
1. raggruppa i dati per repository
2. ordina i commit per data (se `GitPython` e la repo sono disponibili)
3. per ogni repository crea iterazioni walk-forward:
   - iterazione 1: train = commit[0], test = commit[1]
   - iterazione 2: train = commit[0:2], test = commit[2]
   - e così via
4. consente di estrarre un validation set dai commit più recenti del training set

Output:
- `WalkForwardSplit` con `train`, `validation`, `test` e commit usati in ciascuna fase

---

## Esempi d'uso

### Caricare il CSV e esaminare lo squilibrio

```python
import pandas as pd
from gnn.sampling.balance import GraphBalancer

labels = pd.read_csv("output/ansible_rows_successfull_extracted.csv")
balancer = GraphBalancer(random_state=123)
report = balancer.class_balance(
    labels.to_dict("records"),
    label_extractor=lambda row: int(row["failure_prone"]),
)
print(report)
```

### Creare split walk-forward per repository

```python
import pandas as pd
from pathlib import Path
from gnn.sampling.splitter import WalkForwardSplitter

labels = pd.read_csv("output/ansible_rows_successfull_extracted.csv")
splitter = WalkForwardSplitter(labels, repositories_root=Path("input/repositories"))
splits = splitter.all_project_splits()
for split in splits[:3]:
    print(split.repository, split.train_commits, split.test_commit, len(split.train), len(split.test))
```

### Bilanciare il training set solo dopo lo split

```python
from gnn.sampling.balance import GraphBalancer

balancer = GraphBalancer(random_state=42)
balanced_train = balancer.dataframe_oversample(
    split.train,
    label_column="failure_prone",
)
```

### Creare validation set dai commit più recenti del training

```python
split_with_validation = splitter.train_validation_split(split, validation_ratio=0.2)
print("validation commits", split_with_validation.validation_commits)
```

---

## Linea guida operativa

1. usa `WalkForwardSplitter` per ottenere split temporali per ogni repository
2. su ciascun `split.train` applica il bilanciamento con `GraphBalancer`
3. normalizza le feature numeriche dei nodi solo dopo la costruzione del dataset completo
4. mantieni `split.test` invariato e non applicare alcun bilanciamento o normalizzazione che dipenda dai dati di test

---

## Come collegare con `gnn/preprocessing`

- `gnn/preprocessing/dataset.py` costruisce i campioni grafo + label
- `gnn/sampling/splitter.py` crea i sottoinsiemi da questi campioni sulla base del commit e del repository
- `gnn/sampling/balance.py` bilancia il solo training set con oversampling o undersampling
- `gnn/preprocessing/normalization.py` scala i nodi prima dell'addestramento

---

## Note di qualità

- il bilanciamento viene effettuato solo sul training set per evitare data leakage
- le split sono cronologiche, quindi i test rappresentano sempre dati successivi alle versioni di training
- la classica scelta `oversample` è spesso la più sicura quando i grafi sono pochi, mentre `undersample` riduce il training set e può perdere informazioni importanti
- se si desidera sperimentare ulteriormente, è possibile aggiungere un wrapper di SMOTE su feature aggregate a livello di grafo, ma questa versione mantiene il processo più robusto per le GNN
