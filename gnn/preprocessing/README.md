# GNN Preprocessing

Questa cartella contiene la pipeline di preprocessing per i PDG file-level destinati alla classificazione di grafi con GNN.

## Obiettivo

Il preprocessing serve a:
- caricare PDG estratti da file `.graphml` o `.dot`
- normalizzare e ispezionare la struttura dei grafi
- costruire feature nodali e edge feature utili a modelli GNN
- associare a ogni grafo la label binaria `failure_prone`
- mantenere un parser deterministico e idempotente per non perdere informazioni

Questa implementazione è pensata per partire dal CSV `output/ansible_rows_successfull_extracted.csv` e dai relativi file PDG estratti.

---

## Contenuto dei file

### `graph_loader.py`

Responsabilità:
- caricare grafi da file `.graphml`, `.graphml.gz`, `.graphmlz`, `.dot`, `.gv`
- preferire il file GraphML se presente in directory
- normalizzare il grafo in un `networkx.Graph`/`DiGraph`/`MultiDiGraph`
- serializzare tutti gli attributi dei nodi e degli archi come stringhe
- ordinare nodi e archi in modo deterministico

Perché:
- i PDG del dataset possono provenire da estrazioni diverse e usare attributi eterogenei
- la normalizzazione evita errori dovuti a tipi non omogenei durante il parsing
- l’ordinamento deterministico facilita debugging e confronto tra grafi

Funzioni principali:
- `GraphLoader.resolve_graph_file(path)`
- `GraphLoader.load_graph(path)`
- `GraphLoader.load_graphml(path)`
- `GraphLoader.load_dot(path)`
- `GraphLoader.normalize_graph(G)`
- `GraphLoader.inspect(graph_path)`

Risultato atteso:
- grafi caricati in memoria con node/edge attributes puliti e coerenti
- file `.graphml` scelti per primi quando disponibili

---

### `graph_inspector.py`

Responsabilità:
- esaminare la struttura di un grafo
- restituire statistiche generali e riassunti degli attributi

Dettagli:
- conta nodi e archi
- verifica se il grafo è diretto o multigrafo
- raccoglie le chiavi degli attributi più comuni nei nodi e negli archi
- fornisce un esempio di valori per ogni chiave

Perché:
- permette di capire rapidamente cosa contengono i PDG estratti
- utile per valutare se il parser deve essere esteso a nuovi attributi

Output:
```python
{
  "node_count": 123,
  "edge_count": 204,
  "directed": True,
  "multigraph": False,
  "node_attribute_summary": {
      "attribute_keys": [...],
      "key_counts": {...},
      "sample_values": {...},
  },
  "edge_attribute_summary": {...},
}
```

---

### `feature_engineering.py`

Responsabilità:
- generare vettori numerici per i nodi del PDG
- estrarre una feature edge_type per ciascun arco

Scelte principali:
- `node_type` viene codificato con una one-hot su categorie predefinite
- feature strutturali include grado in/out, grado totale
- feature testuali semplificate da `label` o `name`
- se una label contiene `{{...}}`, viene segnalato con `label_has_jinja`
- edge type viene inferito da attributi come `type`, `edge_type`, `kind`, `label`

Tipi node supportati (default):
- Task, Operator, Condition, Data, Variable, Function, Module, Call, Expression, Assignment, Unknown

Tipi edge supportati:
- data, control, depends, guard/condition, task, other

Perché:
- fornisce al modello informazioni strutturali e testuali minimali ma utili
- mantiene il parser robusto anche con grafi che non dichiarano esplicitamente tutti gli attributi

Output nodi:
- vettore `x` con dimensione `4 + len(node_type_categories) + 5 + 2`
- feature names includono: `is_task_node`, `in_degree`, `out_degree`, `degree`, `node_type_*`, `label_length`, `label_token_count`, `label_numeric_token_fraction`, `label_has_jinja`, `label_has_equals`, `attribute_count`, `has_location`

Output archi:
- vettore `edge_attr` di dimensione `(numero_archi, 1)`
- attributo `edge_type`

---

### `dataset.py`

Responsabilità:
- leggere il CSV di riferimento e mappare ogni riga a un campione di grafo
- costruire i dati necessari per un modello di graph classification
- supportare la rimappatura dei percorsi dai percorsi container `/app/...` al filesystem locale

Dettagli:
- usa `graphml_path` per trovare il grafo
- usa `failure_prone` come label binaria
- crea `GraphSample` con repository, commit, filepath, label, graph_path, metadata
- `build_graph_data(sample)` restituisce un dizionario con:
  - `x`: node features
  - `edge_index`: indici degli archi
  - `edge_attr`: edge features
  - `y`: label globale
  - `metadata`: informazioni della riga
  - `pyg_data`: opzionale, se `torch_geometric` è installato

Uso consigliato:
- `builder = GraphDatasetBuilder(label_csv, path_remapper=GraphDatasetBuilder.path_remapper_from_local_root(root))`
- `for sample in builder.samples(): data = builder.build_graph_data(sample)`

Perché:
- mantiene il preprocessing separato dalla parte di training
- garantisce che ogni grafo sia processato sempre alla stessa maniera
- supporta dataset con percorsi salvati in container o su filesystem diversi

---

### `normalization.py`

Responsabilità:
- scalare feature numeriche con StandardScaler o MinMaxScaler

Dettagli:
- `FeatureScaler(method="standard")`
- `fit`, `transform`, `fit_transform`, `inverse_transform`
- usa tipizzazione `np.ndarray`

Perché:
- la normalizzazione migliora convergenza in modelli GNN
- separa il preprocessing delle feature dalla costruzione del dataset

---

### `test_preprocessing.py`

Responsabilità:
- eseguire test locali rapidi per verificare il funzionamento della pipeline

Cosa verifica:
- `GraphLoader` carica un file GraphML e mantiene i nodi/archi
- `GraphInspector` ritorna riassunti corretti
- `GraphFeatureBuilder` elabora feature nodali e edge feature
- `GraphDatasetBuilder` lavora con un CSV di path remappati e crea `x`, `edge_index`, `y`

Perché:
- fornisce un controllo minimale prima di usare la pipeline su dataset reali
- evita regressioni sul caricamento e sulle feature base

---

## Indicazioni d'uso

### 1. Ispezione iniziale dei grafi

Usa `GraphLoader` + `GraphInspector` per esplorare un sottoinsieme di PDG.

```python
from pathlib import Path
from gnn.preprocessing.graph_loader import GraphLoader
from gnn.preprocessing.graph_inspector import GraphInspector

loader = GraphLoader()
G = loader.load_graph(Path("path/to/pdg.graphml"))
summary = GraphInspector.summarize_graph(G)
print(summary)
```

### 2. Costruzione del dataset

```python
from pathlib import Path
from gnn.preprocessing.dataset import GraphDatasetBuilder

builder = GraphDatasetBuilder(
    label_csv=Path("output/ansible_rows_successfull_extracted.csv"),
    path_remapper=GraphDatasetBuilder.path_remapper_from_local_root(Path.cwd(), remote_prefix="/app"),
)
for sample in builder.samples():
    data = builder.build_graph_data(sample)
    print(data["x"].shape, data["edge_index"].shape)
```

### 3. Normalizzazione

```python
from gnn.preprocessing.normalization import FeatureScaler

scaler = FeatureScaler(method="standard")
x_scaled = scaler.fit_transform(data["x"])
```

---

## Scelte implementative chiave

- Il parsing è stato progettato per essere robusto sia ai formati `.graphml` che `.dot`.
- Le feature nodali non si basano su embedding testuali complessi, ma su caratteristiche strutturali e segnali semplici estratti dalle label.
- Le label globali vengono lette direttamente da `failure_prone` per mantenere la pipeline compatibile con la classificazione binaria.
- Il codice è modulare per permettere futuri miglioramenti: è possibile aggiungere un `TokenEmbeddingBuilder` o un `RelationalGNNBuilder` senza cambiare il caricamento dei grafi.

---

## Come proseguire

1. esegui l’ispezione su un campione reale di PDG per identificare attributi nodali/edge specifici
2. estendi `GraphFeatureBuilder` se trovi attributi testuali importanti (`cmd`, `params`, `location`, ecc.)
3. usa `GraphDatasetBuilder` per generare dataset di prova
4. integra il tutto con un trainer GNN e un sistema di validazione walk-forward

Questo file documenta il design e la logica della pipeline di preprocessing creata in `gnn/preprocessing`. È pensato per rendere chiaro cosa fa ogni modulo e come usarlo al meglio.