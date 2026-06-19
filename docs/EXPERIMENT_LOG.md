# Registro degli Esperimenti

**Scopo del documento:** documentare in modo progressivo gli esperimenti eseguiti, i risultati osservati, le anomalie, le ipotesi interpretative e le decisioni operative successive.  
**Stato:** documento di lavoro, non capitolo finale della tesi.  
**Ultimo aggiornamento:** 9 giugno 2026.

---

## 1. Perche mantenere questo registro

Prima di passare al benchmark definitivo conviene usare una fase esplorativa controllata. L'obiettivo non e ancora dichiarare il modello migliore, ma capire:

- se la pipeline sperimentale funziona correttamente end-to-end;
- se gli split, le metriche e gli output sono coerenti;
- quali configurazioni GNN sono promettenti;
- quali problemi metodologici emergono;
- quali scelte vanno corrette prima degli esperimenti finali;
- quali risultati sono robusti e quali dipendono da dettagli come seed, soglie sui grafi piccoli o dimensione dei test set.

Questo documento serve quindi come diario tecnico-scientifico: ogni nuova run dovrebbe aggiungere una sezione con configurazione, risultati, osservazioni e prossimi passi.

---

## 2. Protocollo sperimentale comune

Gli esperimenti devono rispettare il protocollo comune definito in `docs/THESIS_PROJECT_CONTEXT.md` e implementato in `experiments/`.

Punti chiave:

- dataset comune: `datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv`;
- unita di analisi: `(repository, commit, filepath)`;
- label: `failure_prone`;
- confronto principale sul sottoinsieme con PDG valido e GraphML caricabile;
- split within-project walk-forward;
- validation set derivato temporalmente dal training set;
- test set temporalmente successivo al training;
- balancing applicato solo al training;
- validation e test mai bilanciati;
- trasformazioni apprese solo sul training;
- metriche comuni: AUC-PR, AUC-ROC, MCC, precision, recall, F1, accuracy, confusion matrix;
- salvataggio di configurazione, metadata, split manifest, predizioni, metriche e log.

---

## 3. Esperimento E3-001: GraphSAGE baseline completa

### 3.1 Identificazione run

| Campo | Valore |
|---|---|
| ID esperimento | `E3-001` |
| Run name | `e3_graphsage_full_compact` |
| Esperimento | E3, GNN graph-level |
| Modello | GraphSAGE |
| Dataset | `ansible-pdg-defect-dataset_v2026-06-06_final.csv` |
| Seed | `42` |
| Data run | 8 giugno 2026 |
| Output | `experiments/results/exploratory/e3_graphsage_full_compact/` |

Comando eseguito:

```bash
python -m experiments.e3_gnn.run --config experiments/configs/e3_default.yaml --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --run-name e3_graphsage_full_compact --model graphsage --min-nodes 3 --min-edges 2 --balance random_oversampling --epochs 100 --batch-size 32 --seed 42 --log-every-epochs 5 --compact-progress --progress-width 100
```

Nota: `--progress-width 100` era ancora presente nella versione del codice usata per questa run, ma riguardava solo la visualizzazione in console. Il parametro e stato successivamente rimosso per ridurre il numero di opzioni CLI.

### 3.2 Configurazione rilevante

| Parametro | Valore |
|---|---:|
| `model` | `graphsage` |
| `min_nodes` | `3` |
| `min_edges` | `2` |
| `balance_strategy` | `random_oversampling` |
| `epochs` | `100` |
| `batch_size` | `32` |
| `hidden_channels` | `64` |
| `num_layers` | `2` |
| `dropout` | `0.5` |
| `lr` | `0.001` |
| `pooling` | `mean` |
| `early_stopping_metric` | `mcc` |
| `early_stopping_patience` | `10` |
| `class_weights` | `false` |
| `device` | `auto` |

Questa configurazione rappresenta una prima baseline E3 realistica: non e uno smoke test, ma non e ancora un risultato definitivo di benchmark.

---

## 4. Controlli di coerenza della run

### 4.1 Artefatti principali

La run ha prodotto correttamente:

```text
config.yaml
metadata.json
split_manifest.csv
excluded_samples.csv
logs/skipped_splits.csv
logs/run.log
metrics/per_split_metrics.csv
metrics/per_repository_metrics.csv
metrics/aggregated_metrics.csv
predictions/e3_graphsage_predictions.csv
models/*_history.json
models/*_best.pt
reports/run_summary.md
```

### 4.2 Split, predizioni e campioni

| Controllo | Valore |
|---|---:|
| Split validi metricati | 1.706 |
| Split nelle predizioni | 1.706 |
| Predizioni salvate | 25.062 |
| Righe test in `split_manifest.csv` | 25.062 |
| Sample test unici | 25.062 |
| Duplicati nelle predizioni | 0 |
| Sample test mancanti nelle predizioni | 0 |
| Predizioni extra non presenti nel manifest | 0 |
| Graph exclusions | 0 |
| Excluded samples | 0 |

Conclusione: la run e coerente dal punto di vista degli output. Non risultano predizioni mancanti, duplicati o mismatch tra manifest e file predizioni.

### 4.3 Split saltati

La pipeline ha generato:

| Tipo | Numero |
|---|---:|
| Split validi | 1.706 |
| Split saltati | 1.426 |

Motivi degli split saltati:

| Motivo | Numero |
|---|---:|
| `train_single_class` | 817 |
| `test_single_class` | 608 |
| `too_few_commits` | 1 |

Interpretazione: gli split saltati non indicano che meta dataset sia stato eliminato. Uno split e una finestra temporale, non una singola riga. Molti campioni coinvolti in split saltati possono comparire in split successivi come train, validation o test.

---

## 5. Risultati principali

### 5.1 Metriche aggregate per split

Valori da `metrics/aggregated_metrics.csv`:

| Metrica | Media | Mediana |
|---|---:|---:|
| AUC-PR | 0.803 | 0.952 |
| AUC-ROC | 0.826 | 0.929 |
| MCC | 0.611 | 0.632 |
| Precision | 0.690 | 0.750 |
| Recall | 0.828 | 1.000 |
| F1 | 0.702 | 0.750 |
| Accuracy | 0.773 | 0.833 |

Questi valori suggeriscono una capacita predittiva reale, soprattutto in termini di recall. Tuttavia le medie per split vanno interpretate con cautela perche molti test set sono piccoli.

### 5.2 Metriche pooled globali

Calcolando le metriche sulle 25.062 predizioni test aggregate:

| Metrica pooled | Valore |
|---|---:|
| MCC | 0.563 |
| AUC-PR | 0.643 |
| AUC-ROC | 0.818 |
| Precision | 0.646 |
| Recall | 0.842 |
| F1 | 0.731 |
| Accuracy | 0.779 |

Confusion matrix pooled:

|  | Predetto 0 | Predetto 1 |
|---|---:|---:|
| Reale 0 | 11.991 | 4.120 |
| Reale 1 | 1.418 | 7.533 |

Interpretazione: il modello cattura molte istanze positive (`recall = 0.842`), ma genera anche un numero rilevante di falsi positivi. Questo comportamento puo essere accettabile in defect prediction, dove spesso e preferibile individuare molti file rischiosi, ma deve essere confrontato con E1 ed E2.

### 5.3 Distribuzione delle predizioni

| Quantita | Valore |
|---|---:|
| Test label 0 | 16.111 |
| Test label 1 | 8.951 |
| Positive rate reale | 0.357 |
| Predizioni 0 | 13.409 |
| Predizioni 1 | 11.653 |
| Positive rate predetto | 0.465 |

Il modello tende a sovra-predire la classe `failure_prone`.

---

## 6. Analisi overfit e underfit

### 6.1 Training dynamics

Statistiche dai file `models/*_history.json`:

| Indicatore | Media | Mediana |
|---|---:|---:|
| First train loss | 0.685 | 0.693 |
| Last train loss | 0.381 | 0.405 |
| Best validation MCC | 0.672 | 0.707 |
| Best validation AUC-PR | 0.874 | 0.991 |
| Epochs ran | 17.2 | 15 |

Tutti gli split hanno attivato early stopping:

```text
1706 / 1706 split fermati prima di 100 epoche
```

Nessuno split ha raggiunto 90 epoche:

```text
epochs >= 90: 0
```

Questo indica che `epochs=100` e un limite massimo, non il numero effettivo di epoche usate.

### 6.2 Underfit

Non emerge un underfit evidente:

- la train loss scende in modo marcato;
- il modello produce score non banali;
- AUC-ROC pooled e MCC pooled sono sopra una baseline casuale;
- la recall e alta.

### 6.3 Overfit

Non emerge un overfit sistematico forte:

| Indicatore | Valore |
|---|---:|
| Correlazione best validation MCC vs test MCC | 0.852 |
| Correlazione best validation AUC-PR vs test AUC-PR | 0.728 |
| Gap medio validation MCC - test MCC | 0.061 |

Tuttavia esistono split instabili:

| Indicatore | Valore |
|---|---:|
| Split con gap validation-test MCC > 0.5 | 54 |
| Split con validation MCC >= 0.9 e test MCC <= 0.2 | 2 |

Interpretazione: il modello non sembra overfittare in modo generalizzato, ma alcuni split sono fragili, probabilmente per test set piccoli o distribuzioni locali molto sbilanciate.

---

## 7. Problemi e criticita emerse

### 7.1 Test set piccoli

Distribuzione dimensione test:

| Indicatore | Valore |
|---|---:|
| Test size medio | 14.7 |
| Test size mediano | 7 |
| Test size <= 3 | 178 split |
| Test size <= 5 | 463 split |

Metriche per bucket di test size:

| Bucket test size | Split | MCC medio | MCC NaN | AUC-PR media | F1 medio |
|---|---:|---:|---:|---:|---:|
| 1-3 | 178 | 0.705 | 56 | 0.933 | 0.669 |
| 4-5 | 285 | 0.628 | 57 | 0.869 | 0.684 |
| 6-10 | 629 | 0.657 | 48 | 0.807 | 0.730 |
| 11-20 | 265 | 0.568 | 6 | 0.760 | 0.713 |
| 21-50 | 289 | 0.553 | 2 | 0.736 | 0.705 |
| 51-100 | 50 | 0.351 | 0 | 0.582 | 0.538 |
| 101-1000 | 10 | 0.501 | 0 | 0.542 | 0.500 |

Osservazione: gli split piccoli tendono ad avere metriche molto alte o molto instabili. Questo può gonfiare le medie per split.

Possibile correzione:

- riportare sempre metriche pooled;
- riportare metriche pesate per numero di test sample;
- aggiungere analisi con soglia minima su `test_size`;
- valutare aggregazioni per repository invece che solo per split.

### 7.2 MCC non definito

Numero di split con MCC NaN:

```text
169 / 1706
```

Cause osservate:

| Caso | Numero |
|---|---:|
| Predizione tutta positiva | 107 |
| Predizione tutta negativa | 62 |

Questi split non sono errori di pipeline: il modello ha prodotto una sola classe predetta, quindi MCC non e definibile.

Effetto sull'aggregazione:

| Variante MCC | Valore |
|---|---:|
| Media MCC ignorando NaN | 0.611 |
| Media MCC con NaN trattati come 0 | 0.551 |
| MCC pooled globale | 0.563 |

Osservazione: il valore `0.611` e ottimistico se riportato da solo.

Possibile correzione:

- aggiungere `mcc_nan_count`;
- aggiungere `mcc_mean_nan_as_zero`;
- aggiungere metriche pooled;
- indicare esplicitamente quanti split hanno predizione costante.

### 7.3 Sovra-predizione della classe positiva

Il positive rate reale nei test e:

```text
0.357
```

Il positive rate predetto e:

```text
0.465
```

Questo indica una tendenza a classificare più file come `failure_prone` rispetto alla distribuzione reale.

Effetto:

- recall alta;
- falsi positivi elevati;
- precision più bassa della recall.

Possibili correzioni:

- tuning della soglia decisionale invece di usare sempre `0.5`;
- confronto con class weights al posto di oversampling;
- confronto con `balance_strategy=none`;
- analisi precision-recall per scegliere soglie coerenti con l'obiettivo della tesi.

### 7.4 Repository molto diverse tra loro

Repository con molti split e buone performance:

| Repository | Split | MCC medio | AUC-PR media |
|---|---:|---:|---:|
| `gmazoyer/ansible-role-netbox` | 132 | 0.945 | 0.955 |
| `galaxyproject/ansible-galaxy` | 81 | 0.764 | 0.815 |
| `meysam81/ansible-collections` | 43 | 0.917 | 0.949 |

Repository difficili:

| Repository | Split | MCC medio | AUC-PR media |
|---|---:|---:|---:|
| `githubixx/ansible-role-kubernetes-worker` | 12 | 0.043 | 0.496 |
| `robertdebock/ansible-role-tomcat` | 15 | 0.150 | 0.856 |
| `Dynatrace/Dynatrace-OneAgent-Ansible` | 6 | 0.167 | 0.340 |
| `gantsign/ansible-role-java` | 28 | 0.169 | 0.459 |

Osservazione: le performance variano molto tra repository. Il benchmark finale dovrebbe quindi riportare anche risultati per repository e non solo globali.

### 7.5 Repository che dominano i test sample

Top repository per sample test:

| Repository | Test sample | Share |
|---|---:|---:|
| `galaxyproject/ansible-galaxy` | 1.649 | 6.58% |
| `dev-sec/ansible-collection-hardening` | 1.555 | 6.20% |
| `AshAvalanche/ansible-avalanche-collection` | 1.380 | 5.51% |
| `Rosa-Luxemburgstiftung-Berlin/ansible-opnsense` | 1.323 | 5.28% |
| `anthcourtney/ansible-role-cis-amazon-linux` | 1.272 | 5.08% |

Nessuna repository domina da sola il benchmark, ma le prime dieci pesano sensibilmente. Conviene quindi distinguere tra:

- media per split;
- media per repository;
- metriche pooled;
- metriche pesate per numero di sample.

---

## 8. Interpretazione complessiva

La run `E3-001` e una prima baseline GNN reale e utile.

Punti positivi:

- pipeline end-to-end completata;
- nessuna esclusione grafo;
- split, predizioni e metriche coerenti;
- risultati sopra baseline casuale;
- recall alta;
- early stopping funzionante;
- nessun segnale forte di underfit;
- nessun overfit sistematico evidente.

Punti critici:

- molte metriche per split sono su test set piccoli;
- 169 split hanno MCC non definito;
- la media MCC ignorando NaN e ottimistica;
- il modello sovra-predice `failure_prone`;
- performance molto variabili tra repository;
- alcuni split mostrano instabilita validation-test.

Conclusione prudente:

> GraphSAGE mostra capacita predittiva reale sui PDG file-level. La performance globale e promettente, con MCC pooled circa 0.56 e recall circa 0.84. Tuttavia le metriche medie per split sono probabilmente ottimistiche a causa di test set piccoli e MCC non definiti. La run e valida come baseline E3 iniziale, ma non sufficiente da sola per conclusioni definitive.

---

## 9. Azioni consigliate prima del benchmark definitivo

### 9.1 Migliorare il reporting

La pipeline e stata aggiornata per salvare `metrics/pooled_metrics.csv` e includere colonne `pooled_*` in `aggregated_metrics.csv`. Restano da completare o consolidare:

- `mcc_nan_count`;
- `mcc_mean_nan_as_zero`;
- numero di split con predizione tutta positiva;
- numero di split con predizione tutta negativa;
- metriche pesate per `test_size`;
- metriche per bucket di `test_size`;
- top repository per peso e performance.

Motivo: il report attuale non rende abbastanza visibili instabilita e NaN.

### 9.2 Confrontare con E1 ed E2

Eseguire E1 ed E2 sugli stessi split e sullo stesso dataset comune. Solo cosi si puo stabilire se la struttura del grafo aggiunge davvero valore rispetto alle feature tabellari.

Configurazioni minime:

```text
E1 Random Forest
E2 Random Forest con tutte le 11 metriche PDG candidate e feature selection scelta sulla validation
E3 GraphSAGE
```

### 9.3 Ripetere con altri seed

Eseguire almeno:

```text
seed = 7
seed = 123
```

Motivo: una singola run seed 42 non basta per valutare stabilita della GNN.

### 9.4 Small-graph sensitivity

Ripetere E3 con soglie più restrittive:

```text
min_nodes=5, min_edges=4
min_nodes=8, min_edges=6
min_nodes=10, min_edges=6
```

Motivo: i grafi piccoli hanno distribuzioni di label particolari e possono influenzare le prestazioni.

### 9.5 Alternative di balancing

Confrontare:

```text
random_oversampling
none
class_weights=true
random_undersampling
```

Motivo: l'oversampling produce buona recall ma potrebbe aumentare falsi positivi.

### 9.6 Soglia decisionale

Valutare soglie diverse da `0.5` usando validation set:

```text
threshold selected by MCC
threshold selected by F1
threshold selected by target recall
```

Motivo: il modello produce score utili, ma la soglia 0.5 potrebbe non essere ottimale.

---

## 10. Decisione operativa dopo E3-001

La configurazione:

```text
GraphSAGE, 2 layer, hidden 64, mean pooling, random oversampling, seed 42
```

deve essere mantenuta come baseline GNN iniziale.

Non deve ancora essere considerata configurazione finale.

Run esplorative poi eseguite:

1. `E3-002`: small-graph threshold sweep;
2. `E3-003`: split reliability analysis;
3. `E3-004`: balance strategy sweep.

Dopo queste analisi, la configurazione comune preliminare è: soglia grafi `3/2`, tutti gli split validi mantenuti, metriche principali pooled e `random_oversampling` applicato solo al training set. Il prossimo blocco naturale è verificare la stabilità rispetto al seed e poi lanciare E1/E2 sulla stessa configurazione.

---

## 11. E3-002 Small-Graph Threshold Sweep

### Obiettivo

Verificare se i grafi PDG piccoli siano dannosi per GraphSAGE e se sia opportuno usare una soglia più restrittiva della soglia base `3 nodi / 2 archi`.

La motivazione era metodologica: grafi appena sopra la soglia minima potrebbero non contenere abbastanza struttura per una GNN e potrebbero spingere il modello a imparare una scorciatoia legata alla dimensione del grafo, ad esempio "grafo piccolo = file neutral".

### Configurazioni

Sono state confrontate quattro soglie:

```text
3/2
5/4
8/6
10/6
```

Tutte le run usano:

- modello: GraphSAGE;
- balancing: random oversampling solo sul training set;
- epoche massime: 100;
- batch size: 32;
- seed: 42;
- metriche principali: pooled.

Report completo:

```text
experiments/results/exploratory/small_graph_threshold_sweep/threshold_run_summary.md
```

### Risultati Pooled

| Soglia | Campioni mantenuti | Rimossi | Split validi | MCC pooled | AUC-PR pooled | AUC-ROC pooled | F1 pooled |
|---|---:|---:|---:|---:|---:|---:|---:|
| `3/2` | 37.796 | 0 | 1.706 | 0,563 | 0,643 | 0,818 | 0,731 |
| `5/4` | 36.614 | 1.182 | 1.691 | 0,547 | 0,632 | 0,807 | 0,725 |
| `8/6` | 34.834 | 2.962 | 1.675 | 0,560 | 0,628 | 0,808 | 0,735 |
| `10/6` | 33.506 | 4.290 | 1.638 | 0,554 | 0,625 | 0,804 | 0,731 |

### Osservazioni

La soglia base `3/2` resta la migliore su MCC pooled, AUC-PR pooled, AUC-ROC pooled e accuracy pooled.

La soglia `8/6` è la più vicina alla baseline e migliora leggermente precision/F1, ma peggiora MCC, AUC-PR, AUC-ROC e recall. Non è quindi un miglioramento netto.

Le soglie restrittive rimuovono soprattutto campioni neutral: con `5/4`, i 1.182 campioni rimossi hanno positive rate circa `0,148`, molto inferiore alla media globale `0,368`. Questo indica che i grafi piccoli non sono soltanto rumore, ma una fascia reale e informativa del dataset.

### Decisione

Per il benchmark principale si mantiene:

```text
min_nodes = 3
min_edges = 2
```

Motivo: non c'e evidenza che i grafi piccoli danneggino GraphSAGE; rimuoverli riduce campioni, split validi e predizioni test senza migliorare stabilmente le metriche pooled.

La soglia `8/6` puo restare come sensitivity secondaria, ma non sostituisce la soglia principale.

---

## 12. E3-003 Split Reliability Analysis

### Obiettivo

Valutare se gli split walk-forward con test set molto piccoli debbano essere esclusi dal benchmark principale.

Il dubbio nasce dal fatto che molti split hanno pochi campioni nel test. In questi casi MCC, F1 e AUC possono essere instabili. Tuttavia gli split single-class sono già esclusi dalla pipeline e le metriche principali vengono calcolate in forma pooled, quindi gli split piccoli non pesano quanto quelli grandi.

### Metodo

L'analisi è post-hoc: non riaddestra GraphSAGE. Usa le predizioni salvate della run:

```text
experiments/results/exploratory/e3_graphsage_threshold_n3_e2
```

e ricalcola le metriche pooled filtrando gli split secondo soglie minime su:

- numero di campioni nel test set;
- numero di positivi nel test set;
- numero di negativi nel test set.

Report completo:

```text
experiments/results/exploratory/e3_graphsage_threshold_n3_e2/reports/split_reliability/split_reliability_summary.md
```

### Risultati Principali

| Filtro | Split tenuti | Predizioni tenute | Copertura | MCC pooled | AUC-PR pooled | F1 pooled |
|---|---:|---:|---:|---:|---:|---:|
| Tutti gli split validi | 1.706 | 25.062 | 1,000 | 0,563 | 0,643 | 0,731 |
| `test_size >= 5`, `positives >= 1` | 1.375 | 23.980 | 0,957 | 0,567 | 0,642 | 0,731 |
| `test_size >= 10`, `positives >= 1` | 643 | 18.926 | 0,755 | 0,566 | 0,643 | 0,729 |
| `test_size >= 10`, `positives >= 2` | 592 | 17.579 | 0,701 | 0,572 | 0,661 | 0,744 |
| `test_size >= 20`, `positives >= 1` | 362 | 14.877 | 0,594 | 0,563 | 0,627 | 0,712 |
| `test_size >= 50`, `positives >= 1` | 61 | 5.694 | 0,227 | 0,501 | 0,464 | 0,607 |

### Osservazioni

Gli split piccoli sono numerosi, ma pesano poco nelle metriche pooled. Gli split con al massimo 10 campioni sono 1.092 su 1.706, ma rappresentano 6.426 predizioni su 25.062.

Filtrare gli split piccoli produce miglioramenti piccoli o non stabili. Il filtro `test_size >= 10` e `positives >= 2` è il miglior compromesso diagnostico, ma elimina circa il 30% delle predizioni test. È troppo aggressivo per il risultato principale.

Filtri più severi riducono molto la copertura e non producono un miglioramento affidabile.

### Decisione

Per il benchmark principale si mantengono tutti gli split validi prodotti dalla pipeline:

```text
train/test single-class: esclusi
split piccoli ma validi: mantenuti
metriche principali: pooled
```

Gli split piccoli verranno documentati come minaccia alla validità e analizzati con report diagnostici, ma non esclusi automaticamente.

Soglia diagnostica secondaria consigliata:

```text
min_test_size = 10
min_test_positives = 2
min_test_negatives = 1
```

Questa soglia può essere usata per verificare la robustezza dei risultati, non per sostituire il benchmark principale.

---

## 13. E3-004 Balance Strategy Sweep

### Obiettivo

Fissare la strategia di bilanciamento da usare nel benchmark principale.

Dopo `E3-002` ed `E3-003` sono stati fissati due criteri:

- soglia principale dei grafi: `min_nodes=3`, `min_edges=2`;
- gestione split: mantenere tutti gli split validi e usare metriche pooled come risultato principale.

Il test confronta tre strategie applicate solo al training set:

```text
none
random_oversampling
random_undersampling
```

La prima run GraphSAGE completa usava `random_oversampling` e ha mostrato recall alta, ma anche una tendenza a predire più positivi del reale. Per questo il confronto sul bilanciamento deve verificare se oversampling sia davvero la scelta migliore o se una strategia meno aggressiva produca un equilibrio migliore tra precision e recall.

### Identificazione run

| Campo | Valore |
|---|---|
| ID esperimento | `E3-004` |
| Esperimento | E3, GNN graph-level |
| Modello | GraphSAGE |
| Dataset | `ansible-pdg-defect-dataset_v2026-06-06_final.csv` |
| Seed | `42` |
| Output sweep | `experiments/results/exploratory/e3_balance_sweep/` |
| Run `none` | `experiments/results/exploratory/e3_graphsage_balance_none/` |
| Run `random_oversampling` | `experiments/results/exploratory/e3_graphsage_balance_random_oversampling/` |
| Run `random_undersampling` | `experiments/results/exploratory/e3_graphsage_balance_random_undersampling/` |

### Configurazione usata

Tutte le run devono mantenere fissi:

| Parametro | Valore |
|---|---:|
| Modello | `graphsage` |
| `min_nodes` | `3` |
| `min_edges` | `2` |
| `epochs` | `100` |
| `batch_size` | `32` |
| `seed` | `42` |
| `class_weights` | `false` |
| Metriche principali | pooled |

Cambia solo `balance_strategy`.

### Comando

```bash
python -m experiments.sensitivity.run_balance_sweep --config experiments/configs/e3_default.yaml --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --strategies none,random_oversampling,random_undersampling --run-prefix e3_graphsage_balance --model graphsage --min-nodes 3 --min-edges 2 --epochs 100 --batch-size 32 --seed 42 --log-every-epochs 5 --compact-progress
```

Output attesi:

```text
experiments/results/exploratory/e3_balance_sweep/balance_run_summary.csv
experiments/results/exploratory/e3_balance_sweep/balance_run_summary.md
```

### Risultati pooled

Tutte le strategie sono state valutate sugli stessi 1.706 split validi e sulle stesse 25.062 predizioni test. Il positive rate reale del test set aggregato è circa `0,357`.

| Strategia | MCC | AUC-PR | AUC-ROC | Precision | Recall | F1 | Accuracy | Positive rate predetto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0,487 | 0,588 | 0,791 | 0,611 | 0,783 | 0,687 | 0,745 | 0,458 |
| `random_oversampling` | 0,563 | 0,643 | 0,818 | 0,646 | 0,842 | 0,731 | 0,779 | 0,465 |
| `random_undersampling` | 0,430 | 0,543 | 0,743 | 0,573 | 0,763 | 0,655 | 0,713 | 0,476 |

Confusion matrix pooled:

| Strategia | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| `none` | 11.655 | 4.456 | 1.939 | 7.012 |
| `random_oversampling` | 11.991 | 4.120 | 1.418 | 7.533 |
| `random_undersampling` | 11.026 | 5.085 | 2.117 | 6.834 |

### Osservazioni

`random_oversampling` è la strategia migliore su tutte le metriche principali: MCC, AUC-PR, AUC-ROC, precision, recall, F1 e accuracy.

Rispetto a `none`, l'oversampling:

- aumenta MCC da `0,487` a `0,563`;
- aumenta AUC-PR da `0,588` a `0,643`;
- aumenta F1 da `0,687` a `0,731`;
- aumenta recall da `0,783` a `0,842`;
- riduce i falsi positivi da `4.456` a `4.120`;
- riduce i falsi negativi da `1.939` a `1.418`.

Quindi il miglioramento non deriva soltanto dal predire più positivi: il positive rate predetto cresce leggermente rispetto a `none` (`0,458` -> `0,465`), ma migliorano anche precisione e numero di veri negativi.

`random_undersampling` è la strategia peggiore. Probabilmente elimina troppi esempi di training dalla classe maggioritaria, riducendo informazione utile. Ha più falsi positivi, più falsi negativi e metriche pooled inferiori.

### Decisione

Per il benchmark principale si adotta:

```text
balance_strategy = random_oversampling
```

La strategia resta applicata esclusivamente al training set. Validation e test non vengono bilanciati.

`none` può rimanere una baseline diagnostica secondaria. `random_undersampling` non è consigliata per le run definitive.

### Criterio decisionale applicato

La strategia scelta per il benchmark principale deve essere quella con il compromesso migliore tra:

- MCC pooled, metrica guida;
- AUC-PR pooled, metrica secondaria importante per dataset sbilanciato;
- precision e recall pooled;
- positive rate predetto rispetto al positive rate reale;
- stabilità e semplicità metodologica.

Se due strategie sono molto vicine, conviene preferire quella più semplice e meno aggressiva.

In questo caso le strategie non sono vicine: `random_oversampling` è chiaramente migliore. La scelta è quindi sufficientemente solida per procedere allo step successivo.

---

## 14. E3-005 Seed Stability Sweep

### Obiettivo

Verificare che la configurazione E3 scelta non dipenda in modo eccessivo dal seed.

Dopo le analisi precedenti, la configurazione E3 preliminare è:

```text
GraphSAGE
min_nodes = 3
min_edges = 2
balance_strategy = random_oversampling
epochs = 100
batch_size = 32
metriche principali = pooled
```

Il test cambia solo il seed:

```text
42
7
123
```

### Comando

```bash
python -m experiments.sensitivity.run_seed_sweep --config experiments/configs/e3_default.yaml --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --seeds 42,7,123 --run-prefix e3_graphsage_seed --model graphsage --min-nodes 3 --min-edges 2 --balance random_oversampling --epochs 100 --batch-size 32 --log-every-epochs 5 --compact-progress
```

Output attesi:

```text
experiments/results/exploratory/e3_seed_sweep/seed_run_summary.csv
experiments/results/exploratory/e3_seed_sweep/seed_run_summary.md
```

### Risultati pooled

Tutte le run hanno usato gli stessi 1.706 split validi e le stesse 25.062 predizioni test. Il positive rate reale del test set aggregato è `0,357`.

| Seed | MCC | AUC-PR | AUC-ROC | Precision | Recall | F1 | Accuracy | Positive rate predetto |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `42` | 0,563 | 0,643 | 0,818 | 0,646 | 0,842 | 0,731 | 0,779 | 0,465 |
| `7` | 0,572 | 0,639 | 0,820 | 0,647 | 0,855 | 0,737 | 0,782 | 0,472 |
| `123` | 0,567 | 0,625 | 0,816 | 0,649 | 0,844 | 0,734 | 0,781 | 0,464 |

Confusion matrix pooled:

| Seed | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|
| `42` | 11.991 | 4.120 | 1.418 | 7.533 |
| `7` | 11.937 | 4.174 | 1.299 | 7.652 |
| `123` | 12.026 | 4.085 | 1.400 | 7.551 |

### Stabilità

| Metrica | Media | Deviazione standard | Min | Max | Range |
|---|---:|---:|---:|---:|---:|
| MCC | 0,567 | 0,0037 | 0,563 | 0,572 | 0,009 |
| AUC-PR | 0,635 | 0,0077 | 0,625 | 0,643 | 0,018 |
| AUC-ROC | 0,818 | 0,0016 | 0,816 | 0,820 | 0,004 |
| Precision | 0,647 | 0,0011 | 0,646 | 0,649 | 0,002 |
| Recall | 0,847 | 0,0059 | 0,842 | 0,855 | 0,013 |
| F1 | 0,734 | 0,0022 | 0,731 | 0,737 | 0,005 |
| Accuracy | 0,781 | 0,0011 | 0,779 | 0,782 | 0,003 |
| Positive rate predetto | 0,467 | 0,0034 | 0,464 | 0,472 | 0,008 |

### Osservazioni

La configurazione è molto stabile rispetto al seed. MCC varia solo di circa `0,009`, F1 di circa `0,005` e accuracy di circa `0,003`. Anche il positive rate predetto resta coerente, tra `0,464` e `0,472`.

AUC-PR è la metrica più variabile, ma il range resta contenuto (`0,018`) e non indica un crollo di performance su nessuno dei seed.

### Criterio decisionale

La configurazione può essere considerata stabile se:

- MCC pooled, AUC-PR pooled e F1 pooled cambiano poco tra seed;
- il positive rate predetto resta coerente;
- non emergono crolli anomali su uno dei seed.

Se la variabilità è bassa, la configurazione E3 può essere congelata per il benchmark principale. Se la variabilità è alta, bisogna riportare media e deviazione standard su più seed o rivedere la configurazione GNN.

### Decisione

La configurazione GraphSAGE può essere congelata come baseline E3 principale:

```text
model = graphsage
min_nodes = 3
min_edges = 2
balance_strategy = random_oversampling
epochs = 100
batch_size = 32
seed principale = 42
```

Nel report finale è consigliabile riportare anche media e deviazione standard sui tre seed come analisi di robustezza.

---

## 15. E1/E2-001 Tabular Common-Setup Test

### Obiettivo

Eseguire il primo confronto tabellare sulla configurazione comune fissata dalla fase esplorativa E3.

Questo test serve a verificare:

- la baseline E1 con sole feature tabellari non-PDG;
- il valore aggiunto iniziale di E2 con le 11 metriche PDG candidate;
- la correttezza del confronto su stessi campioni, stessi split, stesso bilanciamento e stesse metriche pooled.

### Configurazione

| Parametro | Valore |
|---|---|
| Modello | `random_forest` |
| Dataset | `ansible-pdg-defect-dataset_v2026-06-06_final.csv` |
| Soglia grafi | `min_nodes=3`, `min_edges=2` |
| Split | within-project walk-forward, tutti gli split validi |
| Bilanciamento | `random_oversampling`, solo training set |
| Scaler | `standard` |
| Seed | `42` |
| E1 feature selection | `validation_rfe` nel benchmark aggiornato |
| E2 metriche PDG | `all` |
| E2 feature selection | `validation_rfe` nel benchmark aggiornato |

### Comando

```bash
python -m experiments.sensitivity.run_tabular_e1_e2 --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv --run-prefix tabular_rf_common --models random_forest --pdg-metrics all --balance random_oversampling --scaler standard --e1-feature-selection validation_rfe --e2-feature-selection validation_rfe --seed 42
```

Output attesi:

```text
experiments/results/exploratory/tabular_e1_e2_common/tabular_e1_e2_summary.csv
experiments/results/exploratory/tabular_e1_e2_common/tabular_e1_e2_summary.md
experiments/results/exploratory/tabular_rf_common_e1/
experiments/results/exploratory/tabular_rf_common_e2/
```

### Risultati pooled

Entrambe le run sono terminate correttamente. E1 ed E2 usano gli stessi 1.706 split validi e le stesse 25.062 predizioni test.

| Esperimento | Modello | Feature selection | MCC | AUC-PR | AUC-ROC | Precision | Recall | F1 | Accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | Random Forest | `none` | 0,580 | 0,794 | 0,886 | 0,649 | 0,866 | 0,742 | 0,784 |
| E2 | Random Forest | `rfecv` nella run storica | 0,606 | 0,782 | 0,883 | 0,663 | 0,881 | 0,756 | 0,797 |

Confusion matrix pooled:

| Esperimento | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| E1 | 11.912 | 4.199 | 1.203 | 7.748 |
| E2 | 12.099 | 4.012 | 1.068 | 7.883 |

Delta E2 - E1:

| Metrica | Delta |
|---|---:|
| MCC | +0,026 |
| AUC-PR | -0,012 |
| AUC-ROC | -0,003 |
| Precision | +0,014 |
| Recall | +0,015 |
| F1 | +0,015 |
| Accuracy | +0,013 |
| Falsi positivi | -187 |
| Falsi negativi | -135 |

### Confronto appaiato per split

Il confronto split-by-split conferma che E2 migliora alcune metriche di classificazione, ma non le metriche di ranking:

| Metrica | Split appaiati | Delta medio E2-E1 | E2 migliore | E1 migliore | Pari | Wilcoxon p-value |
|---|---:|---:|---:|---:|---:|---:|
| MCC | 1.512 | +0,020 | 343 | 244 | 925 | 4,27e-04 |
| AUC-PR | 1.706 | -0,036 | 306 | 481 | 919 | 1,23e-17 |
| AUC-ROC | 1.706 | -0,017 | 369 | 437 | 900 | 2,36e-05 |
| Precision | 1.706 | -0,001 | 298 | 238 | 1.170 | 5,50e-01 |
| Recall | 1.706 | +0,010 | 196 | 165 | 1.345 | 2,02e-02 |
| F1 | 1.706 | +0,007 | 368 | 291 | 1.047 | 2,03e-02 |
| Accuracy | 1.706 | +0,009 | 362 | 276 | 1.068 | 1,10e-03 |

### Feature PDG selezionate da RFECV nella run storica

Le 11 metriche PDG sono state candidate in E2 nella run storica. RFECV le ha selezionate con frequenze diverse:

| Metrica PDG | Split in cui è selezionata |
|---|---:|
| `edgesToVerticesRatio` | 713 |
| `verticesCount` | 709 |
| `edgesCount` | 705 |
| `maxPdgVertices` | 657 |
| `globalInput` | 603 |
| `indirectFanIn` | 588 |
| `directFanIn` | 545 |
| `directFanOut` | 497 |
| `indirectFanOut` | 483 |
| `globalOutput` | 241 |
| `lackOfCohesion` | 10 |

Questa distribuzione conferma che non tutte le metriche PDG hanno lo stesso contributo. Le metriche di dimensione e densità del grafo sono selezionate più spesso; `lackOfCohesion` è quasi sempre scartata.

### Criterio decisionale

E2 è considerato utile se migliora E1 soprattutto su MCC pooled e AUC-PR pooled senza aumentare eccessivamente i falsi positivi. Il confronto principale deve usare metriche pooled; le metriche per split e per repository restano diagnostiche.

Se E2 migliora E1, si può procedere a estendere E1/E2 agli altri classificatori classici. Nel benchmark aggiornato, se E2 non migliora o peggiora, bisogna analizzare il feature manifest `validation_rfe` per capire se le metriche PDG vengono effettivamente selezionate o se introducono rumore.

### Decisione

La run storica E2 migliora E1 sulle metriche di classificazione principali: MCC, precision, recall, F1 e accuracy. Inoltre riduce sia falsi positivi sia falsi negativi. Dopo la revisione metodologica richiesta, questi risultati vanno rigenerati usando `validation_rfe` sia per E1 sia per E2.

Il risultato non è però uniforme: E2 peggiora leggermente AUC-PR e AUC-ROC. L'interpretazione prudente è che le metriche PDG aiutano la decisione finale del Random Forest, ma non migliorano il ranking probabilistico complessivo. Questa distinzione va riportata nella tesi.

Il prossimo passo è estendere E1/E2 agli altri classificatori classici oppure, prima, confrontare E1/E2 Random Forest con E3 GraphSAGE nel report comparativo complessivo.

---

## 16. Organizzazione risultati e benchmark finale

### Struttura risultati

I risultati sono stati separati in due aree:

```text
experiments/results/exploratory/
experiments/results/benchmark/
```

`exploratory/` contiene le run usate per fissare il setup sperimentale:

- soglie sui grafi piccoli;
- affidabilità degli split;
- sweep del bilanciamento;
- stabilità sui seed;
- primo confronto E1/E2 Random Forest.

`benchmark/` è riservata alle run definitive.

Revisione pre-benchmark: l'early stopping delle GNN usa MCC sulla validation quando definibile. Negli split in cui la validation è troppo piccola o produce una metrica non definibile, viene usata la validation loss solo come criterio di fallback per il checkpoint. Questo evita run senza best checkpoint e non modifica il calcolo delle metriche finali, che resta basato sulle predizioni test pooled.

### Pipeline benchmark completa

La pipeline definitiva è gestita da:

```text
experiments/run_full_benchmark.py
```

Esegue in sequenza:

- E1 con `decision_tree`, `logistic_regression`, `naive_bayes`, `random_forest`, `svm`;
- E2 con gli stessi cinque modelli, tutte le 11 metriche PDG candidate e `validation_rfe`;
- E3 con `gcn`, `graphsage`, `gat`, `gin`, `rgcn`.

Comando:

```bash
python -m experiments.run_full_benchmark --benchmark-name final_benchmark_v1 --dataset datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv
```

Output:

```text
experiments/results/benchmark/final_benchmark_v1/
```

La pipeline è riavviabile: ogni modello ha una cartella separata e una run viene considerata completa quando esiste `metrics/pooled_metrics.csv`. Se la pipeline viene interrotta, rilanciando lo stesso comando vengono saltate le run già complete e riparte dalla prima run mancante o incompleta.

Riepiloghi:

```text
experiments/results/benchmark/final_benchmark_v1/_summary/benchmark_plan.csv
experiments/results/benchmark/final_benchmark_v1/_summary/benchmark_summary.csv
experiments/results/benchmark/final_benchmark_v1/_summary/benchmark_summary.md
```

La utility `experiments/compare_results.py` può leggere sia una singola run sia la root del benchmark finale. Nel secondo caso concatena le metriche per split delle cartelle modello e salva `comparison_summary.csv` e `comparison_summary.md` in `_summary/`.

---

## 17. Template per prossimi esperimenti

### Esperimento `<ID>`

| Campo | Valore |
|---|---|
| ID esperimento |  |
| Run name |  |
| Esperimento |  |
| Modello |  |
| Dataset |  |
| Seed |  |
| Output |  |

Comando:

```bash

```

Configurazione rilevante:

| Parametro | Valore |
|---|---|
|  |  |

Risultati principali:

| Metrica | Valore |
|---|---:|
|  |  |

Osservazioni:

- 

Problemi riscontrati:

- 

Decisione:

- 

Prossimi passi:

- 
