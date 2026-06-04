# File-level PDG Extraction Pipeline

Questa cartella contiene una pipeline isolata per estrarre Program Dependence Graph
file-level da un dataset di file Ansible etichettati.

La pipeline prende in input un CSV come:

- `input/first_pdg_extraction/ansible.csv`;
- `input/second_pdg_extraction/ansible-2.csv`;
- un altro dataset RADON compatibile.

Per ogni riga, la pipeline usa almeno:

```text
repository, commit, filepath, failure_prone
```

Il risultato principale è un `pdg.graphml` associato alla stessa tupla
`(repository, commit, filepath)` e un CSV di stato che documenta tutte le righe
processate.

## Perché questa pipeline è separata

La fase di estrazione dei PDG richiede Scansible, Git, Graphviz e alcune
dipendenze Python specifiche. Non richiede invece le dipendenze usate per il
training dei classificatori o delle GNN.

Questa cartella fornisce quindi:

- un runner batch configurabile da linea di comando;
- parallelismo sicuro per repository;
- un Dockerfile dedicato;
- requirements dedicati;
- `.dockerignore` dedicato;
- output riproducibili, metadati di run e resume;
- test unitari per le parti indipendenti da Scansible.

## Autonomia della fase

La pipeline di estrazione file-level diretta è completamente contenuta in questa
cartella. Lo script principale non importa moduli da `../scripts` e incorpora
direttamente:

- derivazione dell'URL Git da `repository` o uso diretto di `repo_url`;
- clone temporaneo delle repository;
- checkout dei commit;
- creazione di un playbook wrapper temporaneo per i file `tasks/`;
- invocazione di Scansible;
- conversione DOT -> GraphML;
- costruzione dei percorsi di output;
- generazione del CSV di stato e del CSV dei successi;
- resume, metadati, report e parallelismo.

Gli script rimasti nella cartella root `scripts/` non sono necessari per eseguire
questa fase:

| Script root | Motivo per cui non appartiene alla pipeline diretta |
|---|---|
| `file_level_pdg_baseline_comparator.py` | Calcola e confronta metriche dopo l'estrazione dei grafi |
| `pdg_repo_level_extractor.py` | Estrae PDG repository-level, non PDG file-level diretti |
| `pdg_combined_extractor.py` | Implementa una modalità alternativa repo-level -> file-level |
| `clean.py` | Contiene utility legacy; il CSV dei successi è già generato dal runner |
| `clone.py` | Utility legacy sostituita dal clone temporaneo interno al runner |
| `change_commit.py` | Utility legacy sostituita dalla gestione Git interna al runner |

Il vecchio entrypoint `scripts/pdg_file_level_extractor.py` è stato rimosso per
evitare due implementazioni divergenti della stessa fase.

## Struttura

```text
pdg_file_level_extraction/
├── .dockerignore
├── Dockerfile
├── README.md
├── requirements.txt
├── scripts/
│   └── pdg_file_level_extractor.py
└── tests/
    └── test_pdg_file_level_extractor.py
```

## Requisiti dell'input

Il CSV deve contenere le colonne:

| Colonna | Descrizione |
|---|---|
| `repository` oppure `repo_url` | Identificativo `owner/repo` o URL Git della repository |
| `commit` | Commit da usare per il checkout |
| `filepath` | Percorso del file Ansible all'interno della repository |
| `failure_prone` | Label binaria associata al campione |

Le altre colonne del dataset possono essere presenti, ma non sono necessarie per
l'estrazione del grafo.

Se è presente `repo_url`, la pipeline usa direttamente quel valore. Se è presente
soltanto `repository`, ad esempio `owner/repo`, viene costruito l'URL
`https://github.com/owner/repo.git`.

Non è necessario preparare o montare una directory di clone locali.
Il processo deve però avere accesso di rete alle repository Git e le eventuali
credenziali necessarie per repository private.

Per la PDG extraction, i file `tasks/` vengono eseguiti tramite un playbook
wrapper temporaneo creato nella stessa directory del file task. Questa scelta è
importante: Scansible segue `import_tasks` usando le regole di risoluzione dei
path di Ansible; se il wrapper viene creato altrove, il file importato può non
essere trovato e il risultato diventa un grafo placeholder composto solo da
`import_tasks` e dal path del file. I file `vars/`, `defaults/`, `meta/` e
`handlers/` vengono esclusi come `UNSUPPORTED_FILE_TYPE`, perché non rappresentano
un flusso di task eseguibile da Scansible come playbook.

La pipeline applica anche una validazione minima del grafo estratto. Un DOT
prodotto correttamente da Scansible non viene marcato automaticamente come
`SUCCESS`: se è troppo piccolo o corrisponde al placeholder di un include non
risolto, viene classificato come `LOW_QUALITY_GRAPH` e non entra nel CSV dei
successi usabile per training e metriche PDG.

## Sicurezza del parallelismo

L'estrazione non può elaborare in parallelo due righe della stessa repository,
perché ogni riga richiede un checkout potenzialmente diverso.

Il runner evita questa race condition nel seguente modo:

1. raggruppa tutte le righe che usano la stessa repository;
2. ogni worker clona una repository in una directory temporanea della run;
3. elabora sequenzialmente tutte le righe della repository;
4. elimina il clone appena l'ultima riga è stata processata;
5. elabora in parallelo soltanto repository differenti;
6. lascia al solo processo coordinatore la scrittura di `extraction_status.csv`,
   del CSV dei successi e del report.

Il parametro `--workers` indica quindi il numero massimo di repository differenti
elaborate contemporaneamente, non il numero di file della stessa repository.

La pipeline crea inoltre un file `.run.lock` nella directory della run. Una
seconda esecuzione con lo stesso `--run-name` viene rifiutata finché la prima è
attiva, perché due container non possono condividere in sicurezza gli stessi
clone temporanei e gli stessi file di stato.

## Build Docker

Eseguire il build dalla cartella `pdg_file_level_extraction`:

```powershell
cd pdg_file_level_extraction
docker build -t pdg-file-level-extraction .
```

Dopo modifiche al Dockerfile o alle dipendenze, usare un build senza cache per
evitare di riutilizzare layer con versioni incompatibili di Ansible:

```powershell
docker build --no-cache -t pdg-file-level-extraction .
```

Il Dockerfile:

- usa Python 3.12;
- installa Git e certificati TLS per clonare repository;
- installa le librerie proprie della pipeline definite in `requirements.txt`;
- installa `ansible-core==2.15.12`, versione compatibile con Scansible;
- clona e installa Scansible senza risolvere nuovamente le dipendenze;
- verifica in fase di build che Scansible sia importabile;
- configura lo script principale come entrypoint Python.

Per usare un fork o un branch specifico di Scansible:

```powershell
docker build `
  --build-arg SCANSIBLE_REPOSITORY=https://github.com/softwarelanguageslab/scansible.git `
  --build-arg SCANSIBLE_REF=<branch-tag-o-commit> `
  -t pdg-file-level-extraction .
```

Se `SCANSIBLE_REF` non viene specificato, viene usato il branch predefinito della
repository Scansible.

## Esecuzione con Docker

### Esempio: prima estrazione

Dalla cartella `pdg_file_level_extraction`:

```powershell
docker run --rm -it `
  -v "${PWD}\..\input\first_pdg_extraction\ansible.csv:/app/input/ansible.csv:ro" `
  -v "${PWD}\..\output:/app/output" `
  pdg-file-level-extraction `
  scripts/pdg_file_level_extractor.py `
    --input /app/input/ansible.csv `
    --output-dir /app/output `
    --run-name first_pdg_extraction `
    --workers 4 `
    --timeout 600
```

L'output viene scritto in:

```text
output/first_pdg_extraction/
```

### Esempio: seconda estrazione

```powershell
docker run --rm -it `
  -v "${PWD}\..\input\second_pdg_extraction\ansible-2.csv:/app/input/ansible-2.csv:ro" `
  -v "${PWD}\..\output:/app/output" `
  pdg-file-level-extraction `
  scripts/pdg_file_level_extractor.py `
    --input /app/input/ansible-2.csv `
    --output-dir /app/output `
    --run-name second_pdg_extraction `
    --workers 8 `
    --timeout 600 `
    --min-pdg-nodes 3 `
    --min-pdg-edges 2
```

### Esempio Linux/macOS

```bash
docker run --rm -it \
  -v "$PWD/../input/first_pdg_extraction/ansible.csv:/app/input/ansible.csv:ro" \
  -v "$PWD/../output:/app/output" \
  pdg-file-level-extraction \
  scripts/pdg_file_level_extractor.py \
    --input /app/input/ansible.csv \
    --output-dir /app/output \
    --run-name first_pdg_extraction \
    --workers 4 \
    --timeout 600
```

## Output della run

La directory di output è sempre specifica per la run:

```text
<output-dir>/<run-name>/
```

Eseguendo il comando con `--output-dir /app/output --run-name second_pdg_extraction`,
la seconda estrazione viene quindi salvata in `output/second_pdg_extraction`.

Ogni run produce:

```text
<output-dir>/<run-name>/
├── input_dataset.csv
├── run_metadata.json
├── extraction_status.csv
├── <nome-csv-input>_rows_successfull_extracted.csv
├── extraction_report.txt
├── logs/
│   └── <owner>__<repo>.log
└── pdg_file_level/
    └── <owner>/<repo>/<commit>/<filepath>/PDG_FILE_LEVEL/
        ├── pdg.dot
        └── pdg.graphml
```

### `input_dataset.csv`

È lo snapshot dell'input usato dalla run. Al resume viene riutilizzato per evitare
che una modifica successiva del CSV originale cambi il significato degli indici di
riga.

### `run_metadata.json`

Contiene comando, parametri, percorsi, conteggi e storico delle esecuzioni della
run.

### `extraction_status.csv`

Contiene una riga per ogni entry completata del dataset:

```text
row_index,repository,commit,filepath,failure_prone,status,nodes,edges,graphml_path,error
```

Gli stati terminali possibili sono:

| Stato | Significato |
|---|---|
| `SUCCESS` | PDG estratto, salvato e sopra le soglie minime di qualità configurate |
| `LOW_QUALITY_GRAPH` | DOT prodotto da Scansible ma non utilizzabile come PDG di training, ad esempio placeholder `import_tasks` non risolto o grafo sotto soglia |
| `MISSING_REQUIRED_FIELD` | Mancano repository, commit o filepath |
| `CLONE_FAILURE` | La repository non è stata clonata correttamente |
| `REPOSITORY_NOT_FOUND` | Stato legacy riconosciuto durante il resume di run precedenti |
| `CHECKOUT_FAILURE` | Il commit non è disponibile o il checkout è fallito |
| `INVALID_FILEPATH` | Il percorso del file non è valido |
| `UNSUPPORTED_FILE_TYPE` | File `meta/`, `handlers/`, `vars/` o `defaults/`, esclusi perché non sono play/task file eseguibili |
| `FILE_NOT_FOUND` | Il file non esiste al commit richiesto |
| `EXTRACTION_TIMEOUT` | Scansible ha superato il timeout |
| `REAL_EXTRACTION_FAILURE` | Scansible ha restituito un errore |
| `GRAPH_PARSE_FAILURE` | Il DOT prodotto non è stato interpretato correttamente |
| `EMPTY_GRAPH` | Il grafo non contiene nodi o archi |
| `UNEXPECTED_ERROR` | Errore inatteso documentato nel log |

`GRAPH_PARSE_FAILURE` non viene considerato terminale durante il resume: se un
DOT era già stato estratto ma la conversione GraphML era fallita, la pipeline può
riprovarne la conversione con il parser aggiornato.

### `<nome-csv-input>_rows_successfull_extracted.csv`

Contiene soltanto le righe di `extraction_status.csv` con stato `SUCCESS`. Le
righe `LOW_QUALITY_GRAPH` restano tracciate nello status CSV e nel report, ma non
entrano nel dataset dei PDG candidati per training. Il nome deriva dal CSV di
input:

```text
ansible.csv   -> ansible_rows_successfull_extracted.csv
ansible-2.csv -> ansible-2_rows_successfull_extracted.csv
```

### `pdg_file_level/`

Contiene i grafi ordinati per repository, commit e file. Il percorso è compatibile
con la struttura consolidata in `output/first_pdg_extraction`.

## Resume, restart e refresh dell'input

### Resume

Rilanciare lo stesso comando con lo stesso `--run-name`:

```powershell
docker run --rm -it `
  -v "${PWD}\..\input\first_pdg_extraction\ansible.csv:/app/input/ansible.csv:ro" `
  -v "${PWD}\..\output:/app/output" `
  pdg-file-level-extraction `
  scripts/pdg_file_level_extractor.py `
    --input /app/input/ansible.csv `
    --output-dir /app/output `
    --run-name first_pdg_extraction `
    --workers 4
```

La pipeline:

- riutilizza `input_dataset.csv`;
- salta gli indici di riga già presenti con stato terminale;
- riutilizza i `pdg.graphml` già esistenti;
- aggiorna gli output in modo atomico.

Non avviare il comando di resume mentre la run precedente è ancora attiva.

### Ripartenza da zero

Usare `--force`:

```text
--force
```

La directory della run viene eliminata prima dell'avvio.

Usare `--force` anche dopo una run fallita che deve essere ripetuta: gli stati
come `CHECKOUT_FAILURE` e `CLONE_FAILURE` sono terminali e verrebbero saltati da
un normale resume.

### Sostituire lo snapshot di input

Usare `--refresh-input` soltanto quando si vuole usare un nuovo CSV mantenendo i
grafi già presenti nella run:

```text
--refresh-input
```

Gli output di stato vengono azzerati, mentre i grafi esistenti possono essere
riutilizzati se la stessa tupla ricompare nel nuovo dataset.

## Parametri principali

| Parametro | Descrizione |
|---|---|
| `--input` | CSV di input |
| `--run-name` | Nome della directory della run |
| `--output-dir` | Directory che contiene le run; localmente il default è il folder root `output` |
| `--workers` | Repository differenti elaborate contemporaneamente |
| `--timeout` | Timeout per singolo file; `0` lo disabilita |
| `--scansible-command` | Eseguibile Scansible da invocare |
| `--keep-dot` / `--no-keep-dot` | Mantiene o elimina il file DOT dopo la conversione |
| `--min-pdg-nodes` | Numero minimo di nodi per considerare il PDG utilizzabile; default `3` |
| `--min-pdg-edges` | Numero minimo di archi per considerare il PDG utilizzabile; default `2` |
| `--refresh-input` | Sostituisce lo snapshot di input e azzera gli stati |
| `--force` | Elimina la run e riparte da zero |
| `--max-rows` | Limita le righe per test piccoli |

All'avvio il runner esegue un preflight di Scansible con `scansible --help`. Se
Scansible non parte, la pipeline termina subito senza clonare repository e senza
produrre migliaia di fallimenti ripetitivi.

## Test rapido

Per validare la configurazione senza processare l'intero dataset:

```powershell
docker run --rm -it `
  -v "${PWD}\..\input\first_pdg_extraction\ansible.csv:/app/input/ansible.csv:ro" `
  -v "${PWD}\..\output:/app/output" `
  pdg-file-level-extraction `
  scripts/pdg_file_level_extractor.py `
    --input /app/input/ansible.csv `
    --output-dir /app/output `
    --run-name pdg_extraction_smoke_test `
    --workers 2 `
    --max-rows 20 `
    --force
```

## Esecuzione senza Docker

È possibile eseguire la pipeline localmente se:

- `scansible` è installato ed è disponibile nel `PATH`;
- Git e Graphviz sono installati;
- le dipendenze in `requirements.txt` sono installate.

Esempio:

```powershell
python scripts/pdg_file_level_extractor.py `
  --input ..\input\first_pdg_extraction\ansible.csv `
  --output-dir ..\output `
  --run-name first_pdg_extraction `
  --workers 4
```

## Test unitari

Dalla cartella `pdg_file_level_extraction`:

```powershell
python -m unittest discover -s tests -v
```

I test non invocano Scansible: verificano normalizzazione dei percorsi,
costruzione degli URL di clone, raggruppamento sicuro per repository, ciclo di
vita dei clone temporanei e costruzione dei percorsi di output.

## Note operative

- Aumentare `--workers` gradualmente: ogni worker può usare CPU, RAM e disco.
- Il numero utile di worker non può superare il numero di repository differenti
  ancora da processare.
- Ogni worker mantiene su disco al massimo un clone temporaneo; il clone viene
  eliminato dopo il completamento della repository.
- Repository molto grandi o file complessi possono richiedere timeout superiori.
- I log per repository permettono di diagnosticare errori Scansible senza
  mescolare l'output dei worker.
- Per il confronto sperimentale finale, usare il CSV dei successi come base per
  associare label e percorsi ai grafi estratti.
