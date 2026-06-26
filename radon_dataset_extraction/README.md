# RADON Ansible Dataset Pipeline

Repository per costruire dataset di defect prediction a partire da repository
Ansible pubbliche.

Il progetto contiene due pipeline distinte ma compatibili:

1. **Pipeline di discovery GitHub**: cerca automaticamente repository candidate
   legate ad Ansible e genera un file `repo_url,branch`.
2. **Pipeline RADON**: usa quel file, oppure una lista manuale equivalente, per
   estrarre dataset per singola repository, unirli e produrre il dataset finale
   filtrato.

La discovery serve a costruire in modo riproducibile una lista ampia di
candidate. La pipeline RADON resta la fase di validazione effettiva: una
repository scoperta automaticamente non è considerata utile a priori, ma solo se
produce un dataset valido e supera i filtri finali.

## Struttura

```text
.
├── Dockerfile
├── input_repos_mixed_final.txt
├── requirements.txt
└── scripts
    ├── discover_ansible_repos.py
    ├── run_full_pipeline.py
    ├── batch_export_ansible_datasets_run.py
    ├── export_ansible_dataset.py
    └── merge_run_datasets.py
```

La pipeline va eseguita tramite Docker. In questo modo usa Python 3.8 dentro al
container e non dipende dalla versione di Python installata sul sistema.

La cartella `output/` viene creata automaticamente durante l'esecuzione ed è
ignorata da Git.

## Build Docker

Requisiti base:

- Docker installato e in esecuzione;
- accesso a Internet durante la build, per scaricare l'immagine Python e le
  dipendenze Python;
- esecuzione del comando dalla cartella `radon_dataset_extraction`, dove si
  trova il Dockerfile dedicato alla fase.

Dalla cartella `radon_dataset_extraction`:

```powershell
docker build -t radon-ansible-pipeline .
```

## Pipeline 1: Discovery GitHub

La discovery interroga GitHub Search API usando più strategie, non una singola
query. L'obiettivo è trovare molte repository candidate che nei metadata
sembrano collegate ad Ansible.

Passaggi principali:

1. esegue query GitHub su termini e topic Ansible-related;
2. recupera i metadata delle repository restituite da GitHub;
3. usa il `default_branch` reale indicato da GitHub, quindi non assume sempre
   `master`;
4. esclude repository archiviate, disabilitate, troppo poco pertinenti o
   probabilmente didattiche/generiche;
5. deduplica le repository trovate da più query usando `full_name`;
6. assegna un punteggio di priorità basato su stelle, fork, recency, topic e
   numero di strategie abbinate;
7. genera il file `repo_url,branch` compatibile con la pipeline RADON;
8. salva report documentali sulle repository selezionate, escluse e sui criteri
   applicati.

Le strategie includono segnali come:

- `ansible role`, `ansible-role`;
- `ansible collection`, `ansible-collection`;
- `playbook`, `molecule`;
- `infrastructure automation`, `hardening`, `security`, `monitoring`;
- `prometheus`, `grafana`, `docker`, `nginx`, `mysql`, `postgresql`,
  `kubernetes`, `openstack`;
- topic GitHub come `ansible`, `ansible-role`, `ansible-collection`.

La discovery esclude, quando possibile:

- repository archiviate o disabilitate;
- repository sotto la soglia minima di stelle configurata;
- repository probabilmente didattiche o generiche, ad esempio `tutorial`,
  `example`, `workshop`, `course`, `cheatsheet`, `exercise`, `interview`,
  `awesome`, `demo`;
- repository senza segnali Ansible chiari nei metadata disponibili.

Il filtro non è volutamente troppo aggressivo. La discovery produce candidate
plausibili; la pipeline RADON decide poi quali repository generano davvero dati
utili.

### Comando Solo Discovery

Esempio per generare una lista ampia di repository candidate:

```powershell
docker run --rm -it `
  -e GITHUB_TOKEN="${env:GITHUB_TOKEN}" `
  -v "${PWD}\output:/app/output" `
  radon-ansible-pipeline `
  scripts/discover_ansible_repos.py `
    --max-repos 500 `
    --min-stars 1 `
    --max-pages 3 `
    --per-page 100 `
    --output /app/output/discovery/ansible_candidate_repos.txt `
    --selected-report /app/output/discovery/ansible_candidate_repos_selected.csv `
    --excluded-report /app/output/discovery/ansible_candidate_repos_excluded.csv `
    --methodology-report /app/output/discovery/ansible_discovery_methodology.md
```

Il file importante da passare alla pipeline RADON e:

```text
output/discovery/ansible_candidate_repos.txt
```

Formato:

```text
repo_url,branch
```

Esempio:

```text
https://github.com/geerlingguy/ansible-role-apache,master
https://github.com/elastic/ansible-elasticsearch,main
```

### Parametri Discovery Principali

- `--max-repos`: numero massimo di repository da salvare nel file finale.
  Utile per passare da test piccoli a run grandi, ad esempio `50`, `500`,
  `1000`.
- `--min-stars`: soglia minima di stelle. `0` include anche repository senza
  stelle; `1` o valori superiori riducono un po' il rumore.
- `--max-pages`: numero massimo di pagine GitHub Search per ogni strategia.
  Aumentarlo trova più candidate, ma aumenta tempi e consumo di rate limit.
- `--per-page`: risultati per pagina GitHub. Il valore tipico è `100`.
- `--output`: path del file `repo_url,branch` generato.
- `--selected-report`: path del CSV con le repository selezionate e le
  motivazioni.
- `--excluded-report`: path del CSV con repository scartate o duplicate e
  motivazione sintetica.
- `--methodology-report`: path del report Markdown che documenta query, soglie,
  criteri di inclusione/esclusione, deduplica e ordinamento.
- `--token`: token GitHub esplicito. Se non viene passato, lo script usa
  `GITHUB_TOKEN` o `GH_TOKEN` se presenti.
- `--strategies`: opzionale, limita le strategie da eseguire. Utile per test
  piccoli, ad esempio `"ansible-role,topic ansible"`. Se non viene passato,
  vengono usate tutte le strategie.

### Output Discovery

- `ansible_candidate_repos.txt`: file `repo_url,branch` da usare come input
  RADON.
- `ansible_candidate_repos_selected.csv`: report delle repository selezionate,
  con URL, branch, `full_name`, stelle, fork, linguaggio, date, descrizione,
  topic, query abbinate, score e motivo sintetico della selezione.
- `ansible_candidate_repos_excluded.csv`: report delle repository escluse o
  duplicate, con motivo sintetico.
- `ansible_discovery_methodology.md`: report metodologico con strategie, soglie,
  criteri di inclusione/esclusione, deduplicazione e prioritizzazione.

## Pipeline 2: RADON

La pipeline RADON è la pipeline già presente nel progetto. Prende in input un
file con una repository per riga:

```text
repo_url,branch
```

Il file può essere:

- una lista manuale, ad esempio `input_repos_mixed_final.txt`;
- il file prodotto dalla discovery, ad esempio
  `output/discovery/ansible_candidate_repos.txt`.

Passaggi principali:

1. legge la lista `repo_url,branch`;
2. per ogni repository conta i tag Git disponibili;
3. scarta le repository con meno di 2 tag;
4. clona e analizza la repository con RADON/repository-miner;
5. estrae fixing commit e file Ansible coinvolti;
6. etichetta gli snapshot dei file;
7. calcola metriche product, process e delta;
8. produce un CSV per ogni repository processata con successo;
9. genera `batch_summary.csv` con lo stato di ogni repository;
10. crea `merged_dataset.csv` con tutte le repository `SUCCESS`;
11. crea `merged_dataset_filtered.csv` tenendo solo repository con dati minimi
    sufficienti.

Alla prima esecuzione di una run, il file di input viene copiato dentro la
directory della run:

```text
output/runs/<run-name>/input_repos.csv
```

Se la pipeline viene interrotta e poi rilanciata con lo stesso `--run-name`,
viene usato questo snapshot interno. In questo modo la run resta riproducibile
anche se il file di input originale viene modificato dopo il primo avvio.

`batch_summary.csv` e anche il report che documenta le repository che non sono
andate a buon fine nella fase RADON: per ogni riga salva lo stato, eventuali
metriche intermedie e l'errore sintetico quando disponibile.

Ogni esecuzione salva anche metadati della run nella stessa directory:

- `pipeline_metadata.json`: comando e parametri usati per avviare
  `run_full_pipeline.py`;
- `run_metadata.json`: comando e parametri della fase batch RADON, snapshot
  input usato, numero di repository totali, già completate e ancora pending.

Questi file sono pensati per documentare la run e ricostruire facilmente il
comando di resume o di restart pulito.

Gli stati principali in `batch_summary.csv` sono:

- `SUCCESS`: estrazione riuscita e dataset generato;
- `FAILED_EXPORT`: errore durante l'export RADON;
- `FAILED_TIMEOUT`: timeout della singola repository;
- `EMPTY_DATASET`: export completato ma senza righe utili;
- `SKIPPED_TOO_FEW_TAGS`: repository scartata perché ha meno di 2 tag.

I filtri minimi del dataset finale sono:

- almeno 100 righe;
- almeno 20 esempi positivi;
- almeno 20 esempi negativi.

Per gli esperimenti è consigliato usare:

```text
merged_dataset_filtered.csv
```

## Comando RADON Con Lista Manuale

```powershell
docker run --rm -it `
  -v "${PWD}\output:/app/output" `
  -v "${PWD}\input_repos_mixed_final.txt:/app/input_repos_mixed_final.txt" `
  radon-ansible-pipeline `
  scripts/run_full_pipeline.py `
    --input /app/input_repos_mixed_final.txt `
    --run-name run_manual `
    --timeout 3600
```

## Comando RADON Con File Generato Dalla Discovery

```powershell
docker run --rm -it `
  -v "${PWD}\output:/app/output" `
  radon-ansible-pipeline `
  scripts/run_full_pipeline.py `
    --input /app/output/discovery/ansible_candidate_repos.txt `
    --run-name run_from_discovery `
    --timeout 3600 `
    --workers 1
```

Se la run viene interrotta, puoi rilanciare lo stesso comando con lo stesso
`--run-name`: la pipeline riprende automaticamente dalle repository mancanti,
usando `output/runs/<run-name>/input_repos.csv` e `batch_summary.csv`.

Se invece vuoi cancellare la run e ripartire da zero, aggiungi `--force`:

```powershell
docker run --rm -it `
  -v "${PWD}\output:/app/output" `
  radon-ansible-pipeline `
  scripts/run_full_pipeline.py `
    --input /app/output/discovery/ansible_candidate_repos.txt `
    --run-name run_from_discovery `
    --timeout 3600 `
    --workers 1 `
    --force
```

### Parametri RADON Principali

- `--input`: file `repo_url,branch` da processare. Può essere manuale o generato
  dalla discovery.
- `--run-name`: nome della run. Gli output finiscono in
  `output/runs/<run-name>/`.
- `--output-dir`: directory che contiene tutte le run. Default:
  `output/runs/`.
- `--timeout`: timeout in secondi per ogni repository. Per repository grandi
  conviene usare valori alti, ad esempio `3600`.
- `--workers`: numero di repository processate in parallelo. Il default è `1`,
  cioè esecuzione sequenziale.
- `--keep-clones`: conserva le repository clonate in `clone_dirs/`. Di default
  è disattivato, quindi i cloni vengono cancellati dopo ogni repository
  processata per risparmiare spazio disco.
- `--force`: elimina la directory della run se esiste già e rilancia tutto.

## Esecuzione RADON Parallela

Per processare più repository contemporaneamente, usa `--workers N`:

```powershell
docker run --rm -it `
  -v "${PWD}\output:/app/output" `
  radon-ansible-pipeline `
  scripts/run_full_pipeline.py `
    --input /app/output/discovery/ansible_candidate_repos.txt `
    --run-name run_from_discovery_parallel `
    --timeout 3600 `
    --workers 4
```

Con `--workers 4`, la pipeline mantiene al massimo 4 repository in lavorazione
contemporaneamente. Ogni worker usa una directory di clone separata dentro la
run, quindi due processi non lavorano sulla stessa repository e non condividono
la stessa cartella temporanea. Il file `batch_summary.csv` viene scritto solo dal
processo principale e aggiornato in modo atomico dopo ogni repository completata.
Il dataset aggregato `merged_dataset.csv` e il dataset finale
`merged_dataset_filtered.csv` vengono generati solo dopo la fine della fase di
export, quindi non vengono riscritti continuamente durante il processamento.

La console mostra righe compatte `START`/`DONE`, una progress bar complessiva e
l'elenco delle repository ancora in esecuzione. Durante le attese lunghe stampa
un aggiornamento periodico con tempo trascorso e ultima riga di log nota per
ogni worker. I log dettagliati prodotti dall'export di ogni repository sono
salvati in streaming in `output/runs/<run-name>/logs/`.

Per default le directory di clone in `output/runs/<run-name>/clone_dirs/`
vengono eliminate automaticamente dopo ogni repository processata. All'avvio di
una run vengono rimossi anche eventuali cloni rimasti da esecuzioni precedenti o
interrotte. Se vuoi conservarle per debug, aggiungi `--keep-clones`.

Consigli pratici:

- inizia con `--workers 2` o `--workers 4`;
- su una macchina con pochi core o poca RAM, resta su `2`;
- su una macchina più potente puoi provare `4`, `6` o `8`;
- evita valori troppo alti: ogni repository può clonare molto codice, usare CPU,
  RAM, disco e rete;
- una regola semplice e usare circa metà dei core logici disponibili, poi
  aumentare solo se CPU, RAM e disco restano sotto controllo.

## Arresto E Resume

Per fermare una run in modo semplice, premi `Ctrl+C` nel terminale dove sta
girando Docker. La pipeline intercetta l'interruzione, termina gli export attivi
e mantiene in `batch_summary.csv` tutte le repository già completate.

Per riprendere, rilancia lo stesso comando con lo stesso `--run-name` e senza
`--force`:

```powershell
docker run --rm -it `
  -v "${PWD}\output:/app/output" `
  radon-ansible-pipeline `
  scripts/run_full_pipeline.py `
    --input /app/output/discovery/ansible_candidate_repos.txt `
    --run-name run_from_discovery_parallel `
    --timeout 3600 `
    --workers 4
```

Durante il resume:

- le repository con stato terminale in `batch_summary.csv` vengono saltate;
- le repository mancanti vengono processate;
- il file di input originale non viene riletto se esiste già lo snapshot
  `output/runs/<run-name>/input_repos.csv`;
- `--force` disabilita il resume, cancella la run precedente e riparte da zero.

## Discovery + RADON In Un Solo Comando

Se si vuole eseguire discovery e RADON senza passaggi manuali, `run_full_pipeline.py`
supporta anche `--discover`:

```powershell
docker run --rm -it `
  -e GITHUB_TOKEN="${env:GITHUB_TOKEN}" `
  -v "${PWD}\output:/app/output" `
  radon-ansible-pipeline `
  scripts/run_full_pipeline.py `
    --run-name run_discovery_auto `
    --discover `
    --discover-max-repos 500 `
    --discover-min-stars 1 `
    --discover-max-pages 3 `
    --timeout 3600 `
    --workers 4 `
    --force
```

In questa modalità:

- la discovery salva i propri file in
  `output/runs/<run-name>/discovery/`;
- il file `ansible_candidate_repos.txt` generato viene passato automaticamente
  alla pipeline RADON;
- i report RADON restano in `output/runs/<run-name>/`.
- se la run viene interrotta durante RADON, al resume viene usato lo snapshot
  `output/runs/<run-name>/input_repos.csv` e la discovery non viene rieseguita,
  salvo uso di `--force`.

Parametri discovery disponibili dentro `run_full_pipeline.py`:

- `--discover-max-repos`;
- `--discover-min-stars`;
- `--discover-max-pages`;
- `--discover-per-page`;
- `--github-token`;
- `--discovery-input-output`;
- `--discovery-selected-report`;
- `--discovery-excluded-report`;
- `--discovery-methodology-report`.

## Output RADON

Ogni run RADON viene salvata in:

```text
output/runs/<run-name>/
```

File principali:

- `datasets/`: un CSV per ogni repository processata con successo;
- `logs/`: log dettagliato dell'export per singola repository;
- `pipeline_metadata.json`: comando e parametri dell'orchestratore completo;
- `run_metadata.json`: comando e parametri della fase batch RADON;
- `batch_summary.csv`: esito della fase di estrazione per ogni repository,
  incluse quelle non andate a buon fine;
- `merged_dataset.csv`: merge di tutte le repository con stato `SUCCESS`;
- `merged_dataset_filtered.csv`: dataset finale consigliato per gli esperimenti;
- `batch_summary_filtered_kept.csv`: repository mantenute dal filtro finale;
- `batch_summary_filtered_discarded.csv`: repository scartate dal filtro finale,
  con motivazioni come `dataset_rows<100`, `positives<20`,
  `negatives<20`.

## Note Per Run Grandi

Per run grandi conviene procedere gradualmente:

1. eseguire prima una discovery piccola, ad esempio `--max-repos 20` o `50`;
2. controllare `ansible_candidate_repos_selected.csv` e
   `ansible_discovery_methodology.md`;
3. lanciare una run RADON piccola;
4. aumentare progressivamente a `500`, `800` o `1000` candidate.

La discovery è relativamente veloce, ma consuma rate limit GitHub. Per run grandi
è consigliato impostare `GITHUB_TOKEN`.

La fase RADON può richiedere molto tempo: ogni repository viene interrogata,
clonata e processata separatamente. Il dataset realmente utile non coincide con
il numero di repository scoperte, ma con le repository che risultano `SUCCESS` e
superano il filtering finale.
