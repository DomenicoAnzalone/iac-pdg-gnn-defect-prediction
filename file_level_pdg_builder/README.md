# File-level PDG Builder

Questo modulo genera un PDG a livello di file a partire dal PDG repository-level generato da Scansible.

## Uso

Esegui lo script con:

```bash
python build_file_level_pdg.py --repo /path/to/ansible/repo --file relative/path/to/file.yml
```

Output generati:

- `output/file_level.graphml`
- `output/file_level.dot`

## Struttura

I file inclusi in questo modulo sono:

- `build_file_level_pdg.py`: script principale isolato.
- `project_pdg_info.py`: lettura del PDG repository-level.
- `dictionary_file_tasknode.py`: mappatura task -> file sulla base della provenienza del nodo.
- `extract_task_subgraph.py`: estrazione del sottografo task-level.
- `writer_reader.py`: I/O GraphML.

Questo package è isolato e può essere trasferito in un altro progetto mantenendo la logica principale del repository originale.