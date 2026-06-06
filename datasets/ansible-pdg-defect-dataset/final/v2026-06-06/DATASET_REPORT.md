# Dataset Report - ansible-pdg-defect-dataset v2026-06-06

Generated at: 2026-06-06T15:37:12.772507+00:00

## Inputs

- RADON input: `output\second_pdg_extraction\input_dataset.csv`
- PDG extraction status: `datasets\ansible-pdg-defect-dataset\pdg_extraction\extraction_status.csv`
- Graph base directory: `output`

## Versioned Outputs

- Final dataset: `datasets\ansible-pdg-defect-dataset\final\v2026-06-06\ansible-pdg-defect-dataset_v2026-06-06_final.csv`
- Exclusions: `datasets\ansible-pdg-defect-dataset\final\v2026-06-06\ansible-pdg-defect-dataset_v2026-06-06_exclusions.csv`
- PDG metrics: `datasets\ansible-pdg-defect-dataset\final\v2026-06-06\ansible-pdg-defect-dataset_v2026-06-06_pdg_metrics.csv`
- Reports: `datasets\ansible-pdg-defect-dataset\final\v2026-06-06\reports`

## Filtering Story

- Rows after RADON filtering: 61542
- Rows with a successful PDG extraction: 37834
- Rows with extracted PDG metrics: 37796
- Final rows after graph-quality filtering: 37796
- Final repositories: 123
- Final label distribution: {'0': 23880, '1': 13916}

## Exclusion Reasons

- `pdg_status_UNSUPPORTED_FILE_TYPE`: 19128
- `pdg_status_REAL_EXTRACTION_FAILURE`: 2002
- `pdg_status_EMPTY_GRAPH`: 1603
- `pdg_status_LOW_QUALITY_GRAPH`: 535
- `pdg_status_CLONE_FAILURE`: 431
- `graphml_parse_failure`: 38
- `pdg_status_EXTRACTION_TIMEOUT`: 9

## Graph Quality Policy

- Minimum nodes: 3
- Minimum edges: 2
- Rationale: a graph used by a message-passing GNN must contain nodes and connectivity. PyTorch Geometric represents graph connectivity through `edge_index`; DGL describes graph classification as message passing over nodes/edges followed by graph-level readout. Empty graphs, edgeless graphs, and tiny placeholder graphs do not provide meaningful dependence structure for this study.
- Online references checked for this policy: PyTorch Geometric data/isolated-node documentation (https://pytorch-geometric.readthedocs.io/en/1.3.0/modules/data.html), DGL message passing documentation (https://www.dgl.ai/dgl_docs/guide/message.html), and NetworkX empty/null graph definitions (https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.is_empty.html).
- The selected threshold matches the PDG extraction run configuration and acts as a conservative technical filter: it excludes empty or placeholder outputs without removing small but valid Ansible task graphs.

## PDG Metric Semantics

- The dataset includes the 11 PDG metric columns used in the Iuliano/Pontillo line of work.
- `verticesCount`, `edgesCount`, and `edgesToVerticesRatio` are directly measured on the file-level GraphML.
- `directFanIn`, `directFanOut`, `indirectFanIn`, and `indirectFanOut` are computed from direct and transitive graph reachability around task nodes.
- `globalInput` and `globalOutput` are file-level proxies based on non-task source/sink nodes connected to tasks.
- `maxPdgVertices` is equal to the file-level graph size because the final artifact stores one graph per file snapshot.
- `lackOfCohesion` is a normalized file-level task connectivity proxy; exact task-slice overlap is not reconstructable from the current GraphML alone.
- The column `pdg_metric_semantics=file_level_proxy_v1` marks these semantics explicitly.

## Dataset Checks

- Duplicate RADON key rows removed after first occurrence: 0
- Duplicate status key rows removed after first occurrence: 0
- Final unique keys: 37796
- Final rows with missing label: 0
- Status distribution before filtering: {'SUCCESS': 37834, 'UNSUPPORTED_FILE_TYPE': 19128, 'REAL_EXTRACTION_FAILURE': 2002, 'EMPTY_GRAPH': 1603, 'LOW_QUALITY_GRAPH': 535, 'CLONE_FAILURE': 431, 'EXTRACTION_TIMEOUT': 9}
