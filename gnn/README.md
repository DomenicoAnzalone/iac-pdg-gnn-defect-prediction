# GNN Pipeline for PDG-based Defect Prediction

End-to-end machine learning pipeline for predicting software defects in Ansible Infrastructure-as-Code (IaC) using Graph Neural Networks (GNNs) and Program Dependence Graphs (PDGs).

## Features

- **Robust Ingest**: Reads and normalizes Ansible CSV dataset (227K+ rows) with walk-forward temporal splits
- **PDG Extraction**: Integrates existing PDG extraction modules (repository-level and file-level slicing)
- **Feature Engineering**: Node encoding with degree normalization, type vocabulary, and label hashing
- **PyG Conversion**: Converts GraphML PDGs to PyTorch Geometric Data objects with engineered features
- **Sampling Strategies**: Random undersampling and oversampling for class imbalance
- **Walk-Forward Validation**: Temporal splits per repository to respect commit order and avoid data leakage
- **Multiple Architectures**: GCN, GraphSAGE, and GAT models with configurable hyperparameters
- **Comprehensive Logging**: Structured logging and detailed manifests tracking failures and feature schemas
- **Reporting**: Aggregated metrics, confusion matrices, and JSON/CSV exports

## Installation

```bash
# Install dependencies (from project root)
pip install -r requirements.txt

# If PyTorch/PyG installation fails, try:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
# or see https://pytorch-geometric.readthedocs.io for CUDA-specific instructions
```

## Quick Start

### 1. Ingest and Inspect Data

```bash
python -m gnn.cli --step ingest
```

Outputs:
- `output/gnn/manifests/dataset_manifest.json`: Summary of dataset

### 2. Create Walk-Forward Splits

```bash
python -m gnn.cli --step splits
```

Outputs:
- `output/gnn/splits/walk_forward/<repository>/fold_NNN_train.csv`
- `output/gnn/splits/walk_forward/<repository>/fold_NNN_test.csv`

### 3. Full Pipeline (All Steps)

```bash
python gnn/orchestrate.py --step all --max-repos 10 --balance none --seed 42 --device cpu
```

Or for production (all 227K+ rows):

```bash
python gnn/orchestrate.py --step all --balance undersample
```

## Pipeline Steps in Detail

### A. Ingestion (`gnn/ingest.py`)

Normalizes and indexes the Ansible CSV:
- Detects label column (`failure_prone` or similar)
- Sorts by repository, commit timestamp, commit hash, filepath
- Adds `row_id` for tracking
- Outputs: normalized DataFrame with columns `[row_id, repository, commit, committed_at, filepath, label]`

### B. Walk-Forward Splits (`gnn/splits.py`)

Creates temporal splits per repository:
- For each repo, sorts by `committed_at` and creates folds
- `fold_001_train.csv`: all rows up to index t-1
- `fold_001_test.csv`: row at index t
- Prevents temporal leakage and respects domain structure

### C. PDG Extraction & Slicing (`gnn/extract.py`, `gnn/slice.py`)

Orchestrates existing extraction:
- `extract_for_commit()`: Checkout commit, run scansible PDG extraction, copy outputs
- `slice_repository_file_level()`: Use existing `dictionary_file_tasknode` to slice file-level PDGs
- Outputs to `output/gnn/raw_pdg/` and `output/gnn/file_level/`

### D. Feature Engineering (`gnn/convert.py`)

Converts GraphML → PyG with node features:
- **Degree normalization** (1 dim)
- **Node type one-hot** (vocabulary size dims)
- **Label hashing** (16 dims by default)
- Saves feature schema to `output/gnn/manifests/feature_schema.json`

### E. Dataset Preparation

- Loads PyG Data from `.pt` files
- Applies class balancing (none / undersample / oversample) to training folds only
- Never applies balancing to test set

### F. Walk-Forward Training & Evaluation (`gnn/pipeline.py`)

For each fold and model:
1. Load train/test datasets
2. Train model with early stopping (default: 20 epochs, stop after 5 epochs without improvement)
3. Evaluate on test set
4. Save metrics: precision, recall, F1, MCC, AUC-PR, confusion matrix
5. Export to `output/gnn/models/<model>/<repo>/<fold>/metrics.json`

### G. Reporting (`gnn/reporting.py`)

Aggregates results:
- `output/gnn/reports/final_report.json`: Comprehensive summary
- Per-model, per-repository, and overall metrics

## Configuration

Edit `gnn/config.py` to customize:

```python
@dataclass
class Config:
    seed: int = 42
    data_root: str = "input/repositories"
    ansible_csv: str = "input/ansible.csv"
    output_root: str = "output/gnn"
    torch_device: Optional[str] = None  # "cuda:0" for GPU
    balance_strategy: str = "none"  # "undersample", "oversample"
    max_retries: int = 1
    node_label_hash_dim: int = 16
```

## Output Structure

```
output/gnn/
├── raw_pdg/
│   └── <repository>/<commit>/
│       ├── repository_level/pdg.dot
│       └── task_level/task_*.graphml
├── file_level/
│   └── <repository>/<commit>/<filepath_sanitized>.graphml
├── manifests/
│   ├── dataset_manifest.json
│   ├── feature_schema.json
│   ├── failed_repositories.json
│   ├── failed_commits.json
│   └── failed_files.json
├── splits/
│   └── walk_forward/<repository>/
│       ├── fold_001_train.csv
│       ├── fold_001_test.csv
│       └── ...
├── models/
│   ├── GCN/<repo>/<fold>/
│   │   ├── best.pt
│   │   └── metrics.json
│   ├── GraphSAGE/<repo>/<fold>/...
│   └── GAT/<repo>/<fold>/...
├── reports/
│   ├── final_report.json
│   └── pipeline.log
└── pipeline.log
```

## Example: Running Tests

```bash
# Test on 2 repos (quick validation)
python test_pipeline_full.py

# Or via orchestration:
python gnn/orchestrate.py --step all --max-repos 2 --device cpu
```

## Metrics Explained

- **Precision**: TP / (TP + FP) — fraction of predicted defects that are true
- **Recall**: TP / (TP + FN) — fraction of actual defects that are detected
- **F1**: 2 * (P * R) / (P + R) — harmonic mean of precision and recall
- **MCC**: Matthews Correlation Coefficient — correlation between actual and predicted
- **AUC-PR**: Area under Precision-Recall curve — robust for imbalanced data

## Extending the Pipeline

### Add a New GNN Model

Edit `gnn/models/gnn_models.py` and add your class:

```python
class MyGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=1):
        super().__init__()
        # ... your layers
    
    def forward(self, x, edge_index, batch=None):
        # ... your forward pass
        return out
```

Then add to `gnn/pipeline.py` in the model class dict:

```python
model_cls = {'GCN': GCN, 'GraphSAGE': GraphSAGE, 'GAT': GAT, 'MyGNN': MyGNN}.get(mname)
```

### Use Different Sampler or Augmentation

Edit `gnn/sampling.py` or `gnn/convert.py` to add new strategies.

## Troubleshooting

**Memory issues?** Reduce batch size or use CPU:
```bash
python gnn/orchestrate.py --device cpu --max-repos 5
```

**Slow extraction?** Extraction reuses existing modular code; ensure your system has scansible and graphviz installed.

**Missing PyG data?** Ensure `.graphml` files exist in `output/gnn/file_level/` before training; conversion to `.pt` is automatic.

## References

- PyTorch Geometric: https://pytorch-geometric.readthedocs.io
- Walk-Forward Validation: Bergmeir & Benítez (2012), "On the use of cross-validation for time series"
- PDG-based Software Engineering: https://en.wikipedia.org/wiki/Program_dependence_graph

---

**Author**: Senior ML/Software Engineer  
**Version**: 0.1.0  
**Last Updated**: 2026-05-19

