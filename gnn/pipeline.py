import os
import logging
import json
from typing import List, Dict

logger = logging.getLogger(__name__)

from .paths import file_level_path, models_path, manifests_path
from .config import DEFAULT_CONFIG
from .convert import graphml_to_pyg
from .sampling import undersample, oversample
from .training import train_model
from .evaluation import evaluate_predictions

try:
    import torch
except Exception:
    torch = None


def _load_dataset_from_manifest(manifest_rows: List[Dict], output_root: str) -> List:
    data_list = []
    for row in manifest_rows:
        repo = row.get("repository")
        commit = row.get("commit")
        filename = row.get("filename")
        if not filename:
            continue
        pt_path = os.path.join(file_level_path(output_root, repo, commit), filename.replace('.graphml', '.pt'))
        if os.path.exists(pt_path):
            try:
                d = torch.load(pt_path)
                data_list.append(d)
            except Exception:
                logger.exception("Failed to load %s", pt_path)
    return data_list


def run_walk_forward_experiment(output_root: str, models: List[str] = None):
    if models is None:
        models = ["GCN", "GraphSAGE", "GAT"]

    splits_dir = os.path.join(output_root, "splits", "walk_forward")
    if not os.path.isdir(splits_dir):
        logger.error("Splits dir not found: %s", splits_dir)
        return

    model_base = models_path(output_root)
    os.makedirs(model_base, exist_ok=True)

    # iterate repositories
    for repo in os.listdir(splits_dir):
        repo_dir = os.path.join(splits_dir, repo)
        if not os.path.isdir(repo_dir):
            continue
        for fname in os.listdir(repo_dir):
            if not fname.endswith("_train.csv"):
                continue
            fold = fname.replace("_train.csv", "")
            train_csv = os.path.join(repo_dir, f"{fold}_train.csv")
            test_csv = os.path.join(repo_dir, f"{fold}_test.csv")
            import pandas as pd

            train_rows = pd.read_csv(train_csv).to_dict(orient="records")
            test_rows = pd.read_csv(test_csv).to_dict(orient="records")

            # ensure conversion of graphs to .pt exists; if not, attempt convert
            for row in train_rows + test_rows:
                repo_n = row.get("repository")
                commit = row.get("commit")
                filepath = row.get("filepath")
                # expected graphml filename heuristic
                safe = filepath.replace('/', '_').replace('\\', '_')
                graphml = os.path.join(file_level_path(output_root, repo_n, commit), f"{safe}_0.graphml")
                ptfile = graphml.replace('.graphml', '.pt')
                if os.path.exists(graphml) and not os.path.exists(ptfile):
                    graphml_to_pyg(graphml, ptfile)

            # load datasets
            train_data = _load_dataset_from_manifest(train_rows, output_root)
            test_data = _load_dataset_from_manifest(test_rows, output_root)

            # labels extract
            y_train = [int(getattr(d, 'y', torch.tensor([0])).item()) for d in train_data]

            # apply balancing if requested
            if DEFAULT_CONFIG.balance_strategy == 'undersample':
                train_data, y_train = undersample(train_data, y_train, seed=DEFAULT_CONFIG.seed)
            elif DEFAULT_CONFIG.balance_strategy == 'oversample':
                train_data, y_train = oversample(train_data, y_train, seed=DEFAULT_CONFIG.seed)

            # train each model
            for mname in models:
                mdir = os.path.join(model_base, mname, repo, fold)
                os.makedirs(mdir, exist_ok=True)
                cfg = {"device": DEFAULT_CONFIG.torch_device or "cpu", "epochs": 20, "lr": 1e-3, "early_stopping": 5}

                # create model instance dynamically
                from .models.gnn_models import GCN, GraphSAGE, GAT

                model_cls = {'GCN': GCN, 'GraphSAGE': GraphSAGE, 'GAT': GAT}.get(mname)
                if model_cls is None:
                    continue
                # infer in_dim from first train sample
                if not train_data:
                    logger.info("No train data for %s %s", repo, fold)
                    continue
                in_dim = train_data[0].x.size(1)
                model = model_cls(in_dim=in_dim)

                best_path = train_model(model, train_data, cfg, mdir)

                # evaluate
                if best_path and os.path.exists(best_path):
                    model.load_state_dict(torch.load(best_path, map_location=cfg['device']))
                    model.eval()
                    y_true = []
                    y_scores = []
                    for d in test_data:
                        xb = d.x.to(cfg['device'])
                        out = model(xb, d.edge_index.to(cfg['device']), getattr(d, 'batch', None))
                        import torch.nn.functional as F

                        score = F.sigmoid(out).detach().cpu().numpy()
                        y_scores.append(float(score))
                        y_true.append(int(getattr(d, 'y', torch.tensor([0])).item()))

                    metrics = evaluate_predictions(y_true, y_scores)
                    mpath = os.path.join(mdir, "metrics.json")
                    with open(mpath, "w", encoding="utf-8") as f:
                        json.dump(metrics, f, indent=2)
