from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_DATASET = (
    "datasets/ansible-pdg-defect-dataset/final/v2026-06-06/"
    "ansible-pdg-defect-dataset_v2026-06-06_final.csv"
)


BASE_CONFIG: Dict[str, Any] = {
    "dataset": DEFAULT_DATASET,
    "results_root": "experiments/results",
    "label_column": "failure_prone",
    "time_column": "committed_at",
    "id_columns": ["repository", "commit", "filepath"],
    "graph_path_column": "graphml_local_path",
    "seed": 42,
    "validation_ratio": 0.2,
    "balance_strategy": "none",
    "scaler": "standard",
    "feature_selection": "none",
    "rfecv_cv": 3,
    "rfecv_step": 0.1,
    "rfecv_min_features_to_select": 1,
    "rfe_n_features_to_select": None,
    "remove_constant_features": True,
    "model_selection_scoring": "mcc",
    "hyperparameter_search": False,
    "random_search_iter": 10,
    "n_jobs": -1,
    "min_nodes": 3,
    "min_edges": 2,
    "max_repositories": None,
    "max_splits": None,
    "max_samples": None,
    "dry_run": False,
    "quiet": False,
    "progress": True,
    "log_level": "INFO",
    "log_every_epochs": 1,
    "compact_progress": False,
}


def load_config(path: Optional[str | Path]) -> Dict[str, Any]:
    if not path:
        return deepcopy(BASE_CONFIG)
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
    except Exception:
        loaded = json.loads(text or "{}")
    config = deepcopy(BASE_CONFIG)
    deep_update(config, loaded)
    return config


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_list(value: Optional[str | Iterable[str]], default: Optional[List[str]] = None) -> List[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        if not value:
            return list(default or [])
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config")
    parser.add_argument("--dataset")
    parser.add_argument("--results-root")
    parser.add_argument("--run-name", default="smoke_run")
    parser.add_argument("--balance", choices=["none", "random_undersampling", "random_oversampling"])
    parser.add_argument("--scaler", choices=["none", "min-max", "standard"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-repositories", type=int)
    parser.add_argument("--max-splits", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--log-every-epochs", type=int)
    parser.add_argument("--compact-progress", action="store_true")
    parser.add_argument("--feature-selection", choices=["none", "variance_threshold", "rfe", "rfecv"])


def apply_common_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "dataset": "dataset",
        "results_root": "results_root",
        "balance_strategy": "balance",
        "scaler": "scaler",
        "seed": "seed",
        "max_repositories": "max_repositories",
        "max_splits": "max_splits",
        "max_samples": "max_samples",
        "log_level": "log_level",
        "log_every_epochs": "log_every_epochs",
        "feature_selection": "feature_selection",
    }
    for config_key, arg_key in mapping.items():
        value = getattr(args, arg_key, None)
        if value is not None:
            config[config_key] = value
    if getattr(args, "dry_run", False):
        config["dry_run"] = True
    if getattr(args, "quiet", False):
        config["quiet"] = True
    if getattr(args, "no_progress", False):
        config["progress"] = False
    if getattr(args, "compact_progress", False):
        config["compact_progress"] = True
        config["progress"] = True
    return config
