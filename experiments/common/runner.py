from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .data_loading import filter_common_valid_samples, load_dataset
from .progress import get_logger, setup_logging
from .reporting import prepare_run_dir, save_common_manifests, save_run_metadata
from .splitting import assert_no_overlap, create_walk_forward_splits, split_manifest_rows


def prepare_common_run(config: Dict[str, Any], run_name: str) -> Tuple[Path, pd.DataFrame, list, pd.DataFrame]:
    config["run_name"] = run_name
    run_dir = prepare_run_dir(config.get("results_root", "experiments/results"), run_name)
    setup_logging(run_dir, level=str(config.get("log_level", "INFO")), quiet=bool(config.get("quiet", False)))
    logger = get_logger("experiments.common.runner")
    logger.info("Run avviata: %s", run_name)
    logger.info("Dataset: %s", config["dataset"])
    logger.info("Soglie grafo: min_nodes=%s min_edges=%s", config.get("min_nodes", 3), config.get("min_edges", 2))
    df = load_dataset(
        config["dataset"],
        max_repositories=config.get("max_repositories"),
        max_samples=config.get("max_samples"),
    )
    logger.info("Dataset caricato: %s righe, %s repository", len(df), df["repository"].nunique())
    df, excluded = filter_common_valid_samples(
        df,
        min_nodes=int(config.get("min_nodes", 3)),
        min_edges=int(config.get("min_edges", 2)),
        graph_path_column=config.get("graph_path_column", "graphml_local_path"),
    )
    logger.info("Campioni comuni mantenuti: %s; esclusi: %s", len(df), len(excluded))
    splits, skipped = create_walk_forward_splits(
        df,
        validation_ratio=float(config.get("validation_ratio", 0.2)),
        max_splits=config.get("max_splits"),
    )
    for split in splits:
        assert_no_overlap(split)
    logger.info("Split walk-forward validi: %s; split saltati: %s", len(splits), len(skipped))
    manifest = split_manifest_rows(splits, df) if splits else pd.DataFrame()
    save_run_metadata(run_dir, config, Path.cwd())
    save_common_manifests(run_dir, manifest, excluded, skipped)
    logger.info("Manifest salvati in: %s", run_dir)
    return run_dir, df, splits, excluded
