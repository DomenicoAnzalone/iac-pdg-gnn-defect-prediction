#!/usr/bin/env python3
"""
End-to-end orchestration: ingest, extract, slice, convert, split, train, evaluate.
"""
import os
import sys
import logging
from typing import Optional
import json

from gnn.config import Config, DEFAULT_CONFIG
from gnn.logging_utils import setup_logging
from gnn.paths import manifests_path
from gnn.ingest import normalize_and_index, save_dataset_manifest
from gnn.splits import walk_forward_splits
from gnn.pipeline import run_walk_forward_experiment
import pandas as pd


def orchestrate(
    config: Optional[Config] = None,
    step: str = "all",
    max_repos: Optional[int] = None,
) -> None:
    """Run pipeline steps.
    
    Args:
        config: Config object (default: DEFAULT_CONFIG)
        step: "ingest", "splits", "train", "all"
        max_repos: Limit number of repositories for testing
    """
    cfg = config or DEFAULT_CONFIG
    
    # Setup logging
    log_path = os.path.join(cfg.output_root, "pipeline.log")
    setup_logging(log_path)
    logger = logging.getLogger(__name__)
    
    logger.info("=== GNN Pipeline Orchestration ===")
    logger.info("Config: seed=%d, balance=%s, device=%s", cfg.seed, cfg.balance_strategy, cfg.torch_device)
    
    if step in ("ingest", "all"):
        logger.info("Step 1: Ingest and index CSV")
        df = normalize_and_index(cfg.ansible_csv)
        logger.info("Ingested %d rows, label distribution: %s", len(df), df["label"].value_counts().to_dict())
        
        # save manifest
        save_dataset_manifest(df, cfg.output_root)
        
        # optionally limit to N repos for testing
        if max_repos:
            repos = df["repository"].unique()[:max_repos]
            df = df[df["repository"].isin(repos)]
            logger.info("Limited to %d repos, %d rows", len(repos), len(df))
    else:
        logger.info("Skipping ingest; loading from CSV")
        df = normalize_and_index(cfg.ansible_csv)
    
    if step in ("splits", "all"):
        logger.info("Step 2: Create walk-forward splits")
        walk_forward_splits(df, cfg.output_root)
        splits_dir = os.path.join(cfg.output_root, "splits", "walk_forward")
        n_repos = len([d for d in os.listdir(splits_dir) if os.path.isdir(os.path.join(splits_dir, d))])
        logger.info("Created splits for %d repositories", n_repos)
    
    if step in ("train", "all"):
        logger.info("Step 3: Run walk-forward training")
        models = ["GCN", "GraphSAGE", "GAT"]
        try:
            run_walk_forward_experiment(cfg.output_root, models=models)
            logger.info("Training completed for models: %s", ", ".join(models))
        except Exception as e:
            logger.exception("Training failed: %s", e)
    
    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GNN pipeline orchestration")
    parser.add_argument("--step", choices=["ingest", "splits", "train", "all"], default="all")
    parser.add_argument("--max-repos", type=int, default=None, help="Limit to N repositories for testing")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--balance", choices=["none", "undersample", "oversample"], default=DEFAULT_CONFIG.balance_strategy)
    parser.add_argument("--device", default=DEFAULT_CONFIG.torch_device, help="torch device (cuda:0, cpu, etc)")
    
    args = parser.parse_args()
    
    cfg = Config(
        seed=args.seed,
        balance_strategy=args.balance,
        torch_device=args.device,
    )
    
    orchestrate(config=cfg, step=args.step, max_repos=args.max_repos)
