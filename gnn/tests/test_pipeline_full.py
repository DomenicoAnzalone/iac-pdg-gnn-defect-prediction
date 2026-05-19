#!/usr/bin/env python3
"""
Full pipeline test: ingest → splits → smoke train
"""
import os
import sys
import json
import logging

from gnn.config import Config
from gnn.orchestrate import orchestrate

if __name__ == "__main__":
    # custom config for test run
    cfg = Config(
        seed=42,
        balance_strategy="none",
        torch_device="cpu",
    )
    
    print("\n" + "="*60)
    print("FULL PIPELINE TEST (small subset)")
    print("="*60 + "\n")
    
    # Run orchestration on small subset
    orchestrate(config=cfg, step="all", max_repos=2)  # only 2 repos for quick test
    
    # Check outputs
    print("\n" + "="*60)
    print("OUTPUT VERIFICATION")
    print("="*60 + "\n")
    
    manifest_dir = os.path.join(cfg.output_root, "manifests")
    if os.path.isdir(manifest_dir):
        print(f"OK - Manifests directory: {manifest_dir}")
        for fname in os.listdir(manifest_dir):
            fpath = os.path.join(manifest_dir, fname)
            if fname.endswith(".json"):
                try:
                    with open(fpath, "r") as f:
                        data = json.load(f)
                    print(f"  OK {fname} ({len(data)} items)" if isinstance(data, list) else f"  OK {fname}")
                except Exception as e:
                    print(f"  FAIL {fname}: {e}")
    else:
        print(f"FAIL - Manifests directory not found: {manifest_dir}")
    
    splits_dir = os.path.join(cfg.output_root, "splits", "walk_forward")
    if os.path.isdir(splits_dir):
        print(f"\nOK - Splits directory: {splits_dir}")
        for repo in os.listdir(splits_dir):
            repo_dir = os.path.join(splits_dir, repo)
            if os.path.isdir(repo_dir):
                n_folds = len([f for f in os.listdir(repo_dir) if f.endswith(".csv")])
                print(f"  OK {repo}: {n_folds // 2} folds")
    else:
        print(f"FAIL - Splits directory not found: {splits_dir}")
    
    # Check logs
    log_file = os.path.join(cfg.output_root, "pipeline.log")
    if os.path.exists(log_file):
        print(f"\nOK - Log file: {log_file}")
        with open(log_file, "r") as f:
            lines = f.readlines()
        print(f"  ({len(lines)} lines)")
        print("\n  Last 10 lines:")
        for line in lines[-10:]:
            print(f"    {line.rstrip()}")
    
    print("\n" + "="*60)
    print("OK - FULL PIPELINE TEST COMPLETE")
    print("="*60 + "\n")
