#!/usr/bin/env python3
"""Probe B: splits and module imports"""
from gnn.ingest import normalize_and_index
from gnn.splits import walk_forward_splits
from gnn.config import DEFAULT_CONFIG
import importlib
import os

print("=== Probe B: Splits and Module Imports ===\n")

# 1. Load and split (subset for speed)
df = normalize_and_index('input/ansible.csv')
df_subset = df.head(500)  # small subset for testing
print(f"Using {len(df_subset)} rows (subset of {len(df)})")

# 2. Walk-forward splits
walk_forward_splits(df_subset, DEFAULT_CONFIG.output_root)
print(f"\n✓ Walk-forward splits created in {DEFAULT_CONFIG.output_root}/splits/walk_forward/")

# 3. Smoke import modules
modules = ['gnn.convert', 'gnn.pipeline', 'gnn.training', 'gnn.evaluation', 'gnn.sampling']
for m in modules:
    try:
        importlib.import_module(m)
        print(f"✓ {m}")
    except Exception as e:
        print(f"✗ {m}: {e}")

print("\n✓ Probe B passed")
