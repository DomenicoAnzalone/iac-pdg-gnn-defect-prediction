#!/usr/bin/env python3
"""
Final smoke test: import all modules and verify they work.
"""
import sys
import importlib

MODULES = [
    'gnn',
    'gnn.config',
    'gnn.paths',
    'gnn.logging_utils',
    'gnn.ingest',
    'gnn.extract',
    'gnn.slice',
    'gnn.convert',
    'gnn.sampling',
    'gnn.splits',
    'gnn.training',
    'gnn.evaluation',
    'gnn.reporting',
    'gnn.pipeline',
    'gnn.cli',
    'gnn.orchestrate',
    'gnn.models',
    'gnn.models.gnn_models',
]

print("\n" + "="*70)
print("FINAL SMOKE TEST: All Module Imports")
print("="*70 + "\n")

failed = []
for mod_name in MODULES:
    try:
        mod = importlib.import_module(mod_name)
        print(f"OK   {mod_name}")
    except Exception as e:
        print(f"FAIL {mod_name}: {e}")
        failed.append(mod_name)

print("\n" + "="*70)
if not failed:
    print("SUCCESS: All modules imported successfully!")
else:
    print(f"FAILED: {len(failed)} modules failed:")
    for mod in failed:
        print(f"  - {mod}")
    sys.exit(1)
print("="*70 + "\n")

# Quick sanity checks
print("Sanity checks:")
from gnn.config import DEFAULT_CONFIG, Config
print(f"OK   Config: seed={DEFAULT_CONFIG.seed}, device={DEFAULT_CONFIG.torch_device}")

from gnn.ingest import normalize_and_index
import os
if os.path.exists('input/ansible.csv'):
    df = normalize_and_index('input/ansible.csv')
    print(f"OK   Ingest: loaded {len(df)} rows")
else:
    print(f"SKIP Ingest: CSV not found (expected in full pipeline)")

print("\n" + "="*70)
print("OK - FINAL SMOKE TEST PASSED")
print("="*70 + "\n")
