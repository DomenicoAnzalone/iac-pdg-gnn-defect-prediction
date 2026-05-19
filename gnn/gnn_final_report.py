#!/usr/bin/env python3
"""
Generate final report from pipeline outputs.
"""
import os
import sys
from gnn.config import DEFAULT_CONFIG
from gnn.reporting import generate_report

if __name__ == "__main__":
    report = generate_report(DEFAULT_CONFIG.output_root)
    
    print("\n" + "="*70)
    print("FINAL PIPELINE REPORT")
    print("="*70 + "\n")
    
    print(f"Timestamp: {report.get('timestamp', 'N/A')}")
    
    dataset = report.get("dataset", {})
    print(f"\nDataset:")
    print(f"  Total rows: {dataset.get('total_rows', 'N/A')}")
    print(f"  Repositories: {len(dataset.get('repositories', []))}")
    print(f"  Label distribution: {dataset.get('label_counts', {})}")
    
    print(f"\nFailed repositories: {report.get('failed_repositories', 0)}")
    
    models = report.get("models", {})
    if models:
        print(f"\nModels trained: {list(models.keys())}")
        for model_name, model_data in models.items():
            repos = model_data.get("repos", {})
            total_folds = sum(len(r.get("folds", [])) for r in repos.values())
            print(f"  {model_name}: {len(repos)} repos, {total_folds} folds")
    else:
        print("\nNo models trained yet (expected if no PyG data available)")
    
    print("\n" + "="*70)
    print("Report saved to:", os.path.join(DEFAULT_CONFIG.output_root, "reports", "final_report.json"))
    print("="*70 + "\n")
