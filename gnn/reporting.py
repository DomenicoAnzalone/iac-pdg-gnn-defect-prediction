import os
import json
import csv
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def save_summary(output_root: str, summaries: List[Dict]):
    reports_dir = os.path.join(output_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    csv_path = os.path.join(reports_dir, "summary.csv")
    json_path = os.path.join(reports_dir, "summary.json")
    # write csv
    if summaries:
        keys = sorted(summaries[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for s in summaries:
                writer.writerow(s)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    
    logger.info("Saved summary: %d items to %s and %s", len(summaries), csv_path, json_path)


def aggregate_fold_metrics(output_root: str) -> Dict[str, Any]:
    """Aggregate metrics from all folds and models."""
    models_dir = os.path.join(output_root, "models")
    agg = {}
    
    if not os.path.isdir(models_dir):
        logger.warning("Models directory not found: %s", models_dir)
        return agg
    
    for model_name in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        
        agg[model_name] = {"repos": {}}
        
        for repo_name in os.listdir(model_dir):
            repo_dir = os.path.join(model_dir, repo_name)
            if not os.path.isdir(repo_dir):
                continue
            
            agg[model_name]["repos"][repo_name] = {"folds": []}
            
            for fold_name in os.listdir(repo_dir):
                fold_dir = os.path.join(repo_dir, fold_name)
                if not os.path.isdir(fold_dir):
                    continue
                
                metrics_file = os.path.join(fold_dir, "metrics.json")
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, "r", encoding="utf-8") as f:
                            metrics = json.load(f)
                        agg[model_name]["repos"][repo_name]["folds"].append({
                            "fold": fold_name,
                            "metrics": metrics,
                        })
                    except Exception as e:
                        logger.warning("Failed to load metrics from %s: %s", metrics_file, e)
    
    return agg


def generate_report(output_root: str) -> Dict[str, Any]:
    """Generate comprehensive final report."""
    reports_dir = os.path.join(output_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load manifests
    manifests_dir = os.path.join(output_root, "manifests")
    dataset_manifest = {}
    failed_repos = []
    
    if os.path.isdir(manifests_dir):
        dataset_file = os.path.join(manifests_dir, "dataset_manifest.json")
        if os.path.exists(dataset_file):
            with open(dataset_file, "r", encoding="utf-8") as f:
                dataset_manifest = json.load(f)
        
        failed_repos_file = os.path.join(manifests_dir, "failed_repositories.json")
        if os.path.exists(failed_repos_file):
            with open(failed_repos_file, "r", encoding="utf-8") as f:
                failed_repos = json.load(f)
    
    # Aggregate metrics
    metrics_agg = aggregate_fold_metrics(output_root)
    
    report = {
        "timestamp": str(__import__("datetime").datetime.now()),
        "dataset": dataset_manifest,
        "failed_repositories": len(failed_repos),
        "models": metrics_agg,
    }
    
    # Save report
    report_path = os.path.join(reports_dir, "final_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    logger.info("Saved final report to %s", report_path)
    return report
