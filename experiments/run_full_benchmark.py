from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from experiments.common.config import DEFAULT_DATASET


CLASSICAL_MODELS = ["decision_tree", "logistic_regression", "naive_bayes", "random_forest", "svm"]
GNN_MODELS = ["gcn", "graphsage", "gat", "gin", "rgcn"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete thesis benchmark sequentially and resumably.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--benchmark-name", default="final_benchmark")
    parser.add_argument("--results-root", default="experiments/results/benchmark")
    parser.add_argument("--e1-config", default="experiments/configs/e1_default.yaml")
    parser.add_argument("--e2-config", default="experiments/configs/e2_default.yaml")
    parser.add_argument("--e3-config", default="experiments/configs/e3_default.yaml")
    parser.add_argument("--e1-models", default=",".join(CLASSICAL_MODELS))
    parser.add_argument("--e2-models", default=",".join(CLASSICAL_MODELS))
    parser.add_argument("--e3-models", default=",".join(GNN_MODELS))
    parser.add_argument("--pdg-metrics", default="all")
    parser.add_argument("--balance", default="random_oversampling", choices=["none", "random_undersampling", "random_oversampling"])
    parser.add_argument("--scaler", default="standard", choices=["none", "min-max", "standard"])
    parser.add_argument("--e1-feature-selection", default="validation_rfe", choices=["none", "variance_threshold", "rfe", "rfecv", "validation_rfe"])
    parser.add_argument("--e2-feature-selection", default="validation_rfe", choices=["none", "variance_threshold", "rfe", "rfecv", "validation_rfe"])
    parser.add_argument("--min-nodes", type=int, default=3)
    parser.add_argument("--min-edges", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-epochs", type=int, default=5)
    parser.add_argument("--max-repositories", type=int)
    parser.add_argument("--max-splits", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--rerun-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--no-compact-progress", dest="compact_progress", action="store_false", default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    benchmark_root = Path(args.results_root) / args.benchmark_name
    summary_dir = benchmark_root / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plan = _build_plan(args, benchmark_root)
    _write_plan(summary_dir, args, plan)

    rows: List[Dict[str, object]] = []
    for index, step in enumerate(plan, start=1):
        run_dir = Path(step["run_dir"])
        print(f"\n=== [{index}/{len(plan)}] {step['experiment'].upper()} {step['model']} ===", flush=True)
        if args.skip_existing and (run_dir / "metrics" / "pooled_metrics.csv").exists():
            print(f"Skipping completed run: {run_dir}", flush=True)
            rows.append(_run_row(step, status="completed_existing", return_code=0))
            _write_summary(summary_dir, rows)
            continue

        command = _build_command(args, step, benchmark_root)
        print(" ".join(command), flush=True)
        started_at = datetime.now().isoformat(timespec="seconds")
        stderr_path = run_dir / "logs" / "stderr.log"
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        with stderr_path.open("a", encoding="utf-8") as stderr_file:
            completed = subprocess.run(command, cwd=Path.cwd(), env=_child_env(), stderr=stderr_file)
        return_code = int(completed.returncode)
        status = "dry_run" if args.dry_run and return_code == 0 else "completed" if return_code == 0 else "failed"
        row = _run_row(step, status=status, return_code=return_code)
        row["started_at"] = started_at
        row["finished_at"] = datetime.now().isoformat(timespec="seconds")
        rows.append(row)
        _write_summary(summary_dir, rows)
        if return_code != 0:
            print(f"Benchmark stopped after failed run: {step['run_name']}", flush=True)
            print(f"stderr saved to: {stderr_path}", flush=True)
            sys.exit(return_code)

    _write_summary(summary_dir, rows)
    print(f"\nBenchmark summary written to {summary_dir / 'benchmark_summary.md'}", flush=True)


def _build_plan(args: argparse.Namespace, benchmark_root: Path) -> List[Dict[str, object]]:
    plan: List[Dict[str, object]] = []
    for model in _parse_list(args.e1_models):
        plan.append({
            "experiment": "e1",
            "module": "experiments.e1_tabular_baseline.run",
            "config": args.e1_config,
            "model": model,
            "run_name": f"e1_{model}",
            "run_dir": benchmark_root / f"e1_{model}",
            "feature_selection": args.e1_feature_selection,
        })
    for model in _parse_list(args.e2_models):
        plan.append({
            "experiment": "e2",
            "module": "experiments.e2_tabular_pdg.run",
            "config": args.e2_config,
            "model": model,
            "run_name": f"e2_{model}",
            "run_dir": benchmark_root / f"e2_{model}",
            "feature_selection": args.e2_feature_selection,
        })
    for model in _parse_list(args.e3_models):
        plan.append({
            "experiment": "e3",
            "module": "experiments.e3_gnn.run",
            "config": args.e3_config,
            "model": model,
            "run_name": f"e3_{model}",
            "run_dir": benchmark_root / f"e3_{model}",
            "feature_selection": "",
        })
    return plan


def _build_command(args: argparse.Namespace, step: Dict[str, object], benchmark_root: Path) -> List[str]:
    command = [
        sys.executable,
        "-m",
        str(step["module"]),
        "--config",
        str(step["config"]),
        "--dataset",
        args.dataset,
        "--results-root",
        str(benchmark_root),
        "--run-name",
        str(step["run_name"]),
        "--balance",
        args.balance,
        "--seed",
        str(args.seed),
    ]
    if args.compact_progress:
        command.append("--compact-progress")
    experiment = str(step["experiment"])
    if experiment in {"e1", "e2"}:
        command.extend([
            "--models",
            str(step["model"]),
            "--scaler",
            args.scaler,
            "--feature-selection",
            str(step["feature_selection"]),
        ])
        if experiment == "e2":
            command.extend(["--pdg-metrics", args.pdg_metrics])
    else:
        command.extend([
            "--model",
            str(step["model"]),
            "--min-nodes",
            str(args.min_nodes),
            "--min-edges",
            str(args.min_edges),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--log-every-epochs",
            str(args.log_every_epochs),
        ])
    if args.max_repositories is not None:
        command.extend(["--max-repositories", str(args.max_repositories)])
    if args.max_splits is not None:
        command.extend(["--max-splits", str(args.max_splits)])
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.dry_run:
        command.append("--dry-run")
    return command


def _child_env() -> Dict[str, str]:
    env = os.environ.copy()
    warning_filter = "ignore:.*sklearn.utils.parallel.delayed.*:UserWarning"
    existing = env.get("PYTHONWARNINGS", "")
    if warning_filter not in existing:
        env["PYTHONWARNINGS"] = ",".join(item for item in [existing, warning_filter] if item)
    return env


def _run_row(step: Dict[str, object], status: str, return_code: int) -> Dict[str, object]:
    run_dir = Path(step["run_dir"])
    row: Dict[str, object] = {
        "experiment": step["experiment"],
        "model": step["model"],
        "run_name": step["run_name"],
        "status": status,
        "return_code": return_code,
        "run_dir": str(run_dir),
    }
    pooled_path = run_dir / "metrics" / "pooled_metrics.csv"
    split_path = run_dir / "metrics" / "per_split_metrics.csv"
    if pooled_path.exists():
        pooled = _read_csv_or_empty(pooled_path)
        if not pooled.empty:
            for key, value in pooled.iloc[0].to_dict().items():
                if key not in {"experiment", "model"}:
                    row[key] = value
    if split_path.exists():
        per_split = _read_csv_or_empty(split_path)
        row["valid_metric_splits"] = len(per_split)
    return row


def _write_plan(summary_dir: Path, args: argparse.Namespace, plan: List[Dict[str, object]]) -> None:
    rows = []
    for index, step in enumerate(plan, start=1):
        rows.append({
            "order": index,
            "experiment": step["experiment"],
            "model": step["model"],
            "run_name": step["run_name"],
            "run_dir": step["run_dir"],
        })
    pd.DataFrame(rows).to_csv(summary_dir / "benchmark_plan.csv", index=False)
    config_rows = [{
        "dataset": args.dataset,
        "balance": args.balance,
        "scaler": args.scaler,
        "e1_feature_selection": args.e1_feature_selection,
        "e2_feature_selection": args.e2_feature_selection,
        "pdg_metrics": args.pdg_metrics,
        "min_nodes": args.min_nodes,
        "min_edges": args.min_edges,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }]
    pd.DataFrame(config_rows).to_csv(summary_dir / "benchmark_config.csv", index=False)


def _write_summary(summary_dir: Path, rows: List[Dict[str, object]]) -> None:
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_dir / "benchmark_summary.csv", index=False)
    (summary_dir / "benchmark_summary.md").write_text(_render_markdown(summary), encoding="utf-8")


def _render_markdown(summary: pd.DataFrame) -> str:
    if summary.empty:
        table = "_Nessuna run completata._"
    else:
        ordered = [
            "experiment",
            "model",
            "status",
            "valid_metric_splits",
            "pooled_sample_count",
            "pooled_mcc",
            "pooled_auc_pr",
            "pooled_auc_roc",
            "pooled_precision",
            "pooled_recall",
            "pooled_f1",
            "pooled_accuracy",
            "run_name",
        ]
        display = summary[[column for column in ordered if column in summary.columns]].copy()
        try:
            table = display.to_markdown(index=False)
        except Exception:
            table = "```\n" + display.to_csv(index=False) + "\n```"
    return "\n".join([
        "# Full Benchmark Summary",
        "",
        "Questo file viene rigenerato dopo ogni run completata. Se il benchmark viene interrotto, riavviare lo stesso comando: le run con `pooled_metrics.csv` vengono saltate automaticamente.",
        "",
        table,
        "",
    ])


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
