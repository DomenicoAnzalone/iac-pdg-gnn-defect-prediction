from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from experiments.common.config import DEFAULT_DATASET
from experiments.common.data_loading import load_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E3 sequentially over multiple minimum graph-size thresholds.")
    parser.add_argument("--config", default="experiments/configs/e3_default.yaml")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--thresholds", default="3:2,5:4,8:6,10:6")
    parser.add_argument("--run-prefix", default="e3_graphsage_threshold")
    parser.add_argument("--model", default="graphsage")
    parser.add_argument("--balance", default="random_oversampling", choices=["none", "random_undersampling", "random_oversampling"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-epochs", type=int, default=5)
    parser.add_argument("--results-root", default="experiments/results")
    parser.add_argument("--summary-dir", default="experiments/results/small_graph_threshold_sweep")
    parser.add_argument("--max-repositories", type=int)
    parser.add_argument("--max-splits", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compact-progress", action="store_true", default=True)
    parser.add_argument("--no-compact-progress", dest="compact_progress", action="store_false")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    thresholds = [_parse_threshold(item) for item in args.thresholds.split(",") if item.strip()]
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    dataset_summary = _threshold_dataset_summary(args.dataset, thresholds)
    dataset_summary.to_csv(summary_dir / "threshold_dataset_summary.csv", index=False)

    run_rows: List[Dict[str, object]] = []
    for min_nodes, min_edges in thresholds:
        run_name = f"{args.run_prefix}_n{min_nodes}_e{min_edges}"
        run_dir = Path(args.results_root) / run_name
        if args.skip_existing and (run_dir / "metrics" / "pooled_metrics.csv").exists():
            status = "skipped_existing"
            return_code = 0
        else:
            command = _build_e3_command(args, run_name, min_nodes, min_edges)
            print(f"\n=== Running {run_name} ({min_nodes} nodes / {min_edges} edges) ===", flush=True)
            print(" ".join(command), flush=True)
            completed = subprocess.run(command, cwd=Path.cwd())
            return_code = int(completed.returncode)
            status = "completed" if return_code == 0 else "failed"
            if return_code != 0:
                run_rows.append(_run_row(run_name, min_nodes, min_edges, status, return_code, run_dir))
                break
        run_rows.append(_run_row(run_name, min_nodes, min_edges, status, return_code, run_dir))

    run_summary = pd.DataFrame(run_rows)
    merged = dataset_summary.merge(run_summary, on=["min_nodes", "min_edges"], how="left")
    summary_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(summary_dir / "threshold_run_summary.csv", index=False)
    (summary_dir / "threshold_run_summary.md").write_text(_render_markdown(merged), encoding="utf-8")
    print(f"\nSummary written to {summary_dir / 'threshold_run_summary.md'}", flush=True)


def _build_e3_command(args: argparse.Namespace, run_name: str, min_nodes: int, min_edges: int) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.e3_gnn.run",
        "--config",
        args.config,
        "--dataset",
        args.dataset,
        "--run-name",
        run_name,
        "--model",
        args.model,
        "--min-nodes",
        str(min_nodes),
        "--min-edges",
        str(min_edges),
        "--balance",
        args.balance,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--log-every-epochs",
        str(args.log_every_epochs),
    ]
    if args.compact_progress:
        command.append("--compact-progress")
    if args.max_repositories is not None:
        command.extend(["--max-repositories", str(args.max_repositories)])
    if args.max_splits is not None:
        command.extend(["--max-splits", str(args.max_splits)])
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.dry_run:
        command.append("--dry-run")
    return command


def _threshold_dataset_summary(dataset: str, thresholds: List[tuple[int, int]]) -> pd.DataFrame:
    df = load_dataset(dataset)
    nodes = pd.to_numeric(df["nodes"], errors="coerce")
    edges = pd.to_numeric(df["edges"], errors="coerce")
    rows = []
    for min_nodes, min_edges in thresholds:
        kept = df[(nodes >= min_nodes) & (edges >= min_edges)]
        removed = df[~df["_sample_id"].isin(kept["_sample_id"])]
        rows.append({
            "min_nodes": min_nodes,
            "min_edges": min_edges,
            "kept_samples": len(kept),
            "removed_samples": len(removed),
            "kept_positive_rate": kept["failure_prone"].mean() if len(kept) else None,
            "removed_positive_rate": removed["failure_prone"].mean() if len(removed) else None,
            "kept_repositories": kept["repository"].nunique(),
            "removed_repositories": removed["repository"].nunique(),
        })
    return pd.DataFrame(rows)


def _run_row(run_name: str, min_nodes: int, min_edges: int, status: str, return_code: int, run_dir: Path) -> Dict[str, object]:
    row: Dict[str, object] = {
        "min_nodes": min_nodes,
        "min_edges": min_edges,
        "run_name": run_name,
        "status": status,
        "return_code": return_code,
        "run_dir": str(run_dir),
    }
    pooled_path = run_dir / "metrics" / "pooled_metrics.csv"
    skipped_path = run_dir / "logs" / "skipped_splits.csv"
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
    if skipped_path.exists():
        skipped = _read_csv_or_empty(skipped_path)
        row["skipped_splits"] = len(skipped)
    return row


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _parse_threshold(value: str) -> tuple[int, int]:
    left, right = value.split(":")
    return int(left), int(right)


def _render_markdown(summary: pd.DataFrame) -> str:
    try:
        table = summary.to_markdown(index=False)
    except Exception:
        table = "```\n" + summary.to_csv(index=False) + "\n```"
    return "\n".join([
        "# Small-Graph Threshold Sweep",
        "",
        "Each row corresponds to one E3 run with a different minimum graph-size threshold.",
        "Final model quality should be read primarily from the `pooled_*` columns.",
        "",
        table,
        "",
    ])


if __name__ == "__main__":
    main()
