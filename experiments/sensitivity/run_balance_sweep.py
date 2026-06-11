from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from experiments.common.config import DEFAULT_DATASET


BALANCE_STRATEGIES = ["none", "random_oversampling", "random_undersampling"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E3 sequentially over multiple training balancing strategies.")
    parser.add_argument("--config", default="experiments/configs/e3_default.yaml")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--strategies", default="none,random_oversampling,random_undersampling")
    parser.add_argument("--run-prefix", default="e3_graphsage_balance")
    parser.add_argument("--model", default="graphsage")
    parser.add_argument("--min-nodes", type=int, default=3)
    parser.add_argument("--min-edges", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-epochs", type=int, default=5)
    parser.add_argument("--results-root", default="experiments/results/exploratory")
    parser.add_argument("--summary-dir", default="experiments/results/exploratory/e3_balance_sweep")
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
    strategies = [_parse_strategy(item) for item in args.strategies.split(",") if item.strip()]
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for strategy in strategies:
        run_name = f"{args.run_prefix}_{strategy}"
        run_dir = Path(args.results_root) / run_name
        if args.skip_existing and (run_dir / "metrics" / "pooled_metrics.csv").exists():
            status = "completed_existing"
            return_code = 0
        else:
            command = _build_e3_command(args, run_name, strategy)
            print(f"\n=== Running {run_name} ({strategy}) ===", flush=True)
            print(" ".join(command), flush=True)
            completed = subprocess.run(command, cwd=Path.cwd())
            return_code = int(completed.returncode)
            status = "completed" if return_code == 0 else "failed"
            if return_code != 0:
                rows.append(_run_row(run_name, strategy, status, return_code, run_dir))
                break
        rows.append(_run_row(run_name, strategy, status, return_code, run_dir))

    summary = pd.DataFrame(rows)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_dir / "balance_run_summary.csv", index=False)
    (summary_dir / "balance_run_summary.md").write_text(_render_markdown(summary), encoding="utf-8")
    print(f"\nSummary written to {summary_dir / 'balance_run_summary.md'}", flush=True)


def _build_e3_command(args: argparse.Namespace, run_name: str, strategy: str) -> List[str]:
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
        str(args.min_nodes),
        "--min-edges",
        str(args.min_edges),
        "--balance",
        strategy,
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


def _run_row(run_name: str, strategy: str, status: str, return_code: int, run_dir: Path) -> Dict[str, object]:
    row: Dict[str, object] = {
        "balance_strategy": strategy,
        "run_name": run_name,
        "status": status,
        "return_code": return_code,
        "run_dir": str(run_dir),
    }
    pooled_path = run_dir / "metrics" / "pooled_metrics.csv"
    skipped_path = run_dir / "logs" / "skipped_splits.csv"
    split_path = run_dir / "metrics" / "per_split_metrics.csv"
    predictions_path = run_dir / "predictions" / "e3_graphsage_predictions.csv"
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
    if predictions_path.exists():
        predictions = _read_csv_or_empty(predictions_path)
        if not predictions.empty:
            row["predicted_positive_rate"] = pd.to_numeric(predictions["y_pred"], errors="coerce").mean()
            row["true_positive_rate"] = pd.to_numeric(predictions["y_true"], errors="coerce").mean()
    return row


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _parse_strategy(value: str) -> str:
    strategy = value.strip().lower()
    if strategy not in BALANCE_STRATEGIES:
        raise ValueError(f"Unsupported balance strategy: {strategy}. Supported values: {', '.join(BALANCE_STRATEGIES)}")
    return strategy


def _render_markdown(summary: pd.DataFrame) -> str:
    ordered_columns = [
        "balance_strategy",
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
        "true_positive_rate",
        "predicted_positive_rate",
        "pooled_tn",
        "pooled_fp",
        "pooled_fn",
        "pooled_tp",
        "run_name",
    ]
    display = summary[[column for column in ordered_columns if column in summary.columns]].copy()
    try:
        table = display.to_markdown(index=False)
    except Exception:
        table = "```\n" + display.to_csv(index=False) + "\n```"
    return "\n".join([
        "# E3 Balance Strategy Sweep",
        "",
        "Ogni riga corrisponde a una run E3 con la stessa configurazione GraphSAGE e una diversa strategia di bilanciamento applicata solo al training set.",
        "I risultati principali vanno letti dalle metriche pooled, calcolate aggregando tutte le predizioni dei test walk-forward.",
        "",
        table,
        "",
        "Criterio di lettura consigliato: usare MCC come metrica guida, AUC-PR come metrica secondaria e controllare precision/recall per evitare strategie che migliorano solo perche predicono troppi positivi.",
        "",
    ])


if __name__ == "__main__":
    main()
