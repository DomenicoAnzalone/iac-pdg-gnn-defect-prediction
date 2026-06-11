from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from experiments.common.config import DEFAULT_DATASET


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E1 and E2 tabular experiments sequentially with a shared setup.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--e1-config", default="experiments/configs/e1_default.yaml")
    parser.add_argument("--e2-config", default="experiments/configs/e2_default.yaml")
    parser.add_argument("--run-prefix", default="tabular_rf_common")
    parser.add_argument("--models", default="random_forest")
    parser.add_argument("--pdg-metrics", default="all")
    parser.add_argument("--balance", default="random_oversampling", choices=["none", "random_undersampling", "random_oversampling"])
    parser.add_argument("--scaler", default="standard", choices=["none", "min-max", "standard"])
    parser.add_argument("--e1-feature-selection", default="none", choices=["none", "variance_threshold", "rfe", "rfecv"])
    parser.add_argument("--e2-feature-selection", default="rfecv", choices=["none", "variance_threshold", "rfe", "rfecv"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", default="experiments/results/exploratory")
    parser.add_argument("--summary-dir", default="experiments/results/exploratory/tabular_e1_e2_common")
    parser.add_argument("--max-repositories", type=int)
    parser.add_argument("--max-splits", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        {
            "experiment": "e1",
            "module": "experiments.e1_tabular_baseline.run",
            "config": args.e1_config,
            "run_name": f"{args.run_prefix}_e1",
            "feature_selection": args.e1_feature_selection,
        },
        {
            "experiment": "e2",
            "module": "experiments.e2_tabular_pdg.run",
            "config": args.e2_config,
            "run_name": f"{args.run_prefix}_e2",
            "feature_selection": args.e2_feature_selection,
        },
    ]

    rows: List[Dict[str, object]] = []
    for run in runs:
        run_dir = Path(args.results_root) / str(run["run_name"])
        if args.skip_existing and (run_dir / "metrics" / "pooled_metrics.csv").exists():
            status = "completed_existing"
            return_code = 0
        else:
            command = _build_command(args, run)
            print(f"\n=== Running {run['experiment'].upper()} ({args.models}) ===", flush=True)
            print(" ".join(command), flush=True)
            completed = subprocess.run(command, cwd=Path.cwd())
            return_code = int(completed.returncode)
            status = "completed" if return_code == 0 else "failed"
            if return_code != 0:
                rows.append(_run_row(run, status, return_code, run_dir))
                break
        rows.append(_run_row(run, status, return_code, run_dir))

    summary = pd.DataFrame(rows)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_dir / "tabular_e1_e2_summary.csv", index=False)
    (summary_dir / "tabular_e1_e2_summary.md").write_text(_render_markdown(summary), encoding="utf-8")
    print(f"\nSummary written to {summary_dir / 'tabular_e1_e2_summary.md'}", flush=True)


def _build_command(args: argparse.Namespace, run: Dict[str, object]) -> List[str]:
    command = [
        sys.executable,
        "-m",
        str(run["module"]),
        "--config",
        str(run["config"]),
        "--dataset",
        args.dataset,
        "--run-name",
        str(run["run_name"]),
        "--models",
        args.models,
        "--balance",
        args.balance,
        "--scaler",
        args.scaler,
        "--feature-selection",
        str(run["feature_selection"]),
        "--seed",
        str(args.seed),
    ]
    if run["experiment"] == "e2":
        command.extend(["--pdg-metrics", args.pdg_metrics])
    if args.max_repositories is not None:
        command.extend(["--max-repositories", str(args.max_repositories)])
    if args.max_splits is not None:
        command.extend(["--max-splits", str(args.max_splits)])
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.dry_run:
        command.append("--dry-run")
    return command


def _run_row(run: Dict[str, object], status: str, return_code: int, run_dir: Path) -> Dict[str, object]:
    row: Dict[str, object] = {
        "experiment": run["experiment"],
        "run_name": run["run_name"],
        "feature_selection": run["feature_selection"],
        "status": status,
        "return_code": return_code,
        "run_dir": str(run_dir),
    }
    pooled_path = run_dir / "metrics" / "pooled_metrics.csv"
    split_path = run_dir / "metrics" / "per_split_metrics.csv"
    feature_manifest_path = run_dir / "feature_manifest.csv"
    if pooled_path.exists():
        pooled = _read_csv_or_empty(pooled_path)
        if not pooled.empty:
            for key, value in pooled.iloc[0].to_dict().items():
                if key not in {"experiment"}:
                    row[key] = value
    if split_path.exists():
        per_split = _read_csv_or_empty(split_path)
        row["valid_metric_splits"] = len(per_split)
    if feature_manifest_path.exists():
        feature_manifest = _read_csv_or_empty(feature_manifest_path)
        if not feature_manifest.empty and "status" in feature_manifest.columns:
            used = feature_manifest[feature_manifest["status"].eq("used")]
            removed = feature_manifest[feature_manifest["status"].eq("removed")]
            row["feature_rows_used"] = len(used)
            row["feature_rows_removed"] = len(removed)
            if "feature" in used.columns:
                row["unique_features_used"] = used["feature"].nunique()
    return row


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _render_markdown(summary: pd.DataFrame) -> str:
    ordered_columns = [
        "experiment",
        "model",
        "feature_selection",
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
        "pooled_tn",
        "pooled_fp",
        "pooled_fn",
        "pooled_tp",
        "unique_features_used",
        "run_name",
    ]
    display = summary[[column for column in ordered_columns if column in summary.columns]].copy()
    try:
        table = display.to_markdown(index=False)
    except Exception:
        table = "```\n" + display.to_csv(index=False) + "\n```"
    delta = _pooled_delta_table(summary)
    return "\n".join([
        "# E1/E2 Tabular Common-Setup Test",
        "",
        "Ogni riga corrisponde a una run tabellare eseguita con la configurazione comune fissata nella fase esplorativa.",
        "E1 usa solo feature tabellari non-PDG. E2 usa le stesse feature più le 11 metriche PDG candidate e, di default, RFECV train-only.",
        "",
        table,
        "",
        delta,
        "",
        "Criterio di lettura consigliato: confrontare E1 ed E2 soprattutto su MCC pooled e AUC-PR pooled. E2 è utile se migliora in modo coerente senza aumentare eccessivamente i falsi positivi.",
        "",
    ])


def _pooled_delta_table(summary: pd.DataFrame) -> str:
    if "experiment" not in summary.columns:
        return ""
    e1 = summary[summary["experiment"].astype(str).eq("e1")]
    e2 = summary[summary["experiment"].astype(str).eq("e2")]
    if e1.empty or e2.empty:
        return ""
    left = e1.iloc[0]
    right = e2.iloc[0]
    metrics = [
        "pooled_mcc",
        "pooled_auc_pr",
        "pooled_auc_roc",
        "pooled_precision",
        "pooled_recall",
        "pooled_f1",
        "pooled_accuracy",
        "pooled_fp",
        "pooled_fn",
    ]
    rows = []
    for metric in metrics:
        if metric not in summary.columns:
            continue
        e1_value = pd.to_numeric(pd.Series([left.get(metric)]), errors="coerce").iloc[0]
        e2_value = pd.to_numeric(pd.Series([right.get(metric)]), errors="coerce").iloc[0]
        if pd.isna(e1_value) or pd.isna(e2_value):
            continue
        rows.append({"metric": metric, "e1": e1_value, "e2": e2_value, "delta_e2_minus_e1": e2_value - e1_value})
    if not rows:
        return ""
    delta = pd.DataFrame(rows)
    try:
        table = delta.to_markdown(index=False)
    except Exception:
        table = "```\n" + delta.to_csv(index=False) + "\n```"
    return "\n".join(["## E2 - E1 pooled delta", "", table])


if __name__ == "__main__":
    main()
