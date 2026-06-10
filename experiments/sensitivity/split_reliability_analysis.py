from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from experiments.common.evaluation import pooled_metrics_from_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze how small walk-forward test splits affect pooled metrics.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--min-test-sizes", default="2,5,10,20,30,50")
    parser.add_argument("--min-test-positives", default="1,2,3,5")
    parser.add_argument("--min-test-negatives", default="1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "reports" / "split_reliability"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = _load_predictions(results_dir)
    if predictions.empty:
        raise FileNotFoundError(f"No prediction rows found under {results_dir / 'predictions'}")

    split_stats = _split_stats(predictions)
    split_stats.to_csv(output_dir / "split_test_size_distribution.csv", index=False)

    thresholds = [
        (min_size, min_pos, min_neg)
        for min_size in _parse_int_list(args.min_test_sizes)
        for min_pos in _parse_int_list(args.min_test_positives)
        for min_neg in _parse_int_list(args.min_test_negatives)
    ]

    rows: List[Dict[str, object]] = []
    for (experiment, model), group_predictions in predictions.groupby(["experiment", "model"], dropna=False):
        system_split_stats = split_stats[
            split_stats["experiment"].eq(experiment) & split_stats["model"].eq(model)
        ]
        baseline_metrics = pooled_metrics_from_predictions(group_predictions)
        baseline = {
            "all_splits": system_split_stats["split_id"].nunique(),
            "all_predictions": len(group_predictions),
            "all_positive_rate": float(group_predictions["y_true"].mean()) if len(group_predictions) else None,
            **baseline_metrics,
        }
        for min_size, min_pos, min_neg in thresholds:
            eligible = system_split_stats[
                (system_split_stats["test_size"] >= min_size)
                & (system_split_stats["test_positives"] >= min_pos)
                & (system_split_stats["test_negatives"] >= min_neg)
            ]
            kept_split_ids = set(eligible["split_id"])
            kept_predictions = group_predictions[group_predictions["split_id"].isin(kept_split_ids)]
            removed_predictions = group_predictions[~group_predictions["split_id"].isin(kept_split_ids)]
            row: Dict[str, object] = {
                "experiment": experiment,
                "model": model,
                "min_test_size": min_size,
                "min_test_positives": min_pos,
                "min_test_negatives": min_neg,
                "kept_splits": len(kept_split_ids),
                "removed_splits": baseline["all_splits"] - len(kept_split_ids),
                "kept_predictions": len(kept_predictions),
                "removed_predictions": len(removed_predictions),
                "kept_prediction_rate": len(kept_predictions) / baseline["all_predictions"] if baseline["all_predictions"] else 0,
                "kept_positive_rate": float(kept_predictions["y_true"].mean()) if len(kept_predictions) else None,
                "removed_positive_rate": float(removed_predictions["y_true"].mean()) if len(removed_predictions) else None,
            }
            row.update(pooled_metrics_from_predictions(kept_predictions))
            for metric in ["pooled_mcc", "pooled_auc_pr", "pooled_auc_roc", "pooled_precision", "pooled_recall", "pooled_f1", "pooled_accuracy"]:
                if metric in row and metric in baseline and pd.notna(row[metric]) and pd.notna(baseline[metric]):
                    row[f"{metric}_delta_vs_all"] = float(row[metric]) - float(baseline[metric])
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["experiment", "model", "min_test_size", "min_test_positives", "min_test_negatives"])
    summary.to_csv(output_dir / "split_reliability_summary.csv", index=False)
    (output_dir / "split_reliability_summary.md").write_text(_render_markdown(summary, split_stats), encoding="utf-8")


def _load_predictions(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((results_dir / "predictions").glob("*_predictions.csv")):
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _split_stats(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, model, repository, split_id), group in predictions.groupby(
        ["experiment", "model", "repository", "split_id"],
        dropna=False,
    ):
        y_true = group["y_true"].astype(int)
        rows.append({
            "experiment": experiment,
            "model": model,
            "repository": repository,
            "split_id": split_id,
            "test_size": len(group),
            "test_positives": int(y_true.sum()),
            "test_negatives": int((1 - y_true).sum()),
            "positive_rate": float(y_true.mean()) if len(group) else None,
        })
    return pd.DataFrame(rows)


def _parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _render_markdown(summary: pd.DataFrame, split_stats: pd.DataFrame) -> str:
    compact = summary[
        [
            "experiment",
            "model",
            "min_test_size",
            "min_test_positives",
            "min_test_negatives",
            "kept_splits",
            "removed_splits",
            "kept_predictions",
            "kept_prediction_rate",
            "kept_positive_rate",
            "removed_positive_rate",
            "pooled_mcc",
            "pooled_auc_pr",
            "pooled_auc_roc",
            "pooled_precision",
            "pooled_recall",
            "pooled_f1",
            "pooled_accuracy",
        ]
    ]
    distribution = _distribution_table(split_stats)
    try:
        summary_table = compact.to_markdown(index=False)
        distribution_table = distribution.to_markdown(index=False)
    except Exception:
        summary_table = "```\n" + compact.to_csv(index=False) + "\n```"
        distribution_table = "```\n" + distribution.to_csv(index=False) + "\n```"
    return "\n".join([
        "# Split Reliability Analysis",
        "",
        "This report evaluates how final pooled metrics change when walk-forward test splits with very small test sets are excluded after training.",
        "",
        "The analysis is post-hoc: it does not retrain models. It filters saved test predictions by split and recomputes pooled metrics.",
        "",
        "## Test Split Distribution",
        "",
        distribution_table,
        "",
        "## Threshold Results",
        "",
        summary_table,
        "",
        "Read `kept_prediction_rate` together with the pooled metrics: an apparent metric improvement is useful only if it does not discard too much of the evaluation set.",
        "",
    ])


def _distribution_table(split_stats: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 2, 5, 10, 20, 30, 50, 100, float("inf")]
    labels = ["1-2", "3-5", "6-10", "11-20", "21-30", "31-50", "51-100", ">100"]
    rows = []
    for (experiment, model), group in split_stats.groupby(["experiment", "model"], dropna=False):
        bucket = pd.cut(group["test_size"], bins=bins, labels=labels, include_lowest=True)
        bucketed = (
            group.assign(test_size_bucket=bucket.astype(str))
            .groupby("test_size_bucket", observed=False)
            .agg(
                splits=("split_id", "count"),
                test_predictions=("test_size", "sum"),
                median_test_size=("test_size", "median"),
                median_positives=("test_positives", "median"),
            )
            .reset_index()
        )
        bucketed.insert(0, "model", model)
        bucketed.insert(0, "experiment", experiment)
        rows.append(bucketed)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


if __name__ == "__main__":
    main()
