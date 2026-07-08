from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from .reproducibility import metadata, write_json


def prepare_run_dir(results_root: str | Path, run_name: str) -> Path:
    run_dir = Path(results_root) / run_name
    for child in ["predictions", "metrics", "models", "logs", "reports"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(run_dir: Path, config: Dict[str, Any], cwd: Path) -> None:
    write_json(run_dir / "metadata.json", metadata(config, cwd))
    write_json(run_dir / "config.yaml", config)


def save_common_manifests(
    run_dir: Path,
    split_manifest: pd.DataFrame,
    excluded_samples: pd.DataFrame,
    skipped_splits: pd.DataFrame,
) -> None:
    split_manifest.to_csv(run_dir / "split_manifest.csv", index=False)
    excluded_samples.to_csv(run_dir / "excluded_samples.csv", index=False)
    skipped_splits.to_csv(run_dir / "logs" / "skipped_splits.csv", index=False)


def save_experiment_outputs(
    run_dir: Path,
    experiment: str,
    model: str,
    predictions: pd.DataFrame,
    metrics_rows: List[Dict[str, Any]],
    feature_manifest: pd.DataFrame | None = None,
) -> None:
    predictions.to_csv(run_dir / "predictions" / f"{experiment}_{model}_predictions.csv", index=False)
    metrics_df = pd.DataFrame(metrics_rows)
    per_split_path = run_dir / "metrics" / "per_split_metrics.csv"
    if per_split_path.exists():
        existing = pd.read_csv(per_split_path)
        if not existing.empty and {"experiment", "model"}.issubset(existing.columns):
            existing = existing[
                ~(
                    existing["experiment"].astype(str).eq(str(experiment))
                    & existing["model"].astype(str).eq(str(model))
                )
            ]
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
    metrics_df.to_csv(per_split_path, index=False)
    if feature_manifest is not None:
        feature_manifest.to_csv(run_dir / f"feature_manifest_{experiment}_{model}.csv", index=False)
        merged_feature_path = run_dir / "feature_manifest.csv"
        merged = feature_manifest
        if merged_feature_path.exists():
            existing_features = pd.read_csv(merged_feature_path)
            if not existing_features.empty and {"experiment", "model"}.issubset(existing_features.columns):
                existing_features = existing_features[
                    ~(
                        existing_features["experiment"].astype(str).eq(str(experiment))
                        & existing_features["model"].astype(str).eq(str(model))
                    )
                ]
            merged = pd.concat([existing_features, feature_manifest], ignore_index=True)
        merged.to_csv(merged_feature_path, index=False)
    rebuild_aggregates(run_dir)


def rebuild_aggregates(run_dir: Path) -> None:
    from .evaluation import aggregate_metrics

    per_split_path = run_dir / "metrics" / "per_split_metrics.csv"
    if not per_split_path.exists():
        return
    per_split = pd.read_csv(per_split_path)
    if per_split.empty:
        return
    repo = (
        per_split.groupby(["experiment", "model", "repository"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    repo.to_csv(run_dir / "metrics" / "per_repository_metrics.csv", index=False)
    pooled_by_key = _build_pooled_metrics(run_dir)
    agg_rows = []
    for (experiment, model), group in per_split.groupby(["experiment", "model"]):
        row = {"experiment": experiment, "model": model}
        row.update(aggregate_metrics(group.to_dict("records")))
        row.update(pooled_by_key.get((str(experiment), str(model)), {}))
        agg_rows.append(row)
    pd.DataFrame(agg_rows).to_csv(run_dir / "metrics" / "aggregated_metrics.csv", index=False)


def _build_pooled_metrics(run_dir: Path) -> Dict[tuple[str, str], Dict[str, Any]]:
    from .evaluation import pooled_metrics_from_predictions

    prediction_files = sorted((run_dir / "predictions").glob("*_predictions.csv"))
    rows: List[Dict[str, Any]] = []
    repo_rows: List[Dict[str, Any]] = []
    for path in prediction_files:
        predictions = pd.read_csv(path)
        if predictions.empty or not {"experiment", "model", "y_true", "y_pred"}.issubset(predictions.columns):
            continue
        for (experiment, model), group in predictions.groupby(["experiment", "model"], dropna=False):
            row: Dict[str, Any] = {"experiment": str(experiment), "model": str(model)}
            row.update(pooled_metrics_from_predictions(group))
            rows.append(row)
        for (experiment, model, repository), group in predictions.groupby(["experiment", "model", "repository"], dropna=False):
            row = {"experiment": str(experiment), "model": str(model), "repository": str(repository)}
            row.update(pooled_metrics_from_predictions(group))
            repo_rows.append(row)
    if rows:
        pooled = pd.DataFrame(rows).drop_duplicates(subset=["experiment", "model"], keep="last")
        pooled.to_csv(run_dir / "metrics" / "pooled_metrics.csv", index=False)
        if repo_rows:
            pd.DataFrame(repo_rows).drop_duplicates(
                subset=["experiment", "model", "repository"],
                keep="last",
            ).to_csv(run_dir / "metrics" / "per_repository_pooled_metrics.csv", index=False)
        return {
            (str(row["experiment"]), str(row["model"])): {
                key: value
                for key, value in row.items()
                if key not in {"experiment", "model"}
            }
            for row in pooled.to_dict("records")
        }
    return {}


def write_summary(run_dir: Path, title: str, config: Dict[str, Any]) -> None:
    agg_path = run_dir / "metrics" / "aggregated_metrics.csv"
    lines = [f"# {title}", "", f"- Dataset: `{config.get('dataset')}`", f"- Seed: `{config.get('seed')}`"]
    if agg_path.exists():
        lines.extend(["", "## Final pooled metrics", ""])
        pooled_path = run_dir / "metrics" / "pooled_metrics.csv"
        if pooled_path.exists():
            pooled = pd.read_csv(pooled_path)
            try:
                lines.append(pooled.to_markdown(index=False))
            except Exception:
                lines.append("```")
                lines.append(pooled.to_csv(index=False))
                lines.append("```")
        lines.extend(["", "## Per-split aggregate diagnostics", ""])
        agg = pd.read_csv(agg_path)
        try:
            lines.append(agg.to_markdown(index=False))
        except Exception:
            lines.append("```")
            lines.append(agg.to_csv(index=False))
            lines.append("```")
    lines.extend([
        "",
        "## Notes",
        "",
        "Validation and test sets are never balanced. Transformations are fitted only on the training partition of each walk-forward split.",
    ])
    (run_dir / "reports" / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")
