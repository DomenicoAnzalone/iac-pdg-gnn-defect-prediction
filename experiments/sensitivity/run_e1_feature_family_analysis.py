from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List
import warnings


_SKLEARN_PARALLEL_WARNING = r".*sklearn\.utils\.parallel\.delayed.*"
_PYTHONWARNINGS_FILTER = "ignore:.*sklearn.utils.parallel.delayed.*:UserWarning"

warnings.filterwarnings("ignore", message=_SKLEARN_PARALLEL_WARNING, category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.utils\.parallel")
if _PYTHONWARNINGS_FILTER not in os.environ.get("PYTHONWARNINGS", ""):
    os.environ["PYTHONWARNINGS"] = ",".join(
        item for item in [os.environ.get("PYTHONWARNINGS", ""), _PYTHONWARNINGS_FILTER] if item
    )

import pandas as pd

from experiments.common.config import DEFAULT_DATASET, apply_common_overrides, load_config
from experiments.common.feature_sets import e1_features_by_family, unmapped_e1_features
from experiments.common.reporting import save_experiment_outputs, write_summary
from experiments.common.reproducibility import set_global_seed
from experiments.common.runner import prepare_common_run


FEATURE_FAMILIES = ["process", "product", "iac_oriented", "delta"]
MODEL_ALIASES = {
    "random_forest": "random_forest",
    "rf": "random_forest",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or run E1 Random Forest sensitivity analysis over individual feature families."
    )
    parser.add_argument("--config", default="experiments/configs/e1_default.yaml")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--families", default="process,product,iac_oriented,delta")
    parser.add_argument("--run-prefix", default="e1_feature_family")
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--balance", default="random_oversampling", choices=["none", "random_undersampling", "random_oversampling"])
    parser.add_argument("--scaler", default="standard", choices=["none", "min-max", "standard"])
    parser.add_argument("--feature-selection", default="validation_rfe", choices=["none", "variance_threshold", "rfe", "rfecv", "validation_rfe"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--results-root", default="experiments/results/exploratory")
    parser.add_argument("--summary-dir", default="experiments/results/exploratory/e1_feature_family_analysis")
    parser.add_argument("--max-repositories", type=int)
    parser.add_argument("--max-splits", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    families = [_parse_family(item) for item in args.families.split(",") if item.strip()]
    model = MODEL_ALIASES.get(args.model.lower(), args.model.lower())
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for family in families:
        run_name = f"{args.run_prefix}_{family}_{model}"
        run_dir = Path(args.results_root) / run_name
        if args.skip_existing and (run_dir / "metrics" / "pooled_metrics.csv").exists():
            rows.append(_run_row(run_name, family, model, "completed_existing", 0, run_dir))
            continue

        config = _config_for_family(args, run_name)
        set_global_seed(int(config.get("seed", 42)))
        run_dir, df, splits, _ = prepare_common_run(config, run_name)
        features = e1_features_by_family(df, family)
        unmapped = unmapped_e1_features(df)
        _write_feature_family_manifest(run_dir, family, features, unmapped)
        print(
            f"[{family}] run={run_name} features={len(features)} splits={len(splits)} dry_run={bool(config.get('dry_run'))}",
            flush=True,
        )

        if not features:
            rows.append(_run_row(run_name, family, model, "failed_no_features", 1, run_dir, features, unmapped))
            break
        if config.get("dry_run"):
            write_summary(run_dir, f"E1 feature-family dry run: {family}", config)
            rows.append(_run_row(run_name, family, model, "dry_run", 0, run_dir, features, unmapped, len(splits)))
            print(f"[{family}] dry-run completato", flush=True)
            continue

        from experiments.common.classical import run_tabular_experiment

        predictions, metrics_rows, feature_manifest = run_tabular_experiment(
            df=df,
            splits=splits,
            feature_columns=features,
            model_names=[model],
            experiment=f"e1_{family}",
            config=config,
        )
        model_predictions = predictions[predictions["model"].eq(model)] if not predictions.empty else predictions
        model_metrics = [row for row in metrics_rows if row["model"] == model]
        model_features = feature_manifest[feature_manifest["model"].eq(model)] if not feature_manifest.empty else feature_manifest
        save_experiment_outputs(run_dir, f"e1_{family}", model, model_predictions, model_metrics, model_features)
        write_summary(run_dir, f"E1 feature-family analysis: {family}", config)
        rows.append(_run_row(run_name, family, model, "completed", 0, run_dir, features, unmapped, len(splits)))
        print(f"[{family}] completato", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(summary_dir / "e1_feature_family_summary.csv", index=False)
    (summary_dir / "e1_feature_family_summary.md").write_text(_render_markdown(summary), encoding="utf-8")
    print(f"Summary written to {summary_dir / 'e1_feature_family_summary.md'}", flush=True)


def _config_for_family(args: argparse.Namespace, run_name: str) -> Dict[str, object]:
    config = load_config(args.config)
    namespace = argparse.Namespace(
        dataset=args.dataset,
        results_root=args.results_root,
        balance=args.balance,
        scaler=args.scaler,
        seed=args.seed,
        max_repositories=args.max_repositories,
        max_splits=args.max_splits,
        max_samples=args.max_samples,
        log_level=None,
        log_every_epochs=None,
        feature_selection=args.feature_selection,
        n_jobs=args.n_jobs,
        dry_run=args.dry_run,
        quiet=args.quiet,
        no_progress=args.no_progress,
        compact_progress=False,
    )
    config = apply_common_overrides(config, namespace)
    config["run_name"] = run_name
    config["n_jobs"] = args.n_jobs
    return config


def _write_feature_family_manifest(run_dir: Path, family: str, features: List[str], unmapped: List[str]) -> None:
    manifest = pd.DataFrame(
        [{"family": family, "feature": feature, "status": "included"} for feature in features]
        + [{"family": family, "feature": feature, "status": "unmapped_e1_feature"} for feature in unmapped]
    )
    manifest.to_csv(run_dir / "feature_family_manifest.csv", index=False)


def _run_row(
    run_name: str,
    family: str,
    model: str,
    status: str,
    return_code: int,
    run_dir: Path,
    features: List[str] | None = None,
    unmapped: List[str] | None = None,
    valid_splits: int | None = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "family": family,
        "model": model,
        "status": status,
        "return_code": return_code,
        "run_name": run_name,
        "run_dir": str(run_dir),
    }
    if features is not None:
        row["feature_count"] = len(features)
    if unmapped is not None:
        row["unmapped_e1_feature_count"] = len(unmapped)
        row["unmapped_e1_features"] = ",".join(unmapped)
    if valid_splits is not None:
        row["valid_splits"] = valid_splits
    pooled_path = run_dir / "metrics" / "pooled_metrics.csv"
    split_path = run_dir / "metrics" / "per_split_metrics.csv"
    family_manifest = run_dir / "feature_family_manifest.csv"
    if family_manifest.exists() and features is None:
        manifest = _read_csv_or_empty(family_manifest)
        row["feature_count"] = int(manifest["status"].eq("included").sum()) if "status" in manifest.columns else None
        row["unmapped_e1_feature_count"] = int(manifest["status"].eq("unmapped_e1_feature").sum()) if "status" in manifest.columns else None
    if pooled_path.exists():
        pooled = _read_csv_or_empty(pooled_path)
        if not pooled.empty:
            for key, value in pooled.iloc[0].to_dict().items():
                if key not in {"experiment", "model"}:
                    row[key] = value
    if split_path.exists():
        row["valid_metric_splits"] = len(_read_csv_or_empty(split_path))
    return row


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _parse_family(value: str) -> str:
    family = value.strip().lower().replace("-", "_")
    if family == "iac":
        family = "iac_oriented"
    if family not in FEATURE_FAMILIES:
        raise ValueError(f"Unsupported family: {value}. Supported values: {', '.join(FEATURE_FAMILIES)}")
    return family


def _render_markdown(summary: pd.DataFrame) -> str:
    ordered_columns = [
        "family",
        "model",
        "status",
        "feature_count",
        "valid_splits",
        "pooled_sample_count",
        "pooled_mcc",
        "pooled_auc_pr",
        "pooled_auc_roc",
        "pooled_precision",
        "pooled_recall",
        "pooled_f1",
        "pooled_accuracy",
        "run_name",
        "unmapped_e1_feature_count",
    ]
    display = summary[[column for column in ordered_columns if column in summary.columns]].copy()
    try:
        table = display.to_markdown(index=False)
    except Exception:
        table = "```\n" + display.to_csv(index=False) + "\n```"
    return "\n".join(
        [
            "# E1 Feature-Family Analysis",
            "",
            "Ogni riga corrisponde a una run E1 Random Forest con una sola famiglia di metriche candidate.",
            "Le run usano lo stesso dataset, split walk-forward, bilanciamento, scaling, seed e procedura di feature selection della baseline E1.",
            "",
            table,
            "",
            "Nota: `unmapped_e1_feature_count` segnala eventuali feature E1 non assegnate a nessuna famiglia esplicita.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
