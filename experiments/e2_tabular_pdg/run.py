from __future__ import annotations

import argparse

from experiments.common.classical import run_tabular_experiment
from experiments.common.config import add_common_args, apply_common_overrides, load_config, parse_list
from experiments.common.feature_sets import e2_features
from experiments.common.reporting import save_experiment_outputs, write_summary
from experiments.common.reproducibility import set_global_seed
from experiments.common.runner import prepare_common_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E2 tabular baseline with PDG metrics.")
    add_common_args(parser)
    parser.add_argument("--models", default="random_forest")
    parser.add_argument("--pdg-metrics", default="all")
    parser.add_argument("--pdg-only", action="store_true")
    parser.add_argument("--hyperparameter-search", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = apply_common_overrides(load_config(args.config), args)
    config["pdg_metrics"] = args.pdg_metrics
    config["pdg_only"] = bool(args.pdg_only)
    if args.hyperparameter_search:
        config["hyperparameter_search"] = True
    set_global_seed(int(config.get("seed", 42)))
    run_dir, df, splits, _ = prepare_common_run(config, args.run_name)
    features = e2_features(df, pdg_metrics=args.pdg_metrics, pdg_only=args.pdg_only)
    models = parse_list(args.models, ["random_forest"])
    if config.get("dry_run"):
        write_summary(run_dir, "E2 dry run", config)
        return
    predictions, metrics_rows, feature_manifest = run_tabular_experiment(
        df=df,
        splits=splits,
        feature_columns=features,
        model_names=models,
        experiment="e2",
        config=config,
    )
    for model in models:
        model_predictions = predictions[predictions["model"].eq(model)] if not predictions.empty else predictions
        model_metrics = [row for row in metrics_rows if row["model"] == model]
        model_features = feature_manifest[feature_manifest["model"].eq(model)] if not feature_manifest.empty else feature_manifest
        save_experiment_outputs(run_dir, "e2", model, model_predictions, model_metrics, model_features)
    write_summary(run_dir, "E2 Tabular PDG", config)


if __name__ == "__main__":
    main()

