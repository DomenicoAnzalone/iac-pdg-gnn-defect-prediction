from __future__ import annotations

import argparse

import pandas as pd

from experiments.common.balancing import balance_sequence
from experiments.common.config import add_common_args, apply_common_overrides, load_config, parse_list
from experiments.common.reporting import save_experiment_outputs, write_summary
from experiments.common.reproducibility import set_global_seed
from experiments.common.runner import prepare_common_run
from experiments.common.splitting import materialize_split
from experiments.e3_gnn.graph_data import GraphDataBuilder
from experiments.e3_gnn.training import GNN_ALIASES, run_gnn_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E3 graph-level GNN experiment.")
    add_common_args(parser)
    parser.add_argument("--model", default="graphsage")
    parser.add_argument("--models")
    parser.add_argument("--min-nodes", type=int)
    parser.add_argument("--min-edges", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = apply_common_overrides(load_config(args.config), args)
    for key in ["min_nodes", "min_edges", "epochs", "batch_size", "device"]:
        value = getattr(args, key.replace("_", "-"), None) if False else None
    if args.min_nodes is not None:
        config["min_nodes"] = args.min_nodes
    if args.min_edges is not None:
        config["min_edges"] = args.min_edges
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    config["device"] = args.device
    set_global_seed(int(config.get("seed", 42)))
    run_dir, df, splits, excluded = prepare_common_run(config, args.run_name)
    models = parse_list(args.models or args.model, ["graphsage"])
    if config.get("dry_run"):
        write_summary(run_dir, "E3 dry run", config)
        return
    builder = GraphDataBuilder(
        graph_path_column=config.get("graph_path_column", "graphml_local_path"),
        min_nodes=int(config.get("min_nodes", 3)),
        min_edges=int(config.get("min_edges", 2)),
    )
    all_graph_exclusions = []
    for model in models:
        canonical = GNN_ALIASES.get(model.lower(), model.lower())
        model_predictions = []
        model_metrics = []
        for split in splits:
            train_df, val_df, test_df = materialize_split(df, split)
            train_data, train_ex = builder.build_partition(train_df)
            val_data, val_ex = builder.build_partition(val_df)
            test_data, test_ex = builder.build_partition(test_df)
            all_graph_exclusions.extend([train_ex, val_ex, test_ex])
            if not train_data or not val_data or not test_data or len({int(data.y.item()) for data in train_data}) < 2:
                continue
            scaler = builder.fit_scaler(train_data)
            train_data = builder.apply_scaler(train_data, scaler)
            val_data = builder.apply_scaler(val_data, scaler)
            test_data = builder.apply_scaler(test_data, scaler)
            labels = [int(data.y.item()) for data in train_data]
            train_data, balance_report = balance_sequence(
                train_data,
                labels,
                strategy=config.get("balance_strategy", "none"),
                seed=int(config.get("seed", 42)),
            )
            predictions, metrics = run_gnn_model(
                canonical,
                train_data,
                val_data,
                test_data,
                split_id=split.split_id,
                repository=split.repository,
                config=config,
                model_dir=run_dir / "models",
            )
            metrics["balance_before"] = str(balance_report["before"])
            metrics["balance_after"] = str(balance_report["after"])
            model_predictions.append(predictions)
            model_metrics.append(metrics)
        predictions_df = pd.concat(model_predictions, ignore_index=True) if model_predictions else pd.DataFrame()
        save_experiment_outputs(run_dir, "e3", canonical, predictions_df, model_metrics, None)
    if all_graph_exclusions:
        graph_ex = pd.concat([ex for ex in all_graph_exclusions if ex is not None and not ex.empty], ignore_index=True) if any(not ex.empty for ex in all_graph_exclusions if ex is not None) else pd.DataFrame()
        if not graph_ex.empty:
            graph_ex.to_csv(run_dir / "logs" / "graph_exclusions.csv", index=False)
    write_summary(run_dir, "E3 GNN", config)


if __name__ == "__main__":
    main()

