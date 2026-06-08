from __future__ import annotations

import argparse

import pandas as pd

from experiments.common.balancing import balance_sequence
from experiments.common.config import add_common_args, apply_common_overrides, load_config, parse_list
from experiments.common.progress import get_logger
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
    logger = get_logger("experiments.e3_gnn.run")
    models = parse_list(args.models or args.model, ["graphsage"])
    logger.info(
        "E3: run_dir=%s modelli=%s split=%s campioni=%s progress=%s",
        run_dir,
        ",".join(models),
        len(splits),
        len(df),
        config.get("progress", True),
    )
    if config.get("dry_run"):
        logger.info("Dry run richiesto: training non eseguito")
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
        logger.info("E3: avvio modello %s", canonical)
        model_predictions = []
        model_metrics = []
        for split_index, split in enumerate(splits, start=1):
            train_df, val_df, test_df = materialize_split(df, split)
            logger.info(
                "E3/%s split %s/%s split_id=%s repo=%s: righe train=%s val=%s test=%s",
                canonical,
                split_index,
                len(splits),
                split.split_id,
                split.repository,
                len(train_df),
                len(val_df),
                len(test_df),
            )
            train_data, train_ex = builder.build_partition(train_df, desc=f"load train {split.split_id}", show_progress=bool(config.get("progress", True)))
            val_data, val_ex = builder.build_partition(val_df, desc=f"load val {split.split_id}", show_progress=bool(config.get("progress", True)))
            test_data, test_ex = builder.build_partition(test_df, desc=f"load test {split.split_id}", show_progress=bool(config.get("progress", True)))
            all_graph_exclusions.extend([train_ex, val_ex, test_ex])
            if not train_data or not val_data or not test_data or len({int(data.y.item()) for data in train_data}) < 2:
                logger.warning(
                    "E3/%s split=%s saltato dopo caricamento grafi: train=%s val=%s test=%s classi_train=%s",
                    canonical,
                    split.split_id,
                    len(train_data),
                    len(val_data),
                    len(test_data),
                    sorted({int(data.y.item()) for data in train_data}) if train_data else [],
                )
                continue
            logger.info(
                "E3/%s split=%s grafi caricati: train=%s val=%s test=%s esclusioni=%s",
                canonical,
                split.split_id,
                len(train_data),
                len(val_data),
                len(test_data),
                len(train_ex) + len(val_ex) + len(test_ex),
            )
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
            logger.info(
                "E3/%s split=%s balancing training: before=%s after=%s",
                canonical,
                split.split_id,
                balance_report["before"],
                balance_report["after"],
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
        logger.info("E3/%s output salvati: predizioni=%s split_metriche=%s", canonical, len(predictions_df), len(model_metrics))
    if all_graph_exclusions:
        graph_ex = pd.concat([ex for ex in all_graph_exclusions if ex is not None and not ex.empty], ignore_index=True) if any(not ex.empty for ex in all_graph_exclusions if ex is not None) else pd.DataFrame()
        if not graph_ex.empty:
            graph_ex.to_csv(run_dir / "logs" / "graph_exclusions.csv", index=False)
    write_summary(run_dir, "E3 GNN", config)
    logger.info("E3 completato. Report: %s", run_dir / "reports" / "run_summary.md")


if __name__ == "__main__":
    main()
