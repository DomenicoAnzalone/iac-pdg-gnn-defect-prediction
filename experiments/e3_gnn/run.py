from __future__ import annotations

import argparse
import time

import pandas as pd

from experiments.common.balancing import balance_sequence
from experiments.common.config import add_common_args, apply_common_overrides, load_config, parse_list
from experiments.common.progress import CompactStatusLine, get_logger, progress
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
    logger.info(
        "E3 pronto: modelli=%s split_validi=%s device=%s epochs=%s batch_size=%s. Dettagli completi in %s",
        ",".join(models),
        len(splits),
        config.get("device", "auto"),
        config.get("epochs", 100),
        config.get("batch_size", 32),
        run_dir / "logs" / "run.log",
        extra={"console": True},
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
        logger.info("Training E3/%s avviato", canonical, extra={"console": True})
        model_predictions = []
        model_metrics = []
        model_start = time.time()
        split_times = []
        compact = bool(config.get("compact_progress", False))
        compact_status = CompactStatusLine(len(splits), max_width=100, enabled=compact and bool(config.get("progress", True)))
        if compact:
            split_iter = list(enumerate(splits, start=1))
        else:
            split_iter = progress(
                list(enumerate(splits, start=1)),
                total=len(splits),
                desc="Split",
                unit="split",
                enabled=bool(config.get("progress", True)),
                leave=True,
                position=0,
            )
        for split_index, split in split_iter:
            split_start = time.time()
            if compact:
                compact_status.update(split_index=split_index, completed_splits=split_index - 1, status="loading")
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
            show_graph_progress = bool(config.get("progress", True)) and not bool(config.get("compact_progress", False))
            train_data, train_ex = builder.build_partition(train_df, desc=f"load train {split.split_id}", show_progress=show_graph_progress)
            val_data, val_ex = builder.build_partition(val_df, desc=f"load val {split.split_id}", show_progress=show_graph_progress)
            test_data, test_ex = builder.build_partition(test_df, desc=f"load test {split.split_id}", show_progress=show_graph_progress)
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
                if hasattr(split_iter, "set_postfix"):
                    split_iter.set_postfix(status="skip")
                if compact:
                    compact_status.update(split_index=split_index, completed_splits=split_index, status="skipped")
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
                compact_status=compact_status,
                split_index=split_index,
                total_splits=len(splits),
            )
            metrics["balance_before"] = str(balance_report["before"])
            metrics["balance_after"] = str(balance_report["after"])
            model_predictions.append(predictions)
            model_metrics.append(metrics)
            split_seconds = time.time() - split_start
            split_times.append(split_seconds)
            if hasattr(split_iter, "set_postfix"):
                split_iter.set_postfix(
                    mcc=_format_metric(metrics.get("mcc")),
                    avg=f"{sum(split_times) / len(split_times):.1f}s",
                )
            if compact:
                compact_status.update(
                    split_index=split_index,
                    completed_splits=split_index,
                    mcc=_format_metric(metrics.get("mcc")),
                    avg=f"{sum(split_times) / len(split_times):.1f}s",
                    eta=_eta(sum(split_times) / len(split_times), len(splits) - split_index),
                )
        compact_status.close()
        predictions_df = pd.concat(model_predictions, ignore_index=True) if model_predictions else pd.DataFrame()
        save_experiment_outputs(run_dir, "e3", canonical, predictions_df, model_metrics, None)
        logger.info("E3/%s output salvati: predizioni=%s split_metriche=%s", canonical, len(predictions_df), len(model_metrics))
        logger.info(
            "Training E3/%s completato: split_metriche=%s predizioni=%s tempo=%.1fs",
            canonical,
            len(model_metrics),
            len(predictions_df),
            time.time() - model_start,
            extra={"console": True},
        )
    if all_graph_exclusions:
        graph_ex = pd.concat([ex for ex in all_graph_exclusions if ex is not None and not ex.empty], ignore_index=True) if any(not ex.empty for ex in all_graph_exclusions if ex is not None) else pd.DataFrame()
        if not graph_ex.empty:
            graph_ex.to_csv(run_dir / "logs" / "graph_exclusions.csv", index=False)
    write_summary(run_dir, "E3 GNN", config)
    logger.info("E3 completato. Report: %s", run_dir / "reports" / "run_summary.md", extra={"console": True})


def _format_metric(value: object) -> str:
    try:
        val = float(value)
        if val != val:
            return "nan"
        return f"{val:.3f}"
    except Exception:
        return "nan"


def _eta(avg_seconds: float, remaining: int) -> str:
    total = max(0, int(avg_seconds * remaining))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


if __name__ == "__main__":
    main()
