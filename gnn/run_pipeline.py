from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from gnn.preprocessing.dataset import GraphDatasetBuilder
from gnn.sampling.balance import GraphBalancer
from gnn.sampling.splitter import WalkForwardSplitter, WalkForwardSplit


@dataclass
class PipelineSummary:
    repository: str
    test_commit: str
    train_size: int
    validation_size: int
    test_size: int
    balanced_train_size: int
    train_failure_rate: float
    validation_failure_rate: float
    test_failure_rate: float


def build_sample_index(builder: GraphDatasetBuilder) -> Dict[str, object]:
    sample_index = {}
    for sample in builder.samples():
        sample_index[str(sample.graph_path)] = sample
    return sample_index


def metadata_dataframe(builder: GraphDatasetBuilder) -> pd.DataFrame:
    records = []
    for sample in builder.samples():
        records.append(
            {
                "repository": sample.repository,
                "commit": sample.commit,
                "failure_prone": sample.label,
                "graph_path": str(sample.graph_path),
                "filepath": sample.filepath,
            }
        )
    return pd.DataFrame(records)


def summarize_dataframe(df: pd.DataFrame, label_column: str = "failure_prone") -> str:
    counts = df[label_column].value_counts(dropna=False).to_dict()
    total = len(df)
    rates = {label: counts.get(label, 0) / total for label in sorted(counts)} if total else {}
    return f"samples={total}, counts={counts}, rates={rates}"


def build_graph_data_for_rows(
    builder: GraphDatasetBuilder,
    sample_index: Dict[str, object],
    source_df: pd.DataFrame,
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    results = []
    for _, row in source_df.iterrows():
        graph_path = str(row["graph_path"])
        sample = sample_index.get(graph_path)
        if sample is None:
            logging.warning("Skipping graph not found in builder samples: %s", graph_path)
            continue
        try:
            results.append(builder.build_graph_data(sample))
        except RuntimeError as exc:
            logging.warning("Failed to build graph data for %s: %s", graph_path, exc)
        if limit is not None and len(results) >= limit:
            break
    return results


def run_pipeline(
    label_csv: Path,
    repositories_root: Optional[Path],
    balance_strategy: str,
    validation_ratio: float,
    preview_per_split: int,
    max_splits: Optional[int],
    remote_prefix: str,
) -> List[PipelineSummary]:
    builder = GraphDatasetBuilder(
        label_csv=label_csv,
        path_remapper=GraphDatasetBuilder.path_remapper_from_local_root(Path.cwd(), remote_prefix=remote_prefix),
    )
    metadata = metadata_dataframe(builder)
    if metadata.empty:
        raise RuntimeError("Nessun campione valido trovato nel CSV delle label.")

    sample_index = build_sample_index(builder)
    splitter = WalkForwardSplitter(metadata, repositories_root=repositories_root)
    all_splits = splitter.all_project_splits()
    if not all_splits:
        raise RuntimeError("Nessuno split walk-forward generato. Verifica repository e commit nel CSV.")

    if max_splits is not None:
        all_splits = all_splits[:max_splits]

    balancer = GraphBalancer(random_state=42)
    summaries: List[PipelineSummary] = []

    for split in all_splits:
        split_with_val = splitter.train_validation_split(split, validation_ratio=validation_ratio)
        balanced_train = (
            balancer.dataframe_oversample(split_with_val.train, label_column="failure_prone")
            if balance_strategy == "oversample"
            else balancer.dataframe_undersample(split_with_val.train, label_column="failure_prone")
        )

        validation_source = split_with_val.validation if split_with_val.validation is not None else pd.DataFrame(
            columns=split_with_val.train.columns
        )
        train_data = build_graph_data_for_rows(builder, sample_index, balanced_train, limit=preview_per_split)
        validation_data = build_graph_data_for_rows(builder, sample_index, validation_source, limit=preview_per_split)
        test_data = build_graph_data_for_rows(builder, sample_index, split_with_val.test, limit=preview_per_split)

        summaries.append(
            PipelineSummary(
                repository=split.repository,
                test_commit=split_with_val.test_commit,
                train_size=len(split_with_val.train),
                validation_size=len(split_with_val.validation) if split_with_val.validation is not None else 0,
                test_size=len(split_with_val.test),
                balanced_train_size=len(balanced_train),
                train_failure_rate=float(split_with_val.train["failure_prone"].mean()) if len(split_with_val.train) else 0.0,
                validation_failure_rate=float(split_with_val.validation["failure_prone"].mean()) if split_with_val.validation is not None and len(split_with_val.validation) else 0.0,
                test_failure_rate=float(split_with_val.test["failure_prone"].mean()) if len(split_with_val.test) else 0.0,
            )
        )

        validation_count = len(split_with_val.validation) if split_with_val.validation is not None else 0
        logging.info(
            "Split %s/%s: train=%s, validation=%s, test=%s",
            split.repository,
            split.test_commit,
            len(split_with_val.train),
            validation_count,
            len(split_with_val.test),
        )
        validation_source = split_with_val.validation if split_with_val.validation is not None else pd.DataFrame(columns=split_with_val.train.columns)
        logging.info("Train summary: %s", summarize_dataframe(balanced_train))
        logging.info("Validation summary: %s", summarize_dataframe(validation_source))
        logging.info("Test summary: %s", summarize_dataframe(split_with_val.test))
        logging.info("Preview build: train=%s, validation=%s, test=%s", len(train_data), len(validation_data), len(test_data))

    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GNN preprocessing + sampling pipeline for PDG defect prediction.")
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=Path("output/ansible_rows_successfull_extracted.csv"),
        help="CSV delle label e dei percorsi ai grafi estratti.",
    )
    parser.add_argument(
        "--repositories-root",
        type=Path,
        default=Path("input/repositories"),
        help="Radice delle repository usate per calcolare gli split cronologici.",
    )
    parser.add_argument(
        "--balance-strategy",
        choices=["oversample", "undersample"],
        default="oversample",
        help="Strategia di bilanciamento applicata solo al training set.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Percentuale di validation set estratta dal training set.",
    )
    parser.add_argument(
        "--preview-per-split",
        type=int,
        default=5,
        help="Numero massimo di grafi da costruire per split in anteprima.",
    )
    parser.add_argument(
        "--max-splits",
        type=int,
        default=5,
        help="Numero massimo di split walk-forward da elaborare.",
    )
    parser.add_argument(
        "--remote-prefix",
        type=str,
        default="/app",
        help="Prefisso remoto usato per rimappare i percorsi salvati all'interno del CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Abilita log dettagliati.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG, format="%(levelname)s: %(message)s")

    summaries = run_pipeline(
        label_csv=args.label_csv,
        repositories_root=args.repositories_root,
        balance_strategy=args.balance_strategy,
        validation_ratio=args.validation_ratio,
        preview_per_split=args.preview_per_split,
        max_splits=args.max_splits,
        remote_prefix=args.remote_prefix,
    )

    for summary in summaries:
        print("---")
        print(f"repository: {summary.repository}")
        print(f"test_commit: {summary.test_commit}")
        print(f"train_size: {summary.train_size}")
        print(f"validation_size: {summary.validation_size}")
        print(f"test_size: {summary.test_size}")
        print(f"balanced_train_size: {summary.balanced_train_size}")
        print(f"train_failure_rate: {summary.train_failure_rate:.3f}")
        print(f"validation_failure_rate: {summary.validation_failure_rate:.3f}")
        print(f"test_failure_rate: {summary.test_failure_rate:.3f}")


if __name__ == "__main__":
    main()
