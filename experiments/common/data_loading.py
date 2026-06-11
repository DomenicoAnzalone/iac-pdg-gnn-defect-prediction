from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .feature_sets import PDG_METRICS


REQUIRED_COLUMNS = ["repository", "commit", "filepath", "failure_prone", "committed_at"]
GRAPH_COLUMNS = ["graphml_local_path", "graphml_path"]


def load_dataset(
    dataset_path: str | Path,
    max_repositories: int | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")
    df = pd.read_csv(path)
    validate_dataset_schema(df)
    df = df.copy()
    df["failure_prone"] = pd.to_numeric(df["failure_prone"], errors="coerce").fillna(0).astype(int)
    df["committed_at"] = pd.to_datetime(df["committed_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["committed_at"])
    df["_sample_id"] = make_sample_ids(df)
    if max_repositories:
        repos = df["repository"].drop_duplicates().head(max_repositories).tolist()
        df = df[df["repository"].isin(repos)].copy()
    if max_samples:
        df = df.head(max_samples).copy()
    return df.reset_index(drop=True)


def validate_dataset_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if not any(col in df.columns for col in GRAPH_COLUMNS):
        raise ValueError(f"Dataset must contain one graph path column among {GRAPH_COLUMNS}")
    missing_pdg = [col for col in PDG_METRICS if col not in df.columns]
    if missing_pdg:
        raise ValueError(f"Dataset is missing expected PDG metric columns: {missing_pdg}")
    duplicated = df.duplicated(["repository", "commit", "filepath"]).sum()
    if duplicated:
        raise ValueError(f"Dataset has duplicated (repository, commit, filepath) rows: {duplicated}")


def make_sample_ids(df: pd.DataFrame) -> pd.Series:
    return (
        df["repository"].astype(str)
        + "::"
        + df["commit"].astype(str)
        + "::"
        + df["filepath"].astype(str)
    )


def filter_common_valid_samples(
    df: pd.DataFrame,
    min_nodes: int = 3,
    min_edges: int = 2,
    graph_path_column: str = "graphml_local_path",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reasons: List[Dict[str, object]] = []
    keep_mask = []
    for _, row in df.iterrows():
        reason = ""
        nodes = pd.to_numeric(row.get("nodes"), errors="coerce")
        edges = pd.to_numeric(row.get("edges"), errors="coerce")
        path_value = row.get(graph_path_column) or row.get("graphml_path")
        graph_path = Path(str(path_value)) if path_value == path_value and path_value else None
        if nodes != nodes or int(nodes) < min_nodes:
            reason = f"below_min_nodes_{min_nodes}"
        elif edges != edges or int(edges) < min_edges:
            reason = f"below_min_edges_{min_edges}"
        elif any(pd.isna(row.get(metric)) for metric in PDG_METRICS):
            reason = "missing_pdg_metric"
        elif graph_path is None or not graph_path.exists():
            reason = "graphml_path_missing"
        if reason:
            keep_mask.append(False)
            reasons.append(_exclusion_row(row, reason))
        else:
            keep_mask.append(True)
    kept = df.loc[keep_mask].copy().reset_index(drop=True)
    excluded = pd.DataFrame(reasons)
    return kept, excluded


def _exclusion_row(row: pd.Series, reason: str) -> Dict[str, object]:
    return {
        "repository": row.get("repository"),
        "commit": row.get("commit"),
        "filepath": row.get("filepath"),
        "failure_prone": row.get("failure_prone"),
        "nodes": row.get("nodes"),
        "edges": row.get("edges"),
        "reason": reason,
    }


def label_distribution(df: pd.DataFrame) -> Dict[int, int]:
    return {int(k): int(v) for k, v in df["failure_prone"].value_counts().sort_index().items()}

