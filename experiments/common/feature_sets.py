from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import pandas as pd


ID_COLUMNS = {
    "repo_url",
    "branch",
    "repository",
    "commit",
    "committed_at",
    "filepath",
    "row_index",
    "status",
    "error",
    "_merge",
    "graphml_path",
    "graphml_local_path",
    "dataset_id",
    "dataset_version",
    "pdg_metric_semantics",
    "pdg_quality_min_nodes",
    "pdg_quality_min_edges",
    "_sample_id",
}

LABEL_COLUMNS = {"failure_prone"}

PDG_METRICS = [
    "maxPdgVertices",
    "lackOfCohesion",
    "verticesCount",
    "edgesCount",
    "edgesToVerticesRatio",
    "globalInput",
    "globalOutput",
    "directFanIn",
    "indirectFanIn",
    "directFanOut",
    "indirectFanOut",
]

PDG_AUX_COLUMNS = {
    "nodes",
    "edges",
    "pdg_task_nodes",
    "pdg_order_edges",
    "pdg_def_edges",
    "pdg_use_edges",
    "pdg_unique_edge_labels",
}


def numeric_columns(df: pd.DataFrame) -> List[str]:
    result = []
    for col in df.columns:
        if col in ID_COLUMNS or col in LABEL_COLUMNS:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            result.append(col)
    return result


def e1_features(df: pd.DataFrame) -> List[str]:
    excluded = set(PDG_METRICS) | PDG_AUX_COLUMNS
    return [col for col in numeric_columns(df) if col not in excluded]


def e2_features(df: pd.DataFrame, pdg_metrics: Sequence[str] | str = "all", pdg_only: bool = False) -> List[str]:
    selected_pdg = resolve_pdg_metrics(pdg_metrics)
    if pdg_only:
        return selected_pdg
    base = e1_features(df)
    return base + [metric for metric in selected_pdg if metric not in base]


def resolve_pdg_metrics(pdg_metrics: Sequence[str] | str = "all") -> List[str]:
    if isinstance(pdg_metrics, str):
        if pdg_metrics.lower() == "all":
            return list(PDG_METRICS)
        requested = [item.strip() for item in pdg_metrics.split(",") if item.strip()]
    else:
        requested = list(pdg_metrics)
    unknown = [metric for metric in requested if metric not in PDG_METRICS]
    if unknown:
        raise ValueError(f"Unknown PDG metrics: {unknown}. Available: {PDG_METRICS}")
    return requested

