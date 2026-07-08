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

PDG_METRICS_TOP4 = [
    "maxPdgVertices",
    "verticesCount",
    "edgesToVerticesRatio",
    "edgesCount",
]

PDG_METRICS_TOP5 = [*PDG_METRICS_TOP4, "globalInput"]

PDG_METRIC_ALIASES: Dict[str, List[str]] = {
    "all": PDG_METRICS,
    "top4": PDG_METRICS_TOP4,
    "iuliano_top4": PDG_METRICS_TOP4,
    "best_pdg": PDG_METRICS_TOP4,
    "top5": PDG_METRICS_TOP5,
}

PDG_AUX_COLUMNS = {
    "nodes",
    "edges",
    "pdg_task_nodes",
    "pdg_order_edges",
    "pdg_def_edges",
    "pdg_use_edges",
    "pdg_unique_edge_labels",
}

E1_FEATURE_FAMILIES: Dict[str, List[str]] = {
    "process": [
        "additions",
        "additions_avg",
        "additions_max",
        "change_set_avg",
        "change_set_max",
        "code_churn_avg",
        "code_churn_count",
        "code_churn_max",
        "commits_count",
        "contributors_count",
        "deletions",
        "deletions_avg",
        "deletions_max",
        "highest_contributor_experience",
        "hunks_median",
        "minor_contributors_count",
    ],
    "product": [
        "lines_blank",
        "lines_code",
        "lines_comment",
        "num_conditions",
        "num_decisions",
        "num_keys",
        "num_parameters",
        "num_paths",
        "num_tokens",
        "num_vars",
        "text_entropy",
    ],
    "iac_oriented": [
        "avg_play_size",
        "avg_task_size",
        "num_authorized_key",
        "num_block_error_handling",
        "num_blocks",
        "num_commands",
        "num_deprecated_keywords",
        "num_deprecated_modules",
        "num_distinct_modules",
        "num_external_modules",
        "num_fact_modules",
        "num_file_exists",
        "num_file_mode",
        "num_file_modules",
        "num_filters",
        "num_ignore_errors",
        "num_import_playbook",
        "num_import_role",
        "num_import_tasks",
        "num_include",
        "num_include_role",
        "num_include_tasks",
        "num_include_vars",
        "num_lookups",
        "num_loops",
        "num_math_operations",
        "num_names_with_vars",
        "num_plays",
        "num_prompts",
        "num_regex",
        "num_roles",
        "num_suspicious_comments",
        "num_tasks",
        "num_unique_names",
        "num_uri",
    ],
}

E1_FEATURE_FAMILY_ALIASES: Dict[str, str] = {
    "process": "process",
    "processo": "process",
    "product": "product",
    "prodotto": "product",
    "iac": "iac_oriented",
    "iac_oriented": "iac_oriented",
    "iac-oriented": "iac_oriented",
    "delta": "delta",
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


def e1_feature_families(df: pd.DataFrame) -> Dict[str, List[str]]:
    available = set(e1_features(df))
    families = {
        family: [feature for feature in features if feature in available]
        for family, features in E1_FEATURE_FAMILIES.items()
    }
    families["delta"] = sorted(feature for feature in available if feature.startswith("delta_"))
    return families


def e1_features_by_family(df: pd.DataFrame, family: str) -> List[str]:
    canonical = E1_FEATURE_FAMILY_ALIASES.get(family.strip().lower())
    if canonical is None:
        supported = ", ".join(sorted(E1_FEATURE_FAMILY_ALIASES))
        raise ValueError(f"Unknown E1 feature family: {family}. Supported values: {supported}")
    families = e1_feature_families(df)
    return list(families[canonical])


def unmapped_e1_features(df: pd.DataFrame) -> List[str]:
    all_features = set(e1_features(df))
    mapped = set()
    for features in e1_feature_families(df).values():
        mapped.update(features)
    return sorted(all_features - mapped)


def e2_features(df: pd.DataFrame, pdg_metrics: Sequence[str] | str = "all", pdg_only: bool = False) -> List[str]:
    selected_pdg = resolve_pdg_metrics(pdg_metrics)
    if pdg_only:
        return selected_pdg
    base = e1_features(df)
    return base + [metric for metric in selected_pdg if metric not in base]


def resolve_pdg_metrics(pdg_metrics: Sequence[str] | str = "all") -> List[str]:
    if isinstance(pdg_metrics, str):
        alias = pdg_metrics.lower()
        if alias in PDG_METRIC_ALIASES:
            return list(PDG_METRIC_ALIASES[alias])
        requested = [item.strip() for item in pdg_metrics.split(",") if item.strip()]
    else:
        requested = list(pdg_metrics)
    unknown = [metric for metric in requested if metric not in PDG_METRICS]
    if unknown:
        raise ValueError(f"Unknown PDG metrics: {unknown}. Available: {PDG_METRICS}")
    return requested
