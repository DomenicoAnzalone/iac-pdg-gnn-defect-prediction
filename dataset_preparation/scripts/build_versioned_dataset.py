from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd


KEY_COLUMNS = ["repository", "commit", "filepath"]
PDG_METRIC_COLUMNS = [
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
GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a versioned final Ansible defect-prediction dataset by merging "
            "RADON rows, PDG extraction status, graph-quality filters and PDG metrics."
        )
    )
    parser.add_argument("--radon-input", required=True, help="RADON CSV with tabular metrics.")
    parser.add_argument("--extraction-status", required=True, help="PDG extraction_status.csv.")
    parser.add_argument("--output-root", required=True, help="Root folder for versioned outputs.")
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Stable dataset id, e.g. ansible-pdg-defect-dataset.",
    )
    parser.add_argument("--version", default="", help="Dataset version. Default: vYYYY-MM-DD.")
    parser.add_argument("--min-pdg-nodes", type=int, default=3, help="Minimum graph nodes.")
    parser.add_argument("--min-pdg-edges", type=int, default=2, help="Minimum graph edges.")
    parser.add_argument(
        "--graph-base-dir",
        default="output",
        help="Local base used to resolve Docker paths like /app/output/...",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the version folder if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    radon_path = Path(args.radon_input)
    status_path = Path(args.extraction_status)
    version = args.version or f"v{datetime.now().date().isoformat()}"
    output_dir = Path(args.output_root) / version

    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output version already exists: {output_dir}. Use --force to overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not radon_path.exists():
        raise SystemExit(f"RADON input not found: {radon_path}")
    if not status_path.exists():
        raise SystemExit(f"Extraction status not found: {status_path}")

    radon = pd.read_csv(radon_path, low_memory=False)
    status = pd.read_csv(status_path, low_memory=False)
    radon = normalize_radon(radon)
    status = normalize_status(status)

    exclusions: list[dict[str, Any]] = []
    stage_counts: dict[str, Any] = {
        "radon_rows": int(len(radon)),
        "radon_unique_keys": int(radon[KEY_COLUMNS].drop_duplicates().shape[0]),
        "status_rows": int(len(status)),
        "status_unique_keys": int(status[KEY_COLUMNS].drop_duplicates().shape[0]),
    }

    radon, duplicate_radon = drop_duplicate_keys(radon, "duplicate_radon_key", exclusions)
    status, duplicate_status = drop_duplicate_keys(status, "duplicate_status_key", exclusions)
    stage_counts["radon_rows_after_duplicate_key_filter"] = int(len(radon))
    stage_counts["status_rows_after_duplicate_key_filter"] = int(len(status))

    merged = radon.merge(
        status,
        on=KEY_COLUMNS,
        how="left",
        suffixes=("", "_pdg_status"),
        indicator=True,
    )
    stage_counts["rows_after_radon_status_merge"] = int(len(merged))
    add_missing_status_exclusions(merged, exclusions)

    successful = merged[merged["status"].astype(str).str.upper().eq("SUCCESS")].copy()
    add_non_success_exclusions(merged, exclusions)
    stage_counts["rows_after_successful_pdg_filter"] = int(len(successful))

    metrics_rows = []
    metric_failures = []
    for row in successful.itertuples(index=False):
        graph_path = getattr(row, "graphml_path", "")
        resolved = resolve_graph_path(str(graph_path), args.graph_base_dir)
        key = {
            "repository": getattr(row, "repository"),
            "commit": getattr(row, "commit"),
            "filepath": getattr(row, "filepath"),
        }
        if not resolved:
            metric_failures.append({**key, "reason": "graphml_path_missing_or_not_found"})
            continue
        try:
            metrics = extract_graphml_metrics(resolved)
        except Exception as exc:
            metric_failures.append({**key, "reason": "graphml_parse_failure", "details": str(exc)[:500]})
            continue
        metrics_rows.append(
            {
                **key,
                "graphml_local_path": str(resolved),
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        raise SystemExit("No graph metrics could be extracted; final dataset would be empty.")
    metric_failures_df = pd.DataFrame(metric_failures)
    add_metric_failure_exclusions(metric_failures, exclusions)
    stage_counts["rows_with_extracted_pdg_metrics"] = int(len(metrics_df))

    enriched = successful.merge(metrics_df, on=KEY_COLUMNS, how="left", suffixes=("", "_computed"))
    enriched_missing_metrics = enriched["verticesCount"].isna()
    enriched = enriched[~enriched_missing_metrics].copy()

    quality_mask = (
        (pd.to_numeric(enriched["verticesCount"], errors="coerce") >= args.min_pdg_nodes)
        & (pd.to_numeric(enriched["edgesCount"], errors="coerce") >= args.min_pdg_edges)
    )
    add_low_quality_exclusions(enriched[~quality_mask], exclusions, args.min_pdg_nodes, args.min_pdg_edges)
    final = enriched[quality_mask].copy()
    final["dataset_id"] = args.dataset_id
    final["dataset_version"] = version
    final["pdg_metric_semantics"] = "file_level_proxy_v1"
    final["pdg_quality_min_nodes"] = args.min_pdg_nodes
    final["pdg_quality_min_edges"] = args.min_pdg_edges
    final["failure_prone"] = pd.to_numeric(final["failure_prone"], errors="coerce").astype("Int64")

    stage_counts["final_rows"] = int(len(final))
    stage_counts["final_columns"] = int(len(final.columns))
    stage_counts["final_repositories"] = int(final["repository"].nunique(dropna=True))
    stage_counts["final_label_distribution"] = label_distribution(final)

    final_path = output_dir / f"{args.dataset_id}_{version}_final.csv"
    final.to_csv(final_path, index=False)

    exclusions_df = pd.DataFrame(exclusions)
    exclusions_path = output_dir / f"{args.dataset_id}_{version}_exclusions.csv"
    exclusions_df.to_csv(exclusions_path, index=False)

    metrics_path = output_dir / f"{args.dataset_id}_{version}_pdg_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    if not metric_failures_df.empty:
        metric_failures_df.to_csv(reports_dir / "pdg_metric_failures.csv", index=False)
    if not duplicate_radon.empty:
        duplicate_radon.to_csv(reports_dir / "duplicate_radon_keys.csv", index=False)
    if not duplicate_status.empty:
        duplicate_status.to_csv(reports_dir / "duplicate_status_keys.csv", index=False)

    quality_tables = build_quality_tables(final, merged, exclusions_df)
    for filename, table in quality_tables.items():
        table.to_csv(reports_dir / filename, index=False)

    manifest = {
        "dataset_id": args.dataset_id,
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "radon_input": str(radon_path),
            "extraction_status": str(status_path),
            "graph_base_dir": args.graph_base_dir,
        },
        "outputs": {
            "final_dataset": str(final_path),
            "exclusions": str(exclusions_path),
            "pdg_metrics": str(metrics_path),
            "reports_dir": str(reports_dir),
        },
        "parameters": {
            "min_pdg_nodes": args.min_pdg_nodes,
            "min_pdg_edges": args.min_pdg_edges,
            "metric_semantics": "file_level_proxy_v1",
        },
        "stage_counts": stage_counts,
        "exclusion_reasons": exclusion_reason_counts(exclusions_df),
        "pdg_metric_columns": PDG_METRIC_COLUMNS,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir / "DATASET_REPORT.md", manifest, final, merged, exclusions_df)
    write_json(reports_dir / "quality_summary.json", build_quality_summary(final, merged, exclusions_df))

    print(f"Versioned dataset built: {final_path}")
    print(f"Rows: {stage_counts['radon_rows']} RADON -> {stage_counts['final_rows']} final")
    return 0


def normalize_radon(df: pd.DataFrame) -> pd.DataFrame:
    if "repository" not in df.columns:
        if "repo_url" not in df.columns:
            raise SystemExit("RADON input must contain either repository or repo_url.")
        df = df.copy()
        df["repository"] = df["repo_url"].apply(repository_from_url)
    require_columns(df, KEY_COLUMNS + ["failure_prone"], "RADON input")
    df["repository"] = df["repository"].astype(str).str.strip()
    df["commit"] = df["commit"].astype(str).str.strip()
    df["filepath"] = df["filepath"].astype(str).str.strip()
    return df


def normalize_status(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, KEY_COLUMNS + ["status", "nodes", "edges", "graphml_path"], "extraction status")
    result = df.copy()
    result["repository"] = result["repository"].astype(str).str.strip()
    result["commit"] = result["commit"].astype(str).str.strip()
    result["filepath"] = result["filepath"].astype(str).str.strip()
    keep_columns = [
        column
        for column in [
            "row_index",
            "repository",
            "commit",
            "filepath",
            "status",
            "nodes",
            "edges",
            "graphml_path",
            "error",
        ]
        if column in result.columns
    ]
    return result[keep_columns]


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {label}: {', '.join(missing)}")


def repository_from_url(value: Any) -> str:
    text = str(value).strip().rstrip("/")
    if text.endswith(".git"):
        text = text[:-4]
    if "github.com/" in text:
        return text.split("github.com/", 1)[1]
    parts = [part for part in text.replace("\\", "/").split("/") if part]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return text


def drop_duplicate_keys(
    df: pd.DataFrame,
    reason: str,
    exclusions: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    duplicated = df[df.duplicated(KEY_COLUMNS, keep=False)].copy()
    if duplicated.empty:
        return df, duplicated
    duplicate_groups = (
        duplicated.groupby(KEY_COLUMNS, dropna=False)
        .size()
        .reset_index(name="duplicate_count")
        .sort_values("duplicate_count", ascending=False)
    )
    to_exclude = duplicated[duplicated.duplicated(KEY_COLUMNS, keep="first")]
    for row in to_exclude.itertuples(index=False):
        exclusions.append(exclusion_from_row(row, reason, "Duplicate logical key; first occurrence kept."))
    return df.drop_duplicates(KEY_COLUMNS, keep="first").copy(), duplicate_groups


def add_missing_status_exclusions(merged: pd.DataFrame, exclusions: list[dict[str, Any]]) -> None:
    missing = merged[merged["_merge"].eq("left_only")]
    for row in missing.itertuples(index=False):
        exclusions.append(exclusion_from_row(row, "missing_pdg_extraction_status", "No matching row in extraction_status.csv."))


def add_non_success_exclusions(merged: pd.DataFrame, exclusions: list[dict[str, Any]]) -> None:
    mask = merged["_merge"].eq("both") & ~merged["status"].astype(str).str.upper().eq("SUCCESS")
    for row in merged[mask].itertuples(index=False):
        status = getattr(row, "status", "")
        error = getattr(row, "error", "")
        exclusions.append(exclusion_from_row(row, f"pdg_status_{status}", str(error)[:500]))


def add_metric_failure_exclusions(
    failures: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
) -> None:
    for item in failures:
        exclusions.append(
            {
                "repository": item.get("repository"),
                "commit": item.get("commit"),
                "filepath": item.get("filepath"),
                "reason": item.get("reason"),
                "details": item.get("details", ""),
            }
        )


def add_enriched_missing_metric_exclusions(df: pd.DataFrame, exclusions: list[dict[str, Any]]) -> None:
    for row in df.itertuples(index=False):
        exclusions.append(exclusion_from_row(row, "pdg_metrics_missing_after_merge", "Graph metrics were not available after merge."))


def add_low_quality_exclusions(
    df: pd.DataFrame,
    exclusions: list[dict[str, Any]],
    min_nodes: int,
    min_edges: int,
) -> None:
    for row in df.itertuples(index=False):
        details = (
            f"Graph below configured quality threshold: nodes={getattr(row, 'verticesCount', None)}, "
            f"edges={getattr(row, 'edgesCount', None)}, min_nodes={min_nodes}, min_edges={min_edges}."
        )
        exclusions.append(exclusion_from_row(row, "low_quality_graph_after_metric_extraction", details))


def exclusion_from_row(row: Any, reason: str, details: str) -> dict[str, Any]:
    return {
        "repository": getattr(row, "repository", ""),
        "commit": getattr(row, "commit", ""),
        "filepath": getattr(row, "filepath", ""),
        "failure_prone": getattr(row, "failure_prone", ""),
        "status": getattr(row, "status", ""),
        "nodes": getattr(row, "nodes", ""),
        "edges": getattr(row, "edges", ""),
        "reason": reason,
        "details": details,
    }


def resolve_graph_path(value: str, graph_base_dir: str) -> Path | None:
    if not value or value.lower() == "nan":
        return None
    raw = Path(value)
    candidates = []
    if raw.exists():
        return raw
    normalized = value.replace("\\", "/")
    base = Path(graph_base_dir)
    if "/app/output/" in normalized:
        candidates.append(base / normalized.split("/app/output/", 1)[1])
    if not raw.is_absolute():
        candidates.append(raw)
        candidates.append(base / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def extract_graphml_metrics(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    key_map = parse_graphml_keys(root)
    nodes = []
    node_attrs = {}
    for node in root.findall(".//g:node", GRAPHML_NS):
        node_id = str(node.attrib["id"])
        attrs = data_attrs(node, key_map)
        nodes.append(node_id)
        node_attrs[node_id] = attrs

    edges = []
    edge_labels = []
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    reverse_adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for edge in root.findall(".//g:edge", GRAPHML_NS):
        source = str(edge.attrib["source"])
        target = str(edge.attrib["target"])
        attrs = data_attrs(edge, key_map)
        label = clean_label(attrs.get("label", ""))
        edges.append((source, target, label))
        edge_labels.append(label)
        adjacency.setdefault(source, set()).add(target)
        reverse_adjacency.setdefault(target, set()).add(source)
        adjacency.setdefault(target, set())
        reverse_adjacency.setdefault(source, set())

    task_nodes = [node for node in nodes if is_task_node(node_attrs.get(node, {}))]
    if not task_nodes and nodes:
        task_nodes = [node for node in nodes if adjacency.get(node) or reverse_adjacency.get(node)]

    vertices_count = len(nodes)
    edges_count = len(edges)
    direct_fan_in = sum(len(reverse_adjacency.get(node, set())) for node in task_nodes)
    direct_fan_out = sum(len(adjacency.get(node, set())) for node in task_nodes)
    indirect_fan_in = sum(max(0, len(reachable(node, reverse_adjacency)) - len(reverse_adjacency.get(node, set()))) for node in task_nodes)
    indirect_fan_out = sum(max(0, len(reachable(node, adjacency)) - len(adjacency.get(node, set()))) for node in task_nodes)
    global_input = count_global_inputs(nodes, task_nodes, adjacency, reverse_adjacency)
    global_output = count_global_outputs(nodes, task_nodes, adjacency, reverse_adjacency)
    lack_of_cohesion = task_lack_of_cohesion(task_nodes, adjacency, reverse_adjacency)

    return {
        "maxPdgVertices": vertices_count,
        "lackOfCohesion": lack_of_cohesion,
        "verticesCount": vertices_count,
        "edgesCount": edges_count,
        "edgesToVerticesRatio": safe_ratio(edges_count, vertices_count),
        "globalInput": global_input,
        "globalOutput": global_output,
        "directFanIn": direct_fan_in,
        "indirectFanIn": indirect_fan_in,
        "directFanOut": direct_fan_out,
        "indirectFanOut": indirect_fan_out,
        "pdg_task_nodes": len(task_nodes),
        "pdg_order_edges": sum(1 for label in edge_labels if label.upper() == "ORDER"),
        "pdg_def_edges": sum(1 for label in edge_labels if label.upper() == "DEF"),
        "pdg_use_edges": sum(1 for label in edge_labels if label.upper() == "USE"),
        "pdg_unique_edge_labels": len(set(edge_labels)),
    }


def parse_graphml_keys(root: ET.Element) -> dict[str, str]:
    result = {}
    for key in root.findall("g:key", GRAPHML_NS):
        result[str(key.attrib.get("id"))] = str(key.attrib.get("attr.name", ""))
    return result


def data_attrs(element: ET.Element, key_map: dict[str, str]) -> dict[str, str]:
    attrs = {}
    for data in element.findall("g:data", GRAPHML_NS):
        key = str(data.attrib.get("key", ""))
        name = key_map.get(key, key)
        attrs[name] = data.text or ""
    return attrs


def is_task_node(attrs: dict[str, str]) -> bool:
    label = attrs.get("label", "")
    shape = attrs.get("shape", "")
    return shape == "ellipse" or "<<B>" in label or "<B>" in label


def clean_label(value: str) -> str:
    return str(value).strip().strip('"')


def reachable(start: str, adjacency: dict[str, set[str]]) -> set[str]:
    seen = set()
    queue = deque(adjacency.get(start, set()))
    while queue:
        node = queue.popleft()
        if node in seen or node == start:
            continue
        seen.add(node)
        queue.extend(adjacency.get(node, set()) - seen)
    return seen


def count_global_inputs(
    nodes: list[str],
    task_nodes: list[str],
    adjacency: dict[str, set[str]],
    reverse_adjacency: dict[str, set[str]],
) -> int:
    task_set = set(task_nodes)
    count = 0
    for node in nodes:
        if node in task_set:
            continue
        if reverse_adjacency.get(node):
            continue
        if reachable(node, adjacency) & task_set:
            count += 1
    return count


def count_global_outputs(
    nodes: list[str],
    task_nodes: list[str],
    adjacency: dict[str, set[str]],
    reverse_adjacency: dict[str, set[str]],
) -> int:
    task_set = set(task_nodes)
    count = 0
    for node in nodes:
        if node in task_set:
            continue
        if adjacency.get(node):
            continue
        if reachable(node, reverse_adjacency) & task_set:
            count += 1
    return count


def task_lack_of_cohesion(
    task_nodes: list[str],
    adjacency: dict[str, set[str]],
    reverse_adjacency: dict[str, set[str]],
) -> float:
    if len(task_nodes) < 2:
        return 0.0
    connected_pairs = 0
    total_pairs = 0
    undirected = defaultdict(set)
    for source, targets in adjacency.items():
        for target in targets:
            undirected[source].add(target)
            undirected[target].add(source)
    for index, left in enumerate(task_nodes):
        left_reachable = reachable(left, undirected)
        for right in task_nodes[index + 1 :]:
            total_pairs += 1
            if right in left_reachable:
                connected_pairs += 1
    return 1.0 - safe_ratio(connected_pairs, total_pairs)


def build_quality_tables(
    final: pd.DataFrame,
    merged: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "final_label_distribution.csv": final["failure_prone"].value_counts(dropna=False).rename_axis("failure_prone").reset_index(name="rows"),
        "final_repository_distribution.csv": final.groupby("repository", dropna=False).size().reset_index(name="rows").sort_values("rows", ascending=False),
        "final_status_distribution.csv": merged["status"].fillna("MISSING_STATUS").value_counts(dropna=False).rename_axis("status").reset_index(name="rows"),
        "exclusion_reason_distribution.csv": exclusions["reason"].value_counts(dropna=False).rename_axis("reason").reset_index(name="rows") if not exclusions.empty else pd.DataFrame(columns=["reason", "rows"]),
        "pdg_metric_summary.csv": final[PDG_METRIC_COLUMNS + ["pdg_task_nodes", "pdg_order_edges", "pdg_def_edges", "pdg_use_edges"]].describe().reset_index(),
    }


def build_quality_summary(
    final: pd.DataFrame,
    merged: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "final_rows": int(len(final)),
        "final_columns": int(len(final.columns)),
        "final_repositories": int(final["repository"].nunique(dropna=True)),
        "final_label_distribution": label_distribution(final),
        "status_distribution_before_filter": {
            str(key): int(value)
            for key, value in merged["status"].fillna("MISSING_STATUS").value_counts(dropna=False).items()
        },
        "exclusion_reasons": exclusion_reason_counts(exclusions),
        "pdg_metrics": {
            column: {
                "min": safe_float(pd.to_numeric(final[column], errors="coerce").min()),
                "median": safe_float(pd.to_numeric(final[column], errors="coerce").median()),
                "max": safe_float(pd.to_numeric(final[column], errors="coerce").max()),
            }
            for column in PDG_METRIC_COLUMNS
        },
    }


def write_report(path: Path, manifest: dict[str, Any], final: pd.DataFrame, merged: pd.DataFrame, exclusions: pd.DataFrame) -> None:
    counts = manifest["stage_counts"]
    reasons = manifest["exclusion_reasons"]
    lines = [
        f"# Dataset Report - {manifest['dataset_id']} {manifest['version']}",
        "",
        f"Generated at: {manifest['generated_at']}",
        "",
        "## Inputs",
        "",
        f"- RADON input: `{manifest['inputs']['radon_input']}`",
        f"- PDG extraction status: `{manifest['inputs']['extraction_status']}`",
        f"- Graph base directory: `{manifest['inputs']['graph_base_dir']}`",
        "",
        "## Versioned Outputs",
        "",
        f"- Final dataset: `{manifest['outputs']['final_dataset']}`",
        f"- Exclusions: `{manifest['outputs']['exclusions']}`",
        f"- PDG metrics: `{manifest['outputs']['pdg_metrics']}`",
        f"- Reports: `{manifest['outputs']['reports_dir']}`",
        "",
        "## Filtering Story",
        "",
        f"- Rows after RADON filtering: {counts['radon_rows']}",
        f"- Rows with a successful PDG extraction: {counts['rows_after_successful_pdg_filter']}",
        f"- Rows with extracted PDG metrics: {counts['rows_with_extracted_pdg_metrics']}",
        f"- Final rows after graph-quality filtering: {counts['final_rows']}",
        f"- Final repositories: {counts['final_repositories']}",
        f"- Final label distribution: {counts['final_label_distribution']}",
        "",
        "## Exclusion Reasons",
        "",
    ]
    if reasons:
        lines.extend(f"- `{reason}`: {count}" for reason, count in reasons.items())
    else:
        lines.append("- No exclusions.")
    lines.extend(
        [
            "",
            "## Graph Quality Policy",
            "",
            f"- Minimum nodes: {manifest['parameters']['min_pdg_nodes']}",
            f"- Minimum edges: {manifest['parameters']['min_pdg_edges']}",
            "- Rationale: a graph used by a message-passing GNN must contain nodes and connectivity. "
            "PyTorch Geometric represents graph connectivity through `edge_index`; DGL describes graph classification "
            "as message passing over nodes/edges followed by graph-level readout. Empty graphs, edgeless graphs, "
            "and tiny placeholder graphs do not provide meaningful dependence structure for this study.",
            "- Online references checked for this policy: "
            "PyTorch Geometric data/isolated-node documentation "
            "(https://pytorch-geometric.readthedocs.io/en/1.3.0/modules/data.html), "
            "DGL message passing documentation "
            "(https://www.dgl.ai/dgl_docs/guide/message.html), and NetworkX empty/null graph definitions "
            "(https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.is_empty.html).",
            "- The selected threshold matches the PDG extraction run configuration and acts as a conservative technical filter: "
            "it excludes empty or placeholder outputs without removing small but valid Ansible task graphs.",
            "",
            "## PDG Metric Semantics",
            "",
            "- The dataset includes the 11 PDG metric columns used in the Iuliano/Pontillo line of work.",
            "- `verticesCount`, `edgesCount`, and `edgesToVerticesRatio` are directly measured on the file-level GraphML.",
            "- `directFanIn`, `directFanOut`, `indirectFanIn`, and `indirectFanOut` are computed from direct and transitive graph reachability around task nodes.",
            "- `globalInput` and `globalOutput` are file-level proxies based on non-task source/sink nodes connected to tasks.",
            "- `maxPdgVertices` is equal to the file-level graph size because the final artifact stores one graph per file snapshot.",
            "- `lackOfCohesion` is a normalized file-level task connectivity proxy; exact task-slice overlap is not reconstructable from the current GraphML alone.",
            "- The column `pdg_metric_semantics=file_level_proxy_v1` marks these semantics explicitly.",
            "",
            "## Dataset Checks",
            "",
            f"- Duplicate RADON key rows removed after first occurrence: {counts['radon_rows'] - counts['radon_rows_after_duplicate_key_filter']}",
            f"- Duplicate status key rows removed after first occurrence: {counts['status_rows'] - counts['status_rows_after_duplicate_key_filter']}",
            f"- Final unique keys: {final[KEY_COLUMNS].drop_duplicates().shape[0]}",
            f"- Final rows with missing label: {int(final['failure_prone'].isna().sum())}",
            f"- Status distribution before filtering: {value_counts_dict(merged['status'].fillna('MISSING_STATUS'))}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def label_distribution(df: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in df["failure_prone"].value_counts(dropna=False).items()}


def exclusion_reason_counts(exclusions: pd.DataFrame) -> dict[str, int]:
    if exclusions.empty or "reason" not in exclusions.columns:
        return {}
    return {str(key): int(value) for key, value in exclusions["reason"].value_counts(dropna=False).items()}


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if math.isinf(result) or math.isnan(result):
        return None
    return result


if __name__ == "__main__":
    sys.exit(main())
