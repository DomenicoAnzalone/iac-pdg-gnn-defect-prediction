import argparse
import json
import os
from pathlib import Path

import networkx as nx

import dictionary_file_tasknode as df
import extract_task_subgraph as es
import project_pdg_info as pi
import writer_reader as wr


def resolve_target_file(repo_root: str, target_file: str) -> str:
    repo_root = os.path.abspath(repo_root)
    target_path = Path(target_file)
    if target_path.is_absolute():
        normalized_target = os.path.normpath(str(target_path))
        if not os.path.commonpath([repo_root, normalized_target]) == repo_root:
            raise ValueError(f"Target file is not inside repository root: {target_file}")
        target_rel = os.path.relpath(normalized_target, repo_root)
    else:
        combined = os.path.normpath(os.path.join(repo_root, target_file))
        target_rel = os.path.relpath(combined, repo_root)
    return target_rel.replace("\\", "/")


def merge_subgraphs(subgraphs):
    if not subgraphs:
        raise ValueError("No subgraphs to merge")

    merged = subgraphs[0].__class__()
    merged.graph.update(subgraphs[0].graph)

    for subgraph in subgraphs:
        merged.graph.update(subgraph.graph)

        for node, data in subgraph.nodes(data=True):
            if merged.has_node(node):
                merged.nodes[node].update({k: v for k, v in data.items() if k not in merged.nodes[node]})
            else:
                merged.add_node(node, **data)

        if subgraph.is_multigraph():
            for u, v, key, edge_data in subgraph.edges(keys=True, data=True):
                if merged.has_edge(u, v, key):
                    continue
                merged.add_edge(u, v, key=key, **edge_data)
        else:
            for u, v, edge_data in subgraph.edges(data=True):
                if merged.has_edge(u, v):
                    continue
                merged.add_edge(u, v, **edge_data)

    return merged


def remove_external_tasks(merged_graph, target_file, repo_root):
    task_nodes_to_remove = []
    for node, data in merged_graph.nodes(data=True):
        if str(data.get("node_type", "")) != "Task":
            continue
        location_file = df.parse_location(data.get("location"))
        location_file_norm = df.normalize_location_file(location_file, repo_root)
        if not df.path_matches_file(location_file_norm, target_file):
            task_nodes_to_remove.append(node)

    if task_nodes_to_remove:
        for node in task_nodes_to_remove:
            merged_graph.remove_node(node)
    return merged_graph


def validate_tasks(merged_graph, target_file, repo_root):
    invalid_tasks = []
    for node, data in merged_graph.nodes(data=True):
        if str(data.get("node_type", "")) != "Task":
            continue
        location_file = df.parse_location(data.get("location"))
        location_file_norm = df.normalize_location_file(location_file, repo_root)
        if not df.path_matches_file(location_file_norm, target_file):
            invalid_tasks.append((node, location_file))
    return invalid_tasks


def write_dot(graph, path):
    try:
        nx.drawing.nx_pydot.write_dot(graph, path)
        return
    except Exception:
        try:
            nx.drawing.nx_agraph.write_dot(graph, path)
            return
        except Exception as exc:
            raise RuntimeError(
                "Unable to write DOT output. Install pydot or pygraphviz."
                f" Details: {exc}"
            ) from exc


def build_file_level_pdg(repo_root: str, target_file: str, output_dir: str):
    target_file_rel = resolve_target_file(repo_root, target_file)
    repo_graph = pi.getPDG(repo_root)

    task_nodes = df.get_task_nodes_for_file(repo_graph, target_file_rel, repo_root)
    if not task_nodes:
        raise RuntimeError(
            f"No task nodes found for file '{target_file_rel}' in repository-level PDG"
        )

    task_subgraphs = [es.getPDG_tasklevel(repo_graph, task_id) for task_id in task_nodes]
    merged = merge_subgraphs(task_subgraphs)
    merged.graph["file_level_source"] = target_file_rel
    merged.graph["repository_root"] = os.path.abspath(repo_root)
    merged.graph["task_node_ids"] = list(task_nodes)

    merged = remove_external_tasks(merged, target_file_rel, repo_root)
    invalid_tasks = validate_tasks(merged, target_file_rel, repo_root)
    if invalid_tasks:
        raise RuntimeError(
            f"Validation failed: found {len(invalid_tasks)} task nodes outside target file"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graphml_path = output_dir / "file_level.graphml"
    dot_path = output_dir / "file_level.dot"

    wr.write_graphml(merged, str(graphml_path))
    write_dot(merged, str(dot_path))

    return {
        "graphml": str(graphml_path),
        "dot": str(dot_path),
        "task_nodes": task_nodes,
        "node_count": merged.number_of_nodes(),
        "edge_count": merged.number_of_edges(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a file-level PDG from a repository-level Scansible GraphML PDG."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Path to the root of the Ansible repository containing PDG/graphml.txt",
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Target Ansible YAML file path inside the repository",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory for file_level.graphml and file_level.dot",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_file_level_pdg(args.repo, args.file, args.output)
    print("File-level PDG generated successfully")
    print(f"GraphML: {result['graphml']}")
    print(f"DOT: {result['dot']}")
    print(f"Task nodes included: {len(result['task_nodes'])}")
    print(f"Node count: {result['node_count']}")
    print(f"Edge count: {result['edge_count']}")


if __name__ == "__main__":
    main()
