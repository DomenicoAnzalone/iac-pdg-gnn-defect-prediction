"""
Combined Repository-level + File-level PDG Extractor

Questo script integra l'estrazione del PDG a livello di repository
e la successiva estrazione del sottografo a livello di file.

Processo:
1. Estrae il PDG repository-level usando scansible
2. Salva il repo-level PDG in output/pdg_repo_level
3. Estrae il sottografo file-level dal repo-level
4. Salva il file-level PDG in output/pdg_file_level_from_repo

Output:
- Repo-level: output/pdg_repo_level/{repository}/{commit}/PDG_REPO_LEVEL/pdg.graphml
- File-level: output/pdg_file_level_from_repo/{repository}/{commit}/{filepath}/PDG_FILE_LEVEL/pdg.graphml
"""

import csv
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path, PurePosixPath

import networkx as nx

import change_commit as change_commit


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT_DIR / "input" / "ansible_core_features.csv"
REPOS_ROOT = ROOT_DIR / "input" / "repositories"

# Output paths
OUTPUT_REPORT = ROOT_DIR / "output" / "extraction_combined_report.txt"
OUTPUT_STATUS_CSV = ROOT_DIR / "output" / "extraction_combined_status.csv"
OUTPUT_PDG_REPO = ROOT_DIR / "output" / "pdg_repo_level"
OUTPUT_PDG_FILE = ROOT_DIR / "output" / "pdg_file_level_from_repo"


def main() -> None:
    report_path = OUTPUT_REPORT
    initialize_report(report_path)
    initialize_status_csv(OUTPUT_STATUS_CSV)

    rows = load_dataset(INPUT_CSV)
    last_processed_index = get_last_processed_row_index(OUTPUT_STATUS_CSV)
    start_index = last_processed_index + 1

    success_count = 0
    failure_count = 0

    print(f"Processing {len(rows)} rows from {INPUT_CSV}")
    print(f"Resuming from dataset row_index {start_index}")

    for idx, row in enumerate(rows, start=1):
        if idx < start_index:
            continue

        repository = row.get("repository")
        commit = row.get("commit")
        filepath = row.get("filepath")
        failure_prone = row.get("failure_prone")

        if not repository or not commit or not filepath:
            failure_count += 1
            message = f"ROW {idx}: Missing required field repository/commit/filepath"
            append_report(report_path, message)
            print(message)
            continue

        print(f"[{idx}/{len(rows)}] {repository}@{commit} -> {filepath}")

        try:
            local_repository = normalize_repository_name(repository)
            repo_path = REPOS_ROOT / local_repository
            if not repo_path.exists():
                failure_count += 1
                message = f"ROW {idx}: repository not found: {repository} (tried local name {local_repository})"
                append_report(report_path, message)
                continue

            if not checkout_commit(local_repository, commit):
                failure_count += 1
                message = f"ROW {idx}: checkout failed for {repository}@{commit}"
                append_report(report_path, message)
                continue

            normalized_filepath = normalize_filepath(filepath)
            if normalized_filepath is None:
                failure_count += 1
                message = f"ROW {idx}: invalid filepath value: {filepath}"
                append_report(report_path, message)
                continue

            normalized_str = normalized_filepath.as_posix()

            # Skip unsupported files
            if normalized_str.startswith("meta/"):
                failure_count += 1
                message = f"ROW {idx}: skipped unsupported meta file: {filepath}"
                append_report(report_path, message)
                append_status_csv(OUTPUT_STATUS_CSV, idx, repository, commit, filepath, 
                                failure_prone, "UNSUPPORTED_FILE_TYPE", error="meta file")
                print(message)
                continue

            if normalized_str.startswith("handlers/"):
                failure_count += 1
                message = f"ROW {idx}: skipped unsupported handlers file: {filepath}"
                append_report(report_path, message)
                append_status_csv(OUTPUT_STATUS_CSV, idx, repository, commit, filepath,
                                failure_prone, "UNSUPPORTED_FILE_TYPE", error="handlers file")
                print(message)
                continue

            target_file = repo_path / normalized_filepath
            if not target_file.exists():
                failure_count += 1
                message = f"ROW {idx}: file not found at commit {commit}: {filepath}"
                append_report(report_path, message)
                continue

            # Step 1: Extract or load repository-level PDG as GraphML
            repo_output_dir = output_path_for_repo(repository, commit)
            repo_output_dir.mkdir(parents=True, exist_ok=True)
            repo_graphml_path = repo_output_dir / "pdg.graphml"

            if not repo_graphml_path.exists():
                success, stderr = extract_repo_pdg(repo_path, repo_graphml_path)
                if not success:
                    failure_count += 1
                    message = f"ROW {idx}: scansible build-pdg repo-level failed: {extract_short_error(stderr)}"
                    append_report(report_path, message)
                    append_status_csv(OUTPUT_STATUS_CSV, idx, repository, commit, filepath,
                                    failure_prone, "REPO_EXTRACTION_FAILURE", error=extract_short_error(stderr))
                    continue

            repo_graph = load_and_sanitize_graph(repo_graphml_path)
            if repo_graph.number_of_nodes() == 0 or repo_graph.number_of_edges() == 0:
                failure_count += 1
                message = f"ROW {idx}: empty repo-level PDG generated"
                append_report(report_path, message)
                append_status_csv(OUTPUT_STATUS_CSV, idx, repository, commit, filepath,
                                failure_prone, "EMPTY_REPO_GRAPH")
                continue

            # Step 2: Extract file-level PDG from repository-level
            file_output_dir = output_path_for_file(repository, commit, normalized_filepath)
            file_output_dir.mkdir(parents=True, exist_ok=True)
            file_graphml_path = file_output_dir / "pdg.graphml"
            file_dot_path = file_output_dir / "pdg.dot"

            if not file_graphml_path.exists():
                try:
                    file_graph = extract_file_level_from_repo(
                        repo_graph, normalized_str, str(repo_path)
                    )

                    if file_graph.number_of_nodes() == 0 or file_graph.number_of_edges() == 0:
                        failure_count += 1
                        message = f"ROW {idx}: empty file-level PDG extracted from repo-level"
                        append_report(report_path, message)
                        append_status_csv(OUTPUT_STATUS_CSV, idx, repository, commit, filepath,
                                        failure_prone, "REPO_SUCCESS_FILE_EMPTY",
                                        repo_nodes=repo_graph.number_of_nodes(),
                                        repo_edges=repo_graph.number_of_edges(),
                                        file_nodes=0,
                                        file_edges=0,
                                        repo_graphml_path=str(repo_graphml_path))
                        continue

                    save_graphml(file_graph, file_graphml_path)
                    try:
                        write_dot(file_graph, str(file_dot_path))
                    except Exception:
                        pass  # DOT writing is optional

                    success_count += 1
                    message = f"ROW {idx}: extracted file-level PDG from repo-level"
                    append_status_csv(OUTPUT_STATUS_CSV, idx, repository, commit, filepath,
                                    failure_prone, "SUCCESS",
                                    repo_nodes=repo_graph.number_of_nodes(),
                                    repo_edges=repo_graph.number_of_edges(),
                                    file_nodes=file_graph.number_of_nodes(),
                                    file_edges=file_graph.number_of_edges(),
                                    repo_graphml_path=str(repo_graphml_path),
                                    file_graphml_path=str(file_graphml_path))
                    append_report(report_path, message)
                    print(message)

                except Exception as exc:
                    failure_count += 1
                    message = f"ROW {idx}: file-level extraction failed: {exc}"
                    append_report(report_path, message)
                    append_status_csv(OUTPUT_STATUS_CSV, idx, repository, commit, filepath,
                                    failure_prone, "FILE_EXTRACTION_FAILURE",
                                    repo_nodes=repo_graph.number_of_nodes(),
                                    repo_edges=repo_graph.number_of_edges(),
                                    repo_graphml_path=str(repo_graphml_path),
                                    error=str(exc)[:500])
                    continue
            else:
                success_count += 1
                message = f"ROW {idx}: file-level PDG already processed"
                append_report(report_path, message)
                print(message)

        except Exception as exc:
            failure_count += 1
            tb = traceback.format_exc()
            message = f"ROW {idx}: extraction failed: {exc}\n{tb}"
            append_report(report_path, message)
            print(f"ROW {idx}: failed: {exc}")
            continue

    summary = (
        "\n"
        "================ FINAL SUMMARY ================\n"
        f"Successful extractions: {success_count}\n"
        f"Failed rows           : {failure_count}\n"
        f"Total processed       : {success_count + failure_count}\n"
        "================================================\n"
    )
    append_report(report_path, summary)
    print(summary)


def extract_file_level_from_repo(
    repo_graph: nx.Graph, target_file_rel: str, repo_root: str
) -> nx.Graph:
    """Extract file-level subgraph from repository-level graph."""

    # Find all task nodes in the target file
    task_nodes = get_task_nodes_for_file(repo_graph, target_file_rel, repo_root)
    if not task_nodes:
        raise ValueError(
            f"No task nodes found for file '{target_file_rel}' in repository-level PDG"
        )

    # Extract subgraphs for each task and merge them
    task_subgraphs = []
    for task_id in task_nodes:
        subgraph = extract_task_subgraph(repo_graph, task_id)
        if subgraph.number_of_nodes() > 0:
            task_subgraphs.append(subgraph)

    if not task_subgraphs:
        return nx.DiGraph()

    merged = merge_subgraphs(task_subgraphs)

    # Remove tasks from other files
    merged = remove_external_tasks(merged, target_file_rel, repo_root)

    return merged


def get_task_nodes_for_file(
    graph: nx.Graph, target_file: str, repo_root: str
) -> list:
    """Find all task nodes belonging to a specific file."""
    task_nodes = []

    for node, data in graph.nodes(data=True):
        if str(data.get("node_type", "")) != "Task":
            continue

        location_value = data.get("location")
        location_file = parse_location(location_value)
        if location_file is None:
            continue

        location_file_norm = normalize_location_file(location_file, repo_root)
        if path_matches_file(location_file_norm, target_file):
            task_nodes.append(node)

    return task_nodes


def extract_task_subgraph(graph: nx.Graph, task_node_id) -> nx.Graph:
    """Extract subgraph centered on a task node (within distance 3)."""
    marked = mark_vertices_distance_3(graph, {task_node_id})
    subgraph = graph.subgraph(marked).copy()
    return subgraph


def mark_vertices_distance_3(graph: nx.Graph, start_nodes: set) -> set:
    """Mark vertices within distance 3 from start nodes."""
    marked = set()
    worklist = set(start_nodes)

    while worklist:
        node = worklist.pop()
        marked.add(node)

        # Explore predecessors and successors
        for neighbor in list(graph.predecessors(node)) + list(graph.successors(node)):
            if neighbor not in marked:
                edge_data = graph.get_edge_data(node, neighbor) or graph.get_edge_data(neighbor, node)
                if edge_data and edge_data.get("type", "") in ("ORDER", "ORDER_TRANS", "ORDER_BACK"):
                    marked.add(neighbor)
                else:
                    worklist.add(neighbor)

    return marked


def remove_external_tasks(
    graph: nx.Graph, target_file: str, repo_root: str
) -> nx.Graph:
    """Remove task nodes that don't belong to the target file."""
    nodes_to_remove = []

    for node, data in graph.nodes(data=True):
        if str(data.get("node_type", "")) != "Task":
            continue

        location_value = data.get("location")
        location_file = parse_location(location_value)
        if location_file is None:
            nodes_to_remove.append(node)
            continue

        location_file_norm = normalize_location_file(location_file, repo_root)
        if not path_matches_file(location_file_norm, target_file):
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        graph.remove_node(node)

    return graph


def merge_subgraphs(subgraphs: list) -> nx.Graph:
    """Merge multiple subgraphs into one."""
    if not subgraphs:
        return nx.DiGraph()

    merged = subgraphs[0].__class__()
    merged.graph.update(subgraphs[0].graph)

    for subgraph in subgraphs:
        for node, data in subgraph.nodes(data=True):
            if not merged.has_node(node):
                merged.add_node(node, **data)

        for u, v, edge_data in subgraph.edges(data=True):
            if not merged.has_edge(u, v):
                merged.add_edge(u, v, **edge_data)

    return merged


def parse_location(location_value):
    """Parse location JSON to extract file path."""
    if location_value is None:
        return None
    if isinstance(location_value, str):
        try:
            location_value = json.loads(location_value)
        except (ValueError, TypeError):
            pass
    if isinstance(location_value, dict):
        return location_value.get("file")
    return None


def normalize_location_file(location_file: str, repo_root: str = None) -> str:
    """Normalize file path from location."""
    if location_file is None:
        return None
    location_file = str(location_file).replace("\\", "/")
    if repo_root is not None:
        repo_root_norm = os.path.abspath(repo_root).replace("\\", "/")
        if location_file.startswith(repo_root_norm):
            location_file = location_file[len(repo_root_norm) :].lstrip("/")
    location_file = os.path.normpath(location_file).replace("\\", "/")
    return location_file


def path_matches_file(location_file: str, target_file: str) -> bool:
    """Check if location file matches target file."""
    if location_file is None or target_file is None:
        return False

    location_norm = location_file.replace("\\", "/")
    target_norm = target_file.replace("\\", "/")

    if location_norm == target_norm:
        return True

    location_parts = location_norm.split("/")
    target_parts = target_norm.split("/")

    if len(target_parts) == 0:
        return False
    if len(location_parts) >= len(target_parts):
        if tuple(location_parts[-len(target_parts) :]) == tuple(target_parts):
            return True

    return False


def extract_repo_pdg(repo_path: Path, output_graphml_path: Path) -> tuple[bool, str]:
    """Extract repository-level PDG using scansible and save GraphML output."""
    command = f"scansible build-pdg -f graphml {repo_path}"

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
    )

    if result.returncode == 0:
        output_graphml_path.write_text(result.stdout, encoding="utf-8")
        return True, ""
    else:
        return False, result.stderr


def load_and_sanitize_graph(path: Path) -> nx.Graph:
    """Load graph from GraphML and preserve node metadata."""
    return nx.read_graphml(str(path))


def sanitize_graph_for_graphml(G: nx.Graph) -> nx.Graph:
    """Sanitize graph for GraphML export."""
    if G.is_multigraph():
        H = nx.MultiDiGraph()
    else:
        H = nx.DiGraph()

    for node, attrs in G.nodes(data=True):
        clean_attrs = {str(k): str(v) for k, v in attrs.items()}
        H.add_node(str(node), **clean_attrs)

    for source, target, attrs in G.edges(data=True):
        clean_attrs = {str(k): str(v) for k, v in attrs.items()}
        H.add_edge(str(source), str(target), **clean_attrs)

    return H


def save_graphml(G: nx.Graph, path: Path) -> None:
    """Save graph to GraphML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(path))


def write_dot(G: nx.Graph, path: str) -> None:
    """Write graph to DOT format."""
    try:
        nx.drawing.nx_pydot.write_dot(G, path)
    except Exception:
        try:
            nx.drawing.nx_agraph.write_dot(G, path)
        except Exception as exc:
            raise RuntimeError(f"Unable to write DOT. Install pydot or pygraphviz. {exc}")


def load_dataset(csv_path: Path) -> list:
    """Load dataset from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


def get_last_processed_row_index(csv_path: Path) -> int:
    """Get the last processed row index from status CSV."""
    if not csv_path.exists():
        return 0

    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if rows:
                return int(rows[-1].get("row_index", 0))
    except Exception:
        pass

    return 0


def initialize_report(path: Path) -> None:
    """Initialize report file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def append_report(path: Path, message: str) -> None:
    """Append message to report."""
    with path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def initialize_status_csv(path: Path) -> None:
    """Initialize status CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row_index",
            "repository",
            "commit",
            "filepath",
            "failure_prone",
            "status",
            "repo_nodes",
            "repo_edges",
            "file_nodes",
            "file_edges",
            "repo_graphml_path",
            "file_graphml_path",
            "error",
        ])


def append_status_csv(
    path: Path,
    row_index: int,
    repository: str,
    commit: str,
    filepath: str,
    failure_prone: str,
    status: str,
    repo_nodes: int = 0,
    repo_edges: int = 0,
    file_nodes: int = 0,
    file_edges: int = 0,
    repo_graphml_path: str = "",
    file_graphml_path: str = "",
    error: str = "",
) -> None:
    """Append row to status CSV."""
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            row_index,
            repository,
            commit,
            filepath,
            failure_prone,
            status,
            repo_nodes,
            repo_edges,
            file_nodes,
            file_edges,
            repo_graphml_path,
            file_graphml_path,
            error,
        ])


def checkout_commit(repository: str, commit: str) -> bool:
    """Checkout repository to specific commit."""
    try:
        return change_commit.checkout_repository(repositoryName=repository, commit=commit)
    except Exception:
        return False


def normalize_filepath(filepath: str) -> Path:
    """Normalize file path."""
    try:
        posix_path = PurePosixPath(filepath)
        parts = [p for p in posix_path.parts if p not in ("/", "\\", ".", "..")]
        if not parts:
            return None
        return Path(*parts)
    except Exception:
        return None


def normalize_repository_name(repository: str) -> str:
    """Normalize repository name."""
    repository = repository.strip()
    if "/" in repository:
        repository = repository.rsplit("/", 1)[-1]
    return repository


def output_path_for_repo(repository: str, commit: str) -> Path:
    """Get output path for repository-level PDG."""
    return OUTPUT_PDG_REPO / repository / commit / "PDG_REPO_LEVEL"


def output_path_for_file(repository: str, commit: str, filepath: Path) -> Path:
    """Get output path for file-level PDG."""
    safe_parts = [p for p in filepath.parts if p not in (".", "..")]
    return OUTPUT_PDG_FILE / repository / commit / Path(*safe_parts) / "PDG_FILE_LEVEL"


def extract_short_error(stderr: str) -> str:
    """Extract short error message from stderr."""
    if not stderr:
        return ""

    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if lines:
        return lines[-1][:500]
    return ""


if __name__ == "__main__":
    main()
