import csv
import subprocess
import traceback
from pathlib import Path, PurePosixPath

import networkx as nx

import change_commit as change_commit


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT_DIR / "input" / "ansible_core_features.csv"
REPOS_ROOT = ROOT_DIR / "input" / "repositories"
OUTPUT_REPORT = ROOT_DIR / "output" / "extraction_repo_report.txt"
OUTPUT_STATUS_CSV = ROOT_DIR / "output" / "extraction_repo_level_status.csv"
OUTPUT_PDG_ROOT = ROOT_DIR / "output" / "pdg_repo_level"

# Path to the file-level extraction status CSV for lookup
FILE_LEVEL_STATUS_CSV = ROOT_DIR / "output" / "extraction_status.csv"


def main() -> None:
    report_path = OUTPUT_REPORT
    initialize_report(report_path)
    initialize_status_csv(OUTPUT_STATUS_CSV)

    # Load file-level status for reference
    file_level_status_lookup = load_file_level_status(FILE_LEVEL_STATUS_CSV)

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

        print(f"[{idx}/{len(rows)}] {repository}@{commit} (file={filepath})")

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
                message = f"ROW {idx}: checkout failed for {repository}@{commit} (local {local_repository})"
                append_report(report_path, message)
                continue

            normalized_filepath = normalize_filepath(filepath)
            if normalized_filepath is None:
                failure_count += 1
                message = f"ROW {idx}: invalid filepath value: {filepath}"
                append_report(report_path, message)
                continue

            normalized_str = normalized_filepath.as_posix()

            # Skip meta files
            if normalized_str.startswith("meta/"):
                failure_count += 1
                message = (
                    f"ROW {idx}: skipped unsupported meta file: "
                    f"{filepath}"
                )
                append_report(report_path, message)
                file_level_status = get_file_level_status(
                    file_level_status_lookup, repository, commit, filepath
                )
                append_status_csv(
                    OUTPUT_STATUS_CSV,
                    idx,
                    repository,
                    commit,
                    filepath,
                    failure_prone,
                    "UNSUPPORTED_FILE_TYPE",
                    file_level_status=file_level_status,
                    error="meta file",
                )
                print(message)
                continue

            # Skip handlers files
            if normalized_str.startswith("handlers/"):
                failure_count += 1
                message = (
                    f"ROW {idx}: skipped unsupported handlers file: "
                    f"{filepath}"
                )
                append_report(report_path, message)
                file_level_status = get_file_level_status(
                    file_level_status_lookup, repository, commit, filepath
                )
                append_status_csv(
                    OUTPUT_STATUS_CSV,
                    idx,
                    repository,
                    commit,
                    filepath,
                    failure_prone,
                    "UNSUPPORTED_FILE_TYPE",
                    file_level_status=file_level_status,
                    error="handlers file",
                )
                print(message)
                continue

            target_file = repo_path / normalized_filepath
            if not target_file.exists():
                failure_count += 1
                message = f"ROW {idx}: file not found at commit {commit}: {filepath}"
                append_report(report_path, message)
                continue

            output_dir = output_path_for_repo(repository, commit)
            output_dir.mkdir(parents=True, exist_ok=True)

            dot_path = output_dir / "pdg.dot"
            graphml_path = output_dir / "pdg.graphml"

            # Skip already processed repos
            if graphml_path.exists():
                message = (
                    f"ROW {idx}: repository PDG already processed -> "
                    f"{graphml_path}"
                )

                append_report(report_path, message)
                print(message)

                success_count += 1
                file_level_status = get_file_level_status(
                    file_level_status_lookup, repository, commit, filepath
                )
                append_status_csv(
                    OUTPUT_STATUS_CSV,
                    idx,
                    repository,
                    commit,
                    filepath,
                    failure_prone,
                    "SUCCESS",
                    file_level_status=file_level_status,
                    nodes=0,
                    edges=0,
                    graphml_path=str(graphml_path),
                )

                continue

            success, stdout, stderr = extract_repo_pdg(
                repo_path,
                dot_path
            )

            if not success:
                failure_count += 1
                message = (
                    f"ROW {idx}: scansible build-pdg failed for "
                    f"{repository}@{commit}. "
                    f"stderr={stderr.strip()}"
                )
                append_report(report_path, message)
                file_level_status = get_file_level_status(
                    file_level_status_lookup, repository, commit, filepath
                )
                append_status_csv(
                    OUTPUT_STATUS_CSV,
                    idx,
                    repository,
                    commit,
                    filepath,
                    failure_prone,
                    "REAL_EXTRACTION_FAILURE",
                    file_level_status=file_level_status,
                    error=extract_short_error(stderr),
                )
                continue

            graph = load_and_sanitize_graph(dot_path)

            # Reject empty graphs
            if (
                graph.number_of_nodes() == 0
                or graph.number_of_edges() == 0
            ):
                failure_count += 1

                message = (
                    f"ROW {idx}: empty PDG generated for "
                    f"{repository}@{commit}"
                )

                append_report(report_path, message)
                file_level_status = get_file_level_status(
                    file_level_status_lookup, repository, commit, filepath
                )
                append_status_csv(
                    OUTPUT_STATUS_CSV,
                    idx,
                    repository,
                    commit,
                    filepath,
                    failure_prone,
                    "EMPTY_GRAPH",
                    file_level_status=file_level_status,
                    nodes=graph.number_of_nodes(),
                    edges=graph.number_of_edges(),
                )
                print(message)

                continue

            save_graphml(graph, graphml_path)

            message = (
                f"ROW {idx}: extracted repository-level PDG -> "
                f"{graphml_path}"
            )

            file_level_status = get_file_level_status(
                file_level_status_lookup, repository, commit, filepath
            )

            append_status_csv(
                OUTPUT_STATUS_CSV,
                idx,
                repository,
                commit,
                filepath,
                failure_prone,
                "SUCCESS",
                file_level_status=file_level_status,
                nodes=graph.number_of_nodes(),
                edges=graph.number_of_edges(),
                graphml_path=str(graphml_path),
            )

            append_report(report_path, message)

            success_count += 1

            print(message)

        except Exception as exc:
            failure_count += 1
            tb = traceback.format_exc()
            message = (
                f"ROW {idx}: extraction failed for {repository}@{commit} file={filepath} "
                f"reason={exc}\n{tb}"
            )
            append_report(report_path, message)
            print(f"ROW {idx}: failed: {exc}")
            continue

    summary = (
        "\n"
        "================ FINAL SUMMARY ================\n"
        f"Successful PDGs : {success_count}\n"
        f"Failed rows     : {failure_count}\n"
        f"Total processed : {success_count + failure_count}\n"
        "================================================\n"
    )
    append_report(report_path, summary)
    print(summary)

def get_last_processed_row_index(csv_path: Path) -> int:

    if not csv_path.exists():
        return 0

    with csv_path.open(newline="", encoding="utf-8") as csvfile:

        reader = csv.DictReader(csvfile)

        rows = list(reader)

        if not rows:
            return 0

        last_row = rows[-1]

        try:
            return int(last_row["row_index"])
        except Exception:
            return 0

def load_file_level_status(csv_path: Path) -> dict[tuple[str, str, str], str]:
    """
    Load file-level extraction status into a lookup dictionary.
    Key: (repository, commit, filepath)
    Value: status value from extraction_status.csv
    """
    lookup = {}

    if not csv_path.exists():
        return lookup

    try:
        with csv_path.open(newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                repo = row.get("repository", "").strip()
                commit = row.get("commit", "").strip()
                filepath = row.get("filepath", "").strip()
                status = row.get("status", "")

                if repo and commit and filepath:
                    key = (repo, commit, filepath)
                    lookup[key] = status
    except Exception:
        pass

    return lookup

def get_file_level_status(
    lookup: dict[tuple[str, str, str], str],
    repository: str,
    commit: str,
    filepath: str
) -> str:
    """
    Retrieve file-level status for a given (repository, commit, filepath).
    Returns the status string or "NULL" if not found.
    """
    key = (repository, commit, filepath)
    return lookup.get(key, "NULL")

def extract_repo_pdg(
    repo_path: Path,
    output_dot_path: Path
) -> tuple[bool, str, str]:

    command = f"scansible build-pdg -f graphviz {repo_path}"

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
    )

    if result.returncode == 0:
        output_dot_path.write_text(
            result.stdout,
            encoding="utf-8"
        )

    return (
        result.returncode == 0,
        result.stdout,
        result.stderr,
    )

def load_and_sanitize_graph(dot_path: Path) -> nx.Graph:

    if not dot_path.exists():
        raise FileNotFoundError(
            f"DOT graph not found: {dot_path}"
        )

    G = nx.nx_pydot.read_dot(str(dot_path))

    return sanitize_graph_for_graphml(G)

def load_dataset(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]


def initialize_report(report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("", encoding="utf-8")


def append_report(report_path: Path, message: str) -> None:
    with report_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")

def initialize_status_csv(csv_path: Path) -> None:

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        return

    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "row_index",
            "repository",
            "commit",
            "filepath",
            "failure_prone",
            "status",
            "file_level_status",
            "nodes",
            "edges",
            "graphml_path",
            "error",
        ])


def append_status_csv(
    csv_path: Path,
    row_index: int,
    repository: str,
    commit: str,
    filepath: str,
    failure_prone: str,
    status: str,
    file_level_status: str = "NULL",
    nodes: int = 0,
    edges: int = 0,
    graphml_path: str = "",
    error: str = "",
) -> None:

    with csv_path.open("a", newline="", encoding="utf-8") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            row_index,
            repository,
            commit,
            filepath,
            failure_prone,
            status,
            file_level_status,
            nodes,
            edges,
            graphml_path,
            error,
        ])

def checkout_commit(repository: str, commit: str) -> bool:
    try:
        return change_commit.checkout_repository(repositoryName=repository, commit=commit)
    except Exception:
        return False


def normalize_filepath(filepath: str) -> Path | None:
    try:
        posix_path = PurePosixPath(filepath)
        parts = [part for part in posix_path.parts if part not in ("/", "\\", ".", "..")]
        if not parts:
            return None
        return Path(*parts)
    except Exception:
        return None


def normalize_repository_name(repository: str) -> str:
    repository = repository.strip()
    if "/" in repository:
        repository = repository.rsplit("/", 1)[-1]
    return repository


def output_path_for_repo(repository: str, commit: str) -> Path:
    return OUTPUT_PDG_ROOT / repository / commit / "PDG_REPO_LEVEL"


def sanitize_graph_for_graphml(G: nx.Graph) -> nx.Graph:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(path))

def extract_short_error(stderr: str) -> str:

    if not stderr:
        return ""

    lines = [
        line.strip()
        for line in stderr.splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    # Keep only last meaningful line
    short_error = lines[-1]

    return short_error[:500]

if __name__ == "__main__":
    main()
