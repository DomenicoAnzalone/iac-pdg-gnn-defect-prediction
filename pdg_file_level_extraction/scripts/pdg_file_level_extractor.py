from __future__ import annotations

import atexit
import argparse
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional

import networkx as nx


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("PDG_FILE_LEVEL_OUTPUT_DIR", str(MODULE_ROOT.parent / "output"))
)

STATUS_FIELDNAMES = [
    "row_index",
    "repository",
    "commit",
    "filepath",
    "failure_prone",
    "status",
    "nodes",
    "edges",
    "graphml_path",
    "error",
]

TERMINAL_STATUSES = {
    "SUCCESS",
    "LOW_QUALITY_GRAPH",
    "MISSING_REQUIRED_FIELD",
    "REPOSITORY_NOT_FOUND",
    "CLONE_FAILURE",
    "CHECKOUT_FAILURE",
    "INVALID_FILEPATH",
    "UNSUPPORTED_FILE_TYPE",
    "FILE_NOT_FOUND",
    "EXTRACTION_TIMEOUT",
    "REAL_EXTRACTION_FAILURE",
    "EMPTY_GRAPH",
    "UNEXPECTED_ERROR",
}

active_processes: dict[str, subprocess.Popen] = {}
active_processes_lock = threading.Lock()
active_repositories: dict[str, dict[str, object]] = {}
active_repositories_lock = threading.Lock()
stop_event = threading.Event()


@dataclass(frozen=True)
class DatasetRow:
    row_index: int
    repository: str
    commit: str
    filepath: str
    failure_prone: str
    raw: dict[str, str]
    clone_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract file-level PDGs from a labeled Ansible dataset. "
            "Each repository is cloned on demand, processed sequentially, and deleted; "
            "different repositories can be processed in parallel."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input CSV containing repository or repo_url, plus commit, filepath "
            "and failure_prone columns."
        ),
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name of the run directory created under --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory that will contain extraction run directories.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum number of different repositories processed concurrently.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each scansible file extraction. Use 0 to disable.",
    )
    parser.add_argument(
        "--scansible-command",
        default="scansible",
        help="Scansible executable name or absolute path.",
    )
    parser.add_argument(
        "--keep-dot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep pdg.dot next to pdg.graphml. Default: true.",
    )
    parser.add_argument(
        "--refresh-input",
        action="store_true",
        help="Replace the run input snapshot with the current --input file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing run directory before starting.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit used for small validation runs.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=30,
        help="Minimum seconds between console progress updates. Default: 30.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress after this many newly completed rows. Default: 250.",
    )
    parser.add_argument(
        "--progress-active-repos",
        type=int,
        default=5,
        help="Maximum active repositories shown in each progress update. Default: 5.",
    )
    parser.add_argument(
        "--min-pdg-nodes",
        type=int,
        default=3,
        help=(
            "Minimum number of nodes required to mark a graph as SUCCESS. "
            "Smaller graphs are saved as LOW_QUALITY_GRAPH. Default: 3."
        ),
    )
    parser.add_argument(
        "--min-pdg-edges",
        type=int,
        default=2,
        help=(
            "Minimum number of edges required to mark a graph as SUCCESS. "
            "Smaller graphs are saved as LOW_QUALITY_GRAPH. Default: 2."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    run_dir = output_dir / args.run_name
    lock_file = run_dir / ".run.lock"

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.timeout < 0:
        raise SystemExit("--timeout cannot be negative")
    if args.max_rows is not None and args.max_rows < 1:
        raise SystemExit("--max-rows must be at least 1")
    if args.progress_interval < 1:
        raise SystemExit("--progress-interval must be at least 1")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be at least 1")
    if args.progress_active_repos < 0:
        raise SystemExit("--progress-active-repos cannot be negative")
    if args.min_pdg_nodes < 1:
        raise SystemExit("--min-pdg-nodes must be at least 1")
    if args.min_pdg_edges < 1:
        raise SystemExit("--min-pdg-edges must be at least 1")
    refuse_concurrent_run(lock_file)
    if run_dir.exists() and args.force:
        print(f"Deleting existing run directory: {run_dir}")
        shutil.rmtree(run_dir)
    elif run_dir.exists():
        print(f"Existing run directory found, resuming: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    lock_contents = acquire_run_lock(lock_file)
    atexit.register(remove_run_lock, lock_file, lock_contents)
    verify_scansible_available(args.scansible_command)
    input_snapshot = prepare_input_snapshot(
        input_file=input_file,
        run_dir=run_dir,
        refresh_input=args.refresh_input,
    )

    rows = load_dataset_rows(input_snapshot, max_rows=args.max_rows)
    status_file = run_dir / "extraction_status.csv"
    success_file = run_dir / success_filename_for_input(input_file)
    report_file = run_dir / "extraction_report.txt"
    metadata_file = run_dir / "run_metadata.json"
    logs_dir = run_dir / "logs"
    pdg_root = run_dir / "pdg_file_level"
    clone_root = run_dir / ".repositories"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pdg_root.mkdir(parents=True, exist_ok=True)
    reset_clone_root(clone_root)

    if args.refresh_input:
        for path in (status_file, success_file, report_file):
            path.unlink(missing_ok=True)

    completed = read_completed_statuses(status_file)
    pending = [row for row in rows if row.row_index not in completed]
    groups = group_rows_by_repository(pending)

    write_run_metadata(
        metadata_file=metadata_file,
        args=args,
        input_file=input_file,
        input_snapshot=input_snapshot,
        run_dir=run_dir,
        rows=rows,
        completed=completed,
        pending=pending,
        groups=groups,
    )

    print(f"Input rows: {len(rows)}")
    print(f"Already completed: {len(completed)}")
    print(f"Pending rows: {len(pending)}")
    print(f"Repository groups pending: {len(groups)}")
    print(f"Workers: {args.workers}")
    print(f"Run directory: {run_dir}")

    results = dict(completed)
    try:
        if pending:
            run_parallel_extraction(
                groups=groups,
                total_rows=len(rows),
                results=results,
                status_file=status_file,
                success_file=success_file,
                report_file=report_file,
                logs_dir=logs_dir,
                pdg_root=pdg_root,
                clone_root=clone_root,
                scansible_command=args.scansible_command,
                timeout=args.timeout,
                keep_dot=args.keep_dot,
                min_pdg_nodes=args.min_pdg_nodes,
                min_pdg_edges=args.min_pdg_edges,
                progress_interval=args.progress_interval,
                progress_every=args.progress_every,
                progress_active_repos=args.progress_active_repos,
                workers=args.workers,
            )
        else:
            write_status_outputs(status_file, success_file, results)
            write_report(report_file, rows, results)
    finally:
        remove_directory_tree(clone_root, run_dir)
        remove_run_lock(lock_file, lock_contents)

    print_final_summary(rows, results, run_dir, success_file)


def prepare_input_snapshot(input_file: Path, run_dir: Path, refresh_input: bool) -> Path:
    if not input_file.exists():
        raise SystemExit(f"Input dataset not found: {input_file}")

    snapshot = run_dir / "input_dataset.csv"
    if snapshot.exists() and not refresh_input:
        print(f"Resume input snapshot found: {snapshot}")
        return snapshot

    shutil.copyfile(input_file, snapshot)
    print(f"Input snapshot saved to: {snapshot}")
    return snapshot


def success_filename_for_input(input_file: Path) -> str:
    return f"{safe_path_component(input_file.stem)}_rows_successfull_extracted.csv"


def refuse_concurrent_run(lock_file: Path) -> None:
    if not lock_file.exists():
        return
    details = lock_file.read_text(encoding="utf-8", errors="replace").strip()
    message = f"Run lock already exists: {lock_file}"
    if details:
        message += f"\n{details}"
    message += "\nAnother extraction may still be active. Do not run the same --run-name twice."
    raise SystemExit(message)


def acquire_run_lock(lock_file: Path) -> str:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    details = (
        f"created_at={datetime.now().isoformat(timespec='seconds')}\n"
        f"command={command_line()}\n"
    )
    try:
        with lock_file.open("x", encoding="utf-8") as stream:
            stream.write(details)
    except FileExistsError:
        refuse_concurrent_run(lock_file)
    return details


def remove_run_lock(lock_file: Path, expected_contents: Optional[str] = None) -> None:
    if expected_contents is not None and lock_file.exists():
        current_contents = lock_file.read_text(encoding="utf-8", errors="replace")
        if current_contents != expected_contents:
            return
    lock_file.unlink(missing_ok=True)


def reset_clone_root(clone_root: Path) -> None:
    remove_directory_tree(clone_root, clone_root.parent)
    clone_root.mkdir(parents=True, exist_ok=True)


def verify_scansible_available(scansible_command: str) -> None:
    try:
        result = subprocess.run(
            [scansible_command, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Scansible executable not found: {scansible_command}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"Scansible preflight timed out: {scansible_command}") from exc

    if result.returncode != 0:
        message = short_error(result.stderr or result.stdout)
        raise SystemExit(f"Scansible preflight failed: {message}")


def load_dataset_rows(
    input_file: Path,
    max_rows: Optional[int],
) -> list[DatasetRow]:
    with input_file.open(newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames or []
        required = {"commit", "filepath", "failure_prone"}
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise SystemExit(
                "Input dataset is missing required columns: " + ", ".join(missing)
            )
        if "repository" not in fieldnames and "repo_url" not in fieldnames:
            raise SystemExit(
                "Input dataset is missing a repository identifier column: "
                "expected repository or repo_url"
            )

        rows: list[DatasetRow] = []
        for row_index, raw in enumerate(reader, start=1):
            if max_rows is not None and row_index > max_rows:
                break

            raw_repository = str(
                raw.get("repository", "") or raw.get("repo_url", "") or ""
            ).strip()
            repository = normalize_repository_identifier(raw_repository)
            clone_url = repository_clone_url(
                repository=repository,
                repo_url=str(raw.get("repo_url", "") or "").strip(),
            )
            commit = str(raw.get("commit", "") or "").strip()
            filepath = str(raw.get("filepath", "") or "").strip()
            failure_prone = str(raw.get("failure_prone", "") or "").strip()
            rows.append(
                DatasetRow(
                    row_index=row_index,
                    repository=repository,
                    commit=commit,
                    filepath=filepath,
                    failure_prone=failure_prone,
                    raw={str(k): str(v or "") for k, v in raw.items()},
                    clone_url=clone_url,
                )
            )
    return rows


def normalize_repository_identifier(repository: str) -> str:
    value = repository.strip().replace("\\", "/").rstrip("/")
    value = re.sub(r"^https?://github\.com/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^git@github\.com:", "", value, flags=re.IGNORECASE)
    if value.endswith(".git"):
        value = value[:-4]
    return value.strip("/")


def repository_clone_url(repository: str, repo_url: str = "") -> str:
    if repo_url:
        return repo_url
    if not repository:
        return ""
    return f"https://github.com/{repository}.git"


def group_rows_by_repository(rows: Iterable[DatasetRow]) -> list[tuple[str, list[DatasetRow]]]:
    groups: dict[str, list[DatasetRow]] = defaultdict(list)
    for row in rows:
        key = row.repository or f"missing::row-{row.row_index}"
        groups[key].append(row)

    ordered = []
    for key, group in groups.items():
        ordered.append((key, sorted(group, key=lambda item: item.row_index)))
    return sorted(ordered, key=lambda item: min(row.row_index for row in item[1]))


def read_completed_statuses(status_file: Path) -> dict[int, dict[str, str]]:
    if not status_file.exists():
        return {}

    completed: dict[int, dict[str, str]] = {}
    with status_file.open(newline="", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            try:
                row_index = int(row.get("row_index", ""))
            except ValueError:
                continue
            if row.get("status") in TERMINAL_STATUSES:
                completed[row_index] = normalize_status_row(row)
    return completed


def run_parallel_extraction(
    groups: list[tuple[str, list[DatasetRow]]],
    total_rows: int,
    results: dict[int, dict[str, str]],
    status_file: Path,
    success_file: Path,
    report_file: Path,
    logs_dir: Path,
    pdg_root: Path,
    clone_root: Path,
    scansible_command: str,
    timeout: int,
    keep_dot: bool,
    min_pdg_nodes: int,
    min_pdg_edges: int,
    progress_interval: int,
    progress_every: int,
    progress_active_repos: int,
    workers: int,
) -> None:
    stop_event.clear()
    max_workers = min(max(workers, 1), max(len(groups), 1))
    result_queue: queue.Queue[dict[str, str]] = queue.Queue()
    executor = ThreadPoolExecutor(max_workers=max_workers)
    future_map = {}
    progress = ConsoleProgress(
        total_rows=total_rows,
        interval_seconds=progress_interval,
        completed_delta=progress_every,
        active_repos_limit=progress_active_repos,
    )

    try:
        for group_index, (group_key, group_rows) in enumerate(groups, start=1):
            future = executor.submit(
                process_repository_group,
                group_index,
                len(groups),
                group_key,
                group_rows,
                logs_dir,
                pdg_root,
                clone_root,
                scansible_command,
                timeout,
                keep_dot,
                min_pdg_nodes,
                min_pdg_edges,
                result_queue,
            )
            future_map[future] = (group_key, group_rows)

        while future_map:
            done, _ = wait(future_map, timeout=1, return_when=FIRST_COMPLETED)
            drained = drain_result_queue(result_queue, results)
            if drained:
                write_status_outputs(status_file, success_file, results)
                write_report(report_file, None, results)
                progress.maybe_print(results)

            if not done:
                continue

            for future in done:
                group_key, group_rows = future_map.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    print(f"Repository group failed unexpectedly: {group_key}: {exc}")

                drain_result_queue(result_queue, results)
                if future.exception() is not None:
                    for row in group_rows:
                        if row.row_index not in results:
                            results[row.row_index] = with_status(
                                status_row_for(row),
                                "UNEXPECTED_ERROR",
                                error=short_error(str(future.exception())),
                            )
                write_status_outputs(status_file, success_file, results)
                write_report(report_file, None, results)
                progress.maybe_print(results, force=True)

        drain_result_queue(result_queue, results)
        write_status_outputs(status_file, success_file, results)
        write_report(report_file, None, results)
        progress.maybe_print(results, force=True)
    except KeyboardInterrupt:
        print("Interruption requested. Stopping active Scansible processes...")
        stop_event.set()
        terminate_active_processes()
        executor.shutdown(wait=True, cancel_futures=True)
        drain_result_queue(result_queue, results)
        write_status_outputs(status_file, success_file, results)
        write_report(report_file, None, results)
        raise SystemExit(130)
    finally:
        executor.shutdown(wait=True, cancel_futures=stop_event.is_set())


def process_repository_group(
    group_index: int,
    total_groups: int,
    group_key: str,
    rows: list[DatasetRow],
    logs_dir: Path,
    pdg_root: Path,
    clone_root: Path,
    scansible_command: str,
    timeout: int,
    keep_dot: bool,
    min_pdg_nodes: int,
    min_pdg_edges: int,
    result_queue: queue.Queue[dict[str, str]],
) -> int:
    repository_label = rows[0].repository or group_key
    process_key = group_key
    clone_path = clone_root / safe_repository_name(repository_label)
    log_file = logs_dir / f"{safe_repository_name(repository_label)}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with active_repositories_lock:
        active_repositories[process_key] = {
            "label": repository_label,
            "started_at": time.time(),
            "completed": 0,
            "total": len(rows),
            "last_line": "starting",
        }

    print(
        f"[repo {group_index}/{total_groups}] start "
        f"{shorten_text(repository_label, 46)} ({len(rows)} rows)",
        flush=True,
    )

    completed_count = 0
    try:
        with log_file.open("a", encoding="utf-8") as log:
            log.write(
                f"\n=== Repository group started at {datetime.now().isoformat(timespec='seconds')} "
                f"rows={len(rows)} ===\n"
            )
            if not rows[0].clone_url:
                for row in rows:
                    result_queue.put(
                        with_status(
                            status_row_for(row),
                            "MISSING_REQUIRED_FIELD",
                            error="repository or repo_url",
                        )
                    )
                    completed_count += 1
                return completed_count

            update_active_repository(process_key, last_line="cloning repository")
            try:
                clone_repository(rows[0].clone_url, clone_path, log)
            except Exception as exc:
                log.write(f"CLONE FAILURE: {exc}\n")
                log.flush()
                for row in rows:
                    result_queue.put(
                        with_status(
                            status_row_for(row),
                            "CLONE_FAILURE",
                            error=short_error(str(exc)),
                        )
                    )
                    completed_count += 1
                return completed_count

            for row in rows:
                if stop_event.is_set():
                    break

                update_active_repository(
                    process_key,
                    last_line=f"row {row.row_index}: {row.commit} {row.filepath}",
                )
                try:
                    result = process_row(
                        row=row,
                        repository_path=clone_path,
                        pdg_root=pdg_root,
                        scansible_command=scansible_command,
                        timeout=timeout,
                        keep_dot=keep_dot,
                        min_pdg_nodes=min_pdg_nodes,
                        min_pdg_edges=min_pdg_edges,
                        log=log,
                        process_key=process_key,
                    )
                except Exception as exc:
                    log.write(traceback.format_exc() + "\n")
                    log.flush()
                    result = with_status(
                        status_row_for(row),
                        "UNEXPECTED_ERROR",
                        error=short_error(str(exc)),
                    )
                result_queue.put(result)
                completed_count += 1
                update_active_repository(
                    process_key,
                    completed=completed_count,
                    last_line=f"row {row.row_index}: {result['status']}",
                )
    finally:
        try:
            remove_directory_tree(clone_path, clone_root)
        except Exception as exc:
            print(f"Could not remove temporary clone {clone_path}: {exc}", flush=True)
        with active_repositories_lock:
            active_repositories.pop(process_key, None)

    print(
        f"[repo {group_index}/{total_groups}] done  "
        f"{shorten_text(repository_label, 46)} ({completed_count}/{len(rows)} rows)",
        flush=True,
    )
    return completed_count


def drain_result_queue(
    result_queue: queue.Queue[dict[str, str]],
    results: dict[int, dict[str, str]],
) -> int:
    drained = 0
    while True:
        try:
            row = result_queue.get_nowait()
        except queue.Empty:
            break
        results[int(row["row_index"])] = row
        drained += 1
    return drained


def process_row(
    row: DatasetRow,
    repository_path: Path,
    pdg_root: Path,
    scansible_command: str,
    timeout: int,
    keep_dot: bool,
    min_pdg_nodes: int,
    min_pdg_edges: int,
    log,
    process_key: str,
) -> dict[str, str]:
    base = status_row_for(row)

    if not row.repository or not row.commit or not row.filepath:
        return with_status(base, "MISSING_REQUIRED_FIELD", error="repository/commit/filepath")

    normalized_filepath = normalize_filepath(row.filepath)
    if normalized_filepath is None:
        return with_status(base, "INVALID_FILEPATH", error=row.filepath)

    normalized_str = normalized_filepath.as_posix()
    unsupported_reason = unsupported_file_reason(normalized_filepath)
    if unsupported_reason:
        return with_status(base, "UNSUPPORTED_FILE_TYPE", error=unsupported_reason)

    try:
        checkout_commit(repository_path, row.commit)
    except Exception as exc:
        return with_status(base, "CHECKOUT_FAILURE", error=short_error(str(exc)))

    target_file = repository_path / normalized_filepath
    if not target_file.exists():
        return with_status(base, "FILE_NOT_FOUND", error=str(target_file))

    output_dir = output_path_for_file(pdg_root, row.repository, row.commit, normalized_filepath)
    output_dir.mkdir(parents=True, exist_ok=True)
    dot_path = output_dir / "pdg.dot"
    graphml_path = output_dir / "pdg.graphml"

    if graphml_path.exists():
        try:
            graph = nx.read_graphml(str(graphml_path))
            quality_error = pdg_quality_error(graph, min_pdg_nodes, min_pdg_edges)
            if quality_error:
                return with_status(
                    base,
                    "LOW_QUALITY_GRAPH",
                    nodes=graph.number_of_nodes(),
                    edges=graph.number_of_edges(),
                    error=quality_error,
                )
            return with_status(
                base,
                "SUCCESS",
                nodes=graph.number_of_nodes(),
                edges=graph.number_of_edges(),
                graphml_path=str(graphml_path),
            )
        except Exception:
            graphml_path.unlink(missing_ok=True)

    use_task_wrapper = should_wrap_task_file(normalized_filepath)
    if dot_path.exists():
        existing_dot_status = graph_status_from_dot(
            base=base,
            dot_path=dot_path,
            graphml_path=graphml_path,
            keep_dot=keep_dot,
            min_pdg_nodes=min_pdg_nodes,
            min_pdg_edges=min_pdg_edges,
        )
        if existing_dot_status["status"] == "SUCCESS" or not use_task_wrapper:
            return existing_dot_status

    log.write(
        f"ROW {row.row_index}: {row.repository}@{row.commit} -> {row.filepath}\n"
    )
    log.flush()

    scansible_target = prepare_scansible_target(
        repository_path=repository_path,
        filepath=normalized_filepath,
        row_index=row.row_index,
        use_task_wrapper=use_task_wrapper,
    )
    try:
        returncode, stdout, stderr = run_scansible(
            scansible_command=scansible_command,
            target_file=scansible_target,
            repository_path=repository_path,
            timeout=timeout,
            process_key=f"{process_key}::{row.row_index}",
        )
    except subprocess.TimeoutExpired:
        return with_status(base, "EXTRACTION_TIMEOUT", error=f"timeout_after_{timeout}_seconds")
    except Exception as exc:
        return with_status(base, "REAL_EXTRACTION_FAILURE", error=short_error(str(exc)))

    if stderr:
        log.write(stderr.rstrip() + "\n")
        log.flush()

    if returncode != 0:
        return with_status(base, "REAL_EXTRACTION_FAILURE", error=short_error(stderr))

    dot_path.write_text(stdout, encoding="utf-8")
    return graph_status_from_dot(
        base=base,
        dot_path=dot_path,
        graphml_path=graphml_path,
        keep_dot=keep_dot,
        min_pdg_nodes=min_pdg_nodes,
        min_pdg_edges=min_pdg_edges,
    )


def unsupported_file_reason(filepath: Path) -> str:
    parts = set(filepath.parts)
    if "meta" in parts:
        return "meta file"
    if "handlers" in parts:
        return "handlers file"
    if "vars" in parts:
        return "vars file"
    if "defaults" in parts:
        return "defaults file"
    return ""


def should_wrap_task_file(filepath: Path) -> bool:
    return "tasks" in filepath.parts


def prepare_scansible_target(
    repository_path: Path,
    filepath: Path,
    row_index: int,
    use_task_wrapper: bool,
) -> Path:
    if not use_task_wrapper:
        return repository_path / filepath

    wrappers_dir = (repository_path / filepath).parent
    wrappers_dir.mkdir(parents=True, exist_ok=True)
    wrapper = wrappers_dir / f".pdg_file_level_wrapper_row_{row_index}.yml"
    relative_target = os.path.relpath(
        repository_path / filepath,
        start=wrapper.parent,
    ).replace(os.sep, "/")
    wrapper.write_text(
        "- hosts: all\n"
        "  gather_facts: false\n"
        "  tasks:\n"
        f"    - import_tasks: {json.dumps(relative_target)}\n",
        encoding="utf-8",
    )
    return wrapper


def graph_status_from_dot(
    base: dict[str, str],
    dot_path: Path,
    graphml_path: Path,
    keep_dot: bool,
    min_pdg_nodes: int,
    min_pdg_edges: int,
) -> dict[str, str]:
    try:
        graph = load_and_sanitize_graph(dot_path)
    except Exception as exc:
        return with_status(base, "GRAPH_PARSE_FAILURE", error=short_error(str(exc)))

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return with_status(
            base,
            "EMPTY_GRAPH",
            nodes=graph.number_of_nodes(),
            edges=graph.number_of_edges(),
        )

    quality_error = pdg_quality_error(graph, min_pdg_nodes, min_pdg_edges)
    if quality_error:
        return with_status(
            base,
            "LOW_QUALITY_GRAPH",
            nodes=graph.number_of_nodes(),
            edges=graph.number_of_edges(),
            error=quality_error,
        )

    nx.write_graphml(graph, str(graphml_path))
    if not keep_dot:
        dot_path.unlink(missing_ok=True)

    return with_status(
        base,
        "SUCCESS",
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        graphml_path=str(graphml_path),
    )


def pdg_quality_error(
    graph: nx.MultiDiGraph,
    min_pdg_nodes: int,
    min_pdg_edges: int,
) -> str:
    if is_unresolved_include_placeholder(graph):
        return "unresolved include/import placeholder graph"
    if graph.number_of_nodes() < min_pdg_nodes:
        return f"nodes below minimum threshold: {graph.number_of_nodes()} < {min_pdg_nodes}"
    if graph.number_of_edges() < min_pdg_edges:
        return f"edges below minimum threshold: {graph.number_of_edges()} < {min_pdg_edges}"
    return ""


def is_unresolved_include_placeholder(graph: nx.MultiDiGraph) -> bool:
    if graph.number_of_nodes() != 2 or graph.number_of_edges() != 1:
        return False

    labels = {
        str(attrs.get("label", "")).lower()
        for _, attrs in graph.nodes(data=True)
    }
    edge_labels = {
        str(attrs.get("label", "")).lower()
        for _, _, attrs in graph.edges(data=True)
    }
    has_include_action = any(
        "import_tasks" in label or "include_tasks" in label for label in labels
    )
    return has_include_action and "_raw_params" in edge_labels


def clone_repository(clone_url: str, destination: Path, log) -> None:
    remove_directory_tree(destination, destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["git", "clone", "--no-checkout", clone_url, str(destination)]
    log.write(f"CLONE: {' '.join(command)}\n")
    log.flush()
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout:
        log.write(result.stdout.rstrip() + "\n")
    if result.stderr:
        log.write(result.stderr.rstrip() + "\n")
    log.flush()
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())


def remove_directory_tree(path: Path, allowed_root: Path) -> None:
    if not path.exists():
        return
    resolved_path = path.resolve()
    resolved_root = allowed_root.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise RuntimeError(f"Refusing to remove path outside temporary clone root: {path}")
    shutil.rmtree(resolved_path)


def checkout_commit(repository_path: Path, commit: str) -> None:
    commands = [
        ["git", "-C", str(repository_path), "checkout", "--force", commit],
        ["git", "-C", str(repository_path), "clean", "-fdx"],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())


def run_scansible(
    scansible_command: str,
    target_file: Path,
    repository_path: Path,
    timeout: int,
    process_key: str,
) -> tuple[int, str, str]:
    command = [scansible_command, "build-pdg", "-f", "graphviz", str(target_file)]
    process = subprocess.Popen(
        command,
        cwd=repository_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with active_processes_lock:
        active_processes[process_key] = process

    try:
        stdout, stderr = process.communicate(timeout=timeout or None)
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    finally:
        with active_processes_lock:
            active_processes.pop(process_key, None)


def terminate_active_processes() -> None:
    with active_processes_lock:
        processes = list(active_processes.values())
    for process in processes:
        if process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass


def normalize_filepath(filepath: str) -> Optional[Path]:
    try:
        path = PurePosixPath(filepath.replace("\\", "/"))
        if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
            return None
        return Path(*path.parts)
    except Exception:
        return None


def output_path_for_file(
    pdg_root: Path,
    repository: str,
    commit: str,
    filepath: Path,
) -> Path:
    repository_parts = safe_repository_parts(repository)
    commit_part = safe_path_component(commit)
    return (
        pdg_root
        / Path(*repository_parts)
        / commit_part
        / filepath
        / "PDG_FILE_LEVEL"
    )


def safe_repository_parts(repository: str) -> list[str]:
    normalized = normalize_repository_identifier(repository)
    parts = [safe_path_component(part) for part in PurePosixPath(normalized).parts]
    return parts or ["unknown_repository"]


def safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "unknown"


def safe_repository_name(repository: str) -> str:
    return "__".join(safe_repository_parts(repository))


def load_and_sanitize_graph(dot_path: Path) -> nx.Graph:
    try:
        graph = nx.nx_pydot.read_dot(str(dot_path))
    except Exception:
        graph = read_scansible_dot_fallback(dot_path)

    if graph.is_multigraph():
        clean_graph: nx.Graph = nx.MultiDiGraph()
    else:
        clean_graph = nx.DiGraph()

    for node, attrs in graph.nodes(data=True):
        clean_graph.add_node(str(node), **{str(k): str(v) for k, v in attrs.items()})
    for source, target, attrs in graph.edges(data=True):
        clean_graph.add_edge(
            str(source),
            str(target),
            **{str(k): str(v) for k, v in attrs.items()},
        )
    return clean_graph


def read_scansible_dot_fallback(dot_path: Path) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for raw_line in dot_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line in {"digraph {", "}"} or line.startswith("node "):
            continue
        edge_match = re.match(
            r"^(?P<source>\S+)\s*->\s*(?P<target>\S+)(?:\s*\[(?P<attrs>.*)\])?$",
            line,
        )
        if edge_match:
            graph.add_edge(
                edge_match.group("source"),
                edge_match.group("target"),
                **parse_dot_attributes(edge_match.group("attrs") or ""),
            )
            continue

        node_match = re.match(r"^(?P<node>\S+)(?:\s*\[(?P<attrs>.*)\])?$", line)
        if node_match:
            graph.add_node(
                node_match.group("node"),
                **parse_dot_attributes(node_match.group("attrs") or ""),
            )
    return graph


def parse_dot_attributes(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        key_start = index
        while index < length and re.match(r"[A-Za-z0-9_.-]", text[index]):
            index += 1
        key = text[key_start:index]
        while index < length and text[index].isspace():
            index += 1
        if not key or index >= length or text[index] != "=":
            break
        index += 1
        while index < length and text[index].isspace():
            index += 1

        if index < length and text[index] == '"':
            index += 1
            value_start = index
            escaped = False
            value_chars = []
            while index < length:
                char = text[index]
                if char == '"' and not escaped:
                    break
                value_chars.append(char)
                escaped = char == "\\" and not escaped
                if char != "\\":
                    escaped = False
                index += 1
            value = "".join(value_chars)
            if index < length and text[index] == '"':
                index += 1
        elif index + 1 < length and text[index:index + 2] == "<<":
            depth = 0
            value_start = index
            while index < length:
                if index + 1 < length and text[index:index + 2] == "<<":
                    depth += 1
                    index += 2
                    continue
                if index + 1 < length and text[index:index + 2] == ">>":
                    depth -= 1
                    index += 2
                    if depth == 0:
                        break
                    continue
                index += 1
            value = text[value_start:index]
        else:
            value_start = index
            while index < length and not text[index].isspace():
                index += 1
            value = text[value_start:index]
        attrs[key] = value
    return attrs


def status_row_for(row: DatasetRow) -> dict[str, str]:
    return {
        "row_index": str(row.row_index),
        "repository": row.repository,
        "commit": row.commit,
        "filepath": row.filepath,
        "failure_prone": row.failure_prone,
        "status": "",
        "nodes": "0",
        "edges": "0",
        "graphml_path": "",
        "error": "",
    }


def with_status(
    row: dict[str, str],
    status: str,
    nodes: int = 0,
    edges: int = 0,
    graphml_path: str = "",
    error: str = "",
) -> dict[str, str]:
    result = dict(row)
    result.update(
        {
            "status": status,
            "nodes": str(nodes),
            "edges": str(edges),
            "graphml_path": graphml_path,
            "error": short_error(error),
        }
    )
    return result


def normalize_status_row(row: dict[str, str]) -> dict[str, str]:
    return {name: str(row.get(name, "") or "") for name in STATUS_FIELDNAMES}


def short_error(value: str, limit: int = 500) -> str:
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    return (lines[-1] if lines else "")[:limit]


def write_status_outputs(
    status_file: Path,
    success_file: Path,
    results: dict[int, dict[str, str]],
) -> None:
    ordered = [normalize_status_row(results[index]) for index in sorted(results)]
    write_csv_atomic(status_file, STATUS_FIELDNAMES, ordered)
    successes = [row for row in ordered if row["status"] == "SUCCESS"]
    write_csv_atomic(success_file, STATUS_FIELDNAMES, successes)


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = path.with_suffix(path.suffix + ".tmp")
    with tmp_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_file, path)


def write_report(
    report_file: Path,
    rows: Optional[list[DatasetRow]],
    results: dict[int, dict[str, str]],
) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in results.values():
        counts[row.get("status", "UNKNOWN")] += 1

    lines = [
        "# File-level PDG extraction report",
        "",
        f"Updated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Completed rows: {len(results)}",
    ]
    if rows is not None:
        lines.append(f"Input rows: {len(rows)}")
    lines.extend(["", "Status counts:"])
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")
    lines.append("")

    tmp_file = report_file.with_suffix(report_file.suffix + ".tmp")
    tmp_file.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp_file, report_file)


def write_run_metadata(
    metadata_file: Path,
    args: argparse.Namespace,
    input_file: Path,
    input_snapshot: Path,
    run_dir: Path,
    rows: list[DatasetRow],
    completed: dict[int, dict[str, str]],
    pending: list[DatasetRow],
    groups: list[tuple[str, list[DatasetRow]]],
) -> None:
    execution = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "command": command_line(),
        "argv": sys.argv,
        "parameters": {
            "input": str(input_file),
            "run_name": args.run_name,
            "output_dir": str(Path(args.output_dir).resolve()),
            "workers": args.workers,
            "timeout": args.timeout,
            "scansible_command": args.scansible_command,
            "keep_dot": args.keep_dot,
            "min_pdg_nodes": args.min_pdg_nodes,
            "min_pdg_edges": args.min_pdg_edges,
            "progress_interval": args.progress_interval,
            "progress_every": args.progress_every,
            "progress_active_repos": args.progress_active_repos,
            "refresh_input": args.refresh_input,
            "force": args.force,
            "max_rows": args.max_rows,
        },
        "run_directory": str(run_dir),
        "input_snapshot": str(input_snapshot),
        "input_rows": len(rows),
        "rows_already_completed": len(completed),
        "rows_pending_at_start": len(pending),
        "repository_groups_pending_at_start": len(groups),
        "repository_lifecycle": "clone_on_demand_then_delete",
        "resume": bool(completed),
    }

    metadata = {
        "run_name": args.run_name,
        "run_directory": str(run_dir),
        "input_snapshot": str(input_snapshot),
        "last_execution": execution,
        "executions": [],
    }
    if metadata_file.exists():
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    metadata["run_name"] = args.run_name
    metadata["run_directory"] = str(run_dir)
    metadata["input_snapshot"] = str(input_snapshot)
    metadata["last_execution"] = execution
    metadata.setdefault("executions", []).append(execution)

    tmp_file = metadata_file.with_suffix(metadata_file.suffix + ".tmp")
    tmp_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_file, metadata_file)


def shell_quote(value: object) -> str:
    text = str(value)
    if not text:
        return "''"
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._/:=+")
    if all(char in safe_chars for char in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def command_line() -> str:
    return " ".join(shell_quote(part) for part in sys.argv)


def update_active_repository(process_key: str, **values: object) -> None:
    with active_repositories_lock:
        if process_key in active_repositories:
            active_repositories[process_key].update(values)


class ConsoleProgress:
    def __init__(
        self,
        total_rows: int,
        interval_seconds: int,
        completed_delta: int,
        active_repos_limit: int,
    ) -> None:
        self.total_rows = total_rows
        self.interval_seconds = interval_seconds
        self.completed_delta = completed_delta
        self.active_repos_limit = active_repos_limit
        self.last_print_at = 0.0
        self.last_completed = 0

    def maybe_print(
        self,
        results: dict[int, dict[str, str]],
        force: bool = False,
    ) -> None:
        completed = len(results)
        now = time.monotonic()
        enough_time = now - self.last_print_at >= self.interval_seconds
        enough_rows = completed - self.last_completed >= self.completed_delta
        if not force and not enough_time and not enough_rows:
            return
        if completed == self.last_completed and not force:
            return

        self.last_print_at = now
        self.last_completed = completed
        print_progress_snapshot(
            total_rows=self.total_rows,
            results=results,
            active_repos_limit=self.active_repos_limit,
        )


def print_progress_snapshot(
    total_rows: int,
    results: dict[int, dict[str, str]],
    active_repos_limit: int,
) -> None:
    completed = len(results)
    width = 24
    filled = int(width * completed / total_rows) if total_rows else width
    bar = "#" * filled + "-" * (width - filled)
    percent = (completed / total_rows * 100) if total_rows else 100.0
    counts: dict[str, int] = defaultdict(int)
    for row in results.values():
        counts[row.get("status", "UNKNOWN")] += 1

    with active_repositories_lock:
        running = sorted(
            list(active_repositories.values()),
            key=lambda item: int(item.get("completed", 0)),
            reverse=True,
        )

    count_text = format_status_counts(counts)
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"[{bar}] {completed}/{total_rows} ({percent:5.1f}%) | "
        f"active={len(running)} | {count_text}",
        flush=True,
    )
    if active_repos_limit <= 0 or not running:
        return

    visible = running[:active_repos_limit]
    repo_text = " | ".join(format_active_repository(item) for item in visible)
    remaining = len(running) - len(visible)
    if remaining > 0:
        repo_text += f" | +{remaining} active"
    print(f"  active repos: {repo_text}", flush=True)


def format_status_counts(counts: dict[str, int]) -> str:
    preferred = [
        "SUCCESS",
        "LOW_QUALITY_GRAPH",
        "UNSUPPORTED_FILE_TYPE",
        "EMPTY_GRAPH",
        "REAL_EXTRACTION_FAILURE",
        "CHECKOUT_FAILURE",
        "CLONE_FAILURE",
    ]
    parts = [f"{key}={counts[key]}" for key in preferred if counts.get(key)]
    extras = [
        f"{key}={value}"
        for key, value in sorted(counts.items())
        if key not in preferred and value
    ]
    return " ".join(parts + extras) or "no completed rows yet"


def format_active_repository(item: dict[str, object]) -> str:
    label = shorten_text(str(item.get("label", "")), 30)
    completed = item.get("completed", 0)
    total = item.get("total", 0)
    return f"{label} {completed}/{total}"


def shorten_text(value: str, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1] + "~"


def print_final_summary(
    rows: list[DatasetRow],
    results: dict[int, dict[str, str]],
    run_dir: Path,
    success_file: Path,
) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in results.values():
        counts[row.get("status", "UNKNOWN")] += 1

    print("=" * 80)
    print("FILE-LEVEL PDG EXTRACTION COMPLETED")
    print("=" * 80)
    print(f"Input rows: {len(rows)}")
    print(f"Completed rows: {len(results)}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    print(f"Run directory: {run_dir}")
    print(f"Status CSV: {run_dir / 'extraction_status.csv'}")
    print(f"Success CSV: {success_file}")
    print(f"Report: {run_dir / 'extraction_report.txt'}")
    print(f"PDG directory: {run_dir / 'pdg_file_level'}")


if __name__ == "__main__":
    main()
