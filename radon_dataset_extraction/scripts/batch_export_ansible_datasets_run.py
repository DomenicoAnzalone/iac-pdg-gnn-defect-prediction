import argparse
import csv
import json
import os
import re
import shutil
import signal
import select
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "input_repos_mixed_final.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "runs"
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "export_ansible_dataset.py"

SUMMARY_FIELDNAMES = [
    "repo_url",
    "branch",
    "status",
    "tags_count",
    "fixing_commits",
    "fixed_files",
    "labeled_snapshots",
    "dataset_rows",
    "dataset_columns",
    "positives",
    "negatives",
    "output_csv",
    "log_file",
    "error",
]

TERMINAL_STATUSES = {
    "SUCCESS",
    "FAILED_EXPORT",
    "FAILED_TIMEOUT",
    "FAILED_EXCEPTION",
    "EMPTY_DATASET",
    "SKIPPED_TOO_FEW_TAGS",
}

active_processes = {}
active_processes_lock = threading.Lock()
active_repos = {}
active_repos_lock = threading.Lock()
HEARTBEAT_SECONDS = 30
stop_event = threading.Event()


def safe_repo_name(repo_url: str) -> str:
    name = repo_url.rstrip("/").replace("https://github.com/", "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name)


def repo_key(repo_url: str, branch: str) -> str:
    return f"{repo_url.strip()}::{branch.strip()}"


def run_tracked_process(cmd, timeout, process_key):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    with active_processes_lock:
        active_processes[process_key] = process

    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
    finally:
        with active_processes_lock:
            active_processes.pop(process_key, None)


def count_tags(repo_url: str, process_key: str) -> int:
    returncode, stdout, stderr = run_tracked_process(
        ["git", "ls-remote", "--tags", "--refs", repo_url],
        timeout=120,
        process_key=process_key,
    )

    if returncode != 0:
        raise RuntimeError(stderr.strip())

    if not stdout.strip():
        return 0

    return len(stdout.strip().splitlines())


def read_repos(input_file: Path):
    repos = []
    seen = set()

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]

            repo_url = parts[0]
            branch = parts[1] if len(parts) > 1 and parts[1] else "master"
            key = repo_key(repo_url, branch)

            if key in seen:
                continue

            seen.add(key)
            repos.append((repo_url, branch))

    return repos


def prepare_run_input(input_file: Path, run_dir: Path, force_input_refresh: bool) -> Path:
    run_input_file = run_dir / "input_repos.csv"

    if run_input_file.exists() and not force_input_refresh:
        print(f"Resume input snapshot found: {run_input_file}")
        return run_input_file

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_file, run_input_file)
    print(f"Input snapshot saved to: {run_input_file}")
    return run_input_file


def empty_summary_row(repo_url: str, branch: str, output_csv: Path, log_file: Path):
    return {
        "repo_url": repo_url,
        "branch": branch,
        "status": "UNKNOWN",
        "tags_count": "",
        "fixing_commits": "",
        "fixed_files": "",
        "labeled_snapshots": "",
        "dataset_rows": "",
        "dataset_columns": "",
        "positives": "",
        "negatives": "",
        "output_csv": str(output_csv),
        "log_file": str(log_file),
        "error": "",
    }


def read_summary(summary_file: Path):
    if not summary_file.exists():
        return {}

    with open(summary_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    completed = {}
    for row in rows:
        key = repo_key(row.get("repo_url", ""), row.get("branch", ""))
        if key and row.get("status") in TERMINAL_STATUSES:
            completed[key] = row

    return completed


def write_summary_atomic(summary_file: Path, rows):
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = summary_file.with_suffix(summary_file.suffix + ".tmp")

    with open(tmp_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(tmp_file, summary_file)


def shell_quote(value):
    value = str(value)
    if not value:
        return "''"

    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._/:=+")
    if all(char in safe_chars for char in value):
        return value

    return "'" + value.replace("'", "'\"'\"'") + "'"


def command_line():
    return " ".join(shell_quote(part) for part in sys.argv)


def write_run_metadata(run_dir: Path, args, run_input_file: Path, repos, completed_rows, pending):
    metadata_file = run_dir / "run_metadata.json"
    now = datetime.now().isoformat(timespec="seconds")

    current_execution = {
        "started_at": now,
        "command": command_line(),
        "argv": sys.argv,
        "parameters": {
            "input": str(Path(args.input).resolve()),
            "run_name": args.run_name,
            "output_dir": str(Path(args.output_dir).resolve()),
            "timeout": args.timeout,
            "workers": max(args.workers, 1),
            "refresh_input": bool(args.refresh_input),
            "keep_clones": bool(args.keep_clones),
        },
        "run_directory": str(run_dir),
        "input_snapshot": str(run_input_file),
        "repositories_total": len(repos),
        "repositories_already_completed": len(completed_rows),
        "repositories_pending_at_start": len(pending),
        "resume": len(completed_rows) > 0,
    }

    metadata = {
        "run_name": args.run_name,
        "run_directory": str(run_dir),
        "input_snapshot": str(run_input_file),
        "last_execution": current_execution,
        "executions": [],
    }

    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {
                "run_name": args.run_name,
                "run_directory": str(run_dir),
                "input_snapshot": str(run_input_file),
                "last_execution": current_execution,
                "executions": [],
            }

    metadata["run_name"] = args.run_name
    metadata["run_directory"] = str(run_dir)
    metadata["input_snapshot"] = str(run_input_file)
    metadata["last_execution"] = current_execution
    metadata.setdefault("executions", []).append(current_execution)

    tmp_file = metadata_file.with_suffix(metadata_file.suffix + ".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    os.replace(tmp_file, metadata_file)
    return metadata_file


def parse_export_stdout(output: str, row):
    for line in output.splitlines():
        line = line.strip()

        if line.startswith("Fixing commits found:"):
            row["fixing_commits"] = line.split(":", 1)[1].strip()

        elif line.startswith("Fixed files found:"):
            row["fixed_files"] = line.split(":", 1)[1].strip()

        elif line.startswith("Labeled file snapshots:"):
            row["labeled_snapshots"] = line.split(":", 1)[1].strip()


def count_labels(output_csv: Path, row):
    with open(output_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []

    row["dataset_rows"] = len(rows)
    row["dataset_columns"] = len(columns)

    positives = 0
    negatives = 0

    for r in rows:
        label = str(r.get("failure_prone", "")).strip()

        if label in {"1", "1.0", "True", "true"}:
            positives += 1
        elif label in {"0", "0.0", "False", "false"}:
            negatives += 1

    row["positives"] = positives
    row["negatives"] = negatives


def write_repo_log(log_file: Path, text: str):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(text or "")


def compact_repo_label(repo_url: str, max_len: int = 58):
    label = repo_url.rstrip("/").replace("https://github.com/", "")
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


def status_counts(rows):
    counts = {}
    for row in rows:
        status = row.get("status") or "UNKNOWN"
        counts[status] = counts.get(status, 0) + 1
    return counts


def format_elapsed(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def print_progress(total, completed_rows):
    completed = len(completed_rows)
    width = 28
    filled = int(width * completed / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    counts = status_counts(completed_rows.values())

    with active_repos_lock:
        running = list(active_repos.values())

    status_part = " ".join(
        f"{status}={count}"
        for status, count in sorted(counts.items())
    )
    print(
        f"Progress [{bar}] {completed}/{total} completed | "
        f"running={len(running)} | {status_part}"
    )

    if running:
        shown = running[:8]
        print("Running:")
        now = time.time()
        for item in shown:
            elapsed = format_elapsed(now - item["started_at"])
            last_line = item.get("last_line") or "starting"
            print(
                f"  - [{item['index']}/{total}] {item['label']} "
                f"elapsed={elapsed} last=\"{last_line}\""
            )
        if len(running) > len(shown):
            print(f"  - ... {len(running) - len(shown)} more")


def update_active_repo(process_key, **values):
    with active_repos_lock:
        if process_key in active_repos:
            active_repos[process_key].update(values)


def normalize_log_line(line):
    line = (line or "").strip()
    if not line:
        return ""

    return line[:140]


def terminate_active_processes():
    with active_processes_lock:
        processes = list(active_processes.values())

    for process in processes:
        if process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass

    for process in processes:
        try:
            process.wait(timeout=10)
        except Exception:
            if process.poll() is None:
                try:
                    process.kill()
                except Exception:
                    pass


def run_export_process(cmd, timeout, process_key, log_file):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    with active_processes_lock:
        active_processes[process_key] = process

    try:
        output_lines = []
        deadline = time.time() + timeout if timeout else None
        last_heartbeat = time.time()

        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as log:
            while True:
                if stop_event.is_set() and process.poll() is None:
                    process.terminate()

                if deadline and time.time() > deadline and process.poll() is None:
                    process.kill()
                    remaining = process.stdout.read() if process.stdout else ""
                    if remaining:
                        log.write(remaining)
                        output_lines.append(remaining)
                    raise subprocess.TimeoutExpired(cmd, timeout, output="".join(output_lines))

                line = ""
                if process.stdout:
                    readable, _, _ = select.select([process.stdout], [], [], 0.2)
                    if readable:
                        line = process.stdout.readline()

                if line:
                    log.write(line)
                    log.flush()
                    output_lines.append(line)

                    normalized = normalize_log_line(line)
                    if normalized:
                        update_active_repo(process_key, last_line=normalized)

                elif process.poll() is not None:
                    remaining = process.stdout.read() if process.stdout else ""
                    if remaining:
                        log.write(remaining)
                        output_lines.append(remaining)
                    break
                else:
                    pass

                if time.time() - last_heartbeat >= HEARTBEAT_SECONDS:
                    update_active_repo(process_key, last_line="still running")
                    last_heartbeat = time.time()

        return process.returncode, "".join(output_lines)
    finally:
        with active_processes_lock:
            active_processes.pop(process_key, None)


def cleanup_clone_dir(clone_dir: Path):
    if clone_dir.exists():
        shutil.rmtree(clone_dir, ignore_errors=True)


def cleanup_clone_dirs_dir(clone_dirs_dir: Path):
    if not clone_dirs_dir.exists():
        return 0

    removed = 0
    for child in clone_dirs_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
        else:
            try:
                child.unlink()
                removed += 1
            except Exception:
                pass

    return removed


def process_repo(index, total, repo_url, branch, datasets_dir, clone_dirs_dir, logs_dir, timeout, keep_clones):
    repo_name = safe_repo_name(repo_url)
    output_csv = datasets_dir / f"{repo_name}.csv"
    log_file = logs_dir / f"{repo_name}.log"
    clone_dir = clone_dirs_dir / repo_name
    row = empty_summary_row(repo_url, branch, output_csv, log_file)
    started_at = time.time()

    process_key = repo_key(repo_url, branch)

    with active_repos_lock:
        active_repos[process_key] = {
            "index": index,
            "label": compact_repo_label(repo_url),
            "started_at": started_at,
            "last_line": "checking tags",
        }

    print(f"START [{index}/{total}] {compact_repo_label(repo_url)} branch={branch}")

    try:
        if stop_event.is_set():
            return None

        tags = count_tags(repo_url, "tags::" + repo_key(repo_url, branch))
        row["tags_count"] = tags
        update_active_repo(process_key, last_line=f"tags found: {tags}")

        if tags < 2:
            row["status"] = "SKIPPED_TOO_FEW_TAGS"
            print(f"DONE  [{index}/{total}] {compact_repo_label(repo_url)} status={row['status']} tags={tags}")
            return row

        cmd = [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--repo-url",
            repo_url,
            "--default-branch",
            branch,
            "--clone-dir",
            str(clone_dir),
            "--product",
            "--process",
            "--delta",
            "--output",
            str(output_csv),
        ]

        update_active_repo(process_key, last_line="running export")
        returncode, stdout = run_export_process(
            cmd,
            timeout=timeout,
            process_key=process_key,
            log_file=log_file,
        )

        parse_export_stdout(stdout, row)

        if returncode != 0:
            row["status"] = "FAILED_EXPORT"
            row["error"] = stdout[-3000:].replace("\n", " | ")
            elapsed = int(time.time() - started_at)
            print(
                f"DONE  [{index}/{total}] {compact_repo_label(repo_url)} "
                f"status={row['status']} elapsed={elapsed}s log={log_file}"
            )
            return row

        if not output_csv.exists() or output_csv.stat().st_size <= 1:
            row["status"] = "EMPTY_DATASET"
            elapsed = int(time.time() - started_at)
            print(
                f"DONE  [{index}/{total}] {compact_repo_label(repo_url)} "
                f"status={row['status']} elapsed={elapsed}s log={log_file}"
            )
            return row

        count_labels(output_csv, row)
        row["status"] = "SUCCESS"
        elapsed = int(time.time() - started_at)

        print(
            f"DONE  [{index}/{total}] {compact_repo_label(repo_url)} "
            f"status={row['status']} rows={row['dataset_rows']} "
            f"pos={row['positives']} neg={row['negatives']} elapsed={elapsed}s"
        )
        return row

    except subprocess.TimeoutExpired as exc:
        row["status"] = "FAILED_TIMEOUT"
        row["error"] = f"timeout_after_{timeout}_seconds"
        if exc.output:
            write_repo_log(log_file, str(exc.output))
            row["error"] += " | " + str(exc.output)[-2500:].replace("\n", " | ")
        elapsed = int(time.time() - started_at)
        print(
            f"DONE  [{index}/{total}] {compact_repo_label(repo_url)} "
            f"status={row['status']} elapsed={elapsed}s log={log_file}"
        )
        return row

    except Exception:
        if stop_event.is_set():
            return None

        row["status"] = "FAILED_EXCEPTION"
        row["error"] = traceback.format_exc()[-3000:].replace("\n", " | ")
        elapsed = int(time.time() - started_at)
        print(
            f"DONE  [{index}/{total}] {compact_repo_label(repo_url)} "
            f"status={row['status']} elapsed={elapsed}s"
        )
        return row

    finally:
        if not keep_clones:
            cleanup_clone_dir(clone_dir)

        with active_repos_lock:
            active_repos.pop(process_key, None)


def build_summary_rows(repos, completed_rows):
    rows = []
    for repo_url, branch in repos:
        key = repo_key(repo_url, branch)
        if key in completed_rows:
            rows.append(completed_rows[key])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of repositories to process concurrently.",
    )
    parser.add_argument(
        "--refresh-input",
        action="store_true",
        help="Replace the run input snapshot with --input before processing.",
    )
    parser.add_argument(
        "--keep-clones",
        action="store_true",
        help="Keep cloned repositories after processing. By default clone directories are deleted.",
    )
    args = parser.parse_args()

    def handle_stop_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_stop_signal)

    input_file = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    workers = max(args.workers, 1)

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")

    run_dir = output_dir / run_name
    datasets_dir = run_dir / "datasets"
    clone_dirs_dir = run_dir / "clone_dirs"
    logs_dir = run_dir / "logs"
    summary_file = run_dir / "batch_summary.csv"

    datasets_dir.mkdir(parents=True, exist_ok=True)
    clone_dirs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    removed_clone_dirs = 0
    if not args.keep_clones:
        removed_clone_dirs = cleanup_clone_dirs_dir(clone_dirs_dir)

    run_input_file = prepare_run_input(input_file, run_dir, args.refresh_input)
    repos = read_repos(run_input_file)
    completed_rows = read_summary(summary_file)
    pending = [
        (index, repo_url, branch)
        for index, (repo_url, branch) in enumerate(repos, start=1)
        if repo_key(repo_url, branch) not in completed_rows
    ]

    write_summary_atomic(summary_file, build_summary_rows(repos, completed_rows))
    metadata_file = write_run_metadata(
        run_dir,
        args,
        run_input_file,
        repos,
        completed_rows,
        pending,
    )

    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Input snapshot: {run_input_file}")
    print(f"Datasets directory: {datasets_dir}")
    print(f"Logs directory: {logs_dir}")
    print(f"Batch summary: {summary_file}")
    print(f"Run metadata: {metadata_file}")
    print(f"Repositories total: {len(repos)}")
    print(f"Repositories already completed: {len(completed_rows)}")
    print(f"Repositories pending: {len(pending)}")
    print(f"Workers: {workers}")
    print(f"Keep clones: {args.keep_clones}")
    if not args.keep_clones:
        print(f"Clone dirs cleaned at startup: {removed_clone_dirs}")
    print("=" * 80)

    if not pending:
        print("Nothing to do: all repositories already have a terminal status.")
        return

    executor = ThreadPoolExecutor(max_workers=workers)
    pending_iter = iter(pending)
    futures = set()

    def submit_next():
        try:
            index, repo_url, branch = next(pending_iter)
        except StopIteration:
            return False

        futures.add(
            executor.submit(
                process_repo,
                index,
                len(repos),
                repo_url,
                branch,
                datasets_dir,
                clone_dirs_dir,
                logs_dir,
                args.timeout,
                args.keep_clones,
            )
        )
        return True

    try:
        for _ in range(workers):
            if not submit_next():
                break

        last_progress_print = 0
        while futures:
            done, futures = wait(
                futures,
                timeout=HEARTBEAT_SECONDS,
                return_when=FIRST_COMPLETED,
            )

            if not done:
                print_progress(len(repos), completed_rows)
                last_progress_print = time.time()
                continue

            for future in done:
                if stop_event.is_set():
                    continue

                row = future.result()
                if row is None:
                    continue

                completed_rows[repo_key(row["repo_url"], row["branch"])] = row
                write_summary_atomic(summary_file, build_summary_rows(repos, completed_rows))
                print_progress(len(repos), completed_rows)
                last_progress_print = time.time()

                if not stop_event.is_set():
                    submit_next()

            if time.time() - last_progress_print >= HEARTBEAT_SECONDS:
                print_progress(len(repos), completed_rows)
                last_progress_print = time.time()

    except KeyboardInterrupt:
        print("=" * 80)
        print("INTERRUPT RECEIVED")
        print("Stopping active exports and preserving completed results for resume.")
        print("=" * 80)
        stop_event.set()
        terminate_active_processes()
        write_summary_atomic(summary_file, build_summary_rows(repos, completed_rows))
        executor.shutdown(wait=False)
        raise SystemExit(130)
    finally:
        if not stop_event.is_set():
            executor.shutdown(wait=True)
        else:
            executor.shutdown(wait=False)

    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Input snapshot: {run_input_file}")
    print(f"Datasets directory: {datasets_dir}")
    print(f"Logs directory: {logs_dir}")
    print(f"Batch summary saved to: {summary_file}")
    print(f"Completed repositories: {len(completed_rows)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
