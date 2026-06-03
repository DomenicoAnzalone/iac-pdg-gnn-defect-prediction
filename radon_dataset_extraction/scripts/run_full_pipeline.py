import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "runs"
DEFAULT_INPUT_FILE = PROJECT_ROOT / "input_repos_mixed_final.txt"


def run_command(command):
    print("=" * 80)
    print("Running:")
    print(" ".join(command))
    print("=" * 80)

    result = subprocess.run(command)

    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit code {result.returncode}")


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


def write_pipeline_metadata(run_dir: Path, args, input_file: Path):
    run_dir_existed = run_dir.exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = run_dir / "pipeline_metadata.json"
    now = datetime.now().isoformat(timespec="seconds")

    current_execution = {
        "started_at": now,
        "command": command_line(),
        "argv": sys.argv,
        "parameters": {
            "input": str(input_file),
            "run_name": args.run_name,
            "output_dir": str(Path(args.output_dir).resolve()),
            "timeout": args.timeout,
            "workers": max(args.workers, 1),
            "keep_clones": bool(args.keep_clones),
            "force": bool(args.force),
            "discover": bool(args.discover),
            "discover_max_repos": args.discover_max_repos,
            "discover_min_stars": args.discover_min_stars,
            "discover_max_pages": args.discover_max_pages,
            "discover_per_page": args.discover_per_page,
            "github_token_provided": bool(args.github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")),
            "discovery_input_output": args.discovery_input_output,
            "discovery_selected_report": args.discovery_selected_report,
            "discovery_excluded_report": args.discovery_excluded_report,
            "discovery_methodology_report": args.discovery_methodology_report,
        },
        "run_directory": str(run_dir),
        "resume": run_dir_existed and not args.force,
    }

    metadata = {
        "run_name": args.run_name,
        "run_directory": str(run_dir),
        "last_execution": current_execution,
        "executions": [],
    }

    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    metadata["run_name"] = args.run_name
    metadata["run_directory"] = str(run_dir)
    metadata["last_execution"] = current_execution
    metadata.setdefault("executions", []).append(current_execution)

    tmp_file = metadata_file.with_suffix(metadata_file.suffix + ".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    os.replace(tmp_file, metadata_file)
    return metadata_file


def main():
    parser = argparse.ArgumentParser(
        description="Run the full RADON Ansible dataset pipeline: export, merge, filter."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_FILE),
        help="Input CSV/TXT file with repo_url,branch rows.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name of the output run directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory that will contain pipeline runs.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for each repository export.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of repositories to process concurrently during RADON export.",
    )
    parser.add_argument(
        "--keep-clones",
        action="store_true",
        help="Keep cloned repositories after processing. By default clone directories are deleted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing run directory if it already exists.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Run GitHub Ansible repository discovery before the RADON pipeline.",
    )
    parser.add_argument(
        "--discover-max-repos",
        type=int,
        default=100,
        help="Maximum number of discovered repositories to keep.",
    )
    parser.add_argument(
        "--discover-min-stars",
        type=int,
        default=0,
        help="Minimum GitHub stars for discovered repositories.",
    )
    parser.add_argument(
        "--discover-max-pages",
        type=int,
        default=2,
        help="Maximum GitHub Search API pages per discovery strategy.",
    )
    parser.add_argument(
        "--discover-per-page",
        type=int,
        default=100,
        help="GitHub Search API results per page during discovery.",
    )
    parser.add_argument(
        "--github-token",
        default=None,
        help="GitHub token for discovery. If omitted, GITHUB_TOKEN or GH_TOKEN is used.",
    )
    parser.add_argument(
        "--discovery-input-output",
        default=None,
        help="Path for the generated repo_url,branch file. Defaults under the run discovery directory.",
    )
    parser.add_argument(
        "--discovery-selected-report",
        default=None,
        help="Path for the selected repositories CSV report.",
    )
    parser.add_argument(
        "--discovery-excluded-report",
        default=None,
        help="Path for the excluded repositories CSV report.",
    )
    parser.add_argument(
        "--discovery-methodology-report",
        default=None,
        help="Path for the discovery methodology Markdown report.",
    )

    args = parser.parse_args()

    input_file = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    run_dir = output_dir / args.run_name

    if run_dir.exists() and args.force:
        print(f"Deleting existing run directory: {run_dir}")
        shutil.rmtree(run_dir)
    elif run_dir.exists():
        print(f"Existing run directory found, resuming: {run_dir}")

    run_input_snapshot = run_dir / "input_repos.csv"

    if args.discover and run_input_snapshot.exists() and not args.force:
        print(f"Existing run input snapshot found, skipping discovery: {run_input_snapshot}")
        input_file = run_input_snapshot

    elif args.discover:
        discovery_dir = run_dir / "discovery"
        discovery_input_file = Path(
            args.discovery_input_output
            or discovery_dir / "ansible_candidate_repos.txt"
        ).resolve()
        discovery_selected_report = Path(
            args.discovery_selected_report
            or discovery_dir / "ansible_candidate_repos_selected.csv"
        ).resolve()
        discovery_excluded_report = Path(
            args.discovery_excluded_report
            or discovery_dir / "ansible_candidate_repos_excluded.csv"
        ).resolve()
        discovery_methodology_report = Path(
            args.discovery_methodology_report
            or discovery_dir / "ansible_discovery_methodology.md"
        ).resolve()

        discovery_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "discover_ansible_repos.py"),
            "--max-repos",
            str(args.discover_max_repos),
            "--min-stars",
            str(args.discover_min_stars),
            "--max-pages",
            str(args.discover_max_pages),
            "--per-page",
            str(args.discover_per_page),
            "--output",
            str(discovery_input_file),
            "--selected-report",
            str(discovery_selected_report),
            "--excluded-report",
            str(discovery_excluded_report),
            "--methodology-report",
            str(discovery_methodology_report),
        ]

        if args.github_token:
            discovery_command.extend(["--token", args.github_token])

        run_command(discovery_command)
        input_file = discovery_input_file

    metadata_file = write_pipeline_metadata(run_dir, args, input_file)
    print(f"Pipeline metadata: {metadata_file}")

    python = sys.executable

    batch_command = [
        python,
        str(PROJECT_ROOT / "scripts" / "batch_export_ansible_datasets_run.py"),
        "--input",
        str(input_file),
        "--run-name",
        args.run_name,
        "--output-dir",
        str(output_dir),
        "--timeout",
        str(args.timeout),
        "--workers",
        str(max(args.workers, 1)),
    ]

    if args.keep_clones:
        batch_command.append("--keep-clones")

    run_command(batch_command)

    run_command([
        python,
        str(PROJECT_ROOT / "scripts" / "merge_run_datasets.py"),
        "--run-name",
        args.run_name,
        "--output-dir",
        str(output_dir),
    ])

    run_command([
        python,
        str(PROJECT_ROOT / "scripts" / "merge_run_datasets.py"),
        "--run-name",
        args.run_name,
        "--output-dir",
        str(output_dir),
        "--filtered",
    ])

    print("=" * 80)
    print("FULL PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Raw merged dataset: {run_dir / 'merged_dataset.csv'}")
    print(f"Filtered merged dataset: {run_dir / 'merged_dataset_filtered.csv'}")
    print(f"Batch summary: {run_dir / 'batch_summary.csv'}")
    print(f"Kept summary: {run_dir / 'batch_summary_filtered_kept.csv'}")
    print(f"Discarded summary: {run_dir / 'batch_summary_filtered_discarded.csv'}")

    if args.discover:
        print(f"Discovery input file: {input_file}")
        print(f"Discovery directory: {run_dir / 'discovery'}")


if __name__ == "__main__":
    main()
