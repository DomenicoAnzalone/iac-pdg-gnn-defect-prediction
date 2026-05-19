import argparse
import logging
from .logging_utils import setup_logging
from .config import DEFAULT_CONFIG
from .ingest import normalize_and_index
from .extract import extract_for_commit
from .slice import slice_repository_file_level
from .splits import walk_forward_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["ingest", "extract", "slice", "splits"], required=True)
    parser.add_argument("--repo", help="Repository name")
    parser.add_argument("--commit", help="Commit hash")
    args = parser.parse_args()

    setup_logging(None)

    if args.step == "ingest":
        df = normalize_and_index(DEFAULT_CONFIG.ansible_csv)
        print(df.head())
    elif args.step == "extract":
        if not args.repo or not args.commit:
            parser.error("--repo and --commit required for extract")
        extract_for_commit(args.repo, args.commit, DEFAULT_CONFIG.output_root)
    elif args.step == "slice":
        if not args.repo or not args.commit:
            parser.error("--repo and --commit required for slice")
        slice_repository_file_level(args.repo, args.commit, DEFAULT_CONFIG.output_root)
    elif args.step == "splits":
        df = normalize_and_index(DEFAULT_CONFIG.ansible_csv)
        walk_forward_splits(df, DEFAULT_CONFIG.output_root)


if __name__ == "__main__":
    main()
