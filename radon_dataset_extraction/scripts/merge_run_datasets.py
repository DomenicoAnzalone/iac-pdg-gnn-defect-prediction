import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "runs"

MIN_ROWS = 100
MIN_POSITIVES = 20
MIN_NEGATIVES = 20


def as_int(value):
    try:
        return int(float(value))
    except Exception:
        return 0


def should_keep(row, filtered: bool) -> bool:
    if row.get("status") != "SUCCESS":
        return False

    if not filtered:
        return True

    dataset_rows = as_int(row.get("dataset_rows"))
    positives = as_int(row.get("positives"))
    negatives = as_int(row.get("negatives"))

    return (
        dataset_rows >= MIN_ROWS
        and positives >= MIN_POSITIVES
        and negatives >= MIN_NEGATIVES
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--filtered", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.output_dir).resolve() / args.run_name
    summary_file = run_dir / "batch_summary.csv"

    if args.filtered:
        output_file = run_dir / "merged_dataset_filtered.csv"
        kept_summary_file = run_dir / "batch_summary_filtered_kept.csv"
        discarded_summary_file = run_dir / "batch_summary_filtered_discarded.csv"
    else:
        output_file = run_dir / "merged_dataset.csv"
        kept_summary_file = None
        discarded_summary_file = None

    with open(summary_file, "r", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))

    selected_summary = []
    discarded_summary = []

    for row in summary_rows:
        if should_keep(row, args.filtered):
            row["filter_reason"] = "kept" if args.filtered else ""
            selected_summary.append(row)
        else:
            reasons = []

            if row.get("status") != "SUCCESS":
                reasons.append("status_not_success")

            if args.filtered:
                if as_int(row.get("dataset_rows")) < MIN_ROWS:
                    reasons.append(f"dataset_rows<{MIN_ROWS}")
                if as_int(row.get("positives")) < MIN_POSITIVES:
                    reasons.append(f"positives<{MIN_POSITIVES}")
                if as_int(row.get("negatives")) < MIN_NEGATIVES:
                    reasons.append(f"negatives<{MIN_NEGATIVES}")

            row["filter_reason"] = ";".join(reasons) if reasons else "discarded"
            discarded_summary.append(row)

    all_rows = []
    fieldnames = None

    for summary in selected_summary:
        output_csv = Path(summary["output_csv"])
        repo_url = summary["repo_url"]
        branch = summary["branch"]

        print(f"Reading {output_csv}")

        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if fieldnames is None:
                fieldnames = ["repo_url", "branch"] + list(reader.fieldnames or [])

            for row in rows:
                merged_row = {
                    "repo_url": repo_url,
                    "branch": branch,
                }
                merged_row.update(row)
                all_rows.append(merged_row)

    if fieldnames is None:
        fieldnames = ["repo_url", "branch"]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    positives = sum(
        1 for r in all_rows
        if str(r.get("failure_prone", "")).strip() in {"1", "1.0", "True", "true"}
    )
    negatives = sum(
        1 for r in all_rows
        if str(r.get("failure_prone", "")).strip() in {"0", "0.0", "False", "false"}
    )

    print(f"Saved merged dataset to: {output_file}")
    print(f"Repos included: {len(selected_summary)}")
    print(f"Rows: {len(all_rows)}")
    print(f"Columns: {len(fieldnames)}")
    print(f"Positives: {positives}")
    print(f"Negatives: {negatives}")

    if args.filtered:
        summary_fieldnames = list(summary_rows[0].keys())
        if "filter_reason" not in summary_fieldnames:
            summary_fieldnames.append("filter_reason")

        with open(kept_summary_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(selected_summary)

        with open(discarded_summary_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(discarded_summary)

        print(f"Saved kept summary to: {kept_summary_file}")
        print(f"Saved discarded summary to: {discarded_summary_file}")


if __name__ == "__main__":
    main()
