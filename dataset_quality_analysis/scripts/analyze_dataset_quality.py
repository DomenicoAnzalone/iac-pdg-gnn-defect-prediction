from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_STATIC_PREFIXES = "static_,delta_,process_"
DEFAULT_PDG_PREFIXES = "pdg_,graph_"
DEFAULT_STATUS_COLUMNS = [
    "pdg_status",
    "extraction_status",
    "graph_status",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the quality of an Ansible defect/failure-proneness dataset "
            "and produce human-readable and structured reports."
        )
    )
    parser.add_argument("--input", required=True, help="Input CSV to analyze.")
    parser.add_argument("--label-column", required=True, help="Binary label column.")
    parser.add_argument("--repo-column", required=True, help="Repository column.")
    parser.add_argument("--commit-column", required=True, help="Commit column.")
    parser.add_argument("--file-column", required=True, help="File path column.")
    parser.add_argument("--date-column", default="", help="Optional commit/date column.")
    parser.add_argument(
        "--static-prefixes",
        default=DEFAULT_STATIC_PREFIXES,
        help="Comma-separated prefixes for static/process/delta metrics.",
    )
    parser.add_argument(
        "--pdg-prefixes",
        default=DEFAULT_PDG_PREFIXES,
        help="Comma-separated prefixes for PDG/graph metrics.",
    )
    parser.add_argument(
        "--status-column",
        default="",
        help="Optional PDG extraction status column. If omitted, it is autodetected.",
    )
    parser.add_argument(
        "--graph-path-column",
        default="",
        help="Optional column containing graph file paths.",
    )
    parser.add_argument(
        "--graph-base-dir",
        default="",
        help="Optional local base directory used to verify graph file existence.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where report files will be written.",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.30,
        help="Missing-value ratio threshold for suspicious features. Default: 0.30.",
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=0.90,
        help="Zero-value ratio threshold for suspicious features. Default: 0.90.",
    )
    parser.add_argument(
        "--constant-threshold",
        type=float,
        default=0.90,
        help="Dominant-value ratio threshold for quasi-constant features. Default: 0.90.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Absolute correlation threshold for feature pairs. Default: 0.95.",
    )
    parser.add_argument(
        "--small-repo-thresholds",
        default="5,10,20",
        help="Comma-separated repository size thresholds. Default: 5,10,20.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    required_columns = [
        args.label_column,
        args.repo_column,
        args.commit_column,
        args.file_column,
    ]
    missing_required = [column for column in required_columns if column not in df.columns]
    if missing_required:
        raise SystemExit(
            "Missing required columns: " + ", ".join(missing_required)
        )

    warnings: list[str] = []
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "parameters": vars(args),
        "warnings": warnings,
    }
    suspicious_rows: list[dict[str, Any]] = []

    label_series = normalize_label_series(df[args.label_column])
    id_columns = {
        args.label_column,
        args.repo_column,
        args.commit_column,
        args.file_column,
        "row_index",
        "index",
        "id",
        "status",
        "error",
        "graphml_path",
        "graph_path",
    }
    if args.date_column:
        id_columns.add(args.date_column)
    if args.status_column:
        id_columns.add(args.status_column)
    if args.graph_path_column:
        id_columns.add(args.graph_path_column)

    dataset_summary = analyze_dataset_shape(df)
    label_summary, label_distribution = analyze_label(df, args.label_column, label_series)
    repository_summary, repository_distribution, label_by_repo = analyze_repositories(
        df,
        repo_column=args.repo_column,
        commit_column=args.commit_column,
        file_column=args.file_column,
        label_column=args.label_column,
        label_series=label_series,
        small_thresholds=parse_int_list(args.small_repo_thresholds),
    )
    key_summary, duplicate_keys = analyze_duplicate_keys(
        df,
        repo_column=args.repo_column,
        commit_column=args.commit_column,
        file_column=args.file_column,
    )
    date_summary, instances_by_year, instances_by_month, label_by_month = analyze_dates(
        df,
        date_column=args.date_column,
        label_series=label_series,
        warnings=warnings,
    )
    numeric_summary, numeric_columns = analyze_numeric_metrics(
        df,
        id_columns=id_columns,
        missing_threshold=args.missing_threshold,
        zero_threshold=args.zero_threshold,
        constant_threshold=args.constant_threshold,
        suspicious_rows=suspicious_rows,
    )
    family_summary = analyze_metric_families(
        df,
        numeric_columns=numeric_columns,
        static_prefixes=parse_prefixes(args.static_prefixes),
        pdg_prefixes=parse_prefixes(args.pdg_prefixes),
        missing_threshold=args.missing_threshold,
        constant_threshold=args.constant_threshold,
    )
    status_column = resolve_status_column(df, args.status_column)
    pdg_summary, pdg_coverage_rows = analyze_pdg_coverage(
        df,
        status_column=status_column,
        pdg_prefixes=parse_prefixes(args.pdg_prefixes),
        label_column=args.label_column,
        label_series=label_series,
        repo_column=args.repo_column,
    )
    graph_summary, gnn_rows = analyze_gnn_coverage(
        df,
        label_series=label_series,
        repo_column=args.repo_column,
        graph_path_column=args.graph_path_column,
        graph_base_dir=args.graph_base_dir,
        status_column=status_column,
        warnings=warnings,
    )
    leakage_summary = analyze_leakage(
        df,
        repo_column=args.repo_column,
        commit_column=args.commit_column,
        file_column=args.file_column,
        label_series=label_series,
    )
    correlation_summary, highly_correlated = analyze_correlations(
        df,
        numeric_columns=numeric_columns,
        threshold=args.correlation_threshold,
    )

    summary.update(
        {
            "dataset": dataset_summary,
            "label": label_summary,
            "repository": repository_summary,
            "keys": key_summary,
            "date": date_summary,
            "numeric_metrics": {
                "numeric_feature_count": len(numeric_columns),
                "summary_row_count": len(numeric_summary),
            },
            "metric_families": family_summary,
            "pdg_coverage": pdg_summary,
            "gnn_coverage": graph_summary,
            "leakage": leakage_summary,
            "correlations": correlation_summary,
        }
    )

    write_csv(
        output_dir / "missing_values_by_column.csv",
        dataset_summary["missing_values_by_column"],
        ["column", "missing_values", "missing_ratio"],
    )
    write_csv(
        output_dir / "label_distribution.csv",
        label_distribution,
        ["label", "count", "percentage"],
    )
    write_dataframe(
        output_dir / "repository_distribution.csv",
        repository_distribution,
    )
    write_dataframe(output_dir / "duplicate_keys.csv", duplicate_keys)
    write_dataframe(output_dir / "numeric_metrics_summary.csv", numeric_summary)
    write_csv(
        output_dir / "suspicious_features.csv",
        suspicious_rows,
        ["feature", "issue", "value", "details"],
    )
    write_dataframe(output_dir / "highly_correlated_features.csv", highly_correlated)
    write_csv(
        output_dir / "pdg_coverage_report.csv",
        pdg_coverage_rows,
        ["section", "key", "total", "valid_pdg", "coverage_ratio"],
    )
    write_csv(
        output_dir / "gnn_coverage_report.csv",
        gnn_rows,
        ["section", "key", "value"],
    )

    optional_tables: dict[str, pd.DataFrame] = {
        "label_distribution_by_repository.csv": label_by_repo,
        "instances_by_year.csv": instances_by_year,
        "instances_by_month.csv": instances_by_month,
        "label_distribution_by_month.csv": label_by_month,
    }
    for filename, table in optional_tables.items():
        if not table.empty:
            write_dataframe(output_dir / filename, table)

    write_json(output_dir / "report_summary.json", summary)
    write_text_report(output_dir / "report_summary.txt", summary)

    print(f"Dataset quality analysis completed: {output_dir}")
    return 0


def analyze_dataset_shape(df: pd.DataFrame) -> dict[str, Any]:
    missing_values = df.isna().sum()
    total_rows = len(df)
    return {
        "total_rows": int(total_rows),
        "total_columns": int(len(df.columns)),
        "columns": list(map(str, df.columns)),
        "duplicate_complete_rows": int(df.duplicated().sum()),
        "rows_with_missing_values": int(df.isna().any(axis=1).sum()),
        "missing_values_by_column": [
            {
                "column": str(column),
                "missing_values": int(count),
                "missing_ratio": safe_ratio(count, total_rows),
            }
            for column, count in missing_values.items()
        ],
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
    }


def normalize_label_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    result[numeric == 0] = 0
    result[numeric == 1] = 1
    return result


def analyze_label(
    df: pd.DataFrame,
    label_column: str,
    label_series: pd.Series,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total = len(df)
    count_0 = int((label_series == 0).sum())
    count_1 = int((label_series == 1).sum())
    missing = int(df[label_column].isna().sum())
    invalid_mask = label_series.isna() & df[label_column].notna()
    invalid_values = sorted(map(str, df.loc[invalid_mask, label_column].dropna().unique()))
    major = max(count_0, count_1)
    minor = min(count_0, count_1)
    imbalance = None if minor == 0 else major / minor
    distribution = [
        {"label": "0", "count": count_0, "percentage": safe_ratio(count_0, total)},
        {"label": "1", "count": count_1, "percentage": safe_ratio(count_1, total)},
    ]
    return (
        {
            "label_column": label_column,
            "count_0": count_0,
            "count_1": count_1,
            "percentage_0": safe_ratio(count_0, total),
            "percentage_1": safe_ratio(count_1, total),
            "imbalance_ratio_majority_minority": imbalance,
            "missing_label_values": missing,
            "invalid_label_values_count": int(invalid_mask.sum()),
            "invalid_label_values": invalid_values[:50],
            "binary_values_only": missing == 0 and not invalid_values,
        },
        distribution,
    )


def analyze_repositories(
    df: pd.DataFrame,
    repo_column: str,
    commit_column: str,
    file_column: str,
    label_column: str,
    label_series: pd.Series,
    small_thresholds: list[int],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    repo_counts = df[repo_column].value_counts(dropna=False)
    commit_counts = df.groupby(repo_column, dropna=False)[commit_column].nunique(dropna=True)
    label_df = df[[repo_column]].copy()
    label_df["_label"] = label_series
    label_by_repo = (
        label_df.pivot_table(
            index=repo_column,
            columns="_label",
            values=repo_column,
            aggfunc="count",
            fill_value=0,
        )
        .rename(columns={0: "label_0", 1: "label_1"})
        .reset_index()
    )
    for column in ["label_0", "label_1"]:
        if column not in label_by_repo:
            label_by_repo[column] = 0
    only_positive = label_by_repo.loc[
        (label_by_repo["label_1"] > 0) & (label_by_repo["label_0"] == 0),
        repo_column,
    ].astype(str)
    only_negative = label_by_repo.loc[
        (label_by_repo["label_0"] > 0) & (label_by_repo["label_1"] == 0),
        repo_column,
    ].astype(str)
    distribution = pd.DataFrame(
        {
            repo_column: repo_counts.index.astype(str),
            "instances": repo_counts.values,
            "unique_commits": [
                int(commit_counts.get(repo, 0)) for repo in repo_counts.index
            ],
        }
    )
    small_repositories = {
        f"lt_{threshold}": int((repo_counts < threshold).sum())
        for threshold in small_thresholds
    }
    return (
        {
            "unique_repositories": int(df[repo_column].nunique(dropna=True)),
            "unique_files": int(df[file_column].nunique(dropna=True)),
            "instances_per_repository_mean": safe_float(repo_counts.mean()),
            "instances_per_repository_median": safe_float(repo_counts.median()),
            "instances_per_repository_min": int(repo_counts.min()) if len(repo_counts) else 0,
            "instances_per_repository_max": int(repo_counts.max()) if len(repo_counts) else 0,
            "top_20_repositories": distribution.head(20).to_dict(orient="records"),
            "repositories_only_positive": only_positive.head(200).tolist(),
            "repositories_only_negative": only_negative.head(200).tolist(),
            "repositories_only_positive_count": int(len(only_positive)),
            "repositories_only_negative_count": int(len(only_negative)),
            "small_repositories": small_repositories,
        },
        distribution,
        label_by_repo,
    )


def analyze_duplicate_keys(
    df: pd.DataFrame,
    repo_column: str,
    commit_column: str,
    file_column: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    keys = [repo_column, commit_column, file_column]
    duplicated_mask = df.duplicated(subset=keys, keep=False)
    duplicated = (
        df.loc[duplicated_mask]
        .groupby(keys, dropna=False)
        .size()
        .reset_index(name="duplicate_count")
        .sort_values("duplicate_count", ascending=False)
    )
    return (
        {
            "unique_keys": int(df[keys].drop_duplicates().shape[0]),
            "duplicate_key_groups": int(len(duplicated)),
            "duplicate_key_rows": int(duplicated_mask.sum()),
        },
        duplicated,
    )


def analyze_dates(
    df: pd.DataFrame,
    date_column: str,
    label_series: pd.Series,
    warnings: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not date_column:
        return {"enabled": False, "reason": "date column not provided"}, empty_df(), empty_df(), empty_df()
    if date_column not in df.columns:
        warnings.append(f"Date column not found, skipped: {date_column}")
        return {"enabled": False, "reason": f"date column not found: {date_column}"}, empty_df(), empty_df(), empty_df()

    parsed = pd.to_datetime(df[date_column], errors="coerce", utc=True)
    invalid_count = int(parsed.isna().sum() - df[date_column].isna().sum())
    now = pd.Timestamp.now(tz="UTC")
    too_old = parsed < pd.Timestamp("1990-01-01", tz="UTC")
    future = parsed > now
    temp = pd.DataFrame({"date": parsed, "label": label_series})
    instances_by_year = (
        temp.dropna(subset=["date"])
        .assign(year=lambda item: item["date"].dt.year)
        .groupby("year")
        .size()
        .reset_index(name="instances")
    )
    instances_by_month = (
        temp.dropna(subset=["date"])
        .assign(month=lambda item: item["date"].dt.strftime("%Y-%m"))
        .groupby("month")
        .size()
        .reset_index(name="instances")
    )
    label_by_month = (
        temp.dropna(subset=["date"])
        .assign(month=lambda item: item["date"].dt.strftime("%Y-%m"))
        .pivot_table(index="month", columns="label", values="date", aggfunc="count", fill_value=0)
        .rename(columns={0: "label_0", 1: "label_1"})
        .reset_index()
    )
    for column in ["label_0", "label_1"]:
        if column not in label_by_month:
            label_by_month[column] = 0
    return (
        {
            "enabled": True,
            "date_column": date_column,
            "unparsable_dates": invalid_count,
            "min_date": maybe_iso(parsed.min()),
            "max_date": maybe_iso(parsed.max()),
            "future_dates": int(future.sum()),
            "too_old_dates_before_1990": int(too_old.sum()),
        },
        instances_by_year,
        instances_by_month,
        label_by_month,
    )


def analyze_numeric_metrics(
    df: pd.DataFrame,
    id_columns: set[str],
    missing_threshold: float,
    zero_threshold: float,
    constant_threshold: float,
    suspicious_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[str]]:
    numeric_columns = []
    numeric_data: dict[str, pd.Series] = {}
    for column in df.columns:
        if column in id_columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        if series.notna().sum() == 0:
            continue
        numeric_columns.append(column)
        numeric_data[column] = series

    rows = []
    total = len(df)
    for column in numeric_columns:
        series = numeric_data[column]
        finite = series.replace([math.inf, -math.inf], pd.NA).dropna()
        missing_count = int(series.isna().sum())
        infinite_count = int((series == math.inf).sum() + (series == -math.inf).sum())
        negative_count = int((series < 0).sum())
        zero_count = int((series == 0).sum())
        zero_ratio = safe_ratio(zero_count, total)
        value_counts = series.dropna().value_counts(normalize=True)
        dominant_ratio = safe_float(value_counts.iloc[0]) if len(value_counts) else 0.0
        variance = safe_float(finite.var()) if len(finite) else None
        rows.append(
            {
                "column": column,
                "count": int(finite.count()),
                "mean": safe_float(finite.mean()),
                "std": safe_float(finite.std()),
                "min": safe_float(finite.min()),
                "25%": safe_float(finite.quantile(0.25)) if len(finite) else None,
                "median": safe_float(finite.median()),
                "75%": safe_float(finite.quantile(0.75)) if len(finite) else None,
                "max": safe_float(finite.max()),
                "missing_values": missing_count,
                "infinite_values": infinite_count,
                "negative_values": negative_count,
                "zero_values": zero_count,
                "zero_percentage": zero_ratio,
                "variance": variance,
                "dominant_value_ratio": dominant_ratio,
            }
        )
        add_numeric_warnings(
            suspicious_rows,
            column,
            total,
            missing_count,
            infinite_count,
            zero_ratio,
            dominant_ratio,
            variance,
            missing_threshold,
            zero_threshold,
            constant_threshold,
            finite,
        )
    return pd.DataFrame(rows), numeric_columns


def add_numeric_warnings(
    suspicious_rows: list[dict[str, Any]],
    column: str,
    total: int,
    missing_count: int,
    infinite_count: int,
    zero_ratio: float,
    dominant_ratio: float,
    variance: float | None,
    missing_threshold: float,
    zero_threshold: float,
    constant_threshold: float,
    finite: pd.Series,
) -> None:
    if missing_count / total > missing_threshold if total else False:
        suspicious_rows.append(row_issue(column, "too_many_missing_values", missing_count, f">{missing_threshold:.2f}"))
    if infinite_count:
        suspicious_rows.append(row_issue(column, "infinite_values", infinite_count, "contains +/-inf"))
    if variance == 0:
        suspicious_rows.append(row_issue(column, "zero_variance", 0, "constant numeric feature"))
    if dominant_ratio >= 1.0:
        suspicious_rows.append(row_issue(column, "constant_feature", dominant_ratio, "all non-missing values are equal"))
    elif dominant_ratio > constant_threshold:
        suspicious_rows.append(row_issue(column, "quasi_constant_feature", dominant_ratio, f">{constant_threshold:.2f}"))
    if zero_ratio > zero_threshold:
        suspicious_rows.append(row_issue(column, "too_many_zero_values", zero_ratio, f">{zero_threshold:.2f}"))
    if len(finite) >= 4:
        q1 = finite.quantile(0.25)
        q3 = finite.quantile(0.75)
        iqr = q3 - q1
        if iqr != 0:
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            outliers = int(((finite < lower) | (finite > upper)).sum())
            if outliers:
                suspicious_rows.append(row_issue(column, "possible_extreme_outliers", outliers, "outside 3*IQR"))


def analyze_metric_families(
    df: pd.DataFrame,
    numeric_columns: list[str],
    static_prefixes: list[str],
    pdg_prefixes: list[str],
    missing_threshold: float,
    constant_threshold: float,
) -> dict[str, Any]:
    families = {
        "static_process_delta": [
            column for column in numeric_columns if starts_with_any(column, static_prefixes)
        ],
        "pdg_graph": [
            column for column in numeric_columns if starts_with_any(column, pdg_prefixes)
        ],
    }
    assigned = set(families["static_process_delta"]) | set(families["pdg_graph"])
    families["other_numeric"] = [column for column in numeric_columns if column not in assigned]
    result: dict[str, Any] = {}
    total_rows = len(df)
    for family, columns in families.items():
        if not columns:
            result[family] = {"column_count": 0}
            continue
        data = df[columns].apply(pd.to_numeric, errors="coerce")
        missing_total = int(data.isna().sum().sum())
        missing_ratio_by_column = data.isna().mean()
        dominant_ratios = data.apply(
            lambda series: safe_float(series.dropna().value_counts(normalize=True).iloc[0])
            if len(series.dropna()) else 0.0
        )
        result[family] = {
            "column_count": len(columns),
            "columns": columns,
            "missing_values_total": missing_total,
            "average_missing_ratio": safe_float(missing_ratio_by_column.mean()),
            "columns_above_missing_threshold": int((missing_ratio_by_column > missing_threshold).sum()),
            "constant_columns": int((dominant_ratios >= 1.0).sum()),
            "quasi_constant_columns": int((dominant_ratios > constant_threshold).sum()),
            "overall_mean": safe_float(data.stack().mean()) if total_rows else None,
            "overall_median": safe_float(data.stack().median()) if total_rows else None,
            "overall_min": safe_float(data.stack().min()) if total_rows else None,
            "overall_max": safe_float(data.stack().max()) if total_rows else None,
        }
    return result


def analyze_pdg_coverage(
    df: pd.DataFrame,
    status_column: str,
    pdg_prefixes: list[str],
    label_column: str,
    label_series: pd.Series,
    repo_column: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    total = len(df)
    if status_column:
        status = df[status_column].astype(str)
        valid = status.str.upper().eq("SUCCESS")
        status_counts = status.value_counts(dropna=False).to_dict()
        method = f"status column: {status_column}"
    else:
        pdg_columns = [column for column in df.columns if starts_with_any(column, pdg_prefixes)]
        if pdg_columns:
            pdg_data = df[pdg_columns].apply(pd.to_numeric, errors="coerce")
            valid = ~(pdg_data.isna().all(axis=1) | (pdg_data.fillna(0) == 0).all(axis=1))
            status_counts = {}
            method = "inferred from PDG metric columns"
        else:
            valid = pd.Series(False, index=df.index)
            status_counts = {}
            method = "not available"

    valid_count = int(valid.sum())
    add_coverage_row(rows, "overall", "dataset", total, valid_count)
    for label in [0, 1]:
        mask = label_series == label
        add_coverage_row(rows, "label", str(label), int(mask.sum()), int((valid & mask).sum()))
    pdg_repo_frame = pd.DataFrame(
        {
            repo_column: df[repo_column],
            "_valid_pdg": valid,
        }
    )
    repo_group = pdg_repo_frame.groupby(repo_column, dropna=False)
    for repo, group in repo_group:
        add_coverage_row(rows, "repository", str(repo), len(group), int(group["_valid_pdg"].sum()))
    return (
        {
            "method": method,
            "status_column": status_column or None,
            "valid_pdg_instances": valid_count,
            "invalid_or_missing_pdg_instances": int(total - valid_count),
            "pdg_coverage_ratio": safe_ratio(valid_count, total),
            "status_distribution": {str(k): int(v) for k, v in status_counts.items()},
            "gnn_usable_instances_by_pdg_status": valid_count,
        },
        rows,
    )


def analyze_gnn_coverage(
    df: pd.DataFrame,
    label_series: pd.Series,
    repo_column: str,
    graph_path_column: str,
    graph_base_dir: str,
    status_column: str,
    warnings: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    total = len(df)
    if not graph_path_column:
        return {"enabled": False, "reason": "graph path column not provided"}, rows
    if graph_path_column not in df.columns:
        warnings.append(f"Graph path column not found, skipped: {graph_path_column}")
        return {"enabled": False, "reason": f"graph path column not found: {graph_path_column}"}, rows

    graph_paths = df[graph_path_column].fillna("").astype(str)
    has_path = graph_paths.str.strip() != ""
    if status_column:
        status_valid = df[status_column].astype(str).str.upper().eq("SUCCESS")
    else:
        status_valid = pd.Series(True, index=df.index)
    if graph_base_dir:
        exists = graph_paths.apply(lambda value: graph_path_exists(value, graph_base_dir))
    else:
        exists = has_path
    usable = has_path & status_valid & exists
    usable_count = int(usable.sum())
    represented = set(df.loc[usable, repo_column].astype(str))
    all_repos = set(df[repo_column].dropna().astype(str))
    excluded = sorted(all_repos - represented)
    rows.extend(
        [
            {"section": "overall", "key": "usable_graphs", "value": usable_count},
            {"section": "overall", "key": "usable_graph_ratio", "value": safe_ratio(usable_count, total)},
            {"section": "label", "key": "0", "value": int((usable & (label_series == 0)).sum())},
            {"section": "label", "key": "1", "value": int((usable & (label_series == 1)).sum())},
            {"section": "repositories", "key": "represented", "value": len(represented)},
            {"section": "repositories", "key": "excluded", "value": len(excluded)},
        ]
    )
    return (
        {
            "enabled": True,
            "graph_path_column": graph_path_column,
            "graph_base_dir": graph_base_dir or None,
            "usable_graphs": usable_count,
            "usable_graphs_label_0": int((usable & (label_series == 0)).sum()),
            "usable_graphs_label_1": int((usable & (label_series == 1)).sum()),
            "usable_graph_ratio": safe_ratio(usable_count, total),
            "represented_repositories": len(represented),
            "excluded_repositories": len(excluded),
            "excluded_repository_examples": excluded[:100],
        },
        rows,
    )


def analyze_leakage(
    df: pd.DataFrame,
    repo_column: str,
    commit_column: str,
    file_column: str,
    label_series: pd.Series,
) -> dict[str, Any]:
    repo_file = [repo_column, file_column]
    repo_file_commit_counts = df.groupby(repo_file, dropna=False)[commit_column].nunique(dropna=True)
    shared_repo_file = repo_file_commit_counts[repo_file_commit_counts > 1]
    temp = df[[repo_column, file_column, commit_column]].copy()
    temp["_label"] = label_series
    multi_commit = temp.groupby(repo_file, dropna=False)[commit_column].nunique(dropna=True) > 1
    label0_multi = temp[temp["_label"] == 0].groupby(repo_file, dropna=False)[commit_column].nunique(dropna=True)
    label1_multi = temp[temp["_label"] == 1].groupby(repo_file, dropna=False)[commit_column].nunique(dropna=True)
    return {
        "files_appearing_in_multiple_commits": int(shared_repo_file.size),
        "repositories_with_multiple_commits": int((df.groupby(repo_column, dropna=False)[commit_column].nunique(dropna=True) > 1).sum()),
        "instances_sharing_same_repo_file_across_commits": int(df.set_index(repo_file).index.isin(shared_repo_file.index).sum()),
        "repo_file_pairs_with_multiple_commits_label_0": int((label0_multi > 1).sum()),
        "repo_file_pairs_with_multiple_commits_label_1": int((label1_multi > 1).sum()),
        "instances_sharing_same_repository": int(df[repo_column].duplicated(keep=False).sum()),
        "warning": (
            "A random row-level split can leak information when the same file or "
            "repository appears in both train and test. Prefer repository-level, "
            "temporal commit-level, or within-project walk-forward validation."
        ),
    }


def analyze_correlations(
    df: pd.DataFrame,
    numeric_columns: list[str],
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if len(numeric_columns) < 2:
        return {"enabled": False, "reason": "fewer than two numeric columns"}, empty_df()
    data = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    rows = []
    duplicate_like = 0
    for i, left in enumerate(numeric_columns):
        for right in numeric_columns[i + 1 :]:
            p_value = pearson.loc[left, right]
            s_value = spearman.loc[left, right]
            max_abs = max(abs_or_zero(p_value), abs_or_zero(s_value))
            if max_abs > threshold:
                left_values = data[left].fillna("__NA__").astype(str)
                right_values = data[right].fillna("__NA__").astype(str)
                equal_ratio = safe_ratio(int((left_values == right_values).sum()), len(data))
                if equal_ratio > 0.99:
                    duplicate_like += 1
                rows.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "pearson": safe_float(p_value),
                        "spearman": safe_float(s_value),
                        "max_abs_correlation": safe_float(max_abs),
                        "equal_value_ratio": equal_ratio,
                    }
                )
    table = pd.DataFrame(rows).sort_values("max_abs_correlation", ascending=False) if rows else empty_df()
    return (
        {
            "enabled": True,
            "numeric_columns_used": len(numeric_columns),
            "highly_correlated_pairs": len(rows),
            "duplicate_or_near_duplicate_pairs": duplicate_like,
            "threshold": threshold,
        },
        table,
    )


def resolve_status_column(df: pd.DataFrame, requested: str) -> str:
    if requested and requested in df.columns:
        return requested
    if requested:
        return ""
    for column in DEFAULT_STATUS_COLUMNS:
        if column in df.columns:
            return column
    return ""


def graph_path_exists(value: str, graph_base_dir: str) -> bool:
    if not value:
        return False
    path = Path(value)
    if path.exists():
        return True
    base = Path(graph_base_dir)
    candidates = []
    text = value.replace("\\", "/")
    if "/app/output/" in text:
        candidates.append(base / text.split("/app/output/", 1)[1])
    candidates.append(base / Path(text).name)
    if not path.is_absolute():
        candidates.append(base / path)
    return any(candidate.exists() for candidate in candidates)


def add_coverage_row(
    rows: list[dict[str, Any]],
    section: str,
    key: str,
    total: int,
    valid: int,
) -> None:
    rows.append(
        {
            "section": section,
            "key": key,
            "total": int(total),
            "valid_pdg": int(valid),
            "coverage_ratio": safe_ratio(valid, total),
        }
    )


def row_issue(feature: str, issue: str, value: Any, details: str) -> dict[str, Any]:
    return {"feature": feature, "issue": issue, "value": value, "details": details}


def parse_prefixes(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    return result


def starts_with_any(column: str, prefixes: list[str]) -> bool:
    return any(str(column).startswith(prefix) for prefix in prefixes)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return safe_float(numerator / denominator)


def safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        value = float(value)
        if math.isinf(value) or math.isnan(value):
            return None
        return value
    except Exception:
        return None


def abs_or_zero(value: Any) -> float:
    converted = safe_float(value)
    return abs(converted) if converted is not None else 0.0


def maybe_iso(value: Any) -> str | None:
    if pd.isna(value):
        return None
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def write_dataframe(path: Path, table: pd.DataFrame) -> None:
    if table.empty:
        table.to_csv(path, index=False)
    else:
        table.to_csv(path, index=False)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    table = pd.DataFrame(rows, columns=fieldnames)
    table.to_csv(path, index=False)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except Exception:
            pass
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value


def write_text_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dataset Quality Report",
        "",
        f"Generated at: {summary['generated_at']}",
        f"Input: {summary['input']}",
        "",
        "## 1. Dimensione generale",
    ]
    dataset = summary["dataset"]
    lines.extend(
        [
            f"- Righe totali: {dataset['total_rows']}",
            f"- Colonne totali: {dataset['total_columns']}",
            f"- Righe duplicate complete: {dataset['duplicate_complete_rows']}",
            f"- Righe con almeno un valore mancante: {dataset['rows_with_missing_values']}",
            "- Colonne presenti: " + ", ".join(dataset["columns"]),
            "",
            "## 2. Label binaria",
        ]
    )
    label = summary["label"]
    lines.extend(
        [
            f"- Colonna label: {label['label_column']}",
            f"- Label 0: {label['count_0']} ({label['percentage_0']:.4f})",
            f"- Label 1: {label['count_1']} ({label['percentage_1']:.4f})",
            f"- Rapporto sbilanciamento maggioritaria/minoritaria: {label['imbalance_ratio_majority_minority']}",
            f"- Label mancanti: {label['missing_label_values']}",
            f"- Label non valide: {label['invalid_label_values_count']}",
            "",
            "## 3. Repository, commit e file",
        ]
    )
    repo = summary["repository"]
    lines.extend(
        [
            f"- Repository uniche: {repo['unique_repositories']}",
            f"- File unici: {repo['unique_files']}",
            f"- Istanze/repository media: {repo['instances_per_repository_mean']}",
            f"- Istanze/repository mediana: {repo['instances_per_repository_median']}",
            f"- Istanze/repository min-max: {repo['instances_per_repository_min']} - {repo['instances_per_repository_max']}",
            f"- Repository solo positive: {repo['repositories_only_positive_count']}",
            f"- Repository solo negative: {repo['repositories_only_negative_count']}",
            f"- Repository piccole: {repo['small_repositories']}",
            "",
            "## 4. Chiave logica repo/commit/file",
        ]
    )
    keys = summary["keys"]
    lines.extend(
        [
            f"- Chiavi uniche: {keys['unique_keys']}",
            f"- Gruppi di chiavi duplicate: {keys['duplicate_key_groups']}",
            f"- Righe coinvolte in chiavi duplicate: {keys['duplicate_key_rows']}",
            "",
            "## 5. Distribuzione temporale",
        ]
    )
    date = summary["date"]
    if date.get("enabled"):
        lines.extend(
            [
                f"- Colonna data: {date['date_column']}",
                f"- Data minima: {date['min_date']}",
                f"- Data massima: {date['max_date']}",
                f"- Date non parsabili: {date['unparsable_dates']}",
                f"- Date future: {date['future_dates']}",
                f"- Date prima del 1990: {date['too_old_dates_before_1990']}",
            ]
        )
    else:
        lines.append(f"- Sezione saltata: {date.get('reason')}")
    lines.extend(
        [
            "",
            "## 6. Metriche numeriche",
            f"- Feature numeriche analizzate: {summary['numeric_metrics']['numeric_feature_count']}",
            f"- Righe nel summary numerico: {summary['numeric_metrics']['summary_row_count']}",
            "",
            "## 7. Famiglie di metriche",
        ]
    )
    for family, family_summary in summary["metric_families"].items():
        lines.append(f"- {family}: {family_summary}")
    lines.extend(["", "## 8. Copertura PDG"])
    pdg = summary["pdg_coverage"]
    lines.extend(
        [
            f"- Metodo: {pdg['method']}",
            f"- PDG validi: {pdg['valid_pdg_instances']}",
            f"- PDG mancanti/non validi: {pdg['invalid_or_missing_pdg_instances']}",
            f"- Copertura PDG: {pdg['pdg_coverage_ratio']:.4f}",
            f"- Status distribution: {pdg['status_distribution']}",
            "",
            "## 9. Dataset GNN",
        ]
    )
    gnn = summary["gnn_coverage"]
    if gnn.get("enabled"):
        lines.extend(
            [
                f"- Grafi utilizzabili: {gnn['usable_graphs']}",
                f"- Grafi label 0: {gnn['usable_graphs_label_0']}",
                f"- Grafi label 1: {gnn['usable_graphs_label_1']}",
                f"- Percentuale grafi utilizzabili: {gnn['usable_graph_ratio']:.4f}",
                f"- Repository rappresentate: {gnn['represented_repositories']}",
                f"- Repository escluse: {gnn['excluded_repositories']}",
            ]
        )
    else:
        lines.append(f"- Sezione saltata: {gnn.get('reason')}")
    leakage = summary["leakage"]
    lines.extend(
        [
            "",
            "## 10. Data leakage",
            f"- File in piu commit: {leakage['files_appearing_in_multiple_commits']}",
            f"- Repository con piu commit: {leakage['repositories_with_multiple_commits']}",
            f"- Istanze con stessa repo/file e commit diversi: {leakage['instances_sharing_same_repo_file_across_commits']}",
            f"- Repo/file multi-commit label 0: {leakage['repo_file_pairs_with_multiple_commits_label_0']}",
            f"- Repo/file multi-commit label 1: {leakage['repo_file_pairs_with_multiple_commits_label_1']}",
            f"- Warning: {leakage['warning']}",
            "",
            "## 11. Correlazioni",
        ]
    )
    corr = summary["correlations"]
    lines.extend(
        [
            f"- Abilitata: {corr.get('enabled')}",
            f"- Coppie altamente correlate: {corr.get('highly_correlated_pairs', 0)}",
            f"- Coppie duplicate/quasi duplicate: {corr.get('duplicate_or_near_duplicate_pairs', 0)}",
            "",
            "## Warning",
        ]
    )
    if summary["warnings"]:
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    else:
        lines.append("- Nessun warning operativo.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
