from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from experiments.common.config import DEFAULT_DATASET
from experiments.common.data_loading import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe small-graph sensitivity thresholds.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="experiments/results/exploratory/small_graph_sensitivity")
    parser.add_argument("--thresholds", default="3:2,5:4,8:6,10:6")
    args = parser.parse_args()
    df = load_dataset(args.dataset)
    thresholds = [_parse_threshold(item) for item in args.thresholds.split(",") if item.strip()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    bucket_rows = []
    for min_nodes, min_edges in thresholds:
        nodes = pd.to_numeric(df["nodes"], errors="coerce")
        edges = pd.to_numeric(df["edges"], errors="coerce")
        kept = df[(nodes >= min_nodes) & (edges >= min_edges)]
        removed = df[~df["_sample_id"].isin(kept["_sample_id"])]
        rows.append({
            "min_nodes": min_nodes,
            "min_edges": min_edges,
            "kept_samples": len(kept),
            "removed_samples": len(removed),
            "kept_positive_rate": kept["failure_prone"].mean() if len(kept) else None,
            "removed_positive_rate": removed["failure_prone"].mean() if len(removed) else None,
            "kept_repositories": kept["repository"].nunique(),
            "removed_repositories": removed["repository"].nunique(),
        })
    bucket = pd.cut(pd.to_numeric(df["nodes"], errors="coerce"), bins=[0, 3, 5, 8, 10, 20, 32, 69, 148, float("inf")], include_lowest=True)
    bucket_summary = (
        df.assign(node_bucket=bucket.astype(str))
        .groupby("node_bucket")["failure_prone"]
        .agg(samples="count", positives="sum", positive_rate="mean")
        .reset_index()
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "small_graph_threshold_summary.csv", index=False)
    bucket_summary.to_csv(out / "label_distribution_by_graph_size.csv", index=False)
    (out / "small_graph_sensitivity.md").write_text(render(summary, bucket_summary), encoding="utf-8")


def _parse_threshold(value: str) -> tuple[int, int]:
    left, right = value.split(":")
    return int(left), int(right)


def render(summary: pd.DataFrame, bucket_summary: pd.DataFrame) -> str:
    try:
        threshold_table = summary.to_markdown(index=False)
        bucket_table = bucket_summary.to_markdown(index=False)
    except Exception:
        threshold_table = "```\n" + summary.to_csv(index=False) + "\n```"
        bucket_table = "```\n" + bucket_summary.to_csv(index=False) + "\n```"
    return "\n".join([
        "# Small-Graph Sensitivity Analysis",
        "",
        "This report describes the sample and label impact of alternative minimum graph-size thresholds.",
        "",
        "## Thresholds",
        "",
        threshold_table,
        "",
        "## Label Distribution By Node Bucket",
        "",
        bucket_table,
    ])


if __name__ == "__main__":
    main()
