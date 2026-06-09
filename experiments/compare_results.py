from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import pandas as pd
from scipy.stats import wilcoxon


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved E1/E2/E3 experiment results.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--metric", default="mcc")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    per_split_path = results_dir / "metrics" / "per_split_metrics.csv"
    if not per_split_path.exists():
        raise FileNotFoundError(f"Missing per-split metrics: {per_split_path}")
    per_split = pd.read_csv(per_split_path)
    pooled_path = results_dir / "metrics" / "pooled_metrics.csv"
    pooled = pd.read_csv(pooled_path) if pooled_path.exists() else None
    summary = build_comparison(per_split, args.metric, pooled)
    out_csv = results_dir / "reports" / "comparison_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    out_md = Path(args.output) if args.output else results_dir / "reports" / "comparison_summary.md"
    out_md.write_text(render_markdown(summary, args.metric), encoding="utf-8")


def build_comparison(per_split: pd.DataFrame, metric: str, pooled: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    systems = sorted(per_split[["experiment", "model"]].drop_duplicates().itertuples(index=False, name=None))
    for exp, model in systems:
        vals = per_split[(per_split["experiment"] == exp) & (per_split["model"] == model)][metric].dropna()
        pooled_value = ""
        pooled_col = f"pooled_{metric}"
        if pooled is not None and pooled_col in pooled.columns:
            match = pooled[(pooled["experiment"] == exp) & (pooled["model"] == model)]
            if not match.empty:
                pooled_value = match.iloc[0][pooled_col]
        rows.append({
            "comparison": "single",
            "left": f"{exp}:{model}",
            "right": "",
            "metric": metric,
            "left_pooled": pooled_value,
            "left_mean": vals.mean(),
            "right_mean": "",
            "mean_delta_left_minus_right": "",
            "paired_n": len(vals),
            "wilcoxon_pvalue": "",
        })
    for (left_exp, left_model), (right_exp, right_model) in combinations(systems, 2):
        left = per_split[(per_split["experiment"] == left_exp) & (per_split["model"] == left_model)][["split_id", metric]].rename(columns={metric: "left_metric"})
        right = per_split[(per_split["experiment"] == right_exp) & (per_split["model"] == right_model)][["split_id", metric]].rename(columns={metric: "right_metric"})
        paired = left.merge(right, on="split_id").dropna()
        pvalue = ""
        if len(paired) >= 2 and (paired["left_metric"] - paired["right_metric"]).abs().sum() > 0:
            try:
                pvalue = float(wilcoxon(paired["left_metric"], paired["right_metric"]).pvalue)
            except Exception:
                pvalue = ""
        rows.append({
            "comparison": "paired",
            "left": f"{left_exp}:{left_model}",
            "right": f"{right_exp}:{right_model}",
            "metric": metric,
            "left_pooled": "",
            "left_mean": paired["left_metric"].mean() if not paired.empty else "",
            "right_mean": paired["right_metric"].mean() if not paired.empty else "",
            "mean_delta_left_minus_right": (paired["left_metric"] - paired["right_metric"]).mean() if not paired.empty else "",
            "paired_n": len(paired),
            "wilcoxon_pvalue": pvalue,
        })
    return pd.DataFrame(rows)


def render_markdown(summary: pd.DataFrame, metric: str) -> str:
    try:
        table = summary.to_markdown(index=False)
    except Exception:
        table = "```\n" + summary.to_csv(index=False) + "\n```"
    return "\n".join([
        "# Comparison Summary",
        "",
        f"Metric used for comparison: `{metric}`.",
        "",
        "Single-system rows report pooled metrics when `metrics/pooled_metrics.csv` is available. Paired rows use split-level values for Wilcoxon diagnostics.",
        "",
        table,
        "",
        "Wilcoxon signed-rank tests are computed only for paired split-level results with at least two non-identical pairs.",
    ])


if __name__ == "__main__":
    main()
