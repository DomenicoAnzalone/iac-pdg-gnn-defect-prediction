from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved E1/E2/E3 experiment results.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--metric", default="mcc")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    per_split, pooled, output_dir = load_results(results_dir)
    summary = build_comparison(per_split, args.metric, pooled)
    out_csv = output_dir / "comparison_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    out_md = Path(args.output) if args.output else output_dir / "comparison_summary.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(summary, args.metric), encoding="utf-8")


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, Path]:
    single_per_split = results_dir / "metrics" / "per_split_metrics.csv"
    if single_per_split.exists():
        pooled_path = results_dir / "metrics" / "pooled_metrics.csv"
        pooled = pd.read_csv(pooled_path) if pooled_path.exists() else None
        return pd.read_csv(single_per_split), pooled, results_dir / "reports"

    per_split_frames = []
    pooled_frames = []
    for child in sorted(path for path in results_dir.iterdir() if path.is_dir() and not path.name.startswith("_")):
        per_split_path = child / "metrics" / "per_split_metrics.csv"
        pooled_path = child / "metrics" / "pooled_metrics.csv"
        if per_split_path.exists():
            per_split_frames.append(pd.read_csv(per_split_path))
        if pooled_path.exists():
            pooled_frames.append(pd.read_csv(pooled_path))
    if not per_split_frames:
        raise FileNotFoundError(f"Missing per-split metrics under: {results_dir}")
    pooled = pd.concat(pooled_frames, ignore_index=True) if pooled_frames else None
    return pd.concat(per_split_frames, ignore_index=True), pooled, results_dir / "_summary"


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
        if wilcoxon is not None and len(paired) >= 2 and (paired["left_metric"] - paired["right_metric"]).abs().sum() > 0:
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
