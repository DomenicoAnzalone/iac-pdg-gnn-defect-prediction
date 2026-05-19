import os
import pandas as pd
import logging
import json
from typing import Tuple, List

logger = logging.getLogger(__name__)


def infer_label_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if "failure" in c.lower() or "defect" in c.lower() or c.lower() == "label"]
    if candidates:
        return candidates[0]
    # fallback to last column
    return df.columns[-1]


def normalize_and_index(ansible_csv: str) -> pd.DataFrame:
    df = pd.read_csv(ansible_csv)

    # normalize names
    df = df.rename(columns={c: c.strip() for c in df.columns})

    # infer required columns
    label_col = infer_label_column(df)

    # Ensure columns exist
    required = ["repository", "commit", "filepath", "committed_at"]
    for r in required:
        if r not in df.columns:
            df[r] = None

    df = df[["repository", "commit", "committed_at", "filepath", label_col]]
    df = df.sort_values(["repository", "committed_at", "commit", "filepath"], na_position="last")
    df = df.reset_index(drop=True)
    df = df.rename(columns={label_col: "label"})
    
    # add row_id for tracking
    df.insert(0, "row_id", range(len(df)))

    logger.info("Ingested %d rows from %s; label_col=%s", len(df), ansible_csv, label_col)
    return df


def save_dataset_manifest(df: pd.DataFrame, output_root: str):
    """Save CSV → row mapping manifest."""
    manifest_dir = os.path.join(output_root, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    
    manifest = {
        "total_rows": len(df),
        "label_counts": df["label"].value_counts().to_dict(),
        "repositories": sorted(df["repository"].unique().tolist()),
        "timestamp": str(__import__("datetime").datetime.now()),
    }
    
    path = os.path.join(manifest_dir, "dataset_manifest.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Saved dataset manifest to %s", path)
    except Exception:
        logger.exception("Failed to save dataset manifest")
