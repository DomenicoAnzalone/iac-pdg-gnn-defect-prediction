from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class Split:
    split_id: str
    repository: str
    train_ids: List[str]
    validation_ids: List[str]
    test_ids: List[str]
    train_commits: List[str]
    validation_commits: List[str]
    test_commit: str


def create_walk_forward_splits(
    df: pd.DataFrame,
    validation_ratio: float = 0.2,
    max_splits: int | None = None,
) -> Tuple[List[Split], pd.DataFrame]:
    splits: List[Split] = []
    skipped: List[Dict[str, object]] = []
    for repository, repo_df in df.sort_values(["repository", "committed_at", "commit"]).groupby("repository", sort=False):
        commit_order = (
            repo_df[["commit", "committed_at"]]
            .drop_duplicates()
            .sort_values(["committed_at", "commit"])["commit"]
            .tolist()
        )
        if len(commit_order) < 3:
            skipped.append({"repository": repository, "reason": "too_few_commits", "commit_count": len(commit_order)})
            continue
        for test_idx in range(2, len(commit_order)):
            prior_commits = commit_order[:test_idx]
            test_commit = commit_order[test_idx]
            validation_count = max(1, int(round(len(prior_commits) * validation_ratio)))
            if validation_count >= len(prior_commits):
                validation_count = 1
            val_commits = prior_commits[-validation_count:]
            train_commits = prior_commits[:-validation_count]
            train = repo_df[repo_df["commit"].isin(train_commits)]
            val = repo_df[repo_df["commit"].isin(val_commits)]
            test = repo_df[repo_df["commit"] == test_commit]
            reason = _invalid_split_reason(train, val, test)
            split_id = f"{_slug(repository)}__wf_{test_idx:04d}"
            if reason:
                skipped.append({
                    "split_id": split_id,
                    "repository": repository,
                    "reason": reason,
                    "train_size": len(train),
                    "validation_size": len(val),
                    "test_size": len(test),
                    "test_commit": test_commit,
                })
                continue
            splits.append(Split(
                split_id=split_id,
                repository=str(repository),
                train_ids=train["_sample_id"].tolist(),
                validation_ids=val["_sample_id"].tolist(),
                test_ids=test["_sample_id"].tolist(),
                train_commits=[str(c) for c in train_commits],
                validation_commits=[str(c) for c in val_commits],
                test_commit=str(test_commit),
            ))
            if max_splits and len(splits) >= max_splits:
                return splits, pd.DataFrame(skipped)
    return splits, pd.DataFrame(skipped)


def _invalid_split_reason(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> str:
    if train.empty or val.empty or test.empty:
        return "empty_partition"
    if train["failure_prone"].nunique() < 2:
        return "train_single_class"
    if test["failure_prone"].nunique() < 2:
        return "test_single_class"
    return ""


def split_manifest_rows(splits: List[Split], df: pd.DataFrame) -> pd.DataFrame:
    index = df.set_index("_sample_id")
    rows = []
    for split in splits:
        for part, ids in [("train", split.train_ids), ("validation", split.validation_ids), ("test", split.test_ids)]:
            for sample_id in ids:
                row = index.loc[sample_id]
                rows.append({
                    "split_id": split.split_id,
                    "repository": split.repository,
                    "partition": part,
                    "sample_id": sample_id,
                    "commit": row["commit"],
                    "committed_at": row["committed_at"],
                    "filepath": row["filepath"],
                    "failure_prone": int(row["failure_prone"]),
                })
    return pd.DataFrame(rows)


def materialize_split(df: pd.DataFrame, split: Split) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        df[df["_sample_id"].isin(split.train_ids)].copy(),
        df[df["_sample_id"].isin(split.validation_ids)].copy(),
        df[df["_sample_id"].isin(split.test_ids)].copy(),
    )


def assert_no_overlap(split: Split) -> None:
    train, val, test = set(split.train_ids), set(split.validation_ids), set(split.test_ids)
    if train & val or train & test or val & test:
        raise ValueError(f"Split {split.split_id} has overlapping train/validation/test samples")


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)[:80]

