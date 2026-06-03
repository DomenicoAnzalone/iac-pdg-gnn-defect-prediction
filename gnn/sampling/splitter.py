from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from git import Repo, GitCommandError, InvalidGitRepositoryError
    GITPYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover
    Repo = None
    GitCommandError = Exception
    InvalidGitRepositoryError = Exception
    GITPYTHON_AVAILABLE = False


@dataclass
class WalkForwardSplit:
    repository: str
    train: pd.DataFrame
    validation: Optional[pd.DataFrame]
    test: pd.DataFrame
    train_commits: List[str]
    validation_commits: List[str]
    test_commit: str


class WalkForwardSplitter:
    """Create walk-forward splits for graph classification datasets by repository."""

    def __init__(self, df: pd.DataFrame, repositories_root: Optional[Path] = None):
        self.df = df.copy()
        self.repositories_root = Path(repositories_root) if repositories_root is not None else None

    def _get_repo_path(self, repository: str) -> Optional[Path]:
        if self.repositories_root is None:
            return None
        return self.repositories_root / repository

    def _commit_dates(self, repository: str, commits: Iterable[str]) -> Dict[str, datetime]:
        if not GITPYTHON_AVAILABLE or self.repositories_root is None:
            return {}

        repo_path = self._get_repo_path(repository)
        if repo_path is None or not repo_path.exists():
            return {}

        try:
            repo = Repo(repo_path)
        except (GitCommandError, InvalidGitRepositoryError):
            return {}

        dates: Dict[str, datetime] = {}
        for commit in commits:
            try:
                commit_obj = repo.commit(commit)
                dates[commit] = datetime.fromtimestamp(commit_obj.committed_date)
            except Exception:
                continue
        return dates

    def _ordered_commits(self, repository: str, commits: List[str]) -> List[str]:
        commit_dates = self._commit_dates(repository, commits)
        if len(commit_dates) >= 2:
            sorted_commits = sorted(commits, key=lambda commit: commit_dates.get(commit, datetime.min))
            return sorted_commits
        return list(commits)

    def project_splits(self, repository: str) -> List[WalkForwardSplit]:
        project_df = self.df[self.df["repository"] == repository].copy()
        if project_df.empty:
            return []

        unique_commits = project_df["commit"].drop_duplicates().tolist()
        ordered_commits = self._ordered_commits(repository, unique_commits)
        result: List[WalkForwardSplit] = []

        for idx in range(1, len(ordered_commits)):
            train_commits = ordered_commits[:idx]
            test_commit = ordered_commits[idx]
            train_df = project_df[project_df["commit"].isin(train_commits)].copy()
            test_df = project_df[project_df["commit"] == test_commit].copy()
            result.append(
                WalkForwardSplit(
                    repository=repository,
                    train=train_df.reset_index(drop=True),
                    validation=None,
                    test=test_df.reset_index(drop=True),
                    train_commits=train_commits,
                    validation_commits=[],
                    test_commit=test_commit,
                )
            )

        return result

    def all_project_splits(self) -> List[WalkForwardSplit]:
        splits: List[WalkForwardSplit] = []
        for repository in self.df["repository"].drop_duplicates().tolist():
            splits.extend(self.project_splits(repository))
        return splits

    def train_validation_split(
        self,
        split: WalkForwardSplit,
        validation_ratio: float = 0.2,
        by_commit: bool = True,
    ) -> WalkForwardSplit:
        if split.validation is not None:
            return split

        train_df = split.train.copy()
        if train_df.empty:
            return split

        if by_commit:
            commit_dates = self._commit_dates(split.repository, train_df["commit"].drop_duplicates())
            commits = train_df["commit"].drop_duplicates().tolist()
            if len(commits) > 1:
                ordered_commits = self._ordered_commits(split.repository, commits)
                validation_count = max(1, int(len(ordered_commits) * validation_ratio))
                val_commits = ordered_commits[-validation_count:]
                validation_df = train_df[train_df["commit"].isin(val_commits)].copy()
                train_df = train_df[~train_df["commit"].isin(val_commits)].copy()
                return WalkForwardSplit(
                    repository=split.repository,
                    train=train_df.reset_index(drop=True),
                    validation=validation_df.reset_index(drop=True),
                    test=split.test.copy().reset_index(drop=True),
                    train_commits=[c for c in ordered_commits if c not in val_commits],
                    validation_commits=val_commits,
                    test_commit=split.test_commit,
                )

        validation_df = train_df.sample(
            frac=validation_ratio,
            random_state=42,
        )
        train_df = train_df.drop(validation_df.index)
        return WalkForwardSplit(
            repository=split.repository,
            train=train_df.reset_index(drop=True),
            validation=validation_df.reset_index(drop=True),
            test=split.test.copy().reset_index(drop=True),
            train_commits=train_df["commit"].drop_duplicates().tolist(),
            validation_commits=validation_df["commit"].drop_duplicates().tolist(),
            test_commit=split.test_commit,
        )

    def project_commit_summary(self, repository: str) -> pd.DataFrame:
        project_df = self.df[self.df["repository"] == repository].copy()
        return (
            project_df.groupby("commit")["failure_prone"]
            .agg(["count", "sum"])
            .rename(columns={"count": "sample_count", "sum": "failure_count"})
            .assign(non_failure_count=lambda df: df["sample_count"] - df["failure_count"])
            .reset_index()
        )
