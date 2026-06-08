from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, TypeVar

import pandas as pd

T = TypeVar("T")


def balance_dataframe(
    train_df: pd.DataFrame,
    strategy: str,
    seed: int,
    label_column: str = "failure_prone",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    before = _counts(train_df[label_column])
    strategy = (strategy or "none").lower()
    if strategy == "none" or train_df[label_column].nunique() < 2:
        result = train_df.copy().reset_index(drop=True)
    elif strategy == "random_undersampling":
        target = train_df[label_column].value_counts().min()
        parts = [group.sample(n=target, replace=False, random_state=seed) for _, group in train_df.groupby(label_column)]
        result = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    elif strategy == "random_oversampling":
        target = train_df[label_column].value_counts().max()
        parts = [group.sample(n=target, replace=True, random_state=seed) for _, group in train_df.groupby(label_column)]
        result = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported balance strategy: {strategy}")
    return result, {"strategy": strategy, "before": before, "after": _counts(result[label_column])}


def balance_sequence(samples: Sequence[T], labels: Sequence[int], strategy: str, seed: int) -> Tuple[List[T], Dict[str, object]]:
    df = pd.DataFrame({"idx": list(range(len(samples))), "failure_prone": list(labels)})
    balanced, report = balance_dataframe(df, strategy=strategy, seed=seed)
    return [samples[int(i)] for i in balanced["idx"].tolist()], report


def _counts(series: pd.Series) -> Dict[int, int]:
    return {int(k): int(v) for k, v in series.value_counts().sort_index().items()}

