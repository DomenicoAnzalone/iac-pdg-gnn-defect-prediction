from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, TypeVar, Union

import numpy as np
import pandas as pd
from sklearn.utils import resample

SampleType = TypeVar("SampleType")
LabelExtractor = Callable[[SampleType], int]


@dataclass
class BalanceReport:
    label_counts: Dict[int, int]
    total_samples: int
    majority_label: Optional[int]
    minority_label: Optional[int]
    ratio_minority_to_majority: Optional[float]


class GraphBalancer:
    """Balance graph-level samples using undersampling or oversampling."""

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.rng = random.Random(random_state)

    def label_counts(self, samples: Sequence[SampleType], label_extractor: LabelExtractor) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for sample in samples:
            label = label_extractor(sample)
            counts[label] = counts.get(label, 0) + 1
        return counts

    def class_balance(self, samples: Sequence[SampleType], label_extractor: LabelExtractor) -> BalanceReport:
        counts = self.label_counts(samples, label_extractor)
        labels = sorted(counts.keys())
        if len(labels) < 2:
            return BalanceReport(counts, len(samples), None, None, None)

        majority_label = max(labels, key=lambda lbl: counts[lbl])
        minority_label = min(labels, key=lambda lbl: counts[lbl])
        majority_count = counts[majority_label]
        minority_count = counts[minority_label]
        ratio = minority_count / majority_count if majority_count else None
        return BalanceReport(
            label_counts=counts,
            total_samples=len(samples),
            majority_label=majority_label,
            minority_label=minority_label,
            ratio_minority_to_majority=ratio,
        )

    def undersample(
        self,
        samples: Sequence[SampleType],
        label_extractor: LabelExtractor,
        target_count: Optional[int] = None,
    ) -> List[SampleType]:
        groups = self._group_by_label(samples, label_extractor)
        if target_count is None:
            target_count = min(len(group) for group in groups.values())

        result: List[SampleType] = []
        for label, group in groups.items():
            if len(group) > target_count:
                result.extend(
                    resample(
                        group,
                        replace=False,
                        n_samples=target_count,
                        random_state=self.random_state,
                    )
                )
            else:
                result.extend(group)

        self.rng.shuffle(result)
        return result

    def oversample(
        self,
        samples: Sequence[SampleType],
        label_extractor: LabelExtractor,
        target_count: Optional[int] = None,
    ) -> List[SampleType]:
        groups = self._group_by_label(samples, label_extractor)
        if target_count is None:
            target_count = max(len(group) for group in groups.values())

        result: List[SampleType] = []
        for label, group in groups.items():
            if len(group) < target_count:
                result.extend(
                    resample(
                        group,
                        replace=True,
                        n_samples=target_count,
                        random_state=self.random_state,
                    )
                )
            else:
                result.extend(group)

        self.rng.shuffle(result)
        return result

    def balance(
        self,
        samples: Sequence[SampleType],
        label_extractor: LabelExtractor,
        strategy: str = "oversample",
        target_count: Optional[int] = None,
    ) -> List[SampleType]:
        strategy = strategy.lower()
        if strategy == "undersample":
            return self.undersample(samples, label_extractor, target_count=target_count)
        if strategy == "oversample":
            return self.oversample(samples, label_extractor, target_count=target_count)
        raise ValueError(f"Unsupported balancing strategy: {strategy}")

    def _group_by_label(self, samples: Sequence[SampleType], label_extractor: LabelExtractor) -> Dict[int, List[SampleType]]:
        groups: Dict[int, List[SampleType]] = {}
        for sample in samples:
            label = label_extractor(sample)
            groups.setdefault(label, []).append(sample)
        return groups

    def dataframe_undersample(
        self,
        df: pd.DataFrame,
        label_column: str = "failure_prone",
        target_count: Optional[int] = None,
    ) -> pd.DataFrame:
        if target_count is None:
            target_count = int(df[label_column].value_counts().min())

        result_frames = [
            grp.sample(n=target_count, replace=False, random_state=self.random_state)
            if len(grp) > target_count
            else grp
            for _, grp in df.groupby(label_column)
        ]
        return pd.concat(result_frames).sample(frac=1, random_state=self.random_state).reset_index(drop=True)

    def dataframe_oversample(
        self,
        df: pd.DataFrame,
        label_column: str = "failure_prone",
        target_count: Optional[int] = None,
    ) -> pd.DataFrame:
        counts = df[label_column].value_counts()
        if target_count is None:
            target_count = int(counts.max())

        result_frames = [
            grp.sample(n=target_count, replace=True, random_state=self.random_state)
            if len(grp) < target_count
            else grp
            for _, grp in df.groupby(label_column)
        ]
        return pd.concat(result_frames).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
