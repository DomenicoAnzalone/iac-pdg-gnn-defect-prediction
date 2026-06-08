from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_COLUMNS = ["auc_pr", "auc_roc", "mcc", "precision", "recall", "f1", "accuracy"]


def compute_binary_metrics(y_true: Iterable[int], y_pred: Iterable[int], y_score: Iterable[float] | None = None) -> Dict[str, object]:
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    ys = None if y_score is None else np.asarray(list(y_score), dtype=float)
    warnings: List[str] = []
    metrics: Dict[str, object] = {}
    if len(yt) == 0:
        return {metric: math.nan for metric in METRIC_COLUMNS} | {"tn": 0, "fp": 0, "fn": 0, "tp": 0, "warnings": "empty_test_set"}
    metrics["precision"] = float(precision_score(yt, yp, zero_division=0))
    metrics["recall"] = float(recall_score(yt, yp, zero_division=0))
    metrics["f1"] = float(f1_score(yt, yp, zero_division=0))
    metrics["accuracy"] = float(accuracy_score(yt, yp))
    metrics["mcc"] = float(matthews_corrcoef(yt, yp)) if len(set(yt)) > 1 and len(set(yp)) > 1 else math.nan
    if len(set(yt)) > 1 and ys is not None and len(ys) == len(yt):
        metrics["auc_pr"] = float(average_precision_score(yt, ys))
        metrics["auc_roc"] = float(roc_auc_score(yt, ys))
    else:
        metrics["auc_pr"] = math.nan
        metrics["auc_roc"] = math.nan
        warnings.append("auc_undefined_single_class_or_missing_score")
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    metrics.update({"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])})
    metrics["warnings"] = ";".join(warnings)
    return metrics


def aggregate_metrics(metrics_rows: List[Dict[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for metric in METRIC_COLUMNS:
        vals = [float(row[metric]) for row in metrics_rows if row.get(metric) == row.get(metric)]
        result[f"{metric}_mean"] = float(np.mean(vals)) if vals else math.nan
        result[f"{metric}_median"] = float(np.median(vals)) if vals else math.nan
    result["split_count"] = len(metrics_rows)
    return result

