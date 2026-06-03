from __future__ import annotations

from typing import List, Dict

from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, average_precision_score


def compute_metrics(y_true: List[int], y_pred: List[int], y_score: List[float]) -> Dict[str, float]:
    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0, "pr_auc": 0.0}
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    try:
        pr_auc = average_precision_score(y_true, y_score) if any(y_score) else 0.0
    except Exception:
        pr_auc = 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "mcc": float(mcc), "pr_auc": float(pr_auc)}
