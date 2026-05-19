import logging
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, average_precision_score, confusion_matrix

logger = logging.getLogger(__name__)


def evaluate_predictions(y_true, y_pred_scores, threshold=0.5):
    y_pred = (y_pred_scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    aucpr = average_precision_score(y_true, y_pred_scores) if len(set(y_true)) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "mcc": float(mcc),
        "aucpr": float(aucpr),
        "confusion_matrix": cm.tolist(),
    }
