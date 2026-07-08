from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, RFE, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .evaluation import compute_binary_metrics


_SKLEARN_PARALLEL_WARNING = r".*sklearn\.utils\.parallel\.delayed.*"
_PYTHONWARNINGS_FILTER = "ignore:.*sklearn.utils.parallel.delayed.*:UserWarning"

warnings.filterwarnings(
    "ignore",
    message=_SKLEARN_PARALLEL_WARNING,
    category=UserWarning,
)
if _PYTHONWARNINGS_FILTER not in os.environ.get("PYTHONWARNINGS", ""):
    os.environ["PYTHONWARNINGS"] = ",".join(
        item for item in [os.environ.get("PYTHONWARNINGS", ""), _PYTHONWARNINGS_FILTER] if item
    )


@dataclass
class TabularPreprocessor:
    feature_columns: List[str]
    scaler: str = "standard"
    feature_selection: str = "none"
    remove_constant_features: bool = True
    seed: int = 42
    n_jobs: int = -1
    rfecv_cv: int = 3
    rfecv_step: int | float = 1
    rfecv_min_features_to_select: int = 1
    rfe_n_features_to_select: int | None = None
    model_selection_scoring: str = "mcc"

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_train: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
        X_train = _numeric_frame(train_df, self.feature_columns)
        X_val = _numeric_frame(validation_df, self.feature_columns)
        X_test = _numeric_frame(test_df, self.feature_columns)

        feature_names = list(X_train.columns)
        removed: List[Dict[str, str]] = []
        all_missing = X_train.isna().all(axis=0)
        if all_missing.any():
            removed.extend({"feature": f, "reason": "all_missing_in_training"} for f, drop in all_missing.items() if drop)
            keep_cols = [f for f, drop in all_missing.items() if not drop]
            X_train = X_train[keep_cols]
            X_val = X_val[keep_cols]
            X_test = X_test[keep_cols]
            feature_names = keep_cols
        if not feature_names:
            empty_train = np.empty((len(train_df), 0))
            empty_val = np.empty((len(validation_df), 0))
            empty_test = np.empty((len(test_df), 0))
            manifest = {
                "input_features": self.feature_columns,
                "used_features": [],
                "removed_features": removed,
                "scaler": self.scaler,
                "feature_selection": self.feature_selection,
                "feature_selection_details": {
                    "status": "skipped",
                    "reason": "no_features_after_all_missing_filter",
                },
                "imputer": "median",
            }
            return empty_train, empty_val, empty_test, manifest
        imputer = SimpleImputer(strategy="median")
        X_train_np = imputer.fit_transform(X_train)
        X_val_np = imputer.transform(X_val)
        X_test_np = imputer.transform(X_test)

        if self.remove_constant_features or self.feature_selection == "variance_threshold":
            selector = VarianceThreshold(threshold=0.0)
            try:
                X_train_np = selector.fit_transform(X_train_np)
                X_val_np = selector.transform(X_val_np)
                X_test_np = selector.transform(X_test_np)
                kept_mask = selector.get_support()
                removed.extend({"feature": f, "reason": "constant_or_zero_variance"} for f, keep in zip(feature_names, kept_mask) if not keep)
                feature_names = [f for f, keep in zip(feature_names, kept_mask) if keep]
            except ValueError:
                removed.extend({"feature": f, "reason": "constant_or_zero_variance"} for f in feature_names)
                feature_names = []
                X_train_np = np.empty((len(train_df), 0))
                X_val_np = np.empty((len(validation_df), 0))
                X_test_np = np.empty((len(test_df), 0))

        scaler_obj = None
        if self.scaler == "standard":
            scaler_obj = StandardScaler()
        elif self.scaler == "min-max":
            scaler_obj = MinMaxScaler()
        elif self.scaler == "none":
            scaler_obj = None
        else:
            raise ValueError(f"Unsupported scaler: {self.scaler}")
        if scaler_obj is not None and X_train_np.shape[1] > 0:
            X_train_np = scaler_obj.fit_transform(X_train_np)
            X_val_np = scaler_obj.transform(X_val_np)
            X_test_np = scaler_obj.transform(X_test_np)

        if self.feature_selection in {"rfecv", "rfe", "validation_rfe"} and X_train_np.shape[1] > 0:
            X_train_np, X_val_np, X_test_np, feature_names, selection_removed, selection_details = self._fit_feature_selector(
                X_train_np,
                X_val_np,
                X_test_np,
                feature_names,
                y_train,
                y_val,
            )
            removed.extend(selection_removed)
        else:
            selection_details = {}

        manifest = {
            "input_features": self.feature_columns,
            "used_features": feature_names,
            "removed_features": removed,
            "scaler": self.scaler,
            "feature_selection": self.feature_selection,
            "feature_selection_details": selection_details,
            "imputer": "median",
        }
        return X_train_np, X_val_np, X_test_np, manifest

    def _fit_feature_selector(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        y_train: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[Dict[str, str]], Dict[str, object]]:
        if y_train is None or len(set(y_train.astype(int).tolist())) < 2:
            return X_train, X_val, X_test, feature_names, [], {
                "status": "skipped",
                "reason": "missing_or_single_class_y_train",
            }
        estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            class_weight=None,
        )
        selector_name = self.feature_selection
        details: Dict[str, object] = {"status": "applied", "method": selector_name}
        try:
            if selector_name == "validation_rfe":
                return self._fit_validation_rfe(
                    estimator,
                    X_train,
                    X_val,
                    X_test,
                    feature_names,
                    y_train,
                    y_val,
                    details,
                )
            if selector_name == "rfecv":
                class_counts = np.bincount(y_train.astype(int))
                positive_counts = class_counts[class_counts > 0]
                max_cv = int(positive_counts.min()) if len(positive_counts) else 0
                cv_splits = min(int(self.rfecv_cv), max_cv)
                if cv_splits < 2:
                    return X_train, X_val, X_test, feature_names, [], {
                        "status": "skipped",
                        "reason": "too_few_training_samples_per_class_for_rfecv",
                    }
                selector = RFECV(
                    estimator=estimator,
                    step=self.rfecv_step,
                    min_features_to_select=max(1, min(int(self.rfecv_min_features_to_select), X_train.shape[1])),
                    cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.seed),
                    scoring=make_scorer(matthews_corrcoef),
                    n_jobs=self.n_jobs,
                )
                details["cv_splits"] = cv_splits
            else:
                n_features = self.rfe_n_features_to_select
                if n_features is None:
                    n_features = max(1, min(20, X_train.shape[1] // 2 or 1))
                selector = RFE(
                    estimator=estimator,
                    n_features_to_select=max(1, min(int(n_features), X_train.shape[1])),
                    step=self.rfecv_step,
                )
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_val_selected = selector.transform(X_val)
            X_test_selected = selector.transform(X_test)
            support = selector.get_support()
            kept_features = [feature for feature, keep in zip(feature_names, support) if keep]
            removed = [
                {"feature": feature, "reason": f"removed_by_{selector_name}"}
                for feature, keep in zip(feature_names, support)
                if not keep
            ]
            details["selected_feature_count"] = len(kept_features)
            details["removed_feature_count"] = len(removed)
            if hasattr(selector, "n_features_"):
                details["n_features_"] = int(selector.n_features_)
            return X_train_selected, X_val_selected, X_test_selected, kept_features, removed, details
        except Exception as exc:
            details["status"] = "skipped"
            details["reason"] = f"{selector_name}_failed:{type(exc).__name__}:{exc}"
            return X_train, X_val, X_test, feature_names, [], details

    def _fit_validation_rfe(
        self,
        estimator: RandomForestClassifier,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        y_train: np.ndarray,
        y_val: np.ndarray | None,
        details: Dict[str, object],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[Dict[str, str]], Dict[str, object]]:
        if y_val is None or len(y_val) == 0:
            details.update({
                "status": "skipped",
                "reason": "missing_validation_labels",
            })
            return X_train, X_val, X_test, feature_names, [], details

        selector = RFE(
            estimator=estimator,
            n_features_to_select=1,
            step=self.rfecv_step,
        )
        selector.fit(X_train, y_train)
        ranking = np.asarray(selector.ranking_, dtype=int)
        candidate_counts = _candidate_feature_counts(
            X_train.shape[1],
            self.rfecv_step,
            int(self.rfecv_min_features_to_select),
        )
        scoring_order = _validation_scoring_order(self.model_selection_scoring)
        best: Dict[str, object] | None = None
        evaluated: List[Dict[str, object]] = []
        ranked_indices = np.argsort(ranking, kind="stable")
        for count in candidate_counts:
            mask = np.zeros(X_train.shape[1], dtype=bool)
            mask[ranked_indices[:count]] = True
            selected_count = int(mask.sum())
            if selected_count == 0:
                continue
            candidate = clone(estimator)
            candidate.fit(X_train[:, mask], y_train)
            y_pred = candidate.predict(X_val[:, mask]).astype(int)
            y_score = _scores(candidate, X_val[:, mask], y_pred)
            metrics = compute_binary_metrics(y_val.astype(int), y_pred, y_score)
            score_metric, score = _first_defined_metric(metrics, scoring_order)
            row = {
                "candidate_feature_count": selected_count,
                "score_metric": score_metric,
                "validation_score": score,
            }
            evaluated.append(row)
            if score == score and (best is None or float(score) > float(best["validation_score"])):
                best = row | {"mask": mask}

        details["candidate_feature_counts"] = [row["candidate_feature_count"] for row in evaluated]
        details["validation_scoring_order"] = scoring_order
        if best is None:
            details.update({
                "status": "skipped",
                "reason": "no_defined_validation_score",
            })
            return X_train, X_val, X_test, feature_names, [], details

        best_mask = np.asarray(best["mask"], dtype=bool)
        kept_features = [feature for feature, keep in zip(feature_names, best_mask) if keep]
        removed = [
            {"feature": feature, "reason": "removed_by_validation_rfe"}
            for feature, keep in zip(feature_names, best_mask)
            if not keep
        ]
        details.update({
            "selected_feature_count": len(kept_features),
            "removed_feature_count": len(removed),
            "best_validation_metric": best["score_metric"],
            "best_validation_score": best["validation_score"],
            "fallback_used": best["score_metric"] != self.model_selection_scoring,
            "ranking_source": "training_only_rfe",
            "selection_source": "validation",
        })
        return (
            X_train[:, best_mask],
            X_val[:, best_mask],
            X_test[:, best_mask],
            kept_features,
            removed,
            details,
        )


def _numeric_frame(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    series = {
        col: pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for col in columns
    }
    return pd.DataFrame(series, index=df.index)


def _candidate_feature_counts(feature_count: int, step: int | float, min_features: int) -> List[int]:
    minimum = max(1, min(int(min_features), feature_count))
    if isinstance(step, float) and 0 < step < 1:
        stride = max(1, int(np.ceil(feature_count * step)))
    else:
        stride = max(1, int(step))
    counts = list(range(minimum, feature_count + 1, stride))
    if feature_count not in counts:
        counts.append(feature_count)
    return sorted(set(counts))


def _validation_scoring_order(primary: str) -> List[str]:
    order = [primary, "f1", "accuracy"]
    result: List[str] = []
    for metric in order:
        if metric and metric not in result:
            result.append(metric)
    return result


def _first_defined_metric(metrics: Dict[str, object], scoring_order: List[str]) -> Tuple[str, float]:
    for metric in scoring_order:
        value = metrics.get(metric, np.nan)
        try:
            score = float(value)
        except Exception:
            score = np.nan
        if score == score:
            return metric, score
    return "", np.nan


def _scores(model: object, X: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if proba.shape[1] > 1:
                return proba[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            raw = model.decision_function(X)
            raw = np.asarray(raw, dtype=float)
            return 1.0 / (1.0 + np.exp(-raw))
        except Exception:
            pass
    return np.asarray(y_pred, dtype=float)
