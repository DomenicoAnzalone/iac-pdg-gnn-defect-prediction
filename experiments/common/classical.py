from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .balancing import balance_dataframe
from .evaluation import compute_binary_metrics
from .preprocessing import TabularPreprocessor
from .progress import get_logger, progress
from .splitting import Split, materialize_split


MODEL_ALIASES = {
    "decision_tree": "decision_tree",
    "dt": "decision_tree",
    "logistic_regression": "logistic_regression",
    "lr": "logistic_regression",
    "naive_bayes": "naive_bayes",
    "nb": "naive_bayes",
    "random_forest": "random_forest",
    "rf": "random_forest",
    "svm": "svm",
    "support_vector_machine": "svm",
}


def model_factory(name: str, seed: int, n_jobs: int = -1) -> BaseEstimator:
    canonical = MODEL_ALIASES.get(name.lower(), name.lower())
    if canonical == "decision_tree":
        return DecisionTreeClassifier(random_state=seed)
    if canonical == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight=None, random_state=seed, n_jobs=None)
    if canonical == "naive_bayes":
        return GaussianNB()
    if canonical == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=n_jobs)
    if canonical == "svm":
        return SVC(kernel="rbf", probability=True, random_state=seed)
    raise ValueError(f"Unknown classical model: {name}")


def param_space(name: str) -> Dict[str, List[Any]]:
    canonical = MODEL_ALIASES.get(name.lower(), name.lower())
    if canonical == "decision_tree":
        return {"max_depth": [None, 4, 8, 16], "min_samples_leaf": [1, 2, 5, 10]}
    if canonical == "logistic_regression":
        return {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs", "liblinear"]}
    if canonical == "random_forest":
        return {"n_estimators": [100, 200, 400], "max_depth": [None, 8, 16], "min_samples_leaf": [1, 2, 5]}
    if canonical == "svm":
        return {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]}
    return {}


def run_tabular_experiment(
    df: pd.DataFrame,
    splits: List[Split],
    feature_columns: List[str],
    model_names: List[str],
    experiment: str,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], pd.DataFrame]:
    logger = get_logger(f"experiments.{experiment}.classical")
    all_predictions: List[pd.DataFrame] = []
    all_metrics: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    show_progress = bool(config.get("progress", True))
    logger.info(
        "%s: avvio training tabellare con %s feature, %s modelli, %s split",
        experiment.upper(),
        len(feature_columns),
        len(model_names),
        len(splits),
    )
    for model_name in model_names:
        canonical_model = MODEL_ALIASES.get(model_name.lower(), model_name.lower())
        logger.info("%s: modello %s", experiment.upper(), canonical_model)
        for split in progress(splits, total=len(splits), desc=f"{experiment}:{canonical_model}", unit="split", enabled=show_progress):
            train_df, val_df, test_df = materialize_split(df, split)
            logger.info(
                "%s/%s split=%s repo=%s train=%s val=%s test=%s",
                experiment,
                canonical_model,
                split.split_id,
                split.repository,
                len(train_df),
                len(val_df),
                len(test_df),
            )
            balanced_train_df, balance_report = balance_dataframe(
                train_df,
                strategy=config.get("balance_strategy", "none"),
                seed=int(config.get("seed", 42)),
            )
            y_train = balanced_train_df["failure_prone"].astype(int).to_numpy()
            y_val = val_df["failure_prone"].astype(int).to_numpy()
            y_test = test_df["failure_prone"].astype(int).to_numpy()
            preprocessor = TabularPreprocessor(
                feature_columns=feature_columns,
                scaler=config.get("scaler", "standard"),
                feature_selection=config.get("feature_selection", "none"),
                remove_constant_features=bool(config.get("remove_constant_features", True)),
                seed=int(config.get("seed", 42)),
                n_jobs=int(config.get("n_jobs", -1)),
                rfecv_cv=int(config.get("rfecv_cv", 3)),
                rfecv_step=config.get("rfecv_step", 1),
                rfecv_min_features_to_select=int(config.get("rfecv_min_features_to_select", 1)),
                rfe_n_features_to_select=config.get("rfe_n_features_to_select"),
            )
            X_train, X_val, X_test, feature_manifest = preprocessor.fit_transform(
                balanced_train_df,
                val_df,
                test_df,
                y_train=y_train,
            )
            if X_train.shape[1] == 0:
                logger.warning("%s/%s split=%s saltato: nessuna feature dopo preprocessing", experiment, canonical_model, split.split_id)
                continue
            model = _select_model(model_name, X_train, y_train, X_val, y_val, config)
            start = time.time()
            logger.info("%s/%s split=%s fitting modello", experiment, canonical_model, split.split_id)
            model.fit(X_train, y_train)
            training_seconds = time.time() - start
            y_pred = model.predict(X_test).astype(int)
            y_score = _scores(model, X_test, y_pred)
            pred_df = _prediction_frame(test_df, split, experiment, canonical_model, config, y_pred, y_score)
            metrics = compute_binary_metrics(y_test, y_pred, y_score)
            metrics.update({
                "experiment": experiment,
                "model": canonical_model,
                "split_id": split.split_id,
                "repository": split.repository,
                "train_size": len(train_df),
                "train_size_after_balance": len(balanced_train_df),
                "validation_size": len(val_df),
                "test_size": len(test_df),
                "training_seconds": training_seconds,
                "balance_before": str(balance_report["before"]),
                "balance_after": str(balance_report["after"]),
            })
            all_predictions.append(pred_df)
            all_metrics.append(metrics)
            logger.info(
                "%s/%s split=%s completato: mcc=%s auc_pr=%s f1=%s tempo=%.2fs",
                experiment,
                canonical_model,
                split.split_id,
                metrics.get("mcc"),
                metrics.get("auc_pr"),
                metrics.get("f1"),
                training_seconds,
            )
            for feature in feature_manifest["used_features"]:
                feature_rows.append({"experiment": experiment, "model": canonical_model, "split_id": split.split_id, "feature": feature, "status": "used", "reason": ""})
            for removed in feature_manifest["removed_features"]:
                feature_rows.append({"experiment": experiment, "model": canonical_model, "split_id": split.split_id, "feature": removed["feature"], "status": "removed", "reason": removed["reason"]})
            details = feature_manifest.get("feature_selection_details", {})
            if details:
                feature_rows.append({
                    "experiment": experiment,
                    "model": canonical_model,
                    "split_id": split.split_id,
                    "feature": "__feature_selection__",
                    "status": str(details.get("status", "")),
                    "reason": str(details),
                })
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    return predictions, all_metrics, pd.DataFrame(feature_rows)


def _select_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, config: Dict[str, Any]) -> BaseEstimator:
    seed = int(config.get("seed", 42))
    n_jobs = int(config.get("n_jobs", -1))
    base = model_factory(model_name, seed=seed, n_jobs=n_jobs)
    if not config.get("hyperparameter_search", False) or not param_space(model_name):
        return base
    best_model = base
    best_score = -np.inf
    sampler = ParameterSampler(param_space(model_name), n_iter=int(config.get("random_search_iter", 10)), random_state=seed)
    for params in sampler:
        candidate = clone(base).set_params(**params)
        try:
            candidate.fit(X_train, y_train)
            y_pred = candidate.predict(X_val).astype(int)
            y_score = _scores(candidate, X_val, y_pred)
            score = compute_binary_metrics(y_val, y_pred, y_score).get(config.get("model_selection_scoring", "mcc"), np.nan)
            if score == score and float(score) > best_score:
                best_score = float(score)
                best_model = clone(base).set_params(**params)
        except Exception:
            continue
    return best_model


def _scores(model: BaseEstimator, X: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
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


def _prediction_frame(
    test_df: pd.DataFrame,
    split: Split,
    experiment: str,
    model: str,
    config: Dict[str, Any],
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame({
        "repository": test_df["repository"].astype(str).tolist(),
        "commit": test_df["commit"].astype(str).tolist(),
        "filepath": test_df["filepath"].astype(str).tolist(),
        "split_id": split.split_id,
        "experiment": experiment,
        "model": model,
        "config_id": config.get("run_name", ""),
        "y_true": test_df["failure_prone"].astype(int).tolist(),
        "y_pred": y_pred.tolist(),
        "y_score": y_score.tolist(),
    })
