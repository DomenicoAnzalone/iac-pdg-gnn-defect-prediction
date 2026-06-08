from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class TabularPreprocessor:
    feature_columns: List[str]
    scaler: str = "standard"
    feature_selection: str = "none"
    remove_constant_features: bool = True

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
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
                pass

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

        manifest = {
            "input_features": self.feature_columns,
            "used_features": feature_names,
            "removed_features": removed,
            "scaler": self.scaler,
            "feature_selection": self.feature_selection,
            "imputer": "median",
        }
        return X_train_np, X_val_np, X_test_np, manifest


def _numeric_frame(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    series = {
        col: pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for col in columns
    }
    return pd.DataFrame(series, index=df.index)
