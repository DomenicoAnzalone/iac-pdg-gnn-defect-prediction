from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class FeatureScaler:
    """Standardize or scale node/edge feature matrices consistently."""

    def __init__(self, method: Literal["standard", "minmax"] = "standard"):
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaling method: {method}")
        self.method = method
        self.fitted = False

    def fit(self, features: np.ndarray) -> FeatureScaler:
        self.scaler.fit(features)
        self.fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("FeatureScaler must be fit before transform.")
        return self.scaler.transform(features).astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.fit(features)
        return self.transform(features)

    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("FeatureScaler must be fit before inverse_transform.")
        return self.scaler.inverse_transform(features)

    def get_params(self) -> dict:
        return {"method": self.method, **self.scaler.get_params()} if self.fitted else {"method": self.method}
