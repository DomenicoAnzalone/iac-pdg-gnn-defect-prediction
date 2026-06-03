from __future__ import annotations

from typing import List, Tuple
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def aggregate_node_features(node_features: np.ndarray) -> np.ndarray:
    # node_features: (n_nodes, f)
    if node_features.size == 0:
        return np.zeros(10, dtype=float)
    mean = node_features.mean(axis=0)
    summ = node_features.sum(axis=0)
    mx = node_features.max(axis=0)
    std = node_features.std(axis=0)
    return np.concatenate([mean, summ, mx, std])


def build_graph_feature_vectors(graph_datas: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for data in graph_datas:
        x = data.get("x")
        if isinstance(x, np.ndarray):
            vec = aggregate_node_features(x)
        else:
            # fallback
            vec = np.zeros(10, dtype=float)
        X.append(vec)
        y.append(int(data.get("y", [0])[0]))
    return np.vstack(X), np.array(y, dtype=int)


def train_baseline(train_datas: List[dict], val_datas: List[dict], out_path=None):
    X_train, y_train = build_graph_feature_vectors(train_datas)
    X_val, y_val = build_graph_feature_vectors(val_datas)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return {"model": rf, "accuracy": float(acc)}
