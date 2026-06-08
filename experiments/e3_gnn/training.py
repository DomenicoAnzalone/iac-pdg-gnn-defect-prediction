from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from experiments.common.evaluation import compute_binary_metrics

try:
    import torch
    from torch_geometric.loader import DataLoader
except Exception:  # pragma: no cover
    torch = None
    DataLoader = None

from experiments.e3_gnn.models import get_model


GNN_ALIASES = {"gcn": "gcn", "graphsage": "graphsage", "sage": "graphsage", "gat": "gat", "gin": "gin", "rgcn": "rgcn", "r-gcn": "rgcn"}


def run_gnn_model(
    model_name: str,
    train_data: List[object],
    validation_data: List[object],
    test_data: List[object],
    split_id: str,
    repository: str,
    config: Dict[str, Any],
    model_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch Geometric is not available")
    canonical = GNN_ALIASES.get(model_name.lower(), model_name.lower())
    device = _device(config.get("device", "auto"))
    in_channels = int(train_data[0].x.shape[1])
    num_relations = max([int(data.edge_type.max().item()) if data.edge_type.numel() else 0 for data in train_data + validation_data + test_data] + [0]) + 1
    model_kwargs = {
        "hidden": int(config.get("hidden_channels", 64)),
        "num_classes": 2,
        "num_layers": int(config.get("num_layers", 2)),
        "dropout": float(config.get("dropout", 0.5)),
        "readout": config.get("pooling", "mean"),
    }
    if canonical == "rgcn":
        model_kwargs["num_relations"] = num_relations
    model = get_model(canonical, in_channels=in_channels, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-3)), weight_decay=float(config.get("weight_decay", 0.0)))
    criterion = _criterion(train_data, config, device)
    train_loader = DataLoader(train_data, batch_size=int(config.get("batch_size", 32)), shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=int(config.get("batch_size", 32)), shuffle=False)
    history = []
    best_score = -np.inf
    best_path = model_dir / f"{canonical}_{split_id}_best.pt"
    patience = 0
    start = time.time()
    for epoch in range(1, int(config.get("epochs", 100)) + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_pred = _predict(model, val_loader, device)
        val_metrics = compute_binary_metrics(val_pred["y_true"], val_pred["y_pred"], val_pred["y_score"])
        score = val_metrics.get(config.get("early_stopping_metric", "mcc"), np.nan)
        score_value = float(score) if score == score else -np.inf
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}})
        if score_value > best_score:
            best_score = score_value
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
        if patience >= int(config.get("early_stopping_patience", 10)):
            break
    training_seconds = time.time() - start
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_loader = DataLoader(test_data, batch_size=int(config.get("batch_size", 32)), shuffle=False)
    test_pred = _predict(model, test_loader, device)
    metrics = compute_binary_metrics(test_pred["y_true"], test_pred["y_pred"], test_pred["y_score"])
    metrics.update({
        "experiment": "e3",
        "model": canonical,
        "split_id": split_id,
        "repository": repository,
        "train_size": len(train_data),
        "validation_size": len(validation_data),
        "test_size": len(test_data),
        "training_seconds": training_seconds,
        "epochs_ran": len(history),
        "best_validation_score": best_score if best_score != -np.inf else np.nan,
    })
    (model_dir / f"{canonical}_{split_id}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    predictions = pd.DataFrame({
        "repository": [getattr(data, "repository", "") for data in test_data],
        "commit": [getattr(data, "commit", "") for data in test_data],
        "filepath": [getattr(data, "filepath", "") for data in test_data],
        "split_id": split_id,
        "experiment": "e3",
        "model": canonical,
        "config_id": config.get("run_name", ""),
        "y_true": test_pred["y_true"],
        "y_pred": test_pred["y_pred"],
        "y_score": test_pred["y_score"],
    })
    return predictions, metrics


def _train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total = 0.0
    graphs = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = batch.y.view(-1).long()
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * int(batch.num_graphs)
        graphs += int(batch.num_graphs)
    return total / max(graphs, 1)


def _predict(model, loader, device) -> Dict[str, List[Any]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            scores = torch.softmax(logits, dim=1)[:, 1]
            pred = (scores >= 0.5).long()
            y_true.extend(batch.y.view(-1).cpu().numpy().astype(int).tolist())
            y_pred.extend(pred.cpu().numpy().astype(int).tolist())
            y_score.extend(scores.cpu().numpy().astype(float).tolist())
    return {"y_true": y_true, "y_pred": y_pred, "y_score": y_score}


def _criterion(train_data: List[object], config: Dict[str, Any], device) -> object:
    if not config.get("class_weights", False):
        return torch.nn.CrossEntropyLoss()
    labels = np.asarray([int(data.y.item()) for data in train_data])
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = counts.sum() / np.maximum(counts, 1.0)
    return torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def _device(value: str):
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
