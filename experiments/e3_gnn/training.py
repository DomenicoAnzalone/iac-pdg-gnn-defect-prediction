from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from experiments.common.evaluation import compute_binary_metrics
from experiments.common.progress import CompactStatusLine, get_logger, progress

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
    compact_status: CompactStatusLine | None = None,
    split_index: int | None = None,
    total_splits: int | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch Geometric is not available")
    logger = get_logger("experiments.e3_gnn.training")
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
    max_epochs = int(config.get("epochs", 100))
    log_every = max(1, int(config.get("log_every_epochs", 1)))
    compact_progress = bool(config.get("compact_progress", False))
    if compact_progress:
        epoch_iter = range(1, max_epochs + 1)
    else:
        epoch_iter = progress(
            range(1, max_epochs + 1),
            total=max_epochs,
            desc="Epoch",
            unit="epoch",
            enabled=bool(config.get("progress", True)),
            leave=False,
            position=None,
            dynamic_ncols=True,
        )
    logger.info(
        "E3/%s split=%s training avviato: train_graphs=%s val_graphs=%s test_graphs=%s device=%s epochs=%s",
        canonical,
        split_id,
        len(train_data),
        len(validation_data),
        len(test_data),
        device,
        max_epochs,
    )
    for epoch in epoch_iter:
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_pred = _evaluate(model, val_loader, criterion, device)
        val_metrics = compute_binary_metrics(val_pred["y_true"], val_pred["y_pred"], val_pred["y_score"])
        score = val_metrics.get(config.get("early_stopping_metric", "mcc"), np.nan)
        metric_score = float(score) if score == score else np.nan
        used_loss_fallback = metric_score != metric_score
        score_value = -val_loss if used_loss_fallback else metric_score
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "early_stopping_score": score_value,
            "early_stopping_used_loss_fallback": used_loss_fallback,
            **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))},
        })
        if score_value > best_score:
            best_score = score_value
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
        if compact_progress and compact_status is not None and split_index is not None:
            compact_status.update(
                split_index=split_index,
                completed_splits=max(0, split_index - 1),
                epoch=epoch,
                total_epochs=max_epochs,
                loss=f"{train_loss:.3f}",
                best=_format_metric(best_score if best_score != -np.inf else score_value),
                patience=f"{patience}/{int(config.get('early_stopping_patience', 10))}",
            )
        elif hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                loss=f"{train_loss:.3f}",
                best=_format_metric(best_score if best_score != -np.inf else score_value),
                pat=f"{patience}/{int(config.get('early_stopping_patience', 10))}",
            )
        if epoch == 1 or epoch % log_every == 0:
            logger.info(
                "E3/%s split=%s epoch=%s loss=%.4f val_loss=%.4f val_mcc=%s val_auc_pr=%s early_stop=%s patience=%s",
                canonical,
                split_id,
                epoch,
                train_loss,
                val_loss,
                _format_metric(val_metrics.get("mcc")),
                _format_metric(val_metrics.get("auc_pr")),
                "val_loss" if used_loss_fallback else config.get("early_stopping_metric", "mcc"),
                patience,
            )
        if patience >= int(config.get("early_stopping_patience", 10)):
            logger.info("E3/%s split=%s early stopping a epoch=%s", canonical, split_id, epoch)
            break
    training_seconds = time.time() - start
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_loader = DataLoader(test_data, batch_size=int(config.get("batch_size", 32)), shuffle=False)
    _, test_pred = _evaluate(model, test_loader, criterion, device)
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
        "best_early_stopping_score": best_score if best_score != -np.inf else np.nan,
        "early_stopping_used_loss_fallback": any(bool(row.get("early_stopping_used_loss_fallback")) for row in history),
        "best_validation_score": _best_defined_validation_metric(history, config.get("early_stopping_metric", "mcc")),
    })
    logger.info(
        "E3/%s split=%s test completato: mcc=%s auc_pr=%s f1=%s tempo=%.2fs",
        canonical,
        split_id,
        _format_metric(metrics.get("mcc")),
        _format_metric(metrics.get("auc_pr")),
        _format_metric(metrics.get("f1")),
        training_seconds,
    )
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


def _evaluate(model, loader, criterion, device) -> Tuple[float, Dict[str, List[Any]]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []
    total_loss = 0.0
    graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            y = batch.y.view(-1).long()
            loss = criterion(logits, y)
            scores = torch.softmax(logits, dim=1)[:, 1]
            pred = (scores >= 0.5).long()
            total_loss += float(loss.item()) * int(batch.num_graphs)
            graphs += int(batch.num_graphs)
            y_true.extend(y.cpu().numpy().astype(int).tolist())
            y_pred.extend(pred.cpu().numpy().astype(int).tolist())
            y_score.extend(scores.cpu().numpy().astype(float).tolist())
    return total_loss / max(graphs, 1), {"y_true": y_true, "y_pred": y_pred, "y_score": y_score}


def _best_defined_validation_metric(history: List[Dict[str, Any]], metric_name: str) -> float:
    values = []
    key = f"val_{metric_name}"
    for row in history:
        value = row.get(key)
        if isinstance(value, (int, float)) and value == value:
            values.append(float(value))
    return max(values) if values else np.nan


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


def _format_metric(value: object) -> str:
    try:
        val = float(value)
        if val != val:
            return "nan"
        return f"{val:.4f}"
    except Exception:
        return "nan"
