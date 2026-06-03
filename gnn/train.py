from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict

import time
import json

import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data
except Exception:  # pragma: no cover
    Data = None

# Prefer the new loader API when available to avoid deprecation warnings.
try:
    from torch_geometric.loader import DataLoader
except Exception:
    try:
        from torch_geometric.data import DataLoader
    except Exception:  # pragma: no cover
        DataLoader = None

from .models import get_model
from .eval import compute_metrics


class Trainer:
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None, lr: float = 1e-3, weight_decay: float = 0.0):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, train_data: List[Data], val_data: Optional[List[Data]] = None, epochs: int = 20, batch_size: int = 16, early_stopping: int = 5, out_dir: Optional[Path] = None):
        if DataLoader is None:
            raise RuntimeError("torch_geometric DataLoader not available")

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) if val_data else None

        best_val = None
        patience = 0
        history = {"train_loss": [], "val_loss": [], "val_f1": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                y = batch.y.view(-1).long().to(self.device)
                if logits.dim() == 1 or logits.shape[1] == 1:
                    logits = logits.view(-1, 2)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch.num_graphs

            avg_train_loss = total_loss / len(train_data)
            history["train_loss"].append(avg_train_loss)

            val_loss = None
            val_f1 = None
            if val_loader is not None:
                self.model.eval()
                ys = []
                preds = []
                probs = []
                total_vloss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        logits = self.model(batch)
                        y = batch.y.view(-1).long().to(self.device)
                        if logits.dim() == 1 or logits.shape[1] == 1:
                            logits = logits.view(-1, 2)
                        loss = self.criterion(logits, y)
                        total_vloss += loss.item() * batch.num_graphs
                        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                        p = (scores >= 0.5).astype(int).tolist()
                        preds.extend(p)
                        probs.extend(scores.tolist())
                        ys.extend(y.cpu().numpy().tolist())

                val_loss = total_vloss / len(val_data)
                val_f1 = compute_metrics(ys, preds, probs)["f1"]
                history["val_loss"].append(val_loss)
                history["val_f1"].append(val_f1)

                if best_val is None or val_f1 > best_val:
                    best_val = val_f1
                    patience = 0
                    if out_dir:
                        torch.save(self.model.state_dict(), out_dir / "best_model.pth")
                else:
                    patience += 1

            if early_stopping and patience >= early_stopping:
                break

        # final save
        if out_dir:
            torch.save(self.model.state_dict(), out_dir / "final_model.pth")
            with open(out_dir / "train_history.json", "w", encoding="utf-8") as fh:
                json.dump(history, fh, indent=2)

        return history

    def predict(self, data_list: List[Data], batch_size: int = 32) -> Dict[str, List]:
        if DataLoader is None:
            raise RuntimeError("torch_geometric DataLoader not available")
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        self.model.eval()
        ys = []
        preds = []
        probs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                if logits.dim() == 1 or logits.shape[1] == 1:
                    logits = logits.view(-1, 2)
                scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                p = (scores >= 0.5).astype(int).tolist()
                preds.extend(p)
                probs.extend(scores.tolist())
                ys.extend(batch.y.view(-1).cpu().numpy().tolist())

        return {"y_true": ys, "y_pred": preds, "y_score": probs}
