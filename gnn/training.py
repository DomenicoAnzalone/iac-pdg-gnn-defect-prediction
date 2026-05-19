import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.nn import BCEWithLogitsLoss
except Exception:
    torch = None


def train_model(model: Any, dataset: Any, cfg: Dict[str, Any], save_dir: str):
    if torch is None:
        logger.warning("PyTorch not available: skipping training")
        return None

    device = torch.device(cfg.get("device", "cpu"))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    criterion = BCEWithLogitsLoss()

    best_loss = float("inf")
    early = cfg.get("early_stopping", 10)
    patience = 0
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(cfg.get("epochs", 50)):
        model.train()
        total_loss = 0.0
        for data in dataset:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, getattr(data, "batch", None))
            loss = criterion(out, data.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / max(1, len(dataset))
        logger.info("Epoch %d loss=%.4f", epoch, avg)
        if avg < best_loss:
            best_loss = avg
            patience = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
        else:
            patience += 1
            if patience >= early:
                logger.info("Early stopping at epoch %d", epoch)
                break

    return os.path.join(save_dir, "best.pt")
