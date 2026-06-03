from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import (
        GCNConv,
        SAGEConv,
        GATConv,
        GINConv,
        RGCNConv,
        global_mean_pool,
        global_add_pool,
        global_max_pool,
        Set2Set,
    )
except Exception:  # pragma: no cover
    GCNConv = SAGEConv = GATConv = GINConv = RGCNConv = None
    global_mean_pool = global_add_pool = global_max_pool = Set2Set = None


def _readout_pool(kind: str):
    if kind == "mean":
        return global_mean_pool
    if kind == "add":
        return global_add_pool
    if kind == "max":
        return global_max_pool
    return global_mean_pool


class BaseGNN(nn.Module):
    def __init__(self):
        super().__init__()


class GCNNet(BaseGNN):
    def __init__(self, in_channels: int, hidden: int = 64, num_classes: int = 2, num_layers: int = 2, dropout: float = 0.5, readout: str = "mean"):
        super().__init__()
        if GCNConv is None:
            raise RuntimeError("torch_geometric not available")
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        self.readout = _readout_pool(readout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        batch = getattr(data, "batch", None)
        if batch is None:
            # assume single graph
            g = x.mean(dim=0, keepdim=True)
        else:
            g = self.readout(x, batch)
        return self.mlp(g).squeeze(1) if g.shape[0] == 1 and g.dim() > 1 else self.mlp(g)


class GraphSAGENet(BaseGNN):
    def __init__(self, in_channels: int, hidden: int = 64, num_classes: int = 2, num_layers: int = 2, dropout: float = 0.5, readout: str = "mean"):
        super().__init__()
        if SAGEConv is None:
            raise RuntimeError("torch_geometric not available")
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        self.readout = _readout_pool(readout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        batch = getattr(data, "batch", None)
        if batch is None:
            g = x.mean(dim=0, keepdim=True)
        else:
            g = self.readout(x, batch)
        return self.mlp(g)


class GATNet(BaseGNN):
    def __init__(self, in_channels: int, hidden: int = 64, num_classes: int = 2, num_layers: int = 2, heads: int = 4, dropout: float = 0.5, readout: str = "mean"):
        super().__init__()
        if GATConv is None:
            raise RuntimeError("torch_geometric not available")
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden // heads, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden, hidden // heads, heads=heads))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        self.readout = _readout_pool(readout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        batch = getattr(data, "batch", None)
        if batch is None:
            g = x.mean(dim=0, keepdim=True)
        else:
            g = self.readout(x, batch)
        return self.mlp(g)


class GINNet(BaseGNN):
    def __init__(self, in_channels: int, hidden: int = 64, num_classes: int = 2, num_layers: int = 2, dropout: float = 0.5, readout: str = "mean"):
        super().__init__()
        if GINConv is None:
            raise RuntimeError("torch_geometric not available")
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(nn.Linear(in_channels if i == 0 else hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        self.readout = _readout_pool(readout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        batch = getattr(data, "batch", None)
        if batch is None:
            g = x.mean(dim=0, keepdim=True)
        else:
            g = self.readout(x, batch)
        return self.mlp(g)


class RGCNNet(BaseGNN):
    def __init__(self, in_channels: int, hidden: int = 64, num_classes: int = 2, num_layers: int = 2, num_relations: int = 4, dropout: float = 0.5, readout: str = "mean"):
        super().__init__()
        if RGCNConv is None:
            raise RuntimeError("torch_geometric not available")
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden, num_relations))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden, hidden, num_relations))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        self.readout = _readout_pool(readout)

    def forward(self, data):
        x = data.x
        # RGCNConv expects edge_index and edge_type
        edge_index = data.edge_index
        edge_type = getattr(data, "edge_type", None)
        for conv in self.convs:
            if edge_type is None:
                # fallback: use zero relation
                et = torch.zeros(edge_index.shape[1], dtype=torch.long, device=x.device)
            else:
                et = edge_type
            x = conv(x, edge_index, et)
            x = torch.relu(x)
            x = self.dropout(x)
        batch = getattr(data, "batch", None)
        if batch is None:
            g = x.mean(dim=0, keepdim=True)
        else:
            g = self.readout(x, batch)
        return self.mlp(g)


def get_model(name: str, in_channels: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "gcn":
        return GCNNet(in_channels, **kwargs)
    if name in ("graphsage", "sage"):
        return GraphSAGENet(in_channels, **kwargs)
    if name == "gat":
        return GATNet(in_channels, **kwargs)
    if name == "gin":
        return GINNet(in_channels, **kwargs)
    if name in ("rgcn", "r-gcn"):
        return RGCNNet(in_channels, **kwargs)
    raise ValueError(f"Unknown model: {name}")
