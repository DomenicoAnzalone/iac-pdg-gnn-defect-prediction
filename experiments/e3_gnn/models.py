from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATConv, GCNConv, GINConv, RGCNConv, SAGEConv, global_add_pool, global_max_pool, global_mean_pool
except Exception:  # pragma: no cover
    GATConv = GCNConv = GINConv = RGCNConv = SAGEConv = None
    global_add_pool = global_max_pool = global_mean_pool = None


def _readout_pool(kind: str):
    if kind == "add":
        return global_add_pool
    if kind == "max":
        return global_max_pool
    return global_mean_pool


class GCNNet(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, num_classes: int = 2, num_layers: int = 2, dropout: float = 0.5, readout: str = "mean"):
        super().__init__()
        if GCNConv is None:
            raise RuntimeError("torch_geometric not available")
        self.convs = nn.ModuleList([GCNConv(in_channels, hidden)])
        self.convs.extend(GCNConv(hidden, hidden) for _ in range(num_layers - 1))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        self.readout = _readout_pool(readout)

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = self.dropout(torch.relu(conv(x, data.edge_index)))
        graph = x.mean(dim=0, keepdim=True) if getattr(data, "batch", None) is None else self.readout(x, data.batch)
        return self.mlp(graph)


class GraphSAGENet(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, num_classes: int = 2, num_layers: int = 2, dropout: float = 0.5, readout: str = "mean"):
        super().__init__()
        if SAGEConv is None:
            raise RuntimeError("torch_geometric not available")
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden)])
        self.convs.extend(SAGEConv(hidden, hidden) for _ in range(num_layers - 1))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        self.readout = _readout_pool(readout)

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = self.dropout(torch.relu(conv(x, data.edge_index)))
        graph = x.mean(dim=0, keepdim=True) if getattr(data, "batch", None) is None else self.readout(x, data.batch)
        return self.mlp(graph)


class GATNet(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, num_classes: int = 2, num_layers: int = 2, heads: int = 4, dropout: float = 0.5, readout: str = "mean"):
        super().__init__()
        if GATConv is None:
            raise RuntimeError("torch_geometric not available")
        hidden_per_head = max(1, hidden // heads)
        self.convs = nn.ModuleList([GATConv(in_channels, hidden_per_head, heads=heads)])
        self.convs.extend(GATConv(hidden, hidden_per_head, heads=heads) for _ in range(num_layers - 1))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        self.readout = _readout_pool(readout)

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = self.dropout(torch.relu(conv(x, data.edge_index)))
        graph = x.mean(dim=0, keepdim=True) if getattr(data, "batch", None) is None else self.readout(x, data.batch)
        return self.mlp(graph)


class GINNet(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, num_classes: int = 2, num_layers: int = 2, dropout: float = 0.5, readout: str = "mean"):
        super().__init__()
        if GINConv is None:
            raise RuntimeError("torch_geometric not available")
        self.convs = nn.ModuleList()
        for idx in range(num_layers):
            mlp = nn.Sequential(nn.Linear(in_channels if idx == 0 else hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        self.readout = _readout_pool(readout)

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = self.dropout(torch.relu(conv(x, data.edge_index)))
        graph = x.mean(dim=0, keepdim=True) if getattr(data, "batch", None) is None else self.readout(x, data.batch)
        return self.mlp(graph)


class RGCNNet(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, num_classes: int = 2, num_layers: int = 2, num_relations: int = 4, dropout: float = 0.5, readout: str = "mean"):
        super().__init__()
        if RGCNConv is None:
            raise RuntimeError("torch_geometric not available")
        self.convs = nn.ModuleList([RGCNConv(in_channels, hidden, num_relations)])
        self.convs.extend(RGCNConv(hidden, hidden, num_relations) for _ in range(num_layers - 1))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        self.readout = _readout_pool(readout)

    def forward(self, data):
        x = data.x
        edge_type = getattr(data, "edge_type", None)
        if edge_type is None:
            edge_type = torch.zeros(data.edge_index.shape[1], dtype=torch.long, device=x.device)
        for conv in self.convs:
            x = self.dropout(torch.relu(conv(x, data.edge_index, edge_type)))
        graph = x.mean(dim=0, keepdim=True) if getattr(data, "batch", None) is None else self.readout(x, data.batch)
        return self.mlp(graph)


def get_model(name: str, in_channels: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "gcn":
        return GCNNet(in_channels, **kwargs)
    if name in {"graphsage", "sage"}:
        return GraphSAGENet(in_channels, **kwargs)
    if name == "gat":
        return GATNet(in_channels, **kwargs)
    if name == "gin":
        return GINNet(in_channels, **kwargs)
    if name in {"rgcn", "r-gcn"}:
        return RGCNNet(in_channels, **kwargs)
    raise ValueError(f"Unknown model: {name}")

