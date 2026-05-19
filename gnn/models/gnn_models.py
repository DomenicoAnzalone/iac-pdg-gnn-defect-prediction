try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.nn import global_mean_pool
except Exception:
    torch = None


if torch is not None:
    class GCN(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim=64, out_dim=1):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.lin = torch.nn.Linear(hidden_dim, out_dim)

        def forward(self, x, edge_index, batch=None):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            if batch is None:
                x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
            else:
                x = global_mean_pool(x, batch)
            return self.lin(x).squeeze(-1)

    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim=64, out_dim=1):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.lin = torch.nn.Linear(hidden_dim, out_dim)

        def forward(self, x, edge_index, batch=None):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            if batch is None:
                x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
            else:
                x = global_mean_pool(x, batch)
            return self.lin(x).squeeze(-1)

    class GAT(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim=64, out_dim=1, heads=4):
            super().__init__()
            self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads)
            self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads)
            self.lin = torch.nn.Linear(hidden_dim, out_dim)

        def forward(self, x, edge_index, batch=None):
            x = F.elu(self.conv1(x, edge_index))
            x = F.elu(self.conv2(x, edge_index))
            if batch is None:
                x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
            else:
                x = global_mean_pool(x, batch)
            return self.lin(x).squeeze(-1)
