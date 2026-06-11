from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from experiments.common.progress import progress
from experiments.e3_gnn.feature_engineering import GraphFeatureBuilder
from experiments.e3_gnn.graph_loader import GraphLoader

try:
    import torch
    from torch_geometric.data import Data
except Exception:  # pragma: no cover
    torch = None
    Data = None


@dataclass
class GraphBuildResult:
    data: object | None
    exclusion: Dict[str, object] | None


class GraphDataBuilder:
    def __init__(self, graph_path_column: str = "graphml_local_path", min_nodes: int = 3, min_edges: int = 2):
        self.graph_path_column = graph_path_column
        self.min_nodes = min_nodes
        self.min_edges = min_edges
        self.loader = GraphLoader()
        self.features = GraphFeatureBuilder()

    def build_raw(self, row: pd.Series) -> GraphBuildResult:
        if Data is None or torch is None:
            return GraphBuildResult(None, self._exclusion(row, "torch_geometric_unavailable"))
        graph_path = self._resolve_path(row)
        if graph_path is None:
            return GraphBuildResult(None, self._exclusion(row, "graphml_path_missing"))
        try:
            graph = self.loader.load_graph(graph_path)
        except Exception as exc:
            return GraphBuildResult(None, self._exclusion(row, f"graphml_parse_error:{exc}"))
        if graph.number_of_nodes() < self.min_nodes:
            return GraphBuildResult(None, self._exclusion(row, f"below_min_nodes_{self.min_nodes}"))
        if graph.number_of_edges() < self.min_edges:
            return GraphBuildResult(None, self._exclusion(row, f"below_min_edges_{self.min_edges}"))
        node_features, feature_names = self.features.build_node_features(graph)
        edge_attr, _ = self.features.build_edge_features(graph)
        edge_index = self._edge_index(graph)
        if node_features.size == 0:
            return GraphBuildResult(None, self._exclusion(row, "empty_node_features"))
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_type=torch.tensor(edge_attr.reshape(-1), dtype=torch.long) if edge_attr.size else torch.zeros(edge_index.shape[1], dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32) if edge_attr.size else None,
            y=torch.tensor([int(row["failure_prone"])], dtype=torch.long),
        )
        data.repository = str(row["repository"])
        data.commit = str(row["commit"])
        data.filepath = str(row["filepath"])
        data.sample_id = str(row["_sample_id"])
        data.node_feature_names = feature_names
        return GraphBuildResult(data, None)

    def fit_scaler(self, train_data: List[object]) -> StandardScaler | None:
        if not train_data:
            return None
        x = np.vstack([data.x.numpy() for data in train_data])
        scaler = StandardScaler()
        scaler.fit(x)
        return scaler

    def apply_scaler(self, data_list: List[object], scaler: StandardScaler | None) -> List[object]:
        if scaler is None:
            return data_list
        for data in data_list:
            data.x = torch.tensor(scaler.transform(data.x.numpy()), dtype=torch.float32)
        return data_list

    def build_partition(self, df: pd.DataFrame, desc: str = "graphml", show_progress: bool = False) -> Tuple[List[object], pd.DataFrame]:
        data_list = []
        exclusions = []
        rows = list(df.iterrows())
        for _, row in progress(rows, total=len(rows), desc=desc, unit="graph", enabled=show_progress):
            result = self.build_raw(row)
            if result.data is not None:
                data_list.append(result.data)
            if result.exclusion is not None:
                exclusions.append(result.exclusion)
        return data_list, pd.DataFrame(exclusions)

    def _resolve_path(self, row: pd.Series) -> Path | None:
        for column in [self.graph_path_column, "graphml_local_path", "graphml_path"]:
            value = row.get(column)
            if value == value and value:
                path = Path(str(value))
                if path.exists():
                    return path
        return None

    def _edge_index(self, graph) -> np.ndarray:
        nodes = sorted(str(node) for node in graph.nodes())
        node_index = {node: idx for idx, node in enumerate(nodes)}
        if graph.is_multigraph():
            edges = sorted((str(u), str(v)) for u, v, _ in graph.edges(keys=True))
        else:
            edges = sorted((str(u), str(v)) for u, v in graph.edges())
        if not edges:
            return np.empty((2, 0), dtype=np.int64)
        return np.asarray([[node_index[u] for u, _ in edges], [node_index[v] for _, v in edges]], dtype=np.int64)

    def _exclusion(self, row: pd.Series, reason: str) -> Dict[str, object]:
        return {
            "repository": row.get("repository"),
            "commit": row.get("commit"),
            "filepath": row.get("filepath"),
            "failure_prone": row.get("failure_prone"),
            "nodes": row.get("nodes"),
            "edges": row.get("edges"),
            "reason": reason,
        }
