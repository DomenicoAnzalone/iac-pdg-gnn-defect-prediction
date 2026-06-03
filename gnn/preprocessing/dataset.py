from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from .feature_engineering import GraphFeatureBuilder
from .graph_loader import GraphLoader, GraphLoadError

try:
    from torch import tensor
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover
    Data = None
    tensor = None


@dataclass
class GraphSample:
    repository: str
    commit: str
    filepath: str
    label: int
    graph_path: Path
    metadata: Dict[str, str]


class GraphDatasetBuilder:
    """Build graph classification samples from PDG metadata and extracted graphs."""

    def __init__(
        self,
        label_csv: Path,
        graph_path_column: str = "graphml_path",
        label_column: str = "failure_prone",
        path_remapper: Optional[Callable[[str], Path]] = None,
    ):
        self.label_csv = Path(label_csv)
        self.graph_path_column = graph_path_column
        self.label_column = label_column
        self.path_remapper = path_remapper
        self.df = self._load_label_table()
        self.loader = GraphLoader()
        self.feature_builder = GraphFeatureBuilder()

    def _load_label_table(self) -> pd.DataFrame:
        if not self.label_csv.exists():
            raise FileNotFoundError(f"Label CSV not found: {self.label_csv}")

        df = pd.read_csv(self.label_csv)
        return df

    def samples(self) -> Iterable[GraphSample]:
        for _, row in self.df.iterrows():
            graph_path = self._resolve_graph_path(str(row[self.graph_path_column]))
            if graph_path is None:
                continue

            try:
                label_value = int(row[self.label_column])
            except Exception:
                label_value = 0

            yield GraphSample(
                repository=str(row.get("repository", "")),
                commit=str(row.get("commit", "")),
                filepath=str(row.get("filepath", "")),
                label=label_value,
                graph_path=graph_path,
                metadata={
                    "repository": str(row.get("repository", "")),
                    "commit": str(row.get("commit", "")),
                    "filepath": str(row.get("filepath", "")),
                },
            )

    def _resolve_graph_path(self, path_str: str) -> Optional[Path]:
        if not path_str or path_str != path_str:
            return None

        candidate = Path(path_str)
        if candidate.exists():
            return candidate

        if self.path_remapper is not None:
            remapped = self.path_remapper(path_str)
            if remapped.exists():
                return remapped

        return None

    def build_graph_data(self, sample: GraphSample) -> Dict[str, object]:
        try:
            G = self.loader.load_graph(sample.graph_path)
        except GraphLoadError as exc:
            raise RuntimeError(f"Unable to load graph {sample.graph_path}: {exc}")

        node_features, node_feature_names = self.feature_builder.build_node_features(G)
        edge_features, edge_feature_names = self.feature_builder.build_edge_features(G)

        edge_index = self._edge_index_from_graph(G)

        data = {
            "x": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_features,
            "y": np.array([sample.label], dtype=np.int64),
            "metadata": sample.metadata,
            "node_feature_names": node_feature_names,
            "edge_feature_names": edge_feature_names,
        }

        if Data is not None and tensor is not None:
            data["pyg_data"] = Data(
                x=tensor(node_features, dtype=tensor(node_features).dtype),
                edge_index=tensor(edge_index, dtype=tensor(edge_index).dtype),
                edge_attr=tensor(edge_features, dtype=tensor(edge_features).dtype),
                y=tensor([sample.label], dtype=tensor([sample.label]).dtype),
            )

        return data

    def _edge_index_from_graph(self, G: nx.Graph) -> np.ndarray:
        node_to_index = {node: idx for idx, node in enumerate(sorted(str(node) for node in G.nodes()))}

        if G.is_multigraph():
            edges = sorted(
                [(str(u), str(v)) for u, v, _ in G.edges(keys=True)],
                key=lambda item: (item[0], item[1])
            )
        else:
            edges = sorted(
                [(str(u), str(v)) for u, v in G.edges()],
                key=lambda item: (item[0], item[1])
            )

        edge_index = np.array(
            [[node_to_index[u] for u, _ in edges], [node_to_index[v] for _, v in edges]],
            dtype=np.int64,
        )
        return edge_index

    def build_all(self) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for sample in self.samples():
            try:
                results.append(self.build_graph_data(sample))
            except RuntimeError:
                continue
        return results

    @staticmethod
    def path_remapper_from_local_root(local_root: Path, remote_prefix: str = "/app") -> Callable[[str], Path]:
        def remapper(path_str: str) -> Path:
            candidate = Path(path_str)
            if candidate.exists():
                return candidate

            if path_str.startswith(remote_prefix):
                relative = path_str[len(remote_prefix) :].lstrip("/")
                if relative.startswith("output/pdg/"):
                    alt_path = local_root / "output" / "pdg_file_level" / relative[len("output/pdg/") :]
                    if alt_path.exists():
                        return alt_path
                if relative.startswith("output/pdg_file_level/"):
                    alt_path = local_root / relative
                    if alt_path.exists():
                        return alt_path
                local_candidate = local_root / relative
                if local_candidate.exists():
                    return local_candidate
                alt_candidate = local_root / "output" / "pdg_file_level" / relative
                if alt_candidate.exists():
                    return alt_candidate
                return local_candidate

            local_candidate = local_root / path_str
            if local_candidate.exists():
                return local_candidate
            alt_candidate = local_root / "output" / "pdg_file_level" / path_str
            if alt_candidate.exists():
                return alt_candidate
            return local_candidate

        return remapper
