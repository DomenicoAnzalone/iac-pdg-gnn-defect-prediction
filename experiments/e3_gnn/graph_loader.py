from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx


class GraphLoadError(Exception):
    pass


class GraphLoader:
    """Load PDG graphs from GraphML or DOT files into deterministic NetworkX graphs."""

    GRAPHML_NAMES = ["pdg.graphml", "graphml.txt", "graphml"]
    DOT_NAMES = ["pdg.dot", "pdg.gv", "graph.dot", "pdg"]
    PREFERRED_EXTS = [".graphml", ".graphml.gz", ".graphmlz", ".dot", ".gv"]

    def resolve_graph_file(self, path: Path) -> Path:
        path = Path(path)
        if path.is_file():
            return path
        if path.is_dir():
            for candidate in (path / name for name in self.GRAPHML_NAMES + self.DOT_NAMES):
                if candidate.exists():
                    return candidate
            for child in sorted(path.iterdir()):
                if child.suffix.lower() in self.PREFERRED_EXTS:
                    return child
        raise GraphLoadError(f"Unable to resolve graph file from path: {path}")

    def load_graph(self, path: Path) -> nx.Graph:
        graph_file = self.resolve_graph_file(path)
        suffix = graph_file.suffix.lower()
        if suffix.endswith(".graphml") or suffix in {".graphml", ".xml"}:
            graph = self.load_graphml(graph_file)
        elif suffix in {".dot", ".gv"}:
            graph = self.load_dot(graph_file)
        elif graph_file.name.lower().endswith("graphml"):
            graph = self.load_graphml(graph_file)
        elif graph_file.name.lower().endswith(("dot", "gv")):
            graph = self.load_dot(graph_file)
        else:
            raise GraphLoadError(f"Unsupported graph file extension: {graph_file}")
        return self.normalize_graph(graph)

    def load_graphml(self, path: Path) -> nx.Graph:
        try:
            return nx.read_graphml(str(path))
        except Exception as exc:
            try:
                return nx.parse_graphml(path.read_text(encoding="utf-8"))
            except Exception as fallback_exc:
                raise GraphLoadError(f"Failed to parse GraphML {path}: {exc}; fallback: {fallback_exc}") from fallback_exc

    def load_dot(self, path: Path) -> nx.Graph:
        try:
            return nx.nx_pydot.read_dot(str(path))
        except Exception as exc:
            raise GraphLoadError(f"Failed to parse DOT {path}: {exc}") from exc

    def normalize_graph(self, graph: nx.Graph) -> nx.Graph:
        if graph.is_multigraph():
            normalized = nx.MultiDiGraph()
        elif graph.is_directed():
            normalized = nx.DiGraph()
        else:
            normalized = nx.Graph()

        for node, attrs in sorted(graph.nodes(data=True), key=lambda item: str(item[0])):
            clean_attrs = {str(key): self._serialize_value(value) for key, value in attrs.items()}
            normalized.add_node(str(node), **clean_attrs)

        if graph.is_multigraph():
            edges = sorted(
                ((str(u), str(v), key, data) for u, v, key, data in graph.edges(keys=True, data=True)),
                key=lambda item: (item[0], item[1], str(item[2])),
            )
            for u, v, key, attrs in edges:
                clean_attrs = {str(k): self._serialize_value(vv) for k, vv in attrs.items()}
                normalized.add_edge(u, v, key=str(key), **clean_attrs)
        else:
            edges = sorted(((str(u), str(v), data) for u, v, data in graph.edges(data=True)), key=lambda item: (item[0], item[1]))
            for u, v, attrs in edges:
                clean_attrs = {str(k): self._serialize_value(vv) for k, vv in attrs.items()}
                normalized.add_edge(u, v, **clean_attrs)

        return normalized

    def sorted_node_list(self, graph: nx.Graph) -> List[str]:
        return sorted(str(node) for node in graph.nodes())

    def sorted_edge_list(self, graph: nx.Graph) -> List[Tuple[str, str, Dict[str, str]]]:
        if graph.is_multigraph():
            return sorted(
                [
                    (str(u), str(v), {str(k): self._serialize_value(vv) for k, vv in data.items()})
                    for u, v, _, data in graph.edges(keys=True, data=True)
                ],
                key=lambda item: (item[0], item[1], tuple(sorted(item[2].items()))),
            )
        return sorted(
            [
                (str(u), str(v), {str(k): self._serialize_value(vv) for k, vv in data.items()})
                for u, v, data in graph.edges(data=True)
            ],
            key=lambda item: (item[0], item[1], tuple(sorted(item[2].items()))),
        )

    def inspect(self, graph_path: Path) -> Dict[str, Any]:
        graph = self.load_graph(graph_path)
        return {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "node_attributes": sorted({key for _, attrs in graph.nodes(data=True) for key in attrs.keys()}),
            "edge_attributes": sorted({key for _, _, attrs in graph.edges(data=True) for key in attrs.keys()}),
        }

    def _serialize_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)

