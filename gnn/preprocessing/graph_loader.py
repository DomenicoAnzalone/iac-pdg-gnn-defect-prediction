from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pydot


class GraphLoadError(Exception):
    pass


class GraphLoader:
    """Load PDG graphs from .graphml or .dot files into deterministic NetworkX graphs."""

    GRAPHML_NAMES = ["pdg.graphml", "graphml.txt", "graphml", "pdg.graphml"]
    DOT_NAMES = ["pdg.dot", "pdg.gv", "graph.dot", "pdg"]
    PREFERRED_EXTS = [".graphml", ".graphml.gz", ".graphmlz", ".dot", ".gv"]

    def resolve_graph_file(self, path: Path) -> Path:
        path = Path(path)

        if path.is_file():
            return path

        if path.is_dir():
            # Prefer GraphML over DOT when both are present
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
            G = self.load_graphml(graph_file)
        elif suffix == ".dot" or suffix == ".gv":
            G = self.load_dot(graph_file)
        else:
            # Fallback by file name if path had no extension
            if graph_file.name.lower().endswith("graphml"):
                G = self.load_graphml(graph_file)
            elif graph_file.name.lower().endswith("dot") or graph_file.name.lower().endswith("gv"):
                G = self.load_dot(graph_file)
            else:
                raise GraphLoadError(f"Unsupported graph file extension: {graph_file}")

        return self.normalize_graph(G)

    def load_graphml(self, path: Path) -> nx.Graph:
        try:
            return nx.read_graphml(str(path))
        except Exception as exc:
            try:
                content = path.read_text(encoding="utf-8")
                return nx.parse_graphml(content)
            except Exception as fallback_exc:
                raise GraphLoadError(
                    f"Failed to parse GraphML {path}: {exc}\nFallback error: {fallback_exc}"
                )

    def load_dot(self, path: Path) -> nx.Graph:
        try:
            return nx.nx_pydot.read_dot(str(path))
        except Exception as exc:
            raise GraphLoadError(f"Failed to parse DOT {path}: {exc}")

    def normalize_graph(self, G: nx.Graph) -> nx.Graph:
        if G.is_multigraph():
            H = nx.MultiDiGraph()
        elif G.is_directed():
            H = nx.DiGraph()
        else:
            H = nx.Graph()

        # Normalize node names and attribute values to strings for deterministic parsing.
        for node, attrs in sorted(G.nodes(data=True), key=lambda item: str(item[0])):
            clean_attrs = {str(key): self._serialize_value(value) for key, value in attrs.items()}
            H.add_node(str(node), **clean_attrs)

        # Normalize edge order deterministically.
        if G.is_multigraph():
            edges = sorted(
                ((str(u), str(v), key, data) for u, v, key, data in G.edges(keys=True, data=True)),
                key=lambda item: (item[0], item[1], str(item[2]))
            )
            for u, v, key, attrs in edges:
                clean_attrs = {str(k): self._serialize_value(v) for k, v in attrs.items()}
                H.add_edge(u, v, key=str(key), **clean_attrs)
        else:
            edges = sorted(
                ((str(u), str(v), data) for u, v, data in G.edges(data=True)),
                key=lambda item: (item[0], item[1])
            )
            for u, v, attrs in edges:
                clean_attrs = {str(k): self._serialize_value(v) for k, v in attrs.items()}
                H.add_edge(u, v, **clean_attrs)

        return H

    def _serialize_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    def sorted_node_list(self, G: nx.Graph) -> List[str]:
        return sorted(str(node) for node in G.nodes())

    def sorted_edge_list(self, G: nx.Graph) -> List[Tuple[str, str, Dict[str, str]]]:
        if G.is_multigraph():
            return sorted(
                [(str(u), str(v), {str(k): self._serialize_value(vv) for k, vv in data.items()})
                 for u, v, key, data in G.edges(keys=True, data=True)],
                key=lambda item: (item[0], item[1], tuple(sorted(item[2].items())))
            )

        return sorted(
            [(str(u), str(v), {str(k): self._serialize_value(vv) for k, vv in data.items()})
             for u, v, data in G.edges(data=True)],
            key=lambda item: (item[0], item[1], tuple(sorted(item[2].items())))
        )

    def inspect(self, graph_path: Path) -> Dict[str, Any]:
        G = self.load_graph(graph_path)
        return {
            "nodes": len(G.nodes),
            "edges": len(G.edges),
            "node_attributes": sorted({key for _, attrs in G.nodes(data=True) for key in attrs.keys()}),
            "edge_attributes": sorted({key for _, _, attrs in G.edges(data=True) for key in attrs.keys()}),
        }
