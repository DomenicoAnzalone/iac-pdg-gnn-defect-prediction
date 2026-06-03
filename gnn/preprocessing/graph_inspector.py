from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List

import networkx as nx


class GraphInspector:
    """Inspect PDG graphs and summarize structure, node attrs and edge attrs."""

    @staticmethod
    def summarize_graph(G: nx.Graph) -> Dict[str, object]:
        return {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "directed": G.is_directed(),
            "multigraph": G.is_multigraph(),
            "node_attribute_summary": GraphInspector.summarize_node_attributes(G),
            "edge_attribute_summary": GraphInspector.summarize_edge_attributes(G),
        }

    @staticmethod
    def summarize_node_attributes(G: nx.Graph) -> Dict[str, object]:
        return GraphInspector._summarize_attributes((attrs for _, attrs in G.nodes(data=True)))

    @staticmethod
    def summarize_edge_attributes(G: nx.Graph) -> Dict[str, object]:
        if G.is_multigraph():
            attrs_iter = (attrs for _, _, _, attrs in G.edges(keys=True, data=True))
        else:
            attrs_iter = (attrs for _, _, attrs in G.edges(data=True))
        return GraphInspector._summarize_attributes(attrs_iter)

    @staticmethod
    def _summarize_attributes(attribute_iter: Iterable[Dict[str, object]]) -> Dict[str, object]:
        key_counts = Counter()
        value_samples = {}

        for attrs in attribute_iter:
            for key, value in attrs.items():
                key_counts[key] += 1
                if key not in value_samples and value is not None:
                    value_samples[key] = str(value)

        return {
            "attribute_keys": sorted(key_counts.keys()),
            "key_counts": dict(key_counts),
            "sample_values": value_samples,
        }
