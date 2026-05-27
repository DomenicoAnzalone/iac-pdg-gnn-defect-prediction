from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np

NODE_TYPE_CATEGORIES = [
    "Task",
    "Operator",
    "Condition",
    "Data",
    "Variable",
    "Function",
    "Module",
    "Call",
    "Expression",
    "Assignment",
    "Unknown",
]

EDGE_TYPE_MAP = {
    "data": 1,
    "control": 2,
    "depends": 3,
    "guard": 4,
    "condition": 5,
    "task": 6,
    "other": 0,
}

TEXT_TOKEN_PATTERN = re.compile(r"\w+|\{\{.*?\}\}")


class GraphFeatureBuilder:
    """Build deterministic node and edge feature vectors for PDG graphs."""

    def __init__(self, node_type_categories: Optional[List[str]] = None):
        self.node_type_categories = node_type_categories or NODE_TYPE_CATEGORIES
        self.node_type_index = {name: idx for idx, name in enumerate(self.node_type_categories)}
        self.default_node_type = "Unknown"

    def build_node_features(self, G: nx.Graph) -> Tuple[np.ndarray, List[str]]:
        node_list = self._sorted_nodes(G)
        rows = []

        for node in node_list:
            attrs = G.nodes[node]
            row = []

            row.append(self._is_task_node(attrs))
            row.append(self._degree_in(G, node))
            row.append(self._degree_out(G, node))
            row.append(self._degree_total(G, node))

            node_type_vector = self._one_hot_node_type(attrs)
            row.extend(node_type_vector)

            label = self._node_label(attrs)
            row.extend(self._text_features(label))

            row.append(len(attrs))
            row.append(self._has_location(attrs))

            rows.append(row)

        feature_names = [
            "is_task_node",
            "in_degree",
            "out_degree",
            "degree",
        ]
        feature_names += [f"node_type_{name}" for name in self.node_type_categories]
        feature_names += [
            "label_length",
            "label_token_count",
            "label_numeric_token_fraction",
            "label_has_jinja",
            "label_has_equals",
            "attribute_count",
            "has_location",
        ]

        return np.array(rows, dtype=np.float32), feature_names

    def build_edge_features(self, G: nx.Graph) -> Tuple[np.ndarray, List[str]]:
        edge_list = self._sorted_edge_list(G)
        rows = []

        for _, _, attrs in edge_list:
            edge_type = self._extract_edge_type(attrs)
            rows.append([edge_type])

        return np.array(rows, dtype=np.int64), ["edge_type"]

    def node_type_distribution(self, G: nx.Graph) -> Counter:
        return Counter(self._node_type(attrs) for _, attrs in G.nodes(data=True))

    def edge_type_distribution(self, G: nx.Graph) -> Counter:
        return Counter(self._extract_edge_type(attrs) for _, _, attrs in self._sorted_edge_list(G))

    def _sorted_nodes(self, G: nx.Graph) -> List[str]:
        return sorted(str(node) for node in G.nodes())

    def _sorted_edge_list(self, G: nx.Graph) -> List[Tuple[str, str, Dict[str, str]]]:
        if G.is_multigraph():
            edges = [
                (str(u), str(v), {str(k): str(vv) for k, vv in data.items()})
                for u, v, _, data in G.edges(keys=True, data=True)
            ]
        else:
            edges = [
                (str(u), str(v), {str(k): str(vv) for k, vv in data.items()})
                for u, v, data in G.edges(data=True)
            ]
        return sorted(edges, key=lambda item: (item[0], item[1], tuple(sorted(item[2].items()))))

    def _node_label(self, attrs: Dict[str, str]) -> str:
        return str(attrs.get("label", attrs.get("name", "")))

    def _is_task_node(self, attrs: Dict[str, str]) -> int:
        node_type = self._node_type(attrs).lower()
        return 1 if node_type == "task" else 0

    def _node_type(self, attrs: Dict[str, str]) -> str:
        candidate = attrs.get("node_type") or attrs.get("type") or attrs.get("kind") or attrs.get("label")
        if candidate is None:
            return self.default_node_type
        candidate = str(candidate).strip()
        if candidate in self.node_type_index:
            return candidate
        reduced = candidate.split()[0].capitalize()
        if reduced in self.node_type_index:
            return reduced
        # Recognize simple control/data edge labels
        lowered = candidate.lower()
        if "task" in lowered:
            return "Task"
        if "operator" in lowered:
            return "Operator"
        if "condition" in lowered or "guard" in lowered:
            return "Condition"
        if "data" in lowered or "input" in lowered or "output" in lowered:
            return "Data"
        return self.default_node_type

    def _one_hot_node_type(self, attrs: Dict[str, str]) -> List[int]:
        node_type = self._node_type(attrs)
        vector = [0] * len(self.node_type_categories)
        index = self.node_type_index.get(node_type, self.node_type_index[self.default_node_type])
        vector[index] = 1
        return vector

    def _degree_in(self, G: nx.Graph, node: str) -> int:
        return int(G.in_degree(node)) if G.is_directed() else int(G.degree(node))

    def _degree_out(self, G: nx.Graph, node: str) -> int:
        return int(G.out_degree(node)) if G.is_directed() else int(G.degree(node))

    def _degree_total(self, G: nx.Graph, node: str) -> int:
        if G.is_directed():
            return self._degree_in(G, node) + self._degree_out(G, node)
        return int(G.degree(node))

    def _text_features(self, text: str) -> List[float]:
        label = text or ""
        tokens = TEXT_TOKEN_PATTERN.findall(label)
        numeric_tokens = [token for token in tokens if token.isdigit() or any(char.isdigit() for char in token)]
        label_length = len(label)
        token_count = len(tokens)
        numeric_fraction = len(numeric_tokens) / token_count if token_count else 0.0
        return [
            float(label_length),
            float(token_count),
            float(numeric_fraction),
            1.0 if "{{" in label and "}}" in label else 0.0,
            1.0 if "=" in label else 0.0,
        ]

    def _has_location(self, attrs: Dict[str, str]) -> int:
        return 1 if "location" in attrs or "file" in attrs else 0

    def _extract_edge_type(self, attrs: Dict[str, str]) -> int:
        candidates = [attrs.get(key, "") for key in ("type", "edge_type", "kind", "label", "relation", "flow", "type_label")]
        text = " ".join(str(value).lower() for value in candidates if value)
        if "control" in text:
            return EDGE_TYPE_MAP["control"]
        if "data" in text:
            return EDGE_TYPE_MAP["data"]
        if "depend" in text or "depends" in text:
            return EDGE_TYPE_MAP["depends"]
        if "guard" in text or "condition" in text:
            return EDGE_TYPE_MAP["guard"]
        if "task" in text:
            return EDGE_TYPE_MAP["task"]
        return EDGE_TYPE_MAP["other"]
