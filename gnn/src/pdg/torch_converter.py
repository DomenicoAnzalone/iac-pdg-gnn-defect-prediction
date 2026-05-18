import torch

from torch_geometric.data import Data

from src.pdg.schema import (
    NodeType,
    EdgeType,
)


def build_node_type_mapping() -> dict:
    """
    Dynamically builds node type encoding.

    Example:
        TASK -> 0
        VARIABLE -> 1
        ...
    """

    return {
        node_type.name: idx
        for idx, node_type in enumerate(NodeType)
    }


def build_edge_type_mapping() -> dict:
    """
    Dynamically builds edge type encoding.
    """

    return {
        edge_type.name: idx
        for idx, edge_type in enumerate(EdgeType)
    }


def build_node_index_map(nodes: list[dict]) -> dict:
    """
    Maps graph node IDs to contiguous tensor indices.
    """

    return {
        node["id"]: idx
        for idx, node in enumerate(nodes)
    }


def build_node_features(
    nodes: list[dict],
    node_type_mapping: dict,
) -> torch.Tensor:
    """
    Encodes node semantic types into tensor features.

    Shape:
        [num_nodes, 1]
    """

    features = []

    for node in nodes:

        encoded_type = node_type_mapping.get(
            node["type"],
            -1
        )

        features.append([encoded_type])

    return torch.tensor(
        features,
        dtype=torch.float
    )


def build_edge_index(
    edges: list[dict],
    node_index_map: dict,
) -> torch.Tensor:
    """
    Builds PyG edge_index tensor.

    Shape:
        [2, num_edges]
    """

    source_indices = []
    target_indices = []

    for edge in edges:

        source_indices.append(
            node_index_map[edge["source"]]
        )

        target_indices.append(
            node_index_map[edge["target"]]
        )

    return torch.tensor(
        [source_indices, target_indices],
        dtype=torch.long
    )


def build_edge_type_tensor(
    edges: list[dict],
    edge_type_mapping: dict,
) -> torch.Tensor:
    """
    Encodes edge semantic types.

    Shape:
        [num_edges]
    """

    edge_features = []

    for edge in edges:

        encoded_type = edge_type_mapping.get(
            edge["type"],
            -1
        )

        edge_features.append(encoded_type)

    return torch.tensor(
        edge_features,
        dtype=torch.long
    )


def canonical_to_pyg_data(
    graph_dict: dict,
) -> Data:
    """
    Converts canonical graph JSON into a
    PyTorch Geometric Data object.
    """

    nodes = graph_dict["nodes"]
    edges = graph_dict["edges"]

    node_type_mapping = build_node_type_mapping()

    edge_type_mapping = build_edge_type_mapping()

    node_index_map = build_node_index_map(nodes)

    x = build_node_features(
        nodes,
        node_type_mapping
    )

    edge_index = build_edge_index(
        edges,
        node_index_map
    )

    edge_type = build_edge_type_tensor(
        edges,
        edge_type_mapping
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
    )

    return data