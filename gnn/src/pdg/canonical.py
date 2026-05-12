from src.pdg.normalizer import (
    infer_node_type,
    infer_edge_type
)


class CanonicalGraphBuilder:

    def __init__(self, graph):
        self.graph = graph

    def build(self):

        canonical_graph = {
            "nodes": [],
            "edges": []
        }

        #
        # Nodes
        #

        for node_id, attrs in self.graph.nodes(data=True):

            node_type = infer_node_type(attrs)

            canonical_node = {
                "id": int(node_id),
                "type": node_type.name,
                "text": attrs.get("label", "")
            }

            canonical_graph["nodes"].append(
                canonical_node
            )

        #
        # Edges
        #

        for source, target, attrs in self.graph.edges(data=True):

            edge_type = infer_edge_type(attrs)

            canonical_edge = {
                "source": int(source),
                "target": int(target),
                "type": edge_type.name
            }

            canonical_graph["edges"].append(
                canonical_edge
            )

        return canonical_graph