import networkx as nx


class PDGParser:

    def __init__(self, graphml_path: str):
        self.graphml_path = graphml_path

    def parse(self):
        graph = nx.read_graphml(self.graphml_path)

        print("=== GRAPH INFO ===")
        print(f"Nodes: {graph.number_of_nodes()}")
        print(f"Edges: {graph.number_of_edges()}")

        return graph


if __name__ == "__main__":
    parser = PDGParser("output/repositories/ansible-nginx/PDG_task_level/task_0.graphml")

    graph = parser.parse()

    print("\n=== SAMPLE NODES ===")

    for node_id, attrs in list(graph.nodes(data=True))[:10]:
        print(node_id, attrs)

    print("\n=== SAMPLE EDGES ===")

    for source, target, attrs in list(graph.edges(data=True))[:10]:
        print(source, target, attrs)