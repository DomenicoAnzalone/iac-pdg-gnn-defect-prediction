import networkx as nx


def read_graphml(path, node_type=str, edge_key_type=int, force_multigraph=False):
    return nx.read_graphml(path, node_type=node_type, edge_key_type=edge_key_type, force_multigraph=force_multigraph)


def write_graphml(G, path):
    nx.write_graphml(G, path)
