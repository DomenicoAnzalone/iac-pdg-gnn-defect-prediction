import os
import networkx as nx


def is_task_node(node_attrs):

    label = str(node_attrs.get("label", "")).lower()

    # task/module nodes prodotti da scansible
    return (
        "shape=ellipse" in str(node_attrs)
        or "<<b>" in label
    )


def sanitize_filename(name):

    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def extract_task_subgraph(G, task_node):

    visited = set()
    worklist = [task_node]

    while worklist:

        current = worklist.pop()

        if current in visited:
            continue

        visited.add(current)

        # predecessori
        for pred in G.predecessors(current):
            if pred not in visited:
                worklist.append(pred)

        # successori
        for succ in G.successors(current):
            if succ not in visited:
                worklist.append(succ)

    return G.subgraph(visited).copy()


def extract_pdg_task_level_from_repo(repository: str):

    pdg_path = os.path.normpath(
        os.path.join(
            "output",
            "repositories",
            repository,
            "PDG",
            "pdg.dot"
        )
    )

    output_dir = os.path.normpath(
        os.path.join(
            "output",
            "repositories",
            repository,
            "PDG_task_level"
        )
    )

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading repository-level PDG: {pdg_path}")

    G = nx.nx_pydot.read_dot(pdg_path)

    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    task_nodes = []

    for node, attrs in G.nodes(data=True):

        label = str(attrs.get("label", "")).lower()

        # euristica semplice per identificare task/module nodes
        if "<<b>" in label:
            task_nodes.append(node)

    print(f"Found {len(task_nodes)} task nodes")

    for idx, task_node in enumerate(task_nodes):

        print(f"Extracting task-level PDG {idx+1}/{len(task_nodes)}")

        subgraph = extract_task_subgraph(G, task_node)

        clean_graph = sanitize_graph_for_graphml(subgraph)

        output_file = os.path.join(
            output_dir,
            f"task_{idx}.graphml"
        )

        nx.write_graphml(clean_graph, output_file)

    print("\nTask-level PDG extraction completed")

def sanitize_graph_for_graphml(G):

    H = nx.DiGraph()

    # nodi
    for node, attrs in G.nodes(data=True):

        clean_attrs = {}

        for k, v in attrs.items():
            clean_attrs[str(k)] = str(v)

        H.add_node(str(node), **clean_attrs)

    # archi
    for source, target, attrs in G.edges(data=True):

        clean_attrs = {}

        for k, v in attrs.items():
            clean_attrs[str(k)] = str(v)

        H.add_edge(
            str(source),
            str(target),
            **clean_attrs
        )

    return H