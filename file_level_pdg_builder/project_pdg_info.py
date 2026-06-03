import os

import writer_reader as wr


def getPDG(repo_root, graphml_filename="graphml.txt"):
    repo_root = os.path.abspath(repo_root)
    path = os.path.normpath(os.path.join(repo_root, "PDG", graphml_filename))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Repository-level GraphML file not found: {path}"
        )
    return wr.read_graphml(path, node_type=int)


def find_repository_graphml(repo_root, graphml_filename="graphml.txt"):
    repo_root = os.path.abspath(repo_root)
    candidate = os.path.normpath(os.path.join(repo_root, "PDG", graphml_filename))
    return candidate if os.path.exists(candidate) else None
