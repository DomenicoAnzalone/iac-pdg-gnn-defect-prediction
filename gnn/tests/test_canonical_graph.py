import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pprint import pprint

from src.pdg.parser import PDGParser
from src.pdg.canonical import CanonicalGraphBuilder


def main():

    parser = PDGParser(
        "gnn/data/raw/task_0.graphml"
    )

    graph = parser.parse()

    builder = CanonicalGraphBuilder(graph)

    canonical_graph = builder.build()

    print("\n========================")
    print("CANONICAL GRAPH")
    print("========================")

    print("\nNODES:\n")

    for node in canonical_graph["nodes"][:10]:
        pprint(node)

    print("\nEDGES:\n")

    for edge in canonical_graph["edges"][:10]:
        pprint(edge)


if __name__ == "__main__":
    main()