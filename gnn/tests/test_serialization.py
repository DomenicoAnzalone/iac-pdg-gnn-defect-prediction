import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pdg.parser import PDGParser
from src.pdg.canonical import CanonicalGraphBuilder
from src.pdg.serializer import GraphSerializer

def main():

    #
    # Parse graph
    #

    parser = PDGParser(
        "gnn/data/raw/task_0.graphml"
    )

    graph = parser.parse()

    #
    # Build canonical graph
    #

    builder = CanonicalGraphBuilder(graph)

    canonical_graph = builder.build()

    #
    # Serialize graph
    #

    serializer = GraphSerializer(
        "gnn/data/serialized"
    )

    serializer.save(
        canonical_graph,
        "task_0.json"
    )


if __name__ == "__main__":
    main()