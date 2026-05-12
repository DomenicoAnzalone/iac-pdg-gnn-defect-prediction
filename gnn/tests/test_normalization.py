import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pdg.parser import PDGParser
from src.pdg.normalizer import infer_node_type
from src.pdg.schema import NodeType


def main():

    parser = PDGParser(
        "gnn/data/raw/task_0.graphml"
    )

    graph = parser.parse()

    print("\n========================")
    print("NODE NORMALIZATION")
    print("========================")

    for node_id, attrs in graph.nodes(data=True):

        node_type = infer_node_type(attrs)

        print(
            f"Node {node_id}"
        )

        print(
            f"  label: {attrs.get('label')}"
        )

        print(
            f"  inferred_type: {node_type.name}"
        )

        print()


if __name__ == "__main__":
    main()