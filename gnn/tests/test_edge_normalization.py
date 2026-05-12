import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pdg.parser import PDGParser
from src.pdg.normalizer import infer_edge_type


def main():

    parser = PDGParser(
        "gnn/data/raw/task_0.graphml"
    )

    graph = parser.parse()

    print("\n========================")
    print("EDGE NORMALIZATION")
    print("========================")

    for source, target, attrs in graph.edges(data=True):

        edge_type = infer_edge_type(attrs)

        print(
            f"{source} -> {target}"
        )

        print(
            f"  raw_label: {attrs.get('label')}"
        )

        print(
            f"  inferred_type: {edge_type.name}"
        )

        print()


if __name__ == "__main__":
    main()