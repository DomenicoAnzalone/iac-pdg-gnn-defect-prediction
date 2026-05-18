import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.pdg.torch_converter import (
    canonical_to_pyg_data,
)


def test_canonical_to_pyg_data():

    with open(
        "gnn/data/serialized/task_0.json",
        "r",
        encoding="utf-8"
    ) as f:

        graph_dict = json.load(f)

    data = canonical_to_pyg_data(graph_dict)

    print("\n=== DATA OBJECT ===")
    print(data)

    print("\n=== NODE FEATURE SHAPE ===")
    print(data.x.shape)

    print("\n=== EDGE INDEX SHAPE ===")
    print(data.edge_index.shape)

    print("\n=== EDGE TYPE SHAPE ===")
    print(data.edge_type.shape)

    print("\n=== FIRST NODE FEATURES ===")
    print(data.x[:10])

    print("\n=== FIRST EDGES ===")
    print(data.edge_index[:, :10])

    print("\n=== FIRST EDGE TYPES ===")
    print(data.edge_type[:10])

    # Basic consistency checks

    assert data.x.shape[0] > 0

    assert data.edge_index.shape[0] == 2

    assert (
        data.edge_index.shape[1]
        ==
        data.edge_type.shape[0]
    )

    print("\nTEST PASSED")