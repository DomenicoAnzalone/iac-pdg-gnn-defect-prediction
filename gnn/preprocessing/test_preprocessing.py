from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import networkx as nx

from .dataset import GraphDatasetBuilder, GraphSample
from .feature_engineering import GraphFeatureBuilder
from .graph_inspector import GraphInspector
from .graph_loader import GraphLoader


class TestPreprocessing(unittest.TestCase):
    def test_graph_loader_and_inspector(self):
        G = nx.DiGraph()
        G.add_node("n1", node_type="Task", label="name=install pkg")
        G.add_node("n2", node_type="Data", label="{{ var }}")
        G.add_edge("n1", "n2", type="control")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "pdg.graphml"
            nx.write_graphml(G, str(temp_path))

            loader = GraphLoader()
            loaded = loader.load_graph(temp_path)
            self.assertEqual(loaded.number_of_nodes(), 2)
            self.assertEqual(loaded.number_of_edges(), 1)

            summary = GraphInspector.summarize_graph(loaded)
            self.assertTrue(summary["directed"])
            self.assertIn("node_type", summary["node_attribute_summary"]["attribute_keys"])

    def test_feature_builder(self):
        G = nx.DiGraph()
        G.add_node("a", node_type="Task", label="command: yum install")
        G.add_node("b", node_type="Data", label="{{ item }}")
        G.add_edge("a", "b", type="data")

        builder = GraphFeatureBuilder()
        x, names = builder.build_node_features(G)
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(x.shape[1], len(names))
        self.assertIn("node_type_Task", names)
        self.assertIn("label_has_jinja", names)

        edge_attr, edge_names = builder.build_edge_features(G)
        self.assertEqual(edge_attr.shape, (1, 1))
        self.assertEqual(edge_names, ["edge_type"])

    def test_dataset_builder_path_remapper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph_dir = root / "output" / "pdg" / "repo" / "commit" / "file" / "PDG_FILE_LEVEL"
            graph_dir.mkdir(parents=True)
            graph_file = graph_dir / "pdg.graphml"

            G = nx.DiGraph()
            G.add_node("x", node_type="Task", label="foo")
            G.add_node("y", node_type="Data", label="bar")
            G.add_edge("x", "y", type="control")
            nx.write_graphml(G, str(graph_file))

            csv_path = root / "input.csv"
            csv_path.write_text(
                "repository,commit,filepath,failure_prone,graphml_path\n"
                "repo,commit,file,1,/app/output/pdg/repo/commit/file/PDG_FILE_LEVEL/pdg.graphml\n",
                encoding="utf-8",
            )

            builder = GraphDatasetBuilder(
                label_csv=csv_path,
                path_remapper=GraphDatasetBuilder.path_remapper_from_local_root(root, remote_prefix="/app"),
            )

            samples = list(builder.samples())
            self.assertEqual(len(samples), 1)
            sample = samples[0]
            self.assertEqual(sample.label, 1)

            data = builder.build_graph_data(sample)
            self.assertEqual(data["x"].shape[0], 2)
            self.assertEqual(data["edge_index"].shape[1], 1)
            self.assertEqual(data["y"][0], 1)


if __name__ == "__main__":
    unittest.main()
