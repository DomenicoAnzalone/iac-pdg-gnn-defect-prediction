from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gnn.sampling.balance import GraphBalancer
from gnn.sampling.splitter import WalkForwardSplitter, WalkForwardSplit


class TestSampling(unittest.TestCase):
    def test_balance_oversample(self):
        samples = [
            {"failure_prone": 0, "value": "a"},
            {"failure_prone": 0, "value": "b"},
            {"failure_prone": 1, "value": "c"},
        ]
        balancer = GraphBalancer(random_state=1)
        balanced = balancer.oversample(samples, lambda row: int(row["failure_prone"]))
        counts = {label: sum(1 for row in balanced if int(row["failure_prone"]) == label) for label in [0, 1]}
        self.assertEqual(counts[0], counts[1])

    def test_balance_undersample(self):
        samples = [
            {"failure_prone": 0, "value": "a"},
            {"failure_prone": 0, "value": "b"},
            {"failure_prone": 1, "value": "c"},
        ]
        balancer = GraphBalancer(random_state=1)
        balanced = balancer.undersample(samples, lambda row: int(row["failure_prone"]))
        counts = {label: sum(1 for row in balanced if int(row["failure_prone"]) == label) for label in [0, 1]}
        self.assertEqual(counts[0], counts[1])

    def test_dataframe_oversample(self):
        df = pd.DataFrame(
            [
                {"failure_prone": 0, "path": "a"},
                {"failure_prone": 0, "path": "b"},
                {"failure_prone": 1, "path": "c"},
            ]
        )
        balancer = GraphBalancer(random_state=1)
        balanced = balancer.dataframe_oversample(df)
        self.assertEqual(len(balanced), 4)
        counts = balanced["failure_prone"].value_counts().to_dict()
        self.assertEqual(counts[0], counts[1])

    def test_splitter_order_commits(self):
        df = pd.DataFrame(
            [
                {"repository": "repo/x", "commit": "c1", "failure_prone": 0},
                {"repository": "repo/x", "commit": "c2", "failure_prone": 1},
                {"repository": "repo/x", "commit": "c3", "failure_prone": 0},
                {"repository": "repo/y", "commit": "d1", "failure_prone": 0},
                {"repository": "repo/y", "commit": "d2", "failure_prone": 1},
            ]
        )
        splitter = WalkForwardSplitter(df)
        splits = splitter.all_project_splits()
        self.assertEqual(len(splits), 3)
        self.assertTrue(all(isinstance(split, WalkForwardSplit) for split in splits))

    def test_train_validation_split_fallback(self):
        df = pd.DataFrame(
            [
                {"repository": "repo/x", "commit": "c1", "failure_prone": 0},
                {"repository": "repo/x", "commit": "c2", "failure_prone": 1},
                {"repository": "repo/x", "commit": "c2", "failure_prone": 1},
                {"repository": "repo/x", "commit": "c3", "failure_prone": 0},
            ]
        )
        splitter = WalkForwardSplitter(df)
        split = splitter.project_splits("repo/x")[1]
        split_with_val = splitter.train_validation_split(split, validation_ratio=0.5)
        self.assertIsNotNone(split_with_val.validation)
        self.assertGreater(len(split_with_val.validation), 0)


if __name__ == "__main__":
    unittest.main()
