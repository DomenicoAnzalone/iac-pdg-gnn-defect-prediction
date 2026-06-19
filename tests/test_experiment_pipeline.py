from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.common.balancing import balance_dataframe
from experiments.common.config import load_config
from experiments.common.data_loading import filter_common_valid_samples, load_dataset
from experiments.common.feature_sets import PDG_METRICS, e1_features, e2_features
from experiments.common.preprocessing import TabularPreprocessor
from experiments.common.splitting import assert_no_overlap, create_walk_forward_splits, materialize_split
from experiments.e3_gnn.graph_data import GraphDataBuilder
from experiments.run_full_benchmark import build_parser as build_benchmark_parser


DATASET = Path("datasets/ansible-pdg-defect-dataset/final/v2026-06-06/ansible-pdg-defect-dataset_v2026-06-06_final.csv")


def test_dataset_schema_and_feature_sets():
    df = load_dataset(DATASET, max_repositories=2, max_samples=500)
    assert {"repository", "commit", "filepath", "failure_prone", "committed_at", "_sample_id"}.issubset(df.columns)
    e1 = e1_features(df)
    e2 = e2_features(df)
    assert e1
    assert set(e1).issubset(set(e2))
    assert all(metric in e2 for metric in PDG_METRICS)


def test_walk_forward_splits_no_overlap_and_temporal_order():
    df = load_dataset(DATASET, max_repositories=2, max_samples=1500)
    df, _ = filter_common_valid_samples(df)
    splits, skipped = create_walk_forward_splits(df, max_splits=3)
    assert splits
    for split in splits:
        assert_no_overlap(split)
        train, val, test = materialize_split(df, split)
        assert train["committed_at"].max() <= val["committed_at"].max()
        assert val["committed_at"].max() <= test["committed_at"].min()


def test_balancing_only_training_partition():
    df = load_dataset(DATASET, max_repositories=2, max_samples=1500)
    df, _ = filter_common_valid_samples(df)
    splits, _ = create_walk_forward_splits(df, max_splits=1)
    train, val, test = materialize_split(df, splits[0])
    val_ids = set(val["_sample_id"])
    test_ids = set(test["_sample_id"])
    balanced, report = balance_dataframe(train, "random_oversampling", seed=42)
    assert set(val["_sample_id"]) == val_ids
    assert set(test["_sample_id"]) == test_ids
    assert len(balanced) >= len(train)
    assert report["before"] != {}


def test_graphml_batch_loads():
    df = load_dataset(DATASET, max_repositories=1, max_samples=20)
    df, excluded = filter_common_valid_samples(df)
    builder = GraphDataBuilder()
    data, graph_excluded = builder.build_partition(df.head(3))
    assert data
    assert data[0].x.shape[0] >= 3
    assert data[0].edge_index.shape[0] == 2


def test_validation_rfe_selects_features_from_validation():
    train = pd.DataFrame({
        "good": [0, 0, 1, 1, 0, 1],
        "noise": [0, 1, 0, 1, 1, 0],
    })
    val = pd.DataFrame({
        "good": [0, 1, 0, 1],
        "noise": [1, 1, 0, 0],
    })
    test = val.copy()
    preprocessor = TabularPreprocessor(
        ["good", "noise"],
        feature_selection="validation_rfe",
        scaler="none",
        rfecv_step=1,
        seed=42,
        n_jobs=1,
    )
    _, _, _, manifest = preprocessor.fit_transform(
        train,
        val,
        test,
        y_train=pd.Series([0, 0, 1, 1, 0, 1]).to_numpy(),
        y_val=pd.Series([0, 1, 0, 1]).to_numpy(),
    )
    assert manifest["used_features"] == ["good"]
    details = manifest["feature_selection_details"]
    assert details["method"] == "validation_rfe"
    assert details["selection_source"] == "validation"


def test_validation_rfe_does_not_use_test_partition_for_selection():
    train = pd.DataFrame({
        "good": [0, 0, 1, 1, 0, 1],
        "noise": [0, 1, 0, 1, 1, 0],
    })
    val = pd.DataFrame({
        "good": [0, 1, 0, 1],
        "noise": [1, 1, 0, 0],
    })
    y_train = pd.Series([0, 0, 1, 1, 0, 1]).to_numpy()
    y_val = pd.Series([0, 1, 0, 1]).to_numpy()
    first_test = pd.DataFrame({"good": [0, 1], "noise": [0, 1]})
    second_test = pd.DataFrame({"good": [999, -999], "noise": [999, -999]})

    def selected_features(test_df: pd.DataFrame) -> list[str]:
        preprocessor = TabularPreprocessor(
            ["good", "noise"],
            feature_selection="validation_rfe",
            scaler="none",
            rfecv_step=1,
            seed=42,
            n_jobs=1,
        )
        _, _, _, manifest = preprocessor.fit_transform(train, val, test_df, y_train=y_train, y_val=y_val)
        return manifest["used_features"]

    assert selected_features(first_test) == selected_features(second_test)


def test_validation_rfe_falls_back_when_mcc_is_undefined():
    train = pd.DataFrame({
        "good": [0, 0, 1, 1, 0, 1],
        "noise": [0, 1, 0, 1, 1, 0],
    })
    val = pd.DataFrame({
        "good": [0, 0, 0, 0],
        "noise": [1, 1, 0, 0],
    })
    preprocessor = TabularPreprocessor(
        ["good", "noise"],
        feature_selection="validation_rfe",
        scaler="none",
        rfecv_step=1,
        seed=42,
        n_jobs=1,
    )
    _, _, _, manifest = preprocessor.fit_transform(
        train,
        val,
        val.copy(),
        y_train=pd.Series([0, 0, 1, 1, 0, 1]).to_numpy(),
        y_val=pd.Series([0, 0, 0, 0]).to_numpy(),
    )
    details = manifest["feature_selection_details"]
    assert details["fallback_used"] is True
    assert details["best_validation_metric"] in {"f1", "accuracy"}


def test_tabular_defaults_use_validation_rfe_and_e3_is_unchanged():
    assert load_config("experiments/configs/e1_default.yaml")["feature_selection"] == "validation_rfe"
    assert load_config("experiments/configs/e2_default.yaml")["feature_selection"] == "validation_rfe"
    args = build_benchmark_parser().parse_args([])
    assert args.e1_feature_selection == "validation_rfe"
    assert args.e2_feature_selection == "validation_rfe"
    assert args.e3_models == "gcn,graphsage,gat,gin,rgcn"
