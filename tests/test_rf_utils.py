"""Tests for Person 3 experiment utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from src.experiments.rf_utils import (
    DatasetBundle,
    add_label_noise,
    load_breast_cancer_bundle,
    load_mnist_binary_bundle,
    prepare_bundle_split,
    random_oversample_minority,
)


def test_add_label_noise_is_exact_and_reproducible():
    y = np.array([0, 1] * 20)
    first = add_label_noise(y, 0.20, random_state=42)
    second = add_label_noise(y, 0.20, random_state=42)
    np.testing.assert_array_equal(first, second)
    assert np.sum(first != y) == 8


def test_random_oversampling_balances_classes():
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.array([0] * 10 + [1] * 2)
    X_resampled, y_resampled = random_oversample_minority(X, y, random_state=42)
    _, counts = np.unique(y_resampled, return_counts=True)
    assert counts.tolist() == [10, 10]
    assert len(X_resampled) == len(y_resampled) == 20


def test_prepare_bundle_treats_severe_imbalance_train_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = np.zeros(400, dtype=int)
    y[:4] = 1
    bundle = DatasetBundle("tiny_imbalanced", X, y, "test", True)
    X_train, X_test, y_train, y_test, treatment = prepare_bundle_split(bundle)
    _, train_counts = np.unique(y_train, return_counts=True)
    assert treatment == "random_oversampling_train_only"
    assert train_counts[0] == train_counts[1]
    # Test split remains untouched and therefore imbalanced.
    _, test_counts = np.unique(y_test, return_counts=True)
    assert test_counts[0] != test_counts[1]
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_load_breast_cancer_bundle_reads_local_wdbc(tmp_path):
    data_path = tmp_path / "wdbc.data"
    data_path.write_text(
        "1001,M,1.0,2.0,3.0\n"
        "1002,B,4.0,5.0,6.0\n"
        "1003,M,7.0,8.0,9.0\n",
        encoding="utf-8",
    )

    bundle = load_breast_cancer_bundle(data_path)

    assert bundle.name == "breast_cancer_wdbc"
    assert bundle.X.shape == (3, 3)
    assert bundle.y.tolist() == [1, 0, 1]


def test_load_mnist_bundle_reads_local_npz(tmp_path):
    data_path = tmp_path / "mnist_3_vs_8.npz"
    X = np.arange(24, dtype=np.float32).reshape(3, 8)
    y = np.array([0, 1, 0], dtype=np.int8)
    np.savez_compressed(data_path, X=X, y=y)

    bundle = load_mnist_binary_bundle(data_path)

    assert bundle.name == "mnist_3_vs_8"
    assert bundle.X.shape == (3, 8)
    np.testing.assert_array_equal(bundle.y, y)


def test_local_loader_has_clear_missing_file_error(tmp_path):
    missing_path = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError, match="download_data.sh"):
        load_mnist_binary_bundle(missing_path)
