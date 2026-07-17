"""Tests for Person 3 experiment utility functions."""

from __future__ import annotations

import numpy as np

from src.experiments.rf_utils import (
    DatasetBundle,
    add_label_noise,
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
