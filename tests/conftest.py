"""Test fixtures for decision tree tests."""

import numpy as np
import pytest


@pytest.fixture
def xor_data():
    """XOR problem (requires depth >= 2)."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    return X, y


@pytest.fixture
def two_gaussians():
    """Two well-separated Gaussian clusters."""
    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=-3, scale=1.0, size=(50, 2))
    X1 = rng.normal(loc=3, scale=1.0, size=(50, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


@pytest.fixture
def single_feature_data():
    """Single-feature dataset (edge case)."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y
