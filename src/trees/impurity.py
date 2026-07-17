"""Impurity criteria (gini, entropy) for decision trees."""

from __future__ import annotations

import numpy as np

_EPSILON = 1e-12


def _class_proportions(y: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
    """Weighted class proportions, summing to 1."""
    classes = np.unique(y)
    total_weight = sample_weight.sum()
    if total_weight <= 0:
        raise ValueError("sample_weight must sum to a positive value")
    proportions: np.ndarray = np.array(
        [sample_weight[y == c].sum() for c in classes]
    ) / total_weight
    return proportions


def gini(y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    """Gini impurity: 1 - sum(p_c^2). Supports weighted samples."""
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=float)
    p = _class_proportions(y, sample_weight)
    result: float = float(1.0 - np.sum(p ** 2))
    return result


def entropy(y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    """Shannon entropy in bits. Uses epsilon to avoid log(0)."""
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=float)
    p = _class_proportions(y, sample_weight)
    return float(-np.sum(p * np.log2(p + _EPSILON)))


_CRITERIA = {"gini": gini, "entropy": entropy}


def impurity(
    y: np.ndarray, sample_weight: np.ndarray | None, criterion: str
) -> float:
    """Return impurity for the given criterion ('gini' or 'entropy')."""
    if criterion not in _CRITERIA:
        raise ValueError(f"Unknown criterion {criterion!r}; use 'gini' or 'entropy'")
    return _CRITERIA[criterion](y, sample_weight)
