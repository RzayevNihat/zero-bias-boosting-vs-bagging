"""Best-split search for decision trees.

Uses sort + incremental sweep instead of naive rescanning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplitResult:
    """The best split found for a node, across all searched features.

    Attributes:
        feature_index: Index of the feature to split on.
        threshold: Samples with feature value <= threshold go left.
        gain: Impurity reduction (delta I) achieved by this split.
    """

    feature_index: int
    threshold: float
    gain: float


def _impurity_from_counts(
    class_weights: np.ndarray, total_weight: float, criterion: str
) -> float:
    """Compute impurity directly from class weights."""
    if total_weight <= 0:
        return 0.0
    p = class_weights / total_weight
    if criterion == "gini":
        return float(1.0 - np.sum(p ** 2))
    elif criterion == "entropy":
        return float(-np.sum(p * np.log2(p + 1e-12)))
    raise ValueError(f"Unknown criterion {criterion!r}")


def best_split_for_feature(
    x_col: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, criterion: str
) -> tuple[float, float] | None:
    """Find the best split threshold for one feature."""
    order = np.argsort(x_col, kind="mergesort")  # stable sort
    x_sorted = x_col[order]
    y_sorted = y[order]
    w_sorted = sample_weight[order]

    classes, y_encoded = np.unique(y_sorted, return_inverse=True)
    n_classes = len(classes)
    total_weight = w_sorted.sum()

    parent_impurity = _impurity_from_counts(
        np.array([w_sorted[y_encoded == c].sum() for c in range(n_classes)]),
        total_weight,
        criterion,
    )

    left_counts = np.zeros(n_classes)
    right_counts = np.array(
        [w_sorted[y_encoded == c].sum() for c in range(n_classes)]
    )
    left_w, right_w = 0.0, total_weight

    best_gain, best_threshold = -np.inf, None
    n = len(x_sorted)

    for i in range(n - 1):
        c = y_encoded[i]
        left_counts[c] += w_sorted[i]
        right_counts[c] -= w_sorted[i]
        left_w += w_sorted[i]
        right_w -= w_sorted[i]

        if x_sorted[i] == x_sorted[i + 1]:
            continue  # identical feature values: no valid boundary here

        left_imp = _impurity_from_counts(left_counts, left_w, criterion)
        right_imp = _impurity_from_counts(right_counts, right_w, criterion)

        gain = (
            parent_impurity
            - (left_w / total_weight) * left_imp
            - (right_w / total_weight) * right_imp
        )

        if gain > best_gain:
            best_gain = gain
            best_threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0

    if best_threshold is None:
        return None
    return best_threshold, best_gain


def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    criterion: str,
    feature_indices: np.ndarray,
) -> SplitResult | None:
    """Find the best split across the given feature subset."""
    best: SplitResult | None = None
    for j in feature_indices:
        result = best_split_for_feature(X[:, j], y, sample_weight, criterion)
        if result is None:
            continue
        threshold, gain = result
        if best is None or gain > best.gain:
            best = SplitResult(feature_index=int(j), threshold=threshold, gain=gain)
    return best
