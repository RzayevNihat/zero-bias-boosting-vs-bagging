"""Shared validation helpers for from-scratch boosting models."""

from __future__ import annotations

from typing import Optional

import numpy as np


def validate_inputs(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """Validate the common 2-D feature matrix / 1-D target interface."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    if y is not None:
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}.")
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}."
            )
