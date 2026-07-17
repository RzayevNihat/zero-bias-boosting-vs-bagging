"""Stopping-criteria checks for CART tree growth.

Kept as a standalone pure function (independent of split search) so it
can be unit-tested in isolation.
"""

from __future__ import annotations

import numpy as np


def should_stop(
    depth: int,
    n_samples: int,
    y: np.ndarray,
    max_depth: int | None,
    min_samples_split: int,
) -> bool:
    """Check if node should stop growing (depth, samples, or pure)."""
    if max_depth is not None and depth >= max_depth:
        return True
    if n_samples < min_samples_split:
        return True
    if len(np.unique(y)) == 1:
        return True  # pure node
    return False
