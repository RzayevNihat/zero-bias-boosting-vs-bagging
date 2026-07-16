"""
K-Means clustering implemented from scratch.
"""

from __future__ import annotations

import numpy as np


class KMeans:
    """
    K-Means clustering using Lloyd's algorithm.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:

        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def _validate_input(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if X.shape[0] < self.n_clusters:
            raise ValueError(
                "Number of samples must be >= n_clusters."
            )

        return X
    
    def _validate_input(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if X.shape[0] < self.n_clusters:
            raise ValueError(
                "Number of samples must be >= n_clusters."
            )

        return X