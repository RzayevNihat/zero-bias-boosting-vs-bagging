"""
DBSCAN clustering implemented from scratch.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise.

    Parameters
    ----------
    eps:
        Maximum distance between two neighboring samples.
    min_samples:
        Minimum number of samples required to form a dense region.
    """

    UNVISITED = -99
    NOISE = -1

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive.")

        if not isinstance(min_samples, int):
            raise TypeError("min_samples must be an integer.")

        if min_samples <= 0:
            raise ValueError("min_samples must be positive.")

        self.eps = eps
        self.min_samples = min_samples

        self.labels_: np.ndarray | None = None
        self.core_sample_indices_: np.ndarray | None = None
        self.n_clusters_: int | None = None

    @staticmethod
    def _validate_input(X: np.ndarray) -> np.ndarray:
        """Validate and convert input data."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X cannot be empty.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")

        return X

    def _region_query(
        self,
        X: np.ndarray,
        sample_index: int,
    ) -> np.ndarray:
        """Return indices of samples inside the epsilon neighborhood."""
        differences = X - X[sample_index]
        squared_distances = np.sum(differences**2, axis=1)

        return np.flatnonzero(
            squared_distances <= self.eps**2
        )

    def _expand_cluster(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        sample_index: int,
        neighbors: np.ndarray,
        cluster_id: int,
        core_mask: np.ndarray,
    ) -> None:
        """Expand one cluster from a core sample."""
        labels[sample_index] = cluster_id
        core_mask[sample_index] = True

        queue = deque(neighbors.tolist())
        queued = set(neighbors.tolist())

        while queue:
            neighbor_index = queue.popleft()

            if labels[neighbor_index] == self.NOISE:
                labels[neighbor_index] = cluster_id

            if labels[neighbor_index] != self.UNVISITED:
                continue

            labels[neighbor_index] = cluster_id

            neighbor_neighbors = self._region_query(
                X,
                neighbor_index,
            )

            if neighbor_neighbors.size >= self.min_samples:
                core_mask[neighbor_index] = True

                for candidate_index in neighbor_neighbors:
                    candidate_index = int(candidate_index)

                    if candidate_index not in queued:
                        queue.append(candidate_index)
                        queued.add(candidate_index)

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """
        Fit DBSCAN and assign cluster labels.

        Noise samples receive the label -1.
        """
        X = self._validate_input(X)

        n_samples = X.shape[0]

        labels = np.full(
            n_samples,
            self.UNVISITED,
            dtype=int,
        )

        core_mask = np.zeros(
            n_samples,
            dtype=bool,
        )

        cluster_id = 0

        for sample_index in range(n_samples):
            if labels[sample_index] != self.UNVISITED:
                continue

            neighbors = self._region_query(
                X,
                sample_index,
            )

            if neighbors.size < self.min_samples:
                labels[sample_index] = self.NOISE
                continue

            self._expand_cluster(
                X,
                labels,
                sample_index,
                neighbors,
                cluster_id,
                core_mask,
            )

            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(core_mask)
        self.n_clusters_ = cluster_id

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit DBSCAN and return cluster labels."""
        self.fit(X)

        if self.labels_ is None:
            raise RuntimeError(
                "DBSCAN fitting did not produce labels."
            )

        return self.labels_.copy()