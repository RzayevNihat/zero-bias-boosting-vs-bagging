"""
K-Means clustering implemented from scratch.
"""

from __future__ import annotations

import numpy as np


class KMeans:
    """
    K-Means clustering using Lloyd's algorithm.

    Parameters
    ----------
    n_clusters:
        Number of clusters.
    max_iter:
        Maximum number of iterations.
    tol:
        Convergence tolerance based on centroid movement.
    random_state:
        Seed for reproducible centroid initialization.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        if not isinstance(n_clusters, int):
            raise TypeError("n_clusters must be an integer.")

        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")

        if not isinstance(max_iter, int):
            raise TypeError("max_iter must be an integer.")

        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        if tol < 0:
            raise ValueError("tol cannot be negative.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    def _validate_input(
        self,
        X: np.ndarray,
        *,
        check_cluster_count: bool = True,
    ) -> np.ndarray:
        """Validate and convert input data."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X cannot be empty.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")

        if check_cluster_count and X.shape[0] < self.n_clusters:
            raise ValueError(
                "Number of samples must be greater than or equal to "
                "n_clusters."
            )

        return X

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Select initial centroids randomly from the input samples."""
        rng = np.random.default_rng(self.random_state)

        indices = rng.choice(
            X.shape[0],
            size=self.n_clusters,
            replace=False,
        )

        return X[indices].copy()

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assign every sample to its nearest centroid."""
        if self.centroids_ is None:
            raise RuntimeError("Centroids have not been initialized.")

        differences = X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]
        squared_distances = np.sum(differences**2, axis=2)

        return np.argmin(squared_distances, axis=1)

    def _update_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Recalculate centroids using assigned samples."""
        if self.centroids_ is None:
            raise RuntimeError("Centroids have not been initialized.")

        new_centroids = np.empty_like(self.centroids_)

        for cluster_index in range(self.n_clusters):
            cluster_points = X[labels == cluster_index]

            if cluster_points.shape[0] == 0:
                new_centroids[cluster_index] = self.centroids_[cluster_index]
            else:
                new_centroids[cluster_index] = np.mean(
                    cluster_points,
                    axis=0,
                )

        return new_centroids

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Fit K-Means using Lloyd's iterative clustering algorithm.
        """
        X = self._validate_input(X)
        self.centroids_ = self._initialize_centroids(X)

        for iteration in range(1, self.max_iter + 1):
            labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, labels)

            centroid_shift = np.linalg.norm(
                new_centroids - self.centroids_,
            )

            self.centroids_ = new_centroids
            self.n_iter_ = iteration

            if centroid_shift <= self.tol:
                break

        self.labels_ = self._assign_clusters(X)

        differences = X - self.centroids_[self.labels_]
        self.inertia_ = float(np.sum(differences**2))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new samples to the nearest fitted centroids."""
        if self.centroids_ is None:
            raise RuntimeError("KMeans must be fitted before predict.")

        X = self._validate_input(
            X,
            check_cluster_count=False,
        )

        if X.shape[1] != self.centroids_.shape[1]:
            raise ValueError(
                "X must have the same number of features as the fitted data."
            )

        return self._assign_clusters(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and return cluster labels."""
        self.fit(X)

        if self.labels_ is None:
            raise RuntimeError("KMeans fitting did not produce labels.")

        return self.labels_.copy()