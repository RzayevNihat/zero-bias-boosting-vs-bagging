"""
K-Means clustering implemented from scratch.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class KMeans:
    """
    K-Means clustering using Lloyd's algorithm.

    Parameters
    ----------
    n_clusters:
        Number of clusters.
    max_iter:
        Maximum number of iterations for each initialization.
    tol:
        Convergence tolerance based on centroid movement.
    n_init:
        Number of independent centroid initializations.
    random_state:
        Seed used for reproducible centroid initialization.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
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

        if not isinstance(n_init, int):
            raise TypeError("n_init must be an integer.")

        if n_init <= 0:
            raise ValueError("n_init must be positive.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
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

    @staticmethod
    def _squared_distances(
        X: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """Return squared distances from samples to every centroid."""
        differences = (
            X[:, np.newaxis, :]
            - centroids[np.newaxis, :, :]
        )

        return np.sum(differences**2, axis=2)

    def _initialize_centroids(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Select initial centroids randomly from the input samples."""
        indices = rng.choice(
            X.shape[0],
            size=self.n_clusters,
            replace=False,
        )

        return X[indices].copy()

    def _assign_clusters(
        self,
        X: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """Assign every sample to its nearest centroid."""
        squared_distances = self._squared_distances(
            X,
            centroids,
        )

        return np.argmin(squared_distances, axis=1)

    def _update_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        old_centroids: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Recalculate centroids using assigned samples."""
        new_centroids = np.empty_like(old_centroids)

        for cluster_index in range(self.n_clusters):
            cluster_points = X[labels == cluster_index]

            if cluster_points.shape[0] == 0:
                replacement_index = rng.integers(
                    low=0,
                    high=X.shape[0],
                )
                new_centroids[cluster_index] = X[
                    replacement_index
                ]
            else:
                new_centroids[cluster_index] = np.mean(
                    cluster_points,
                    axis=0,
                )

        return new_centroids

    @staticmethod
    def _calculate_inertia(
        X: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
    ) -> float:
        """Calculate the within-cluster sum of squared distances."""
        differences = X - centroids[labels]

        return float(np.sum(differences**2))

    def _run_single_initialization(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Run one complete Lloyd iteration sequence."""
        centroids = self._initialize_centroids(X, rng)

        for iteration in range(1, self.max_iter + 1):
            labels = self._assign_clusters(X, centroids)

            new_centroids = self._update_centroids(
                X,
                labels,
                centroids,
                rng,
            )

            centroid_shift = np.linalg.norm(
                new_centroids - centroids,
            )

            centroids = new_centroids

            if centroid_shift <= self.tol:
                break

        labels = self._assign_clusters(X, centroids)

        inertia = self._calculate_inertia(
            X,
            labels,
            centroids,
        )

        return centroids, labels, inertia, iteration

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Fit K-Means using multiple centroid initializations.

        The initialization producing the lowest inertia is retained.
        """
        X = self._validate_input(X)
        rng = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_centroids: np.ndarray | None = None
        best_labels: np.ndarray | None = None
        best_iteration: int | None = None

        for _ in range(self.n_init):
            centroids, labels, inertia, iteration = (
                self._run_single_initialization(X, rng)
            )

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.copy()
                best_labels = labels.copy()
                best_iteration = iteration

        if (
            best_centroids is None
            or best_labels is None
            or best_iteration is None
        ):
            raise RuntimeError(
                "KMeans failed to produce a clustering result."
            )

        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = best_iteration

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new samples to the nearest fitted centroids."""
        if self.centroids_ is None:
            raise RuntimeError(
                "KMeans must be fitted before predict."
            )

        X = self._validate_input(
            X,
            check_cluster_count=False,
        )

        if X.shape[1] != self.centroids_.shape[1]:
            raise ValueError(
                "X must have the same number of features "
                "as the fitted data."
            )

        return self._assign_clusters(
            X,
            self.centroids_,
        )

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and return cluster labels."""
        self.fit(X)

        if self.labels_ is None:
            raise RuntimeError(
                "KMeans fitting did not produce labels."
            )

        return self.labels_.copy()

    def score(self) -> float:
        """Return the fitted model's inertia."""
        if self.inertia_ is None:
            raise RuntimeError(
                "KMeans must be fitted before score."
            )

        return self.inertia_

    @staticmethod
    def elbow_curve(
        X: np.ndarray,
        cluster_range: Iterable[int],
        *,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate inertia for different numbers of clusters.

        Returns
        -------
        cluster_values:
            Cluster counts used for the elbow analysis.
        inertias:
            Best inertia obtained for each cluster count.
        """
        cluster_values = np.asarray(
            list(cluster_range),
            dtype=int,
        )

        if cluster_values.ndim != 1:
            raise ValueError(
                "cluster_range must be one-dimensional."
            )

        if cluster_values.size == 0:
            raise ValueError(
                "cluster_range cannot be empty."
            )

        if np.any(cluster_values <= 0):
            raise ValueError(
                "All cluster counts must be positive."
            )

        inertias = np.empty(
            cluster_values.shape[0],
            dtype=float,
        )

        for index, cluster_count in enumerate(
            cluster_values
        ):
            model = KMeans(
                n_clusters=int(cluster_count),
                max_iter=max_iter,
                tol=tol,
                n_init=n_init,
                random_state=random_state,
            )

            model.fit(X)
            inertias[index] = model.score()

        return cluster_values, inertias