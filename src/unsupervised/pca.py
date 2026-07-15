"""
Principal Component Analysis implemented from scratch.
"""

from __future__ import annotations

import numpy as np


class PCA:
    """
    Principal Component Analysis using covariance eigen-decomposition.

    Parameters
    ----------
    n_components:
        Number of principal components to retain.
    """

    def __init__(self, n_components: int) -> None:
        if not isinstance(n_components, int):
            raise TypeError("n_components must be an integer.")

        if n_components <= 0:
            raise ValueError("n_components must be greater than zero.")

        self.n_components = n_components

        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        """
        Fit PCA using covariance matrix eigen-decomposition.

        Parameters
        ----------
        X:
            Input data with shape (n_samples, n_features).

        Returns
        -------
        PCA
            The fitted PCA instance.
        """
        X = self._validate_input(X)

        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError("PCA requires at least two samples.")

        if self.n_components > n_features:
            raise ValueError(
                "n_components cannot be greater than the number of features."
            )

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        covariance_matrix = np.cov(X_centered, rowvar=False)
        covariance_matrix = np.atleast_2d(covariance_matrix)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        eigenvalues = np.maximum(eigenvalues, 0.0)

        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]

        total_variance = np.sum(eigenvalues)

        if total_variance == 0:
            self.explained_variance_ratio_ = np.zeros(
                self.n_components,
                dtype=float,
            )
        else:
            self.explained_variance_ratio_ = (
                self.explained_variance_ / total_variance
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto the fitted principal components.

        Parameters
        ----------
        X:
            Input data with shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Transformed data with shape (n_samples, n_components).
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                "X must have the same number of features as the fitted data."
            )

        X_centered = X - self.mean_

        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and return the transformed data.
        """
        return self.fit(X).transform(X)

    def _check_is_fitted(self) -> None:
        """Raise an error when PCA has not been fitted."""
        if (
            self.mean_ is None
            or self.components_ is None
            or self.explained_variance_ is None
            or self.explained_variance_ratio_ is None
        ):
            raise RuntimeError("PCA must be fitted before transform.")

    @staticmethod
    def _validate_input(X: np.ndarray) -> np.ndarray:
        """Validate and convert input data to a floating-point array."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X cannot be empty.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")

        return X