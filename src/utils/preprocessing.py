"""
Shared preprocessing utilities for machine-learning experiments.
"""

from __future__ import annotations

import numpy as np


class MeanImputer:
    """
    Replace missing numerical values with feature means.
    """

    def __init__(self) -> None:
        self.statistics_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MeanImputer":
        """
        Compute the mean value of each feature.
        """
        X = self._validate_input(X, allow_nan=True)

        if np.any(np.all(np.isnan(X), axis=0)):
            raise ValueError(
                "Cannot impute a feature containing only missing values."
            )

        self.statistics_ = np.nanmean(X, axis=0)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Replace missing values using the fitted feature means.
        """
        if self.statistics_ is None:
            raise RuntimeError(
                "MeanImputer must be fitted before transform."
            )

        X = self._validate_input(X, allow_nan=True)

        if X.shape[1] != self.statistics_.shape[0]:
            raise ValueError(
                "X must have the same number of features as the fitted data."
            )

        X_imputed = X.copy()

        missing_rows, missing_columns = np.where(
            np.isnan(X_imputed)
        )

        X_imputed[
            missing_rows,
            missing_columns,
        ] = self.statistics_[missing_columns]

        return X_imputed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the imputer and return the transformed data.
        """
        return self.fit(X).transform(X)

    @staticmethod
    def _validate_input(
        X: np.ndarray,
        *,
        allow_nan: bool,
    ) -> np.ndarray:
        """Validate and convert input data."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X cannot be empty.")

        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values.")

        if not allow_nan and np.any(np.isnan(X)):
            raise ValueError("X contains missing values.")

        return X


class StandardScaler:
    """
    Standardize numerical features to zero mean and unit variance.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """
        Compute feature means and standard deviations.
        """
        X = self._validate_input(X)

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        self.scale_ = np.where(
            self.scale_ == 0,
            1.0,
            self.scale_,
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize data using fitted statistics.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError(
                "StandardScaler must be fitted before transform."
            )

        X = self._validate_input(X)

        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                "X must have the same number of features as the fitted data."
            )

        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the scaler and return standardized data.
        """
        return self.fit(X).transform(X)

    @staticmethod
    def _validate_input(X: np.ndarray) -> np.ndarray:
        """Validate and convert input data."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X cannot be empty.")

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contains NaN or infinite values."
            )

        return X


class PreprocessingPipeline:
    """
    Apply mean imputation followed by standard scaling.
    """

    def __init__(self) -> None:
        self.imputer = MeanImputer()
        self.scaler = StandardScaler()
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray) -> "PreprocessingPipeline":
        """
        Fit both preprocessing stages.
        """
        X_imputed = self.imputer.fit_transform(X)
        self.scaler.fit(X_imputed)
        self.is_fitted_ = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted imputation and scaling.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "PreprocessingPipeline must be fitted before transform."
            )

        X_imputed = self.imputer.transform(X)

        return self.scaler.transform(X_imputed)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the pipeline and return processed data.
        """
        return self.fit(X).transform(X)