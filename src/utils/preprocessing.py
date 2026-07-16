"""
Shared preprocessing utilities for machine-learning experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Dataset:
    """
    Container for a feature matrix and target labels.

    Attributes
    ----------
    X:
        Feature matrix with shape (n_samples, n_features).
    y:
        Target labels with shape (n_samples,).
    """

    X: np.ndarray
    y: np.ndarray


def load_wdbc(
    path: str | Path = "data/wdbc.data",
) -> Dataset:
    """
    Load the Wisconsin Diagnostic Breast Cancer dataset.

    The raw WDBC file contains:

    - column 0: sample ID
    - column 1: diagnosis, M or B
    - columns 2 onward: numerical features

    Malignant samples are encoded as 1.
    Benign samples are encoded as 0.

    Parameters
    ----------
    path:
        Path to the raw ``wdbc.data`` file.

    Returns
    -------
    Dataset
        Dataset object containing ``X`` and ``y`` arrays.
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"WDBC dataset was not found at: {dataset_path}"
        )

    raw_data = np.genfromtxt(
        dataset_path,
        delimiter=",",
        dtype=str,
    )

    if raw_data.ndim != 2:
        raise ValueError(
            "WDBC dataset must be a two-dimensional table."
        )

    if raw_data.shape[0] == 0:
        raise ValueError("WDBC dataset cannot be empty.")

    if raw_data.shape[1] < 3:
        raise ValueError(
            "WDBC dataset must contain an ID, diagnosis, "
            "and at least one feature."
        )

    diagnoses = raw_data[:, 1]

    valid_diagnoses = np.isin(
        diagnoses,
        ["M", "B"],
    )

    if not np.all(valid_diagnoses):
        invalid_values = np.unique(
            diagnoses[~valid_diagnoses]
        )

        raise ValueError(
            "WDBC diagnosis column contains invalid values: "
            f"{invalid_values.tolist()}"
        )

    try:
        X = raw_data[:, 2:].astype(float)
    except ValueError as error:
        raise ValueError(
            "WDBC feature columns must contain numerical values."
        ) from error

    y = np.where(
        diagnoses == "M",
        1,
        0,
    ).astype(int)

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "WDBC feature matrix contains NaN or infinite values."
        )

    return Dataset(
        X=X,
        y=y,
    )


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
        X = self._validate_input(
            X,
            allow_nan=True,
        )

        if np.any(
            np.all(
                np.isnan(X),
                axis=0,
            )
        ):
            raise ValueError(
                "Cannot impute a feature containing only missing values."
            )

        self.statistics_ = np.nanmean(
            X,
            axis=0,
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Replace missing values using the fitted feature means.
        """
        if self.statistics_ is None:
            raise RuntimeError(
                "MeanImputer must be fitted before transform."
            )

        X = self._validate_input(
            X,
            allow_nan=True,
        )

        if X.shape[1] != self.statistics_.shape[0]:
            raise ValueError(
                "X must have the same number of features "
                "as the fitted data."
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
        X = np.asarray(
            X,
            dtype=float,
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional array."
            )

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X cannot be empty.")

        if np.any(np.isinf(X)):
            raise ValueError(
                "X contains infinite values."
            )

        if not allow_nan and np.any(np.isnan(X)):
            raise ValueError(
                "X contains missing values."
            )

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

        self.mean_ = np.mean(
            X,
            axis=0,
        )

        self.scale_ = np.std(
            X,
            axis=0,
        )

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
                "X must have the same number of features "
                "as the fitted data."
            )

        return (
            X - self.mean_
        ) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the scaler and return standardized data.
        """
        return self.fit(X).transform(X)

    @staticmethod
    def _validate_input(
        X: np.ndarray,
    ) -> np.ndarray:
        """Validate and convert input data."""
        X = np.asarray(
            X,
            dtype=float,
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional array."
            )

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

    def fit(
        self,
        X: np.ndarray,
    ) -> "PreprocessingPipeline":
        """
        Fit both preprocessing stages.
        """
        X_imputed = self.imputer.fit_transform(X)

        self.scaler.fit(
            X_imputed
        )

        self.is_fitted_ = True

        return self

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Apply fitted imputation and scaling.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "PreprocessingPipeline must be fitted "
                "before transform."
            )

        X_imputed = self.imputer.transform(X)

        return self.scaler.transform(
            X_imputed
        )

    def fit_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Fit the pipeline and return processed data.
        """
        return self.fit(X).transform(X)