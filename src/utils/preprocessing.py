"""
Shared preprocessing utilities for machine-learning experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from sklearn.model_selection import train_test_split as _sklearn_train_test_split


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
    name: str = "dataset"
    task: str = "classification"
    source: str = "unknown"
    notes: str = ""


@dataclass(frozen=True)
class DatasetBundle:
    """Extended dataset container used by the project experiments."""

    name: str
    X: np.ndarray
    y: np.ndarray
    source: str = "unknown"
    task: str = "classification"
    notes: str = ""


def handle_missing_values(X: np.ndarray) -> np.ndarray:
    """Impute missing values with the column median without mutating the input."""
    X_array = np.asarray(X, dtype=float)
    if X_array.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")

    if X_array.shape[0] == 0 or X_array.shape[1] == 0:
        raise ValueError("X cannot be empty.")

    X_filled = X_array.copy()
    for column_index in range(X_filled.shape[1]):
        column = X_filled[:, column_index]
        if np.all(np.isnan(column)):
            raise ValueError("Cannot impute a column containing only missing values.")
        median_value = np.nanmedian(column)
        X_filled[:, column_index] = np.where(np.isnan(column), median_value, column)

    return X_filled


def train_test_split(*args: Any, **kwargs: Any) -> Any:
    """Compatibility wrapper around sklearn's train_test_split."""
    return _sklearn_train_test_split(*args, **kwargs)


def _resolve_test_size(n_samples: int, test_size: float | int) -> int:
    """Resolve a fractional or absolute test size to a validated row count."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if isinstance(test_size, (float, np.floating)):
        if not 0.0 < float(test_size) < 1.0:
            raise ValueError("A fractional test_size must be between 0 and 1.")
        resolved = int(np.ceil(n_samples * float(test_size)))
    elif isinstance(test_size, (int, np.integer)):
        resolved = int(test_size)
    else:
        raise TypeError("test_size must be a float or integer.")
    if resolved <= 0 or resolved >= n_samples:
        raise ValueError("test_size must leave at least one training sample.")
    return resolved


def _open_text(path: str | Path) -> TextIO:
    """Open plain-text or gzip-compressed dataset files as UTF-8 text."""
    dataset_path = Path(path)
    if dataset_path.suffix.lower() == ".gz":
        return gzip.open(dataset_path, mode="rt", encoding="utf-8")
    return dataset_path.open(mode="r", encoding="utf-8")


def _normalise_dataset_name(name: str) -> str:
    """Map accepted dataset aliases to their canonical project name."""
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "adult_income": "adult",
        "covtype": "covertype",
        "covertype_subset": "covertype",
        "digits": "mnist",
        "mnist_binary": "mnist",
    }
    return aliases.get(normalized, normalized)


def _subsample_rows(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a reproducible row subset without replacement."""
    if max_samples is None or len(X) <= max_samples:
        return X, y
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X), size=max_samples, replace=False)
    return X[indices], y[indices]


def _resolve_data_path(path: str | Path, *, data_dir: str | Path | None = None) -> Path:
    """Resolve a dataset path from an explicit path or repository data directory."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    if candidate.exists():
        return candidate

    if data_dir is not None:
        fallback = Path(data_dir) / candidate
        if fallback.exists():
            return fallback

    if candidate.suffix == "":
        directory_candidate = candidate.parent / candidate.name
        if directory_candidate.exists():
            return directory_candidate

    return candidate


def _read_delimited_table(path: str | Path, *, max_rows: int | None = None) -> np.ndarray:
    """Read a delimited text table into a 2D string array."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file was not found at: {dataset_path}")

    raw_data = np.genfromtxt(
        dataset_path,
        delimiter=",",
        dtype=str,
        encoding="utf-8",
        max_rows=max_rows,
    )

    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)

    if raw_data.ndim != 2:
        raise ValueError(f"Dataset at {dataset_path} must be a two-dimensional table.")

    if raw_data.shape[0] == 0:
        raise ValueError(f"Dataset at {dataset_path} cannot be empty.")

    return raw_data


def _read_adult_file(path: str | Path, *, max_rows: int | None = None) -> np.ndarray:
    """Read an Adult-income data file, dropping blank rows."""
    table = _read_delimited_table(path, max_rows=max_rows)
    non_blank = np.any(np.char.strip(table.astype(str)) != "", axis=1)
    return table[non_blank]


def _read_numeric_csv_table(
    path: str | Path,
    *,
    max_rows: int | None = None,
) -> np.ndarray:
    """Read and validate a numeric CSV dataset table."""
    table = _read_delimited_table(path, max_rows=max_rows)
    try:
        numeric = table.astype(float)
    except ValueError as error:
        raise ValueError(f"Dataset at {path} must contain only numeric values.") from error
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"Dataset at {path} contains NaN or infinite values.")
    return numeric


def _coerce_feature_matrix(values: np.ndarray) -> np.ndarray:
    """Convert a text table to a numeric feature matrix."""
    features = np.asarray(values, dtype=str)
    if features.ndim != 2:
        raise ValueError("Features must be a two-dimensional array.")

    matrix = np.empty(features.shape, dtype=float)

    for column_index in range(features.shape[1]):
        column = features[:, column_index]
        converted = np.empty(column.shape[0], dtype=float)

        normalized_column = np.array(
            [str(item).strip() for item in column],
            dtype=str,
        )
        unique_values = np.unique(normalized_column)
        mapping = {
            value: index
            for index, value in enumerate(unique_values)
        }

        for row_index, value in enumerate(normalized_column):
            if value in {"", "?", "nan", "NaN"}:
                converted[row_index] = np.nan
                continue
            try:
                converted[row_index] = float(value)
            except ValueError:
                if value in mapping:
                    converted[row_index] = float(mapping[value])
                    continue
                raise ValueError(
                    f"Feature matrix contains non-numeric values: {value}"
                ) from None

        matrix[:, column_index] = converted

    if not np.all(np.isfinite(matrix[~np.isnan(matrix)])):
        raise ValueError("Feature matrix contains non-numeric values.")

    return matrix


def _encode_mixed_feature_table(values: np.ndarray) -> np.ndarray:
    """Encode mixed numeric/categorical features into a numeric matrix."""
    return handle_missing_values(_coerce_feature_matrix(values))


def _coerce_labels(values: np.ndarray) -> np.ndarray:
    """Convert labels to integer codes."""
    labels = np.asarray(values, dtype=str)
    if labels.ndim != 1:
        raise ValueError("Labels must be a one-dimensional array.")

    unique_labels = np.unique(labels)
    if unique_labels.size == 0:
        raise ValueError("Labels cannot be empty.")

    mapping = {value: index for index, value in enumerate(unique_labels)}
    encoded = np.array([mapping[value] for value in labels], dtype=int)
    return encoded


def _sample_dataset(
    dataset: DatasetBundle,
    *,
    max_samples: int | None = None,
    random_state: int = 42,
) -> DatasetBundle:
    """Sample rows from a dataset when a maximum size is requested."""
    if max_samples is None or dataset.X.shape[0] <= max_samples:
        return dataset

    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")

    rng = np.random.default_rng(random_state)
    indices = rng.choice(dataset.X.shape[0], size=max_samples, replace=False)
    return DatasetBundle(
        name=dataset.name,
        X=dataset.X[indices],
        y=dataset.y[indices],
        source=dataset.source,
        task=dataset.task,
        notes=dataset.notes,
    )


def load_wdbc(
    path: str | Path = "data/wdbc.data",
    *,
    data_dir: str | Path | None = None,
) -> DatasetBundle:
    """
    Load the Wisconsin Diagnostic Breast Cancer dataset.

    The raw WDBC file contains:

    - column 0: sample ID
    - column 1: diagnosis, M or B
    - columns 2 onward: numerical features

    Malignant samples are encoded as 1.
    Benign samples are encoded as 0.

    If the local data file is not present, this loader falls back to the
    breast-cancer dataset provided by scikit-learn for compatibility.
    """
    dataset_path = _resolve_data_path(path)
    if dataset_path.exists() and dataset_path.is_dir():
        dataset_path = dataset_path / "wdbc.data"

    if not dataset_path.exists():
        from sklearn.datasets import load_breast_cancer

        breast_cancer = load_breast_cancer()
        X = np.asarray(breast_cancer.data, dtype=float)
        y = np.asarray(breast_cancer.target, dtype=int)
        return DatasetBundle(
            name="wdbc",
            X=X,
            y=y,
            source="sklearn.datasets.load_breast_cancer",
            task="binary",
            notes="Fallback to sklearn breast cancer dataset because the local WDBC file is unavailable.",
        )

    raw_data = _read_delimited_table(dataset_path, max_rows=None)

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

    X = _coerce_feature_matrix(raw_data[:, 2:])
    X = handle_missing_values(X)
    y = np.where(
        diagnoses == "M",
        1,
        0,
    ).astype(int)

    if not np.all(np.isfinite(X[~np.isnan(X)])):
        raise ValueError(
            "WDBC feature matrix contains NaN or infinite values."
        )

    return DatasetBundle(
        name="wdbc",
        X=X,
        y=y,
        source="wdbc.data",
        task="binary",
        notes="Wisconsin Diagnostic Breast Cancer dataset.",
    )


def load_adult(
    path: str | Path = "data/adult.data",
    *,
    data_dir: str | Path | None = None,
    max_samples: int | None = None,
    random_state: int = 42,
) -> DatasetBundle:
    """Load the Adult dataset from a local CSV/TSV-like file."""
    dataset_path = _resolve_data_path(path, data_dir=data_dir)
    if dataset_path.exists() and dataset_path.is_dir():
        dataset_path = dataset_path / "adult.data"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Adult dataset was not found at: {dataset_path}"
        )

    raw_data = _read_delimited_table(dataset_path, max_rows=max_samples)
    if raw_data.shape[1] < 2:
        raise ValueError("Adult dataset must contain at least one feature column and a label.")

    X = _coerce_feature_matrix(raw_data[:, :-1])
    X = handle_missing_values(X)
    y = _coerce_labels(raw_data[:, -1])
    dataset = DatasetBundle(
        name="adult",
        X=X,
        y=y,
        source=str(dataset_path),
        task="binary",
        notes="Adult-income-style tabular dataset.",
    )
    return _sample_dataset(dataset, max_samples=max_samples, random_state=random_state)


def load_covertype(
    path: str | Path = "data/covtype.data",
    *,
    data_dir: str | Path | None = None,
    max_samples: int | None = None,
    random_state: int = 42,
) -> DatasetBundle:
    """Load the Covertype dataset from a local CSV-like file."""
    dataset_path = _resolve_data_path(path, data_dir=data_dir)
    if dataset_path.exists() and dataset_path.is_dir():
        dataset_path = dataset_path / "covtype.data"

    # Accept the legacy project filename produced by older download scripts.
    if not dataset_path.exists() and dataset_path.name == "covtype.data":
        legacy_path = dataset_path.with_name("covertype.data")
        if legacy_path.exists():
            dataset_path = legacy_path

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Covertype dataset was not found at: {dataset_path}"
        )

    raw_data = _read_delimited_table(dataset_path, max_rows=max_samples)
    if raw_data.shape[1] < 2:
        raise ValueError("Covertype dataset must contain at least one feature column and a label.")

    X = _coerce_feature_matrix(raw_data[:, :-1])
    X = handle_missing_values(X)
    y = _coerce_labels(raw_data[:, -1])
    dataset = DatasetBundle(
        name="covertype",
        X=X,
        y=y,
        source=str(dataset_path),
        task="multiclass",
        notes="Forest cover-type tabular dataset.",
    )
    return _sample_dataset(dataset, max_samples=max_samples, random_state=random_state)


def load_mnist_dataset(
    *,
    path: str | Path | None = None,
    sample_limit: int | None = None,
    random_state: int = 42,
) -> DatasetBundle:
    """Load MNIST from CSV, falling back to OpenML when no local file exists.

    The expected CSV layout is the common MNIST layout: the label is the first
    column and the remaining 784 columns are pixel values.
    """
    if path is None:
        candidate_paths = [Path("data/mnist.csv"), Path("mnist.csv")]
        dataset_path = None
        for candidate in candidate_paths:
            if candidate.exists():
                dataset_path = candidate
                break
    else:
        dataset_path = _resolve_data_path(path)

    if dataset_path is not None and dataset_path.exists():
        raw_data = _read_delimited_table(dataset_path, max_rows=sample_limit)
        if raw_data.shape[1] != 785:
            raise ValueError("MNIST CSV must contain one label and 784 pixel columns.")
        X = _coerce_feature_matrix(raw_data[:, 1:])
        y = _coerce_labels(raw_data[:, 0])
        source = str(dataset_path)
        notes = "MNIST handwritten digits from the local data directory."
    else:
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X = np.asarray(mnist.data, dtype=float)
        y = np.asarray(mnist.target, dtype=int)
        source = "openml:mnist_784:1"
        notes = "MNIST handwritten digits from OpenML."

    if sample_limit is not None:
        if sample_limit <= 0:
            raise ValueError("sample_limit must be positive.")
        sample_limit = min(sample_limit, X.shape[0])
        rng = np.random.default_rng(random_state)
        selected_indices = rng.choice(
            X.shape[0],
            size=sample_limit,
            replace=False,
        )
        X = X[selected_indices]
        y = y[selected_indices]

    return DatasetBundle(
        name="mnist",
        X=X,
        y=y,
        source=source,
        task="multiclass",
        notes=notes,
    )


def load_digits_dataset(
    *,
    path: str | Path | None = None,
    sample_limit: int | None = None,
    random_state: int = 42,
) -> DatasetBundle:
    """Backward-compatible alias for :func:`load_mnist_dataset`."""
    return load_mnist_dataset(
        path=path,
        sample_limit=sample_limit,
        random_state=random_state,
    )


def load_adult_income(
    path: str | Path = "data/adult.data",
    *,
    max_samples: int | None = None,
    random_state: int = 42,
) -> DatasetBundle:
    """Named project API for loading the Adult Income dataset."""
    return load_adult(
        path,
        max_samples=max_samples,
        random_state=random_state,
    )


def load_covertype_subset(
    path: str | Path = "data/covtype.data",
    *,
    max_samples: int | None = 5000,
    random_state: int = 42,
) -> DatasetBundle:
    """Load a reproducible subset of the Covertype dataset."""
    return load_covertype(
        path,
        max_samples=max_samples,
        random_state=random_state,
    )


def load_mnist_binary_subset(
    *,
    digits: tuple[int, int] = (3, 8),
    path: str | Path | None = None,
    max_samples: int | None = 1000,
    random_state: int = 42,
) -> DatasetBundle:
    """Load two MNIST classes and encode them as binary labels 0 and 1."""
    if len(digits) != 2 or digits[0] == digits[1]:
        raise ValueError("digits must contain two distinct MNIST labels.")
    if any(digit < 0 or digit > 9 for digit in digits):
        raise ValueError("MNIST digits must be between 0 and 9.")

    dataset = load_mnist_dataset(path=path, random_state=random_state)
    mask = np.isin(dataset.y, digits)
    X = dataset.X[mask]
    y = np.where(dataset.y[mask] == digits[0], 0, 1).astype(int)
    X, y = _subsample_rows(X, y, max_samples, random_state=random_state)
    return DatasetBundle(
        name=f"mnist_{digits[0]}_vs_{digits[1]}",
        X=X,
        y=y,
        source=dataset.source,
        task="binary",
        notes=f"MNIST binary subset containing digits {digits[0]} and {digits[1]}.",
    )


def load_project_datasets(
    *,
    names: tuple[str, ...] | list[str] = ("wdbc", "adult", "covertype"),
    data_dir: str | Path = "data",
    adult_max_samples: int | None = None,
    covertype_max_samples: int | None = None,
    mnist_max_samples: int | None = None,
    mnist_digits: tuple[str, ...] | list[str] | None = None,
    random_state: int = 42,
) -> list[DatasetBundle]:
    """Load one or more project datasets into DatasetBundle objects."""
    data_directory = Path(data_dir)
    datasets: list[DatasetBundle] = []

    for name in names:
        normalized_name = _normalise_dataset_name(str(name))
        if normalized_name == "wdbc":
            dataset = load_wdbc(
                data_directory / "wdbc.data",
                data_dir=data_directory,
            )
        elif normalized_name == "adult":
            dataset = load_adult(
                data_directory / "adult.data",
                data_dir=data_directory,
                max_samples=adult_max_samples,
                random_state=random_state,
            )
        elif normalized_name == "covertype":
            dataset = load_covertype(
                data_directory / "covtype.data",
                data_dir=data_directory,
                max_samples=covertype_max_samples,
                random_state=random_state,
            )
        elif normalized_name in {"mnist", "digits"}:
            if mnist_digits:
                selected_digits = [int(digit) for digit in mnist_digits]
                if len(selected_digits) != 2:
                    raise ValueError("mnist_digits must contain exactly two labels.")
                dataset = load_mnist_binary_subset(
                    digits=(selected_digits[0], selected_digits[1]),
                    max_samples=mnist_max_samples,
                    random_state=random_state,
                )
            else:
                dataset = load_mnist_dataset(
                    sample_limit=mnist_max_samples,
                    random_state=random_state,
                )
        else:
            raise ValueError(f"Unsupported dataset name: {name}")

        datasets.append(dataset)

    return datasets


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
