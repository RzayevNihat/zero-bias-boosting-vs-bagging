"""
Shared preprocessing utilities for machine-learning experiments.
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Sequence
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
    name:
        Human-readable dataset identifier.
    task:
        Classification task description.
    source:
        Local source path used to load the dataset.
    notes:
        Short provenance or preprocessing note for experiment exports.
    """

    X: np.ndarray
    y: np.ndarray
    name: str = "dataset"
    task: str = "classification"
    source: str = ""
    notes: str = ""


@dataclass(frozen=True)
class DatasetBundle:
    """
    Experiment-ready dataset with metadata used in result exports.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    source: str
    task: str
    notes: str


def handle_missing_values(X: np.ndarray) -> np.ndarray:
    """
    Return a copy of ``X`` with NaN values replaced by column medians.

    The input array is never modified in place. A column containing only
    missing values cannot be imputed and raises ``ValueError``.
    """
    X_array = np.asarray(X, dtype=float)

    if X_array.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")

    if X_array.shape[0] == 0 or X_array.shape[1] == 0:
        raise ValueError("X cannot be empty.")

    if np.any(np.isinf(X_array)):
        raise ValueError("X contains infinite values.")

    if np.any(np.all(np.isnan(X_array), axis=0)):
        raise ValueError("Cannot impute a feature containing only missing values.")

    X_clean = X_array.copy()
    medians = np.nanmedian(X_clean, axis=0)
    missing_rows, missing_columns = np.where(np.isnan(X_clean))
    X_clean[missing_rows, missing_columns] = medians[missing_columns]
    return X_clean


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float | int = 0.2,
    random_state: int | None = None,
    stratify: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays into train and test subsets.

    This project-local helper covers the small subset of sklearn's
    ``train_test_split`` behavior used by the tests and experiment utilities:
    deterministic shuffling, float or integer ``test_size``, and optional
    stratification by class labels.
    """
    X_array = np.asarray(X)
    y_array = np.asarray(y)

    if X_array.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if y_array.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if X_array.shape[0] == 0:
        raise ValueError("Cannot split an empty dataset.")

    n_samples = X_array.shape[0]
    n_test = _resolve_test_size(test_size, n_samples)
    rng = np.random.default_rng(random_state)

    if stratify is None:
        indices = rng.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    else:
        stratify_array = np.asarray(stratify)
        if stratify_array.ndim != 1:
            raise ValueError("stratify must be a one-dimensional array.")
        if stratify_array.shape[0] != n_samples:
            raise ValueError("stratify must have the same length as X and y.")

        train_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        test_fraction = n_test / n_samples

        for label in np.unique(stratify_array):
            class_indices = np.flatnonzero(stratify_array == label)
            if class_indices.size < 2:
                raise ValueError("Every stratified class must contain at least two samples.")

            shuffled = rng.permutation(class_indices)
            class_test_count = int(round(class_indices.size * test_fraction))
            class_test_count = max(1, min(class_test_count, class_indices.size - 1))
            test_parts.append(shuffled[:class_test_count])
            train_parts.append(shuffled[class_test_count:])

        test_indices = np.concatenate(test_parts)
        train_indices = np.concatenate(train_parts)
        test_indices = rng.permutation(test_indices)
        train_indices = rng.permutation(train_indices)

    return (
        X_array[train_indices],
        X_array[test_indices],
        y_array[train_indices],
        y_array[test_indices],
    )


def _resolve_test_size(test_size: float | int, n_samples: int) -> int:
    """Normalize float or integer test_size into a non-empty sample count."""
    if isinstance(test_size, bool):
        raise ValueError("test_size must be a float fraction or integer count.")

    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError("Float test_size must be in the interval (0, 1).")
        n_test = int(np.ceil(n_samples * test_size))
    elif isinstance(test_size, int):
        if not 0 < test_size < n_samples:
            raise ValueError("Integer test_size must be between 1 and n_samples - 1.")
        n_test = int(test_size)
    else:
        raise ValueError("test_size must be a float fraction or integer count.")

    if n_test <= 0 or n_test >= n_samples:
        raise ValueError("test_size leaves an empty train or test split.")
    return n_test


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
    if dataset_path.is_dir():
        dataset_path = dataset_path / "wdbc.data"

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
        name="wdbc",
        task="binary_classification",
        source=str(dataset_path),
        notes=(
            "Wisconsin Diagnostic Breast Cancer dataset loaded from the "
            "local data directory; labels are M=1 and B=0."
        ),
    )


def load_adult_income(
    data_dir: str | Path = "data",
    *,
    max_samples: int | None = None,
    random_state: int | None = 42,
) -> DatasetBundle:
    """
    Load and one-hot encode the local Adult Income dataset files.

    The loader combines ``adult.data`` and ``adult.test`` from ``data/``.
    Rows containing the Adult missing-value marker ``?`` are removed.
    """
    data_path = Path(data_dir)
    train_path = data_path / "adult.data"
    test_path = data_path / "adult.test"

    if not train_path.exists():
        raise FileNotFoundError(f"Adult training file was not found at: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Adult test file was not found at: {test_path}")

    raw_data = np.vstack(
        [
            _read_adult_file(train_path),
            _read_adult_file(test_path),
        ]
    )

    complete_rows = ~np.any(raw_data == "?", axis=1)
    dropped_rows = int(raw_data.shape[0] - np.count_nonzero(complete_rows))
    raw_data = raw_data[complete_rows]
    if raw_data.shape[0] == 0:
        raise ValueError("Adult Income dataset has no complete rows after cleaning.")

    labels = raw_data[:, -1]
    valid_labels = np.isin(labels, ["<=50K", ">50K"])
    if not np.all(valid_labels):
        invalid_values = np.unique(labels[~valid_labels])
        raise ValueError(
            "Adult Income labels contain invalid values: "
            f"{invalid_values.tolist()}"
        )

    X = _encode_mixed_feature_table(raw_data[:, :-1])
    y = (labels == ">50K").astype(int)
    X, y = _subsample_rows(X, y, max_samples, random_state)

    notes = (
        "Adult Income local train/test files combined, rows with '?' removed, "
        "categorical features one-hot encoded."
    )
    if dropped_rows:
        notes += f" Dropped {dropped_rows} rows with missing values."
    if max_samples is not None:
        notes += f" Limited to at most {max_samples} samples."

    return DatasetBundle(
        name="adult",
        X=X,
        y=y,
        source=f"{train_path}; {test_path}",
        task="binary_classification",
        notes=notes,
    )


def load_covertype_subset(
    data_dir: str | Path = "data",
    *,
    max_samples: int | None = None,
    random_state: int | None = 42,
) -> DatasetBundle:
    """
    Load the local Covertype dataset, optionally as a random subset.
    """
    data_path = Path(data_dir)
    dataset_path = data_path / "covtype.data"
    if not dataset_path.exists():
        gz_path = data_path / "covtype.data.gz"
        if gz_path.exists():
            dataset_path = gz_path

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Covertype dataset was not found at: {data_path / 'covtype.data'}"
        )

    raw_data = _read_numeric_csv_table(
        dataset_path,
        expected_columns=55,
        max_samples=max_samples,
        random_state=random_state,
    )
    if raw_data.shape[1] != 55:
        raise ValueError("Covertype data must contain 54 features and 1 label column.")

    X = raw_data[:, :-1].astype(float)
    y = raw_data[:, -1].astype(int)

    notes = (
        "Covertype dataset loaded from the local data directory; original "
        "cover-type labels are kept."
    )
    if max_samples is not None:
        notes += f" Random reservoir subset limited to at most {max_samples} samples."

    return DatasetBundle(
        name="covertype",
        X=X,
        y=y,
        source=str(dataset_path),
        task="multiclass_classification",
        notes=notes,
    )


def load_mnist_binary_subset(
    *,
    max_samples: int | None = None,
    digits: tuple[str, str] = ("3", "8"),
    random_state: int | None = 42,
) -> DatasetBundle:
    """
    Fetch MNIST from OpenML through sklearn and keep a two-digit subset.

    MNIST is not distributed as a static file by ``download_data.sh`` in this
    repository, so the project loader fetches it on demand for the optional GBM
    comparison. The fetched OpenML data is cached by sklearn.
    """
    if len(digits) != 2:
        raise ValueError("digits must contain exactly two digit labels.")
    if digits[0] == digits[1]:
        raise ValueError("digits must contain two different labels.")

    try:
        from sklearn.datasets import fetch_openml
    except ImportError as error:
        raise ImportError(
            "scikit-learn is required to fetch the MNIST OpenML dataset."
        ) from error

    try:
        X_raw, y_raw = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False,
            parser="auto",
        )
    except TypeError:
        X_raw, y_raw = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False,
        )

    y_labels = np.asarray(y_raw, dtype=str)
    selected = np.isin(y_labels, digits)
    if not np.any(selected):
        raise ValueError(f"MNIST contains no samples for requested digits: {digits}")

    X = np.asarray(X_raw[selected], dtype=float) / 255.0
    label_mapping = {digits[0]: 0, digits[1]: 1}
    y = np.array([label_mapping[label] for label in y_labels[selected]], dtype=int)
    X, y = _subsample_rows(X, y, max_samples, random_state)

    notes = (
        "MNIST fetched through sklearn.datasets.fetch_openml because this "
        "repository does not provide a static MNIST file. Kept digit subset "
        f"{digits[0]} -> 0 and {digits[1]} -> 1; pixel values scaled to [0, 1]."
    )
    if max_samples is not None:
        notes += f" Limited to at most {max_samples} samples."

    return DatasetBundle(
        name=f"mnist_{digits[0]}_vs_{digits[1]}",
        X=X,
        y=y,
        source="sklearn.datasets.fetch_openml('mnist_784', version=1)",
        task="binary_classification",
        notes=notes,
    )


def load_project_datasets(
    names: Sequence[str] | None = None,
    data_dir: str | Path = "data",
    *,
    adult_max_samples: int | None = None,
    covertype_max_samples: int | None = None,
    mnist_max_samples: int | None = None,
    mnist_digits: tuple[str, str] = ("3", "8"),
    random_state: int | None = 42,
) -> list[DatasetBundle]:
    """
    Load the project datasets.

    WDBC, Adult, and Covertype are loaded from local files in ``data/``.
    MNIST is fetched on demand through sklearn/OpenML because no static MNIST
    file is provided by ``download_data.sh``.
    """
    selected_names = tuple(names) if names is not None else ("wdbc", "adult", "covertype")
    data_path = Path(data_dir)
    bundles: list[DatasetBundle] = []

    for requested_name in selected_names:
        key = _normalise_dataset_name(requested_name)
        if key == "wdbc":
            dataset = load_wdbc(data_path / "wdbc.data")
            bundles.append(
                DatasetBundle(
                    name=dataset.name,
                    X=dataset.X,
                    y=dataset.y,
                    source=dataset.source,
                    task=dataset.task,
                    notes=dataset.notes,
                )
            )
        elif key == "adult":
            bundles.append(
                load_adult_income(
                    data_path,
                    max_samples=adult_max_samples,
                    random_state=random_state,
                )
            )
        elif key == "covertype":
            bundles.append(
                load_covertype_subset(
                    data_path,
                    max_samples=covertype_max_samples,
                    random_state=random_state,
                )
            )
        elif key == "mnist":
            bundles.append(
                load_mnist_binary_subset(
                    max_samples=mnist_max_samples,
                    digits=mnist_digits,
                    random_state=random_state,
                )
            )
        else:
            raise ValueError(f"Unknown project dataset name: {requested_name}")

    return bundles


def _read_adult_file(path: Path) -> np.ndarray:
    """Read one Adult Income CSV-like file into a string table."""
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, skipinitialspace=True)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("|"):
                continue
            cleaned = [value.strip() for value in row]
            if len(cleaned) != 15:
                raise ValueError(
                    f"Adult file {path} has a row with {len(cleaned)} columns; "
                    "expected 15."
                )
            cleaned[-1] = cleaned[-1].rstrip(".")
            rows.append(cleaned)

    if not rows:
        raise ValueError(f"Adult file is empty after parsing: {path}")
    return np.asarray(rows, dtype=str)


def _encode_mixed_feature_table(features: np.ndarray) -> np.ndarray:
    """Convert numeric columns directly and categorical columns to one-hot."""
    encoded_columns: list[np.ndarray] = []
    for column_index in range(features.shape[1]):
        values = features[:, column_index]
        try:
            encoded_columns.append(values.astype(float)[:, None])
        except ValueError:
            categories = np.unique(values)
            one_hot = (values[:, None] == categories[None, :]).astype(float)
            encoded_columns.append(one_hot)
    return np.hstack(encoded_columns).astype(float)


def _read_numeric_csv_table(
    path: Path,
    *,
    expected_columns: int,
    max_samples: int | None,
    random_state: int | None,
) -> np.ndarray:
    """Read a numeric CSV file, using reservoir sampling when requested."""
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive when provided.")

        rng = np.random.default_rng(random_state)
        reservoir: list[np.ndarray] = []
        seen_rows = 0
        with _open_text(path) as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                values = np.fromstring(stripped, sep=",", dtype=float)
                if values.size != expected_columns:
                    raise ValueError(
                        f"{path} line {line_number} has {values.size} columns; "
                        f"expected {expected_columns}."
                    )
                seen_rows += 1
                if len(reservoir) < max_samples:
                    reservoir.append(values)
                    continue
                replacement_index = int(rng.integers(0, seen_rows))
                if replacement_index < max_samples:
                    reservoir[replacement_index] = values

        if not reservoir:
            raise ValueError(f"Numeric CSV file is empty: {path}")
        return np.vstack(reservoir)

    with _open_text(path) as file:
        data = np.genfromtxt(file, delimiter=",", dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] == 0:
        raise ValueError(f"Numeric CSV file is empty: {path}")
    if data.shape[1] != expected_columns:
        raise ValueError(
            f"{path} has {data.shape[1]} columns; expected {expected_columns}."
        )
    return data


def _open_text(path: Path):
    """Open plain text or gzip-compressed dataset files."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _subsample_rows(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic random row subset when max_samples is set."""
    if max_samples is None or X.shape[0] <= max_samples:
        return X, y
    if max_samples < 1:
        raise ValueError("max_samples must be positive when provided.")

    rng = np.random.default_rng(random_state)
    indices = rng.choice(X.shape[0], size=max_samples, replace=False)
    indices.sort()
    return X[indices], y[indices]


def _normalise_dataset_name(name: str) -> str:
    """Map common aliases to canonical project dataset names."""
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "breast_cancer": "wdbc",
        "breast_cancer_wisconsin": "wdbc",
        "wisconsin": "wdbc",
        "adult_income": "adult",
        "income": "adult",
        "covtype": "covertype",
        "forest_cover_type": "covertype",
    }
    return aliases.get(key, key)


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
