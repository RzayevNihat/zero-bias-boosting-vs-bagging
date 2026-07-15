"""Utility functions for Person 3 Random Forest experiments.

The helpers in this file keep the experiment scripts short and reproducible.
They intentionally use sklearn only for datasets, metrics, preprocessing, and
reference baselines. The project implementation of Random Forest remains the
from-scratch class in ``src.bagging.random_forest``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, make_classification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# This file is located at:
# repository/src/experiments/rf_utils.py
#
# parents[0] -> repository/src/experiments
# parents[1] -> repository/src
# parents[2] -> repository root
ROOT_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = ROOT_DIR / "data"


@dataclass(frozen=True)
class DatasetBundle:
    """Container used by the experiment scripts."""

    name: str
    X: np.ndarray
    y: np.ndarray
    description: str
    is_binary: bool

    @property
    def minority_fraction(self) -> float:
        """Return the fraction of samples belonging to the smallest class."""
        y_array = np.asarray(self.y)

        if y_array.size == 0:
            raise ValueError("Dataset bundle must contain at least one sample.")

        _, counts = np.unique(y_array, return_counts=True)
        return float(counts.min() / counts.sum())


def ensure_output_dirs() -> None:
    """Create output folders for figures and CSV result tables."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_breast_cancer_bundle() -> DatasetBundle:
    """Load a small real-world binary classification dataset."""
    data = load_breast_cancer()

    return DatasetBundle(
        name="breast_cancer",
        X=data.data.astype(float),
        y=data.target.astype(int),
        description=(
            "Breast Cancer Wisconsin Diagnostic dataset from "
            "sklearn.datasets."
        ),
        is_binary=True,
    )


def load_digits_binary_bundle() -> DatasetBundle:
    """Load a high-dimensional binary subset of the Digits dataset.

    Digits has 64 input features, which satisfies the project requirement for
    at least one high-dimensional dataset with more than 20 features.
    """
    data = load_digits()
    mask = np.isin(data.target, [3, 8])

    X = data.data[mask].astype(float)
    y = (data.target[mask] == 8).astype(int)

    return DatasetBundle(
        name="digits_3_vs_8",
        X=X,
        y=y,
        description=(
            "Binary high-dimensional subset of sklearn Digits: "
            "class 3 versus class 8."
        ),
        is_binary=True,
    )


def load_imbalanced_bundle() -> DatasetBundle:
    """Load a real or fallback severely imbalanced binary dataset.

    If ``data/covtype.data`` exists, a one-vs-rest Covertype task is created
    with cover type 4 as the positive class. Otherwise, a deterministic
    synthetic 99:1 dataset is used for offline smoke testing.
    """
    covtype_path = DATA_DIR / "covtype.data"

    if covtype_path.exists():
        columns = [f"feature_{index}" for index in range(54)] + ["target"]
        dataframe = pd.read_csv(
            covtype_path,
            header=None,
            names=columns,
        )

        dataframe = dataframe.sample(
            n=min(5000, len(dataframe)),
            random_state=RANDOM_STATE,
        )

        X = dataframe.iloc[:, :-1].to_numpy(dtype=float)
        y = (dataframe["target"].to_numpy() == 4).astype(int)

        return DatasetBundle(
            name="covertype_type4_imbalanced",
            X=X,
            y=y,
            description=(
                "Covertype one-vs-rest task with rare cover type 4 "
                "as the positive class."
            ),
            is_binary=True,
        )

    X, y = make_classification(
        n_samples=1200,
        n_features=24,
        n_informative=10,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[0.99, 0.01],
        flip_y=0.0,
        class_sep=1.2,
        random_state=RANDOM_STATE,
    )

    return DatasetBundle(
        name="synthetic_imbalanced_99_1",
        X=X.astype(float),
        y=y.astype(int),
        description=(
            "Offline fallback synthetic dataset with 99:1 imbalance. "
            "Use the real Covertype dataset for final reported results."
        ),
        is_binary=True,
    )


def load_default_bundles() -> list[DatasetBundle]:
    """Return the default datasets used by Person 3 experiments."""
    return [
        load_breast_cancer_bundle(),
        load_digits_binary_bundle(),
        load_imbalanced_bundle(),
    ]


def train_test_scaled_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a stratified split and standardize using training data only."""
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y)

    if X_array.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")

    if y_array.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")

    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_array,
        y_array,
        test_size=test_size,
        stratify=y_array,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def random_oversample_minority(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance classes by randomly oversampling minority classes.

    Sampling is performed with replacement until every class has the same
    number of samples as the majority class.
    """
    X_array = np.asarray(X)
    y_array = np.asarray(y)

    if X_array.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")

    if y_array.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")

    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    if y_array.size == 0:
        raise ValueError("Cannot oversample an empty dataset.")

    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y_array, return_counts=True)
    max_count = int(counts.max())

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for class_label in classes:
        indices = np.flatnonzero(y_array == class_label)

        if indices.size < max_count:
            sampled_indices = rng.choice(
                indices,
                size=max_count,
                replace=True,
            )
        else:
            sampled_indices = indices

        X_parts.append(X_array[sampled_indices])
        y_parts.append(y_array[sampled_indices])

    X_resampled = np.vstack(X_parts)
    y_resampled = np.concatenate(y_parts)

    permutation = rng.permutation(y_resampled.shape[0])

    return X_resampled[permutation], y_resampled[permutation]


def prepare_bundle_split(
    bundle: DatasetBundle,
    apply_imbalance_treatment: bool = True,
    severe_threshold: float = 0.01,
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
]:
    """Split, scale, and optionally treat severe class imbalance.

    Standardization is fitted on training data only. Random oversampling is
    also applied only to the training split, leaving the test data unchanged.

    Returns
    -------
    tuple
        ``X_train, X_test, y_train, y_test, treatment_name``.
    """
    if not 0.0 <= severe_threshold <= 1.0:
        raise ValueError("severe_threshold must be between 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_scaled_split(
        bundle.X,
        bundle.y,
        test_size=test_size,
        random_state=random_state,
    )

    treatment = "none"

    if (
        apply_imbalance_treatment
        and bundle.minority_fraction <= severe_threshold
    ):
        X_train, y_train = random_oversample_minority(
            X_train,
            y_train,
            random_state=random_state,
        )
        treatment = "random_oversampling_train_only"

    return X_train, X_test, y_train, y_test, treatment


def evaluate_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute accuracy, macro F1, and ROC-AUC when available."""
    y_pred = model.predict(X_test)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(
            f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }

    if not hasattr(model, "predict_proba"):
        metrics["auc_roc"] = float("nan")
        return metrics

    probabilities = np.asarray(model.predict_proba(X_test))

    try:
        if probabilities.ndim != 2:
            raise ValueError(
                "predict_proba must return a two-dimensional array."
            )

        if probabilities.shape[1] == 2:
            metrics["auc_roc"] = float(
                roc_auc_score(y_test, probabilities[:, 1])
            )
        else:
            metrics["auc_roc"] = float(
                roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        metrics["auc_roc"] = float("nan")

    return metrics


def add_label_noise(
    y: np.ndarray,
    noise_fraction: float,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Randomly flip a specified fraction of binary target labels."""
    if not 0.0 <= noise_fraction <= 1.0:
        raise ValueError("noise_fraction must be between 0 and 1.")

    y_array = np.asarray(y)

    if y_array.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")

    classes = np.unique(y_array)

    if classes.size != 2:
        raise ValueError(
            "add_label_noise currently supports binary targets only."
        )

    noisy = y_array.copy()
    n_flip = int(round(noise_fraction * noisy.shape[0]))

    if n_flip == 0:
        return noisy

    rng = np.random.default_rng(random_state)
    flip_indices = rng.choice(
        noisy.shape[0],
        size=n_flip,
        replace=False,
    )

    first_class, second_class = classes
    noisy[flip_indices] = np.where(
        noisy[flip_indices] == first_class,
        second_class,
        first_class,
    )

    return noisy


def save_results_table(
    rows: list[dict[str, Any]],
    filename: str,
) -> Path:
    """Save experiment rows as CSV and return the generated path."""
    if not filename.lower().endswith(".csv"):
        raise ValueError("filename must use the .csv extension.")

    ensure_output_dirs()

    output_path = RESULTS_DIR / filename
    pd.DataFrame(rows).to_csv(output_path, index=False)

    return output_path
