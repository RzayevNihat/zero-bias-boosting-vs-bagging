"""Shared utilities for Person 3 Random Forest experiments.

The helpers in this module keep dataset preparation, imbalance treatment,
metric calculation, and result serialization reproducible. Scikit-learn is
used only for datasets, preprocessing, metrics, and reference baselines; the
project Random Forest implementation remains ``src.bagging.random_forest``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.preprocessing import load_mnist_binary_subset

RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = ROOT_DIR / "data"


@dataclass(frozen=True)
class DatasetBundle:
    """Dataset container shared by the experiment scripts."""

    name: str
    X: np.ndarray
    y: np.ndarray
    description: str
    is_binary: bool

    @property
    def minority_fraction(self) -> float:
        """Return the smallest observed class fraction."""
        _, counts = np.unique(self.y, return_counts=True)
        return float(counts.min() / counts.sum())


class ClassifierProtocol(Protocol):
    """Minimal classifier interface used by metric helpers."""

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def ensure_output_dirs() -> None:
    """Create canonical root-level output directories."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_breast_cancer_bundle() -> DatasetBundle:
    """Load the Breast Cancer Wisconsin Diagnostic dataset."""
    data = load_breast_cancer()
    return DatasetBundle(
        name="breast_cancer",
        X=data.data.astype(float),
        y=data.target.astype(int),
        description="Breast Cancer Wisconsin Diagnostic from sklearn.datasets.",
        is_binary=True,
    )


def load_mnist_binary_bundle() -> DatasetBundle:
    """Load a high-dimensional binary MNIST subset (class 3 versus class 8)."""
    data = load_mnist_binary_subset(
        digits=(3, 8),
        path=DATA_DIR / "mnist.csv" if (DATA_DIR / "mnist.csv").exists() else None,
        max_samples=1000,
        random_state=RANDOM_STATE,
    )
    return DatasetBundle(
        name="mnist_3_vs_8",
        X=data.X.astype(float),
        y=data.y.astype(int),
        description="High-dimensional 784-feature MNIST subset: class 3 vs class 8.",
        is_binary=True,
    )


def load_digits_binary_bundle() -> DatasetBundle:
    """Backward-compatible name for the MNIST 3-vs-8 bundle."""
    return load_mnist_binary_bundle()


def load_imbalanced_bundle() -> DatasetBundle:
    """Load a real Covertype one-vs-rest task or an offline 99:1 fallback.

    For the final report, place ``covtype.data`` in ``data/``. The synthetic
    fallback is intended only for offline smoke tests and must be identified as
    synthetic in any generated result table.
    """
    covtype_path = DATA_DIR / "covtype.data"
    if covtype_path.exists():
        columns = [f"feature_{index}" for index in range(54)] + ["target"]
        frame = pd.read_csv(covtype_path, header=None, names=columns)
        # Cover type 4 is rare. Keep up to 10,000 rows for manageable runtime.
        frame = frame.sample(n=min(10_000, len(frame)), random_state=RANDOM_STATE)
        X = frame.iloc[:, :-1].to_numpy(dtype=float)
        y = (frame["target"].to_numpy() == 4).astype(int)
        return DatasetBundle(
            name="covertype_type4_imbalanced",
            X=X,
            y=y,
            description="Covertype one-vs-rest: cover type 4 as the positive class.",
            is_binary=True,
        )

    X, y = make_classification(
        n_samples=1_200,
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
            "Offline synthetic 99:1 fallback. Replace it with data/covtype.data "
            "before producing final report results."
        ),
        is_binary=True,
    )


def load_default_bundles() -> list[DatasetBundle]:
    """Return the three datasets used by the Person 3 experiment suite."""
    return [
        load_breast_cancer_bundle(),
        load_mnist_binary_bundle(),
        load_imbalanced_bundle(),
    ]


def train_test_scaled_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a stratified split and fit scaling on training data only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
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
    """Randomly oversample every minority class to the majority count."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be 2D, y must be 1D, and lengths must match.")

    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    target_count = int(counts.max())
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for class_label in classes:
        indices = np.flatnonzero(y == class_label)
        sampled_indices = (
            rng.choice(indices, size=target_count, replace=True)
            if len(indices) < target_count
            else indices
        )
        X_parts.append(X[sampled_indices])
        y_parts.append(y[sampled_indices])

    X_resampled = np.vstack(X_parts)
    y_resampled = np.concatenate(y_parts)
    order = rng.permutation(len(y_resampled))
    return X_resampled[order], y_resampled[order]


def prepare_bundle_split(
    bundle: DatasetBundle,
    apply_imbalance_treatment: bool = True,
    severe_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Split/scale a bundle and optionally oversample severe training imbalance."""
    X_train, X_test, y_train, y_test = train_test_scaled_split(bundle.X, bundle.y)
    treatment = "none"
    if apply_imbalance_treatment and bundle.minority_fraction <= severe_threshold:
        X_train, y_train = random_oversample_minority(X_train, y_train)
        treatment = "random_oversampling_train_only"
    return X_train, X_test, y_train, y_test, treatment


def evaluate_classifier(
    model: ClassifierProtocol,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute accuracy, macro F1, AUC-ROC, and minority-class recall."""
    y_pred = model.predict(X_test)
    classes, counts = np.unique(y_test, return_counts=True)
    minority_class = classes[int(np.argmin(counts))]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "minority_recall": float(
            recall_score(
                y_test,
                y_pred,
                labels=[minority_class],
                average="macro",
                zero_division=0,
            )
        ),
    }

    try:
        proba = model.predict_proba(X_test)
        if proba.shape[1] == 2:
            metrics["auc_roc"] = float(roc_auc_score(y_test, proba[:, 1]))
        else:
            metrics["auc_roc"] = float(
                roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
            )
    except (AttributeError, ValueError):
        metrics["auc_roc"] = float("nan")
    return metrics


def add_label_noise(
    y: np.ndarray,
    noise_fraction: float,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Flip an exact fraction of labels in a binary target."""
    y = np.asarray(y)
    if not 0.0 <= noise_fraction <= 1.0:
        raise ValueError("noise_fraction must be between 0 and 1.")
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("add_label_noise supports binary targets only.")

    noisy = y.copy()
    n_flip = int(round(noise_fraction * len(y)))
    if n_flip == 0:
        return noisy

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(y), size=n_flip, replace=False)
    noisy[indices] = np.where(noisy[indices] == classes[0], classes[1], classes[0])
    return noisy


def save_results_table(rows: list[dict], filename: str) -> Path:
    """Save experiment records to the canonical root-level results directory."""
    ensure_output_dirs()
    output_path = RESULTS_DIR / filename
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
