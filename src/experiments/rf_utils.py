"""Shared utilities for Person 3 Random Forest experiments.

Dataset loading in this module is intentionally file-based. The experiment
modules read the project datasets from the repository-level ``data/`` folder;
they do not call ``sklearn.datasets`` at runtime. Run ``download_data.sh`` once
to prepare the required local files.

Scikit-learn is used only for train/test splitting, standardization, and
metrics. The project Random Forest implementation remains
``src.bagging.random_forest``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.preprocessing import load_wdbc

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
        if self.y.size == 0:
            raise ValueError("Dataset target cannot be empty.")
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


def _required_data_path(path: str | Path, dataset_name: str) -> Path:
    """Return an existing dataset path or raise a useful setup error."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} dataset was not found at: {dataset_path}. "
            "Run 'bash download_data.sh' from the repository root first."
        )
    return dataset_path


def _validate_bundle_arrays(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize arrays loaded from a local dataset file."""
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y)

    if X_array.ndim != 2:
        raise ValueError(f"{dataset_name}: X must be a two-dimensional array.")
    if y_array.ndim != 1:
        raise ValueError(f"{dataset_name}: y must be a one-dimensional array.")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError(f"{dataset_name}: X and y sample counts must match.")
    if X_array.shape[0] == 0 or X_array.shape[1] == 0:
        raise ValueError(f"{dataset_name}: dataset cannot be empty.")
    if not np.all(np.isfinite(X_array)):
        raise ValueError(f"{dataset_name}: X contains NaN or infinite values.")

    return X_array, y_array


def load_breast_cancer_bundle(
    path: str | Path | None = None,
) -> DatasetBundle:
    """Load the local Wisconsin Diagnostic Breast Cancer raw file.

    Expected file: ``data/wdbc.data``. The existing project loader encodes
    malignant as 1 and benign as 0.
    """
    dataset_path = _required_data_path(
        DATA_DIR / "wdbc.data" if path is None else path,
        "Breast Cancer Wisconsin Diagnostic",
    )
    dataset = load_wdbc(dataset_path)
    X, y = _validate_bundle_arrays(dataset.X, dataset.y, "WDBC")

    return DatasetBundle(
        name="breast_cancer_wdbc",
        X=X,
        y=y.astype(int),
        description=(
            "Wisconsin Diagnostic Breast Cancer loaded from data/wdbc.data; "
            "malignant=1 and benign=0."
        ),
        is_binary=True,
    )


def load_mnist_binary_bundle(
    path: str | Path | None = None,
) -> DatasetBundle:
    """Load the local high-dimensional MNIST class 3 vs class 8 subset.

    Expected file: ``data/mnist_3_vs_8.npz`` with arrays named ``X`` and ``y``.
    The download script creates a deterministic balanced 5,000-sample subset,
    encodes digit 3 as 0, and digit 8 as 1.
    """
    dataset_path = _required_data_path(
        DATA_DIR / "mnist_3_vs_8.npz" if path is None else path,
        "MNIST 3-vs-8",
    )

    try:
        with np.load(dataset_path, allow_pickle=False) as archive:
            if "X" not in archive or "y" not in archive:
                raise ValueError("MNIST NPZ file must contain arrays named X and y.")
            X, y = _validate_bundle_arrays(
                archive["X"],
                archive["y"],
                "MNIST 3-vs-8",
            )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Could not read a valid MNIST subset from: {dataset_path}"
        ) from error

    y = y.astype(int)
    if not np.array_equal(np.unique(y), np.array([0, 1])):
        raise ValueError("MNIST 3-vs-8 target must contain exactly labels 0 and 1.")

    return DatasetBundle(
        name="mnist_3_vs_8",
        X=X,
        y=y,
        description=(
            "Local 5,000-sample MNIST binary subset with 784 features: "
            "digit 3 versus digit 8."
        ),
        is_binary=True,
    )


def load_imbalanced_bundle(
    path: str | Path | None = None,
    max_samples: int = 10_000,
) -> DatasetBundle:
    """Load a local Covertype type-4 one-vs-rest task.

    ``download_data.sh`` normally creates ``data/covtype.data`` and keeps the
    compressed ``data/covtype.data.gz`` file as well. This loader accepts either
    format. It uses a deterministic subset for manageable from-scratch runtime.
    No synthetic fallback is used: missing real data raises ``FileNotFoundError``.
    """
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2.")

    if path is None:
        extracted_path = DATA_DIR / "covtype.data"
        compressed_path = DATA_DIR / "covtype.data.gz"
        candidate = extracted_path if extracted_path.exists() else compressed_path
    else:
        candidate = Path(path)

    dataset_path = _required_data_path(candidate, "Covertype")
    columns = [f"feature_{index}" for index in range(54)] + ["target"]

    frame = pd.read_csv(
        dataset_path,
        header=None,
        names=columns,
        compression="infer",
    )
    if frame.empty:
        raise ValueError("Covertype dataset cannot be empty.")

    sample_size = min(max_samples, len(frame))
    if sample_size < len(frame):
        frame = frame.sample(n=sample_size, random_state=RANDOM_STATE)

    X, raw_target = _validate_bundle_arrays(
        frame.iloc[:, :-1].to_numpy(dtype=float),
        frame["target"].to_numpy(),
        "Covertype",
    )
    y = (raw_target.astype(int) == 4).astype(int)

    if np.unique(y).size != 2:
        raise ValueError(
            "The selected Covertype subset does not contain both type 4 and "
            "non-type-4 samples. Increase max_samples or check the source file."
        )

    return DatasetBundle(
        name="covertype_type4_imbalanced",
        X=X,
        y=y,
        description=(
            f"Local Covertype deterministic subset ({len(y)} rows), one-vs-rest: "
            "cover type 4 is the positive minority class."
        ),
        is_binary=True,
    )


def load_default_bundles() -> list[DatasetBundle]:
    """Return the three local datasets used by Person 3 experiments."""
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
