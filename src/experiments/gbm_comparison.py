"""Bonus experiment -- AdaBoost vs. from-scratch Gradient Boosting.

This bonus compares the project AdaBoost implementation with a binary
Gradient Boosting classifier trained by log-loss.  The default configuration
uses the four project datasets named in the brief: WDBC, Adult, Covertype, and
MNIST 2-class subset.  Because the GBM implementation is binary, multiclass
datasets are converted by keeping their two largest classes.

Run from the repository root:

    python experiments/gbm_comparison.py
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoosting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.boosting.adaboost import AdaBoostClassifier
from src.boosting.gradient_boosting import GradientBoostingClassifier
from src.metrics.evaluation import (
    METRIC_NAMES,
    classification_metrics,
    export_csv,
    export_json,
)
from src.utils.preprocessing import DatasetBundle, load_project_datasets


ModelFactory = Callable[[int], Any]


@dataclass(frozen=True)
class ModelSpec:
    """Model metadata used to separate project code from reference baselines."""

    name: str
    role: str
    factory: ModelFactory
    note: str


@dataclass(frozen=True)
class GBMComparisonConfig:
    """Configuration for the GBM bonus experiment."""

    datasets: tuple[str, ...] = ("wdbc", "adult", "covertype", "mnist")
    n_estimators: int = 100
    adaboost_learning_rate: float = 1.0
    gbm_learning_rate: float = 0.1
    gbm_max_depth: int = 3
    gbm_min_samples_split: int = 2
    test_size: float = 0.2
    random_state: int = 42
    adult_max_samples: Optional[int] = 5000
    covertype_max_samples: Optional[int] = 5000
    mnist_max_samples: Optional[int] = 1000
    mnist_digits: tuple[str, str] = ("3", "8")
    include_sklearn_reference: bool = True
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("results"))


def setup_logger(name: str) -> logging.Logger:
    """Create a simple console logger for reproducible experiment logs."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = setup_logger(__name__)


def derive_seed(base_seed: int, offset: int) -> int:
    """Derive deterministic independent seeds for datasets and models."""
    sequence = np.random.SeedSequence([int(base_seed), int(offset)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def dataset_payload(dataset: DatasetBundle) -> Dict[str, Any]:
    """Return report-friendly dataset metadata."""
    return {
        "name": dataset.name,
        "task": dataset.task,
        "source": dataset.source,
        "notes": dataset.notes,
        "n_samples": int(dataset.X.shape[0]),
        "n_features": int(dataset.X.shape[1]),
        "class_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(dataset.y, return_counts=True))
        },
    }


def to_binary_task(dataset: DatasetBundle) -> DatasetBundle:
    """Return a binary dataset, filtering multiclass data when needed."""
    labels, counts = np.unique(dataset.y, return_counts=True)
    if labels.size < 2:
        raise ValueError(f"{dataset.name} has fewer than two classes.")

    if labels.size == 2:
        mapping = {label: index for index, label in enumerate(labels.tolist())}
        y = np.array([mapping[label] for label in dataset.y], dtype=int)
        note = dataset.notes
        if not np.array_equal(labels, np.array([0, 1])):
            note += f" Labels remapped to binary order: {mapping}."
        return DatasetBundle(
            name=dataset.name,
            X=dataset.X,
            y=y,
            source=dataset.source,
            task="binary" if "binary" not in dataset.task else dataset.task,
            notes=note,
        )

    selected = np.sort(labels[np.argsort(counts)[::-1][:2]])
    mask = np.isin(dataset.y, selected)
    mapping = {selected[0]: 0, selected[1]: 1}
    y = np.array([mapping[label] for label in dataset.y[mask]], dtype=int)
    note = (
        f"{dataset.notes} For the binary GBM bonus, kept the two largest "
        f"classes only: {selected[0]} -> 0, {selected[1]} -> 1."
    )
    return DatasetBundle(
        name=f"{dataset.name} ({selected[0]} vs {selected[1]})",
        X=dataset.X[mask],
        y=y,
        source=dataset.source,
        task="binary_from_multiclass",
        notes=note,
    )


def load_comparison_datasets(
    config: GBMComparisonConfig,
) -> tuple[list[DatasetBundle], list[Dict[str, Any]]]:
    """Load requested datasets one by one so a fetch failure is recorded."""
    datasets: list[DatasetBundle] = []
    failures: list[Dict[str, Any]] = []

    for index, name in enumerate(config.datasets):
        try:
            loaded = load_project_datasets(
                names=(name,),
                data_dir=config.data_dir,
                adult_max_samples=config.adult_max_samples,
                covertype_max_samples=config.covertype_max_samples,
                mnist_max_samples=config.mnist_max_samples,
                mnist_digits=config.mnist_digits,
                random_state=derive_seed(config.random_state, index),
            )[0]
            datasets.append(to_binary_task(loaded))
        except Exception as exc:
            logger.warning("Skipping dataset '%s': %s", name, exc)
            failures.append(
                {
                    "dataset": name,
                    "status": "failed_to_load",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return datasets, failures


def build_model_specs(config: GBMComparisonConfig) -> List[ModelSpec]:
    """Construct the bonus models and an optional sklearn sanity check."""
    specs = [
        ModelSpec(
            name="AdaBoost (ours)",
            role="project_implementation",
            factory=lambda seed: AdaBoostClassifier(
                n_estimators=config.n_estimators,
                learning_rate=config.adaboost_learning_rate,
                random_state=seed,
            ),
            note="From-scratch discrete SAMME AdaBoost with weighted stumps.",
        ),
        ModelSpec(
            name="Gradient Boosting (ours, bonus)",
            role="project_bonus",
            factory=lambda seed: GradientBoostingClassifier(
                n_estimators=config.n_estimators,
                learning_rate=config.gbm_learning_rate,
                max_depth=config.gbm_max_depth,
                min_samples_split=config.gbm_min_samples_split,
                random_state=seed,
            ),
            note="From-scratch binary GBM trained by logistic log-loss.",
        ),
    ]

    if config.include_sklearn_reference:
        specs.append(
            ModelSpec(
                name="Gradient Boosting (sklearn reference only)",
                role="reference_only",
                factory=lambda seed: SklearnGradientBoosting(
                    n_estimators=config.n_estimators,
                    learning_rate=config.gbm_learning_rate,
                    max_depth=config.gbm_max_depth,
                    random_state=seed,
                ),
                note="Sanity-check baseline only; not used as the project implementation.",
            )
        )
    return specs


def _predict_scores(model: Any, X: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return probability-like scores and their class-column order."""
    if callable(getattr(model, "predict_proba", None)):
        return np.asarray(model.predict_proba(X), dtype=float), getattr(model, "classes_", None)
    if callable(getattr(model, "decision_function", None)):
        return np.asarray(model.decision_function(X), dtype=float), getattr(model, "classes_", None)
    return None, getattr(model, "classes_", None)


def _score_split(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute the project metric set for one fitted model and one split."""
    prediction = np.asarray(model.predict(X))
    scores, score_labels = _predict_scores(model, X)
    labels = getattr(model, "classes_", None)
    return classification_metrics(
        y,
        prediction,
        scores,
        labels=labels,
        score_labels=score_labels,
    )


def evaluate_model(
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    """Fit one model and return train/test metrics plus diagnostics."""
    logger.info("  training %s", spec.name)
    started = time.perf_counter()
    model = spec.factory(seed)

    try:
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - started
        train_metrics = _score_split(model, X_train, y_train)
        test_metrics = _score_split(model, X_test, y_test)
    except Exception as exc:
        logger.exception("  %s failed", spec.name)
        return {
            "status": "failed",
            "role": spec.role,
            "note": spec.note,
            "error": f"{type(exc).__name__}: {exc}",
            "training_time_seconds": float(time.perf_counter() - started),
        }

    logger.info(
        "  %-42s acc=%.4f f1=%.4f logloss=%.4f",
        spec.name,
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
        test_metrics["log_loss"],
    )

    result: Dict[str, Any] = {
        "status": "ok",
        "role": spec.role,
        "note": spec.note,
        "n_estimators": int(len(getattr(model, "estimators_", [])))
        if hasattr(model, "estimators_")
        else None,
        "training_time_seconds": float(elapsed),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    if hasattr(model, "train_log_loss_"):
        curve = np.asarray(model.train_log_loss_, dtype=float)
        result["train_log_loss_curve"] = {
            "first": float(curve[0]) if curve.size else float("nan"),
            "last": float(curve[-1]) if curve.size else float("nan"),
            "last5": curve[-5:].tolist(),
        }
    return result


def compare_against_adaboost(models: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize whether the GBM bonus improves over AdaBoost on test metrics."""
    reference = "AdaBoost (ours)"
    challenger = "Gradient Boosting (ours, bonus)"
    if reference not in models or challenger not in models:
        return {"available": False, "reason": "Required models are missing."}
    if models[reference].get("status") != "ok" or models[challenger].get("status") != "ok":
        return {"available": False, "reason": "At least one required model failed."}

    lower_is_better = {"log_loss", "brier_score", "expected_calibration_error"}
    metrics: Dict[str, Any] = {}
    for metric in METRIC_NAMES:
        ada_value = models[reference]["test_metrics"][metric]
        gbm_value = models[challenger]["test_metrics"][metric]
        if metric in lower_is_better:
            improvement = ada_value - gbm_value
            improves = bool(np.isfinite(improvement) and improvement > 0.0)
        else:
            improvement = gbm_value - ada_value
            improves = bool(np.isfinite(improvement) and improvement > 0.0)
        metrics[metric] = {
            "adaboost": ada_value,
            "gradient_boosting": gbm_value,
            "improvement": improvement,
            "gbm_improves": improves,
            "lower_is_better": metric in lower_is_better,
        }

    return {
        "available": True,
        "reference": reference,
        "challenger": challenger,
        "metrics": metrics,
    }


def run_dataset(dataset: DatasetBundle, config: GBMComparisonConfig) -> Dict[str, Any]:
    """Run the GBM bonus comparison for one binary project dataset."""
    labels, counts = np.unique(dataset.y, return_counts=True)
    if labels.size != 2:
        raise ValueError(f"{dataset.name} is not binary after preprocessing.")
    if int(counts.min()) < 2:
        raise ValueError(f"{dataset.name} has too few samples in one class.")

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=config.test_size,
        stratify=dataset.y,
        random_state=config.random_state,
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(
        "Dataset: %s | samples=%d, features=%d | train=%d, test=%d",
        dataset.name,
        dataset.X.shape[0],
        dataset.X.shape[1],
        X_train.shape[0],
        X_test.shape[0],
    )

    models: Dict[str, Dict[str, Any]] = {}
    for offset, spec in enumerate(build_model_specs(config)):
        seed = derive_seed(config.random_state, offset)
        models[spec.name] = evaluate_model(
            spec,
            X_train,
            y_train,
            X_test,
            y_test,
            seed,
        )

    return {
        "dataset": {
            **dataset_payload(dataset),
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
        },
        "models": models,
        "comparison_against_adaboost": compare_against_adaboost(models),
    }


def flatten_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert nested result payloads to a report-friendly CSV table."""
    rows: List[Dict[str, Any]] = []
    for dataset_name, dataset_result in results["results"].items():
        dataset_info = dataset_result["dataset"]
        for model_name, model_result in dataset_result["models"].items():
            base = {
                "dataset": dataset_name,
                "task": dataset_info["task"],
                "model": model_name,
                "role": model_result["role"],
                "status": model_result["status"],
                "n_samples": dataset_info["n_samples"],
                "n_features": dataset_info["n_features"],
                "train_size": dataset_info["train_size"],
                "test_size": dataset_info["test_size"],
                "training_time_seconds": model_result.get("training_time_seconds"),
            }
            if model_result["status"] != "ok":
                rows.append({**base, "split": "test", "error": model_result.get("error")})
                continue
            for split in ("train", "test"):
                metrics = model_result[f"{split}_metrics"]
                rows.append(
                    {
                        **base,
                        "split": split,
                        **{metric: metrics[metric] for metric in METRIC_NAMES},
                    }
                )
    return rows


def run_experiment(config: GBMComparisonConfig) -> Dict[str, Any]:
    """Run the GBM bonus comparison across the configured project datasets."""
    datasets, load_failures = load_comparison_datasets(config)
    all_results = {dataset.name: run_dataset(dataset, config) for dataset in datasets}
    return {
        "config": vars(config),
        "load_failures": load_failures,
        "datasets": {
            name: result["dataset"] for name, result in all_results.items()
        },
        "results": all_results,
        "comparison_against_adaboost": {
            name: result["comparison_against_adaboost"]
            for name, result in all_results.items()
        },
    }


def export_results(results: Dict[str, Any], config: GBMComparisonConfig) -> None:
    """Persist JSON and CSV artifacts for report writing."""
    rows = flatten_rows(results)
    export_json(
        {
            "config": results["config"],
            "load_failures": results["load_failures"],
            "datasets": results["datasets"],
            "comparison_against_adaboost": results["comparison_against_adaboost"],
            "results": results["results"],
        },
        config.output_dir / "gbm_comparison.json",
    )
    export_csv(rows, config.output_dir / "gbm_comparison.csv")
    logger.info("Results exported to %s/", config.output_dir)


def main(config: Optional[GBMComparisonConfig] = None) -> Dict[str, Any]:
    """Run the bonus experiment, export artifacts, and return the payload."""
    config = config or GBMComparisonConfig()
    results = run_experiment(config)
    export_results(results, config)
    return results


if __name__ == "__main__":
    main()
