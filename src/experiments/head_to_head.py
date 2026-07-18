"""Experiment 4 -- head-to-head comparison under fixed resources.

The comparison is run on the downloaded project datasets, not on a hidden
in-memory dataset. AdaBoost uses our from-scratch implementation. sklearn
tree/forest models are exported only as explicit reference baselines; they are
not presented as replacements for the team's required Decision Tree or Random
Forest implementations.

Run from the repository root:

    python experiments/head_to_head.py
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.boosting.adaboost import AdaBoostClassifier, AdaBoostSAMMERClassifier
from src.metrics.evaluation import (
    METRIC_NAMES,
    classification_metrics as project_classification_metrics,
    export_csv,
    export_json,
)
from src.utils.preprocessing import DatasetBundle, load_project_datasets

try:
    from src.trees.decision_tree import DecisionTree as OwnDecisionTree
except ImportError:
    OwnDecisionTree = None  # type: ignore[assignment]

try:
    from src.bagging.random_forest import RandomForestClassifier as OwnRandomForest
except ImportError:
    OwnRandomForest = None  # type: ignore[assignment]


CALIBRATION_METRIC_NAMES = (
    "log_loss",
    "brier_score",
    "expected_calibration_error",
)
ModelFactory = Callable[[int], Any]
FoldOutput = Tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class ModelSpec:
    """Model metadata used to avoid mixing project and reference baselines."""

    name: str
    role: str
    factory: ModelFactory
    note: str


@dataclass(frozen=True)
class CrossValidationConfig:
    """Configuration for Experiment 4."""

    datasets: tuple[str, ...] = ("wdbc", "adult", "covertype")
    n_splits: int = 5
    n_estimators: int = 100
    learning_rate: float = 1.0
    sammer_smoothing: float = 1e-3
    random_state: int = 42
    n_jobs: int = 1
    confidence_level: float = 0.95
    alpha: float = 0.05
    adult_max_samples: Optional[int] = 5000
    covertype_max_samples: Optional[int] = 5000
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("results"))
    figure_dir: Path = field(default_factory=lambda: Path("figures"))


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
    """Derive deterministic independent seeds for folds and models."""
    sequence = np.random.SeedSequence([int(base_seed), int(offset)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _is_classifier(model: Any) -> bool:
    """Check the minimal fit/predict interface used by this experiment."""
    return callable(getattr(model, "fit", None)) and callable(getattr(model, "predict", None))


def _predict_scores(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
    """Return probability-like scores when the model exposes them."""
    if callable(getattr(model, "predict_proba", None)):
        return np.asarray(model.predict_proba(X), dtype=float)
    if callable(getattr(model, "decision_function", None)):
        return np.asarray(model.decision_function(X), dtype=float)
    return None


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    model_classes: Optional[np.ndarray],
) -> Dict[str, float]:
    """Compute all metrics required for the head-to-head experiment."""
    labels = np.asarray(model_classes) if model_classes is not None else None
    return project_classification_metrics(
        y_true,
        y_pred,
        y_score,
        labels=labels,
        score_labels=labels,
    )


def _student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Return a t critical value, falling back to normal approximation."""
    try:
        from scipy import stats

        return float(stats.t.ppf(probability, degrees_of_freedom))
    except Exception:
        return 1.96


def _student_t_two_sided_pvalue(t_statistic: float, degrees_of_freedom: int) -> float:
    """Return a two-sided t-test p-value, with a normal fallback."""
    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(t_statistic), degrees_of_freedom))
    except Exception:
        return float(math.erfc(abs(t_statistic) / math.sqrt(2.0)))


def aggregate_scores(
    per_fold: Dict[str, np.ndarray],
    confidence_level: float,
) -> Dict[str, Dict[str, float]]:
    """Summarize per-fold metric arrays as mean/std/confidence interval."""
    bounds = {
        "accuracy": (0.0, 1.0),
        "macro_precision": (0.0, 1.0),
        "macro_recall": (0.0, 1.0),
        "macro_f1": (0.0, 1.0),
        "roc_auc": (0.0, 1.0),
        "log_loss": (0.0, None),
        "brier_score": (0.0, 2.0),
        "expected_calibration_error": (0.0, 1.0),
    }
    summaries: Dict[str, Dict[str, float]] = {}
    for metric, values in per_fold.items():
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summaries[metric] = {
                "mean": float("nan"),
                "std": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
            }
            continue

        mean = float(np.mean(finite))
        std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        if finite.size > 1:
            tail = 0.5 + confidence_level / 2.0
            critical = _student_t_quantile(tail, finite.size - 1)
            half_width = critical * std / math.sqrt(finite.size)
        else:
            half_width = 0.0
        ci_low = mean - half_width
        ci_high = mean + half_width
        lower, upper = bounds.get(metric, (None, None))
        if lower is not None:
            ci_low = max(lower, ci_low)
        if upper is not None:
            ci_high = min(upper, ci_high)
        summaries[metric] = {
            "mean": mean,
            "std": std,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    return summaries


def paired_t_tests_vs_reference(
    per_model: Dict[str, np.ndarray],
    reference: str,
    alpha: float,
) -> List[Dict[str, Any]]:
    """Run paired t-tests and Holm-correct p-values against one reference."""
    if reference not in per_model:
        raise KeyError(f"Reference model '{reference}' is not present in results.")

    reference_scores = np.asarray(per_model[reference], dtype=float)
    tests: List[Dict[str, Any]] = []
    for model_name, model_scores in per_model.items():
        if model_name == reference:
            continue
        model_scores = np.asarray(model_scores, dtype=float)
        mask = np.isfinite(reference_scores) & np.isfinite(model_scores)
        differences = reference_scores[mask] - model_scores[mask]
        if differences.size < 2:
            t_statistic = float("nan")
            p_value = float("nan")
        else:
            mean_difference = float(np.mean(differences))
            std_difference = float(np.std(differences, ddof=1))
            if std_difference <= 1e-15:
                t_statistic = 0.0 if abs(mean_difference) <= 1e-15 else math.inf
                p_value = 1.0 if t_statistic == 0.0 else 0.0
            else:
                t_statistic = mean_difference / (
                    std_difference / math.sqrt(differences.size)
                )
                p_value = _student_t_two_sided_pvalue(t_statistic, differences.size - 1)

        tests.append(
            {
                "model": model_name,
                "n_pairs": int(differences.size),
                "mean_difference_reference_minus_model": float(np.mean(differences))
                if differences.size
                else float("nan"),
                "t_statistic": float(t_statistic),
                "p_value": float(p_value),
                "p_value_holm": float("nan"),
                "significant": False,
            }
        )

    ordered = sorted(
        range(len(tests)),
        key=lambda index: (
            not math.isfinite(tests[index]["p_value"]),
            tests[index]["p_value"],
        ),
    )
    running_adjusted = 0.0
    m = len(tests)
    for rank, index in enumerate(ordered):
        p_value = tests[index]["p_value"]
        if not math.isfinite(p_value):
            adjusted = float("nan")
        else:
            adjusted = min(1.0, (m - rank) * p_value)
            running_adjusted = max(running_adjusted, adjusted)
            adjusted = running_adjusted
        tests[index]["p_value_holm"] = adjusted
        tests[index]["significant"] = bool(math.isfinite(adjusted) and adjusted <= alpha)

    return tests


def evaluate_fold(
    factory: ModelFactory,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, float], FoldOutput]:
    """Train and score one model on one cross-validation fold."""
    scaler = StandardScaler().fit(X[train_idx])
    X_train = scaler.transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    model = factory(seed)
    if not _is_classifier(model):
        raise TypeError(
            f"Factory produced {type(model).__name__}, which lacks fit/predict."
        )

    model.fit(X_train, y[train_idx])
    y_pred = np.asarray(model.predict(X_test))
    y_score = _predict_scores(model, X_test)
    model_classes = getattr(model, "classes_", None)
    scores = compute_metrics(y[test_idx], y_pred, y_score, model_classes)
    return scores, (test_idx, y[test_idx], y_pred)


def cross_validate_model(
    spec: ModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    config: CrossValidationConfig,
) -> Dict[str, Any]:
    """Run stratified k-fold CV for one model."""
    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    folds = list(splitter.split(X, y))
    jobs = [
        (train_idx, test_idx, derive_seed(config.random_state, fold_index))
        for fold_index, (train_idx, test_idx) in enumerate(folds)
    ]

    if config.n_jobs == 1:
        outcomes = [
            evaluate_fold(spec.factory, X, y, train_idx, test_idx, seed)
            for train_idx, test_idx, seed in jobs
        ]
    else:
        outcomes = Parallel(n_jobs=config.n_jobs)(
            delayed(evaluate_fold)(spec.factory, X, y, train_idx, test_idx, seed)
            for train_idx, test_idx, seed in jobs
        )

    for fold_index, (scores, _) in enumerate(outcomes, start=1):
        logger.info(
            "  fold %d | acc=%.4f  f1=%.4f  auc=%.4f  logloss=%.4f  ece=%.4f",
            fold_index,
            scores["accuracy"],
            scores["macro_f1"],
            scores["roc_auc"],
            scores["log_loss"],
            scores["expected_calibration_error"],
        )

    results: Dict[str, Any] = {
        metric: np.array([scores[metric] for scores, _ in outcomes], dtype=float)
        for metric in METRIC_NAMES
    }
    results["raw"] = [raw for _, raw in outcomes]
    results["role"] = spec.role
    results["note"] = spec.note
    return results


def missing_required_custom_models() -> List[str]:
    """Return required project implementations that are not importable yet."""
    missing: List[str] = []
    if OwnDecisionTree is None:
        missing.append("src.trees.decision_tree.DecisionTree")
    if OwnRandomForest is None:
        missing.append("src.bagging.random_forest.RandomForestClassifier")
    return missing


def build_model_specs(config: CrossValidationConfig) -> List[ModelSpec]:
    """Construct project models and clearly labeled reference-only baselines."""
    specs: List[ModelSpec] = [
        ModelSpec(
            name="AdaBoost (ours)",
            role="project_implementation",
            factory=lambda seed: AdaBoostClassifier(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                random_state=seed,
            ),
            note="From-scratch SAMME AdaBoost implemented in src/boosting/adaboost.py.",
        ),
        ModelSpec(
            name="AdaBoost SAMME.R (ours, bonus)",
            role="project_bonus",
            factory=lambda seed: AdaBoostSAMMERClassifier(
                n_estimators=config.n_estimators,
                smoothing=config.sammer_smoothing,
                random_state=seed,
            ),
            note=(
                "Bonus real-valued SAMME.R AdaBoost with smoothed probability "
                "stumps; evaluated with log-loss, Brier score, and ECE."
            ),
        ),
    ]

    if OwnDecisionTree is not None:
        specs.append(
            ModelSpec(
                name="Single tree (ours)",
                role="project_implementation",
                factory=lambda seed: OwnDecisionTree(random_state=seed),
                note="Team DecisionTree implementation.",
            )
        )
    if OwnRandomForest is not None:
        specs.append(
            ModelSpec(
                name="Random Forest (ours)",
                role="project_implementation",
                factory=lambda seed: OwnRandomForest(
                    n_estimators=config.n_estimators,
                    random_state=seed,
                ),
                note="Team RandomForest implementation.",
            )
        )

    specs.extend(
        [
            ModelSpec(
                name="Single tree (sklearn reference only)",
                role="reference_only",
                factory=lambda seed: SklearnDecisionTree(random_state=seed),
                note=(
                    "Reference baseline only; not a substitute for the required "
                    "from-scratch DecisionTree."
                ),
            ),
            ModelSpec(
                name="Random Forest, depth-1 (sklearn reference only)",
                role="reference_only",
                factory=lambda seed: SklearnRandomForest(
                    n_estimators=config.n_estimators,
                    max_depth=1,
                    random_state=seed,
                    n_jobs=1,
                ),
                note=(
                    "Reference baseline only; uses depth-1 sklearn trees to match "
                    "AdaBoost weak-learner capacity."
                ),
            ),
            ModelSpec(
                name="Random Forest, full (sklearn reference only)",
                role="reference_only",
                factory=lambda seed: SklearnRandomForest(
                    n_estimators=config.n_estimators,
                    random_state=seed,
                    n_jobs=1,
                ),
                note=(
                    "Reference baseline only; not a substitute for the required "
                    "from-scratch RandomForest."
                ),
            ),
        ]
    )
    return specs


def summarize(
    results: Dict[str, Dict[str, Any]],
    config: CrossValidationConfig,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Log and return mean/std/CI summaries for every model and metric."""
    logger.info(
        "=== Summary (mean +/- std [%.0f%% CI] over %d folds) ===",
        100 * config.confidence_level,
        config.n_splits,
    )
    summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, scores in results.items():
        per_fold = {metric: scores[metric] for metric in METRIC_NAMES}
        summaries[name] = aggregate_scores(per_fold, config.confidence_level)

    for metric in METRIC_NAMES:
        logger.info("%s:", metric)
        for name, summary in summaries.items():
            entry = summary[metric]
            logger.info(
                "  %-40s %.4f +/- %.4f  [%.4f, %.4f]",
                name,
                entry["mean"],
                entry["std"],
                entry["ci_low"],
                entry["ci_high"],
            )
    return summaries


def calibration_bonus_summary(
    summaries: Dict[str, Dict[str, Dict[str, float]]],
    reference: str = "AdaBoost (ours)",
    challenger: str = "AdaBoost SAMME.R (ours, bonus)",
) -> Dict[str, Any]:
    """Summarize whether SAMME.R improves probability calibration."""
    if reference not in summaries or challenger not in summaries:
        return {
            "available": False,
            "reference": reference,
            "challenger": challenger,
            "reason": "Required AdaBoost models are missing from the summary.",
        }

    metrics: Dict[str, Dict[str, float | bool]] = {}
    for metric in CALIBRATION_METRIC_NAMES:
        reference_mean = summaries[reference][metric]["mean"]
        challenger_mean = summaries[challenger][metric]["mean"]
        improvement = reference_mean - challenger_mean
        metrics[metric] = {
            "reference_mean": reference_mean,
            "sammer_mean": challenger_mean,
            "improvement_reference_minus_sammer": improvement,
            "sammer_improves": bool(math.isfinite(improvement) and improvement > 0.0),
            "lower_is_better": True,
        }

    return {
        "available": True,
        "reference": reference,
        "challenger": challenger,
        "interpretation": (
            "Positive improvement_reference_minus_sammer means SAMME.R has the "
            "lower calibration error on that metric."
        ),
        "metrics": metrics,
    }


def run_significance_tests(
    results: Dict[str, Dict[str, Any]],
    reference: str,
    config: CrossValidationConfig,
    metric: str = "macro_f1",
) -> List[Dict[str, Any]]:
    """Compare every baseline against AdaBoost with paired t-tests."""
    per_model = {name: scores[metric] for name, scores in results.items()}
    tests = paired_t_tests_vs_reference(per_model, reference, config.alpha)

    logger.info(
        "=== Paired t-tests on fold %s vs '%s' (Holm-corrected, alpha=%.2f) ===",
        metric,
        reference,
        config.alpha,
    )
    for test in tests:
        verdict = "significant" if test["significant"] else "not significant"
        logger.info(
            "  vs %-34s diff=%.4f  t=%.3f  p=%.4f  p_holm=%.4f (%s)",
            test["model"],
            test["mean_difference_reference_minus_model"],
            test["t_statistic"],
            test["p_value"],
            test["p_value_holm"],
            verdict,
        )
    return tests


def dataset_payload(dataset: DatasetBundle) -> Dict[str, Any]:
    """Return report-friendly dataset metadata."""
    return {
        "name": dataset.name,
        "task": dataset.task,
        "source": dataset.source,
        "notes": dataset.notes,
        "n_samples": int(dataset.X.shape[0]),
        "n_features": int(dataset.X.shape[1]),
    }


def run_dataset(
    dataset: DatasetBundle,
    config: CrossValidationConfig,
) -> Dict[str, Any]:
    """Run Experiment 4 for one project dataset."""
    logger.info(
        "Dataset: %s (%d samples, %d features)",
        dataset.name,
        dataset.X.shape[0],
        dataset.X.shape[1],
    )
    logger.info(
        "Setup: %d-fold stratified CV | %d estimators | learning_rate=%.2f | seed=%d",
        config.n_splits,
        config.n_estimators,
        config.learning_rate,
        config.random_state,
    )

    results: Dict[str, Dict[str, Any]] = {}
    missing_models = missing_required_custom_models()
    if missing_models:
        logger.warning(
            "Required custom model(s) missing; sklearn rows are reference-only: %s",
            ", ".join(missing_models),
        )

    for spec in build_model_specs(config):
        logger.info("--- %s [%s] ---", spec.name, spec.role)
        results[spec.name] = cross_validate_model(spec, dataset.X, dataset.y, config)

    summaries = summarize(results, config)
    tests = run_significance_tests(results, "AdaBoost (ours)", config)
    bonus_summary = calibration_bonus_summary(summaries)
    return {
        "dataset": dataset_payload(dataset),
        "experiment_status": {
            "status": "partial" if missing_models else "complete",
            "missing_required_custom_models": missing_models,
            "reference_only_models_are_not_project_implementations": True,
        },
        "model_metadata": {
            spec.name: {"role": spec.role, "note": spec.note}
            for spec in build_model_specs(config)
        },
        "results": results,
        "summary": summaries,
        "bonus_calibration_summary": bonus_summary,
        "significance_tests": tests,
    }


def _short_model_label(model_name: str) -> str:
    """Return compact model labels for comparison figures."""
    replacements = {
        "AdaBoost (ours)": "AdaBoost",
        "AdaBoost SAMME.R (ours, bonus)": "SAMME.R",
        "Single tree (ours)": "Tree ours",
        "Random Forest (ours)": "RF ours",
        "Single tree (sklearn reference only)": "Tree sklearn ref",
        "Random Forest, depth-1 (sklearn reference only)": "RF depth-1 ref",
        "Random Forest, full (sklearn reference only)": "RF full ref",
    }
    return replacements.get(model_name, model_name)


def plot_head_to_head_macro_f1(
    all_results: Dict[str, Dict[str, Any]],
    path: Path,
) -> Optional[Path]:
    """Save a heatmap of cross-validation mean macro-F1 by model and dataset."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning(
            "Skipping head-to-head plot export because matplotlib is unavailable: %s",
            exc,
        )
        return None

    dataset_names = list(all_results.keys())
    model_names: list[str] = []
    for dataset_result in all_results.values():
        for model_name in dataset_result["summary"]:
            if model_name not in model_names:
                model_names.append(model_name)

    if not dataset_names or not model_names:
        logger.warning("Skipping head-to-head plot export because there are no results.")
        return None

    values = np.full((len(model_names), len(dataset_names)), np.nan, dtype=float)
    for dataset_index, dataset_name in enumerate(dataset_names):
        summary = all_results[dataset_name]["summary"]
        for model_index, model_name in enumerate(model_names):
            if model_name in summary:
                values[model_index, dataset_index] = float(
                    summary[model_name]["macro_f1"]["mean"]
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure_width = max(8.0, 2.0 + 1.35 * len(dataset_names))
    figure_height = max(5.2, 1.8 + 0.45 * len(model_names))
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    image = axis.imshow(
        np.ma.masked_invalid(values),
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        aspect="auto",
    )

    axis.set_title("Head-to-head comparison: mean CV macro-F1")
    axis.set_xticks(np.arange(len(dataset_names)))
    axis.set_xticklabels(dataset_names, rotation=15, ha="right")
    axis.set_yticks(np.arange(len(model_names)))
    axis.set_yticklabels([_short_model_label(name) for name in model_names])

    for model_index in range(len(model_names)):
        for dataset_index in range(len(dataset_names)):
            value = values[model_index, dataset_index]
            if not np.isfinite(value):
                continue
            text_color = "white" if value < 0.62 else "black"
            axis.text(
                dataset_index,
                model_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Mean macro-F1")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def export_results(
    all_results: Dict[str, Dict[str, Any]],
    config: CrossValidationConfig,
) -> None:
    """Persist the study to results/ as JSON, CSV, and figure artifacts."""
    figure_path = plot_head_to_head_macro_f1(
        all_results,
        config.figure_dir / "head_to_head_macro_f1.png",
    )
    export_json(
        {
            "config": vars(config),
            "datasets": {
                name: result["dataset"] for name, result in all_results.items()
            },
            "summary": {
                name: result["summary"] for name, result in all_results.items()
            },
            "experiment_status": {
                name: result["experiment_status"]
                for name, result in all_results.items()
            },
            "model_metadata": {
                name: result["model_metadata"]
                for name, result in all_results.items()
            },
            "bonus_calibration_summary": {
                name: result["bonus_calibration_summary"]
                for name, result in all_results.items()
            },
            "significance_tests": {
                name: result["significance_tests"]
                for name, result in all_results.items()
            },
            "figure_paths": {
                "macro_f1_heatmap": figure_path,
            },
        },
        config.output_dir / "head_to_head.json",
    )

    rows = [
        {
            "dataset": dataset_name,
            "model": model_name,
            "role": scores["role"],
            "fold": fold + 1,
            **{metric: float(scores[metric][fold]) for metric in METRIC_NAMES},
        }
        for dataset_name, dataset_result in all_results.items()
        for model_name, scores in dataset_result["results"].items()
        for fold in range(config.n_splits)
    ]
    export_csv(rows, config.output_dir / "head_to_head_folds.csv")
    logger.info("Results exported to %s/", config.output_dir)


def main(config: Optional[CrossValidationConfig] = None) -> Dict[str, Dict[str, Any]]:
    """Load downloaded project data, run Experiment 4, and export results."""
    config = config or CrossValidationConfig()
    datasets = load_project_datasets(
        names=config.datasets,
        data_dir=config.data_dir,
        adult_max_samples=config.adult_max_samples,
        covertype_max_samples=config.covertype_max_samples,
        random_state=config.random_state,
    )
    all_results = {
        dataset.name: run_dataset(dataset, config)
        for dataset in datasets
    }
    export_results(all_results, config)
    return all_results


if __name__ == "__main__":
    main()
