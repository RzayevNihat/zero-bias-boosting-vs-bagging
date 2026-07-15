"""Evaluation helpers for the ensemble-methods final project.

The project experiments repeatedly need the same small set of classification
metrics: accuracy, macro precision/recall/F1, ROC-AUC, calibration scores,
fold summaries, paired tests, and bias-variance tables.  This module keeps
those calculations in one place and implements the metric logic with NumPy
rather than delegating to ``sklearn.metrics``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MetricDict = Dict[str, float]
ArrayLike = Sequence[Any] | np.ndarray

METRIC_NAMES: Tuple[str, ...] = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "roc_auc",
    "log_loss",
    "brier_score",
    "expected_calibration_error",
)
EPS = 1e-12


def _as_1d_array(values: ArrayLike, name: str) -> np.ndarray:
    """Convert input to a non-empty 1-D array."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return array


def _validate_same_length(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Raise a clear error when target and prediction lengths differ."""
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred have inconsistent lengths: "
            f"{y_true.shape[0]} vs {y_pred.shape[0]}."
        )


def _check_zero_division(zero_division: int | float) -> float:
    """Normalize the zero-division policy used by precision and recall."""
    if zero_division not in (0, 1):
        raise ValueError("zero_division must be either 0 or 1.")
    return float(zero_division)


def _resolve_labels(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    labels: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    """Return the class labels used for metric rows and score columns."""
    if labels is not None:
        resolved = np.asarray(labels)
        if resolved.ndim != 1 or resolved.shape[0] == 0:
            raise ValueError("labels must be a non-empty 1-D sequence.")
        if np.unique(resolved).shape[0] != resolved.shape[0]:
            raise ValueError("labels must not contain duplicates.")
        return resolved

    if y_pred is None:
        return np.unique(y_true)
    return np.unique(np.concatenate([y_true, y_pred]))


def _label_index(labels: np.ndarray) -> Dict[Any, int]:
    """Build a deterministic label-to-column map."""
    return {label: index for index, label in enumerate(labels.tolist())}


def _encode_labels(values: np.ndarray, labels: np.ndarray, name: str) -> np.ndarray:
    """Encode labels as integer positions in ``labels``."""
    mapping = _label_index(labels)
    encoded = np.empty(values.shape[0], dtype=np.int64)
    unknown: List[Any] = []

    for index, value in enumerate(values.tolist()):
        if value not in mapping:
            unknown.append(value)
        else:
            encoded[index] = mapping[value]

    if unknown:
        preview = ", ".join(repr(value) for value in sorted(set(unknown), key=repr)[:5])
        raise ValueError(f"{name} contains labels not present in labels: {preview}.")
    return encoded


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    zero_division: float,
) -> np.ndarray:
    """Elementwise divide while applying a sklearn-like zero policy."""
    output = np.full_like(numerator, zero_division, dtype=float)
    mask = denominator != 0
    output[mask] = numerator[mask] / denominator[mask]
    return output


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[Sequence[Any]] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Return a class-by-class confusion matrix.

    Rows represent true classes and columns represent predicted classes.  The
    optional ``normalize`` argument accepts ``"true"``, ``"pred"``, or ``"all"``.
    """
    y_true_array = _as_1d_array(y_true, "y_true")
    y_pred_array = _as_1d_array(y_pred, "y_pred")
    _validate_same_length(y_true_array, y_pred_array)

    resolved_labels = _resolve_labels(y_true_array, y_pred_array, labels)
    true_encoded = _encode_labels(y_true_array, resolved_labels, "y_true")
    pred_encoded = _encode_labels(y_pred_array, resolved_labels, "y_pred")

    matrix = np.zeros((resolved_labels.size, resolved_labels.size), dtype=np.int64)
    np.add.at(matrix, (true_encoded, pred_encoded), 1)

    if normalize is None:
        return matrix
    if normalize not in {"true", "pred", "all"}:
        raise ValueError("normalize must be one of None, 'true', 'pred', or 'all'.")

    matrix_float = matrix.astype(float)
    if normalize == "true":
        totals = matrix_float.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        totals = matrix_float.sum(axis=0, keepdims=True)
    else:
        totals = np.array([[matrix_float.sum()]])

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            matrix_float,
            totals,
            out=np.zeros_like(matrix_float),
            where=totals != 0,
        )


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the fraction of correctly classified samples."""
    y_true_array = _as_1d_array(y_true, "y_true")
    y_pred_array = _as_1d_array(y_pred, "y_pred")
    _validate_same_length(y_true_array, y_pred_array)
    return float(np.mean(y_true_array == y_pred_array))


def precision_recall_f1_support(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[Sequence[Any]] = None,
    average: Optional[str] = None,
    zero_division: int | float = 0,
) -> Tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float, np.ndarray | None]:
    """Compute precision, recall, F1, and support.

    ``average`` may be ``None``, ``"macro"``, ``"weighted"``, or ``"micro"``.
    With ``average=None`` the function returns one value per class plus the
    support array.  Averaged modes return scalar precision/recall/F1 and
    ``None`` for support.
    """
    zero_value = _check_zero_division(zero_division)
    y_true_array = _as_1d_array(y_true, "y_true")
    y_pred_array = _as_1d_array(y_pred, "y_pred")
    _validate_same_length(y_true_array, y_pred_array)

    resolved_labels = _resolve_labels(y_true_array, y_pred_array, labels)
    matrix = confusion_matrix(y_true_array, y_pred_array, resolved_labels)
    true_positive = np.diag(matrix).astype(float)
    predicted_positive = matrix.sum(axis=0).astype(float)
    actual_positive = matrix.sum(axis=1).astype(float)

    precision = _safe_divide(true_positive, predicted_positive, zero_value)
    recall = _safe_divide(true_positive, actual_positive, zero_value)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall, zero_value)
    support = actual_positive.astype(np.int64)

    if average is None:
        return precision, recall, f1, support
    if average == "macro":
        return float(np.mean(precision)), float(np.mean(recall)), float(np.mean(f1)), None
    if average == "weighted":
        weight_sum = float(support.sum())
        if weight_sum == 0.0:
            return zero_value, zero_value, zero_value, None
        weights = support / weight_sum
        return (
            float(np.sum(precision * weights)),
            float(np.sum(recall * weights)),
            float(np.sum(f1 * weights)),
            None,
        )
    if average == "micro":
        total_tp = float(true_positive.sum())
        total_predicted = float(predicted_positive.sum())
        total_actual = float(actual_positive.sum())
        micro_precision = zero_value if total_predicted == 0.0 else total_tp / total_predicted
        micro_recall = zero_value if total_actual == 0.0 else total_tp / total_actual
        micro_f1 = (
            zero_value
            if micro_precision + micro_recall == 0.0
            else 2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        )
        return float(micro_precision), float(micro_recall), float(micro_f1), None

    raise ValueError("average must be one of None, 'macro', 'weighted', or 'micro'.")


def macro_f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[Sequence[Any]] = None,
    zero_division: int | float = 0,
) -> float:
    """Return macro-averaged F1."""
    _, _, f1, _ = precision_recall_f1_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=zero_division,
    )
    return float(f1)


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks, assigning tied scores their mean rank."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_ranks = np.empty(scores.shape[0], dtype=float)

    start = 0
    while start < scores.shape[0]:
        end = start + 1
        while end < scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        sorted_ranks[start:end] = average_rank
        start = end

    ranks = np.empty(scores.shape[0], dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def binary_roc_auc_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    positive_label: Optional[Any] = None,
) -> float:
    """Compute binary ROC-AUC from ranks, with correct tie handling."""
    y_true_array = _as_1d_array(y_true, "y_true")
    scores = _as_1d_array(y_score, "y_score").astype(float)
    if y_true_array.shape[0] != scores.shape[0]:
        raise ValueError(
            f"y_true and y_score have inconsistent lengths: "
            f"{y_true_array.shape[0]} vs {scores.shape[0]}."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score contains NaN or infinite values.")

    classes = np.unique(y_true_array)
    if classes.size < 2:
        return float("nan")
    if classes.size > 2 and positive_label is None:
        raise ValueError("positive_label is required when y_true has more than two labels.")

    positive = classes[-1] if positive_label is None else positive_label
    positive_mask = y_true_array == positive
    n_positive = int(positive_mask.sum())
    n_negative = int(y_true_array.shape[0] - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    ranks = _average_ranks(scores)
    positive_rank_sum = float(ranks[positive_mask].sum())
    numerator = positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    return float(numerator / (n_positive * n_negative))


def _align_score_columns(
    y_score: np.ndarray,
    labels: np.ndarray,
    score_labels: Optional[Sequence[Any]],
) -> np.ndarray:
    """Reorder score columns so column ``i`` corresponds to ``labels[i]``."""
    if y_score.ndim != 2:
        raise ValueError(f"y_score must be 2-dimensional, got shape {y_score.shape}.")

    if score_labels is None:
        if y_score.shape[1] != labels.shape[0]:
            raise ValueError(
                "y_score column count must match labels when score_labels is not provided."
            )
        return y_score

    provided_labels = np.asarray(score_labels)
    if provided_labels.ndim != 1:
        raise ValueError("score_labels must be 1-dimensional.")
    if provided_labels.shape[0] != y_score.shape[1]:
        raise ValueError("score_labels length must match y_score columns.")

    provided_index = _label_index(provided_labels)
    missing = [label for label in labels.tolist() if label not in provided_index]
    if missing:
        preview = ", ".join(repr(value) for value in missing[:5])
        raise ValueError(f"y_score is missing score columns for labels: {preview}.")

    order = [provided_index[label] for label in labels.tolist()]
    return y_score[:, order]


def roc_auc_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    labels: Optional[Sequence[Any]] = None,
    score_labels: Optional[Sequence[Any]] = None,
    average: str = "macro",
    positive_label: Optional[Any] = None,
) -> float:
    """Compute binary or one-vs-rest multiclass ROC-AUC.

    For binary tasks, ``y_score`` may be a 1-D positive-class score vector or a
    2-D score matrix.  For multiclass tasks, ``y_score`` must be a 2-D matrix
    and the result is a macro or weighted one-vs-rest average.
    """
    if average not in {"macro", "weighted"}:
        raise ValueError("average must be either 'macro' or 'weighted'.")

    y_true_array = _as_1d_array(y_true, "y_true")
    resolved_labels = _resolve_labels(y_true_array, labels=labels)
    if resolved_labels.size < 2:
        return float("nan")

    scores = np.asarray(y_score, dtype=float)
    if scores.shape[0] != y_true_array.shape[0]:
        raise ValueError(
            f"y_true and y_score have inconsistent lengths: "
            f"{y_true_array.shape[0]} vs {scores.shape[0]}."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score contains NaN or infinite values.")

    if resolved_labels.size == 2:
        positive = positive_label if positive_label is not None else resolved_labels[-1]
        if scores.ndim == 1:
            positive_scores = scores
        else:
            aligned = _align_score_columns(scores, resolved_labels, score_labels)
            positive_index = _label_index(resolved_labels)[positive]
            positive_scores = aligned[:, positive_index]
        return binary_roc_auc_score(y_true_array, positive_scores, positive)

    if scores.ndim != 2:
        raise ValueError("Multiclass ROC-AUC requires a 2-D y_score matrix.")

    aligned_scores = _align_score_columns(scores, resolved_labels, score_labels)
    support = np.array([np.sum(y_true_array == label) for label in resolved_labels], dtype=float)
    auc_values = np.array(
        [
            binary_roc_auc_score(y_true_array == label, aligned_scores[:, index], True)
            for index, label in enumerate(resolved_labels)
        ],
        dtype=float,
    )
    finite = np.isfinite(auc_values)
    if not np.any(finite):
        return float("nan")
    if average == "macro":
        return float(np.mean(auc_values[finite]))

    weights = support[finite]
    if float(weights.sum()) == 0.0:
        return float("nan")
    return float(np.average(auc_values[finite], weights=weights))


def _probability_matrix(
    y_true: np.ndarray,
    y_score: ArrayLike,
    labels: np.ndarray,
    score_labels: Optional[Sequence[Any]] = None,
    positive_label: Optional[Any] = None,
) -> np.ndarray:
    """Return a clipped probability matrix aligned to ``labels``."""
    scores = np.asarray(y_score, dtype=float)
    if scores.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"y_true and y_score have inconsistent lengths: "
            f"{y_true.shape[0]} vs {scores.shape[0]}."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score contains NaN or infinite values.")

    if labels.size == 2 and scores.ndim == 1:
        positive = positive_label if positive_label is not None else labels[-1]
        if positive not in _label_index(labels):
            raise ValueError("positive_label must be present in labels.")
        positive_index = _label_index(labels)[positive]
        probabilities = np.zeros((scores.shape[0], 2), dtype=float)
        probabilities[:, positive_index] = np.clip(scores, EPS, 1.0 - EPS)
        probabilities[:, 1 - positive_index] = 1.0 - probabilities[:, positive_index]
        return probabilities

    aligned = _align_score_columns(scores, labels, score_labels)
    if np.any(aligned < 0.0):
        raise ValueError("Probability scores must be non-negative.")

    probabilities = np.clip(aligned, EPS, None)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= EPS):
        raise ValueError("Each probability row must have positive total mass.")
    return probabilities / row_sums


def log_loss_score(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    labels: Optional[Sequence[Any]] = None,
    score_labels: Optional[Sequence[Any]] = None,
) -> float:
    """Return multiclass negative log-likelihood."""
    y_true_array = _as_1d_array(y_true, "y_true")
    resolved_labels = _resolve_labels(y_true_array, labels=labels)
    encoded = _encode_labels(y_true_array, resolved_labels, "y_true")
    probabilities = _probability_matrix(
        y_true_array,
        y_proba,
        resolved_labels,
        score_labels=score_labels,
    )
    true_probabilities = probabilities[np.arange(y_true_array.shape[0]), encoded]
    return float(-np.mean(np.log(np.clip(true_probabilities, EPS, 1.0))))


def brier_score_loss(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    labels: Optional[Sequence[Any]] = None,
    score_labels: Optional[Sequence[Any]] = None,
) -> float:
    """Return multiclass Brier score as mean squared probability error."""
    y_true_array = _as_1d_array(y_true, "y_true")
    resolved_labels = _resolve_labels(y_true_array, labels=labels)
    encoded = _encode_labels(y_true_array, resolved_labels, "y_true")
    probabilities = _probability_matrix(
        y_true_array,
        y_proba,
        resolved_labels,
        score_labels=score_labels,
    )
    target = np.zeros_like(probabilities)
    target[np.arange(y_true_array.shape[0]), encoded] = 1.0
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def expected_calibration_error(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    labels: Optional[Sequence[Any]] = None,
    score_labels: Optional[Sequence[Any]] = None,
    n_bins: int = 10,
) -> float:
    """Return top-label expected calibration error with equal-width bins."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")

    y_true_array = _as_1d_array(y_true, "y_true")
    resolved_labels = _resolve_labels(y_true_array, labels=labels)
    probabilities = _probability_matrix(
        y_true_array,
        y_proba,
        resolved_labels,
        score_labels=score_labels,
    )

    prediction_indices = np.argmax(probabilities, axis=1)
    confidences = probabilities[np.arange(probabilities.shape[0]), prediction_indices]
    predicted_labels = resolved_labels[prediction_indices]
    correct = predicted_labels == y_true_array

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for index in range(n_bins):
        low = bin_edges[index]
        high = bin_edges[index + 1]
        if index == 0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences > low) & (confidences <= high)
        if not np.any(mask):
            continue
        bin_accuracy = float(np.mean(correct[mask]))
        bin_confidence = float(np.mean(confidences[mask]))
        ece += float(np.mean(mask)) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def classification_report_dict(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[Sequence[Any]] = None,
    zero_division: int | float = 0,
) -> Dict[str, Any]:
    """Return per-class and aggregate classification metrics as a dictionary."""
    y_true_array = _as_1d_array(y_true, "y_true")
    y_pred_array = _as_1d_array(y_pred, "y_pred")
    _validate_same_length(y_true_array, y_pred_array)
    resolved_labels = _resolve_labels(y_true_array, y_pred_array, labels)

    precision, recall, f1, support = precision_recall_f1_support(
        y_true_array,
        y_pred_array,
        labels=resolved_labels,
        average=None,
        zero_division=zero_division,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_f1_support(
        y_true_array,
        y_pred_array,
        labels=resolved_labels,
        average="macro",
        zero_division=zero_division,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_f1_support(
        y_true_array,
        y_pred_array,
        labels=resolved_labels,
        average="weighted",
        zero_division=zero_division,
    )

    per_class = {
        str(label): {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(resolved_labels.tolist())
    }

    return {
        "labels": resolved_labels,
        "per_class": per_class,
        "accuracy": accuracy_score(y_true_array, y_pred_array),
        "macro_avg": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
            "support": int(np.sum(support)),
        },
        "weighted_avg": {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1": float(weighted_f1),
            "support": int(np.sum(support)),
        },
    }


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_score: Optional[np.ndarray] = None,
    labels: Optional[Sequence[Any]] = None,
    score_labels: Optional[Sequence[Any]] = None,
    zero_division: int | float = 0,
) -> MetricDict:
    """Return the metric set required by the project experiments."""
    y_true_array = _as_1d_array(y_true, "y_true")
    y_pred_array = _as_1d_array(y_pred, "y_pred")
    _validate_same_length(y_true_array, y_pred_array)
    resolved_labels = _resolve_labels(y_true_array, y_pred_array, labels)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_f1_support(
        y_true_array,
        y_pred_array,
        labels=resolved_labels,
        average="macro",
        zero_division=zero_division,
    )

    auc = float("nan")
    log_loss = float("nan")
    brier_score = float("nan")
    calibration_error = float("nan")
    if y_score is not None:
        try:
            auc = roc_auc_score(
                y_true_array,
                y_score,
                labels=resolved_labels,
                score_labels=score_labels,
                average="macro",
            )
        except ValueError:
            auc = float("nan")
        try:
            log_loss = log_loss_score(
                y_true_array,
                y_score,
                labels=resolved_labels,
                score_labels=score_labels,
            )
            brier_score = brier_score_loss(
                y_true_array,
                y_score,
                labels=resolved_labels,
                score_labels=score_labels,
            )
            calibration_error = expected_calibration_error(
                y_true_array,
                y_score,
                labels=resolved_labels,
                score_labels=score_labels,
            )
        except ValueError:
            log_loss = float("nan")
            brier_score = float("nan")
            calibration_error = float("nan")

    return {
        "accuracy": accuracy_score(y_true_array, y_pred_array),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "roc_auc": auc,
        "log_loss": log_loss,
        "brier_score": brier_score,
        "expected_calibration_error": calibration_error,
    }


def predict_scores(model: Any, X: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return model scores and class-column labels when the model exposes them."""
    if callable(getattr(model, "predict_proba", None)):
        return np.asarray(model.predict_proba(X), dtype=float), getattr(model, "classes_", None)
    if callable(getattr(model, "decision_function", None)):
        return np.asarray(model.decision_function(X), dtype=float), getattr(model, "classes_", None)
    return None, getattr(model, "classes_", None)


def evaluate_classifier(
    model: Any,
    X: np.ndarray,
    y: ArrayLike,
    labels: Optional[Sequence[Any]] = None,
) -> MetricDict:
    """Predict with a fitted classifier and return project metrics."""
    if not callable(getattr(model, "predict", None)):
        raise TypeError(f"{type(model).__name__} does not expose predict().")
    y_pred = np.asarray(model.predict(X))
    y_score, score_labels = predict_scores(model, X)
    return classification_metrics(y, y_pred, y_score, labels=labels, score_labels=score_labels)


def mean_std_ci(
    values: ArrayLike,
    confidence_level: float = 0.95,
    clip: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Summarize finite values as mean, sample std, and confidence interval."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0.0,
        }

    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    if finite.size > 1:
        critical = _student_t_quantile(0.5 + confidence_level / 2.0, finite.size - 1)
        half_width = critical * std / math.sqrt(finite.size)
    else:
        half_width = 0.0

    ci_low = mean - half_width
    ci_high = mean + half_width
    if clip is not None:
        low, high = clip
        ci_low = max(low, ci_low)
        ci_high = min(high, ci_high)

    return {
        "mean": mean,
        "std": std,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n": float(finite.size),
    }


def aggregate_scores(
    per_fold: Mapping[str, ArrayLike],
    confidence_level: float = 0.95,
    metric_bounds: Tuple[float, float] = (0.0, 1.0),
) -> Dict[str, Dict[str, float]]:
    """Aggregate fold-level scores for every metric in ``per_fold``."""
    return {
        metric: mean_std_ci(values, confidence_level, clip=metric_bounds)
        for metric, values in per_fold.items()
    }


def _student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Return a t critical value, falling back to a normal approximation."""
    if degrees_of_freedom <= 0:
        return 0.0
    try:
        from scipy import stats

        return float(stats.t.ppf(probability, degrees_of_freedom))
    except Exception:
        return 1.96


def _student_t_two_sided_pvalue(t_statistic: float, degrees_of_freedom: int) -> float:
    """Return a two-sided t-test p-value, with a normal fallback."""
    if degrees_of_freedom <= 0:
        return float("nan")
    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(t_statistic), degrees_of_freedom))
    except Exception:
        return float(math.erfc(abs(t_statistic) / math.sqrt(2.0)))


def paired_t_tests_vs_reference(
    per_model: Mapping[str, ArrayLike],
    reference: str,
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Run paired t-tests and Holm-correct p-values against a reference model."""
    if reference not in per_model:
        raise KeyError(f"Reference model '{reference}' is not present in results.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    reference_scores = np.asarray(per_model[reference], dtype=float)
    tests: List[Dict[str, Any]] = []
    for model_name, model_scores_raw in per_model.items():
        if model_name == reference:
            continue

        model_scores = np.asarray(model_scores_raw, dtype=float)
        if model_scores.shape != reference_scores.shape:
            raise ValueError(
                f"Scores for '{model_name}' do not match reference shape "
                f"{reference_scores.shape}."
            )

        mask = np.isfinite(reference_scores) & np.isfinite(model_scores)
        differences = reference_scores[mask] - model_scores[mask]
        if differences.size < 2:
            t_statistic = float("nan")
            p_value = float("nan")
            mean_difference = float(np.mean(differences)) if differences.size else float("nan")
        else:
            mean_difference = float(np.mean(differences))
            std_difference = float(np.std(differences, ddof=1))
            if std_difference <= EPS:
                t_statistic = 0.0 if abs(mean_difference) <= EPS else math.inf
                p_value = 1.0 if t_statistic == 0.0 else 0.0
            else:
                t_statistic = mean_difference / (
                    std_difference / math.sqrt(differences.size)
                )
                p_value = _student_t_two_sided_pvalue(t_statistic, differences.size - 1)

        tests.append(
            {
                "model": str(model_name),
                "n_pairs": int(differences.size),
                "mean_difference_reference_minus_model": mean_difference,
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
        tests[index]["p_value_holm"] = float(adjusted)
        tests[index]["significant"] = bool(math.isfinite(adjusted) and adjusted <= alpha)

    return tests


def classification_bias_variance(
    predictions: np.ndarray,
    y_true: ArrayLike,
    labels: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Compute 0-1 classification bias and variance from replicate predictions.

    ``predictions`` must have shape ``(n_models, n_samples)``.  The main
    prediction is the modal class over models for each sample; ties are resolved
    by the order in ``labels``.
    """
    prediction_array = np.asarray(predictions)
    if prediction_array.ndim != 2:
        raise ValueError("predictions must have shape (n_models, n_samples).")
    if prediction_array.shape[0] == 0 or prediction_array.shape[1] == 0:
        raise ValueError("predictions must contain at least one model and one sample.")

    y_true_array = _as_1d_array(y_true, "y_true")
    if prediction_array.shape[1] != y_true_array.shape[0]:
        raise ValueError(
            f"predictions and y_true have inconsistent sample counts: "
            f"{prediction_array.shape[1]} vs {y_true_array.shape[0]}."
        )

    resolved_labels = _resolve_labels(
        y_true_array,
        prediction_array.reshape(-1),
        labels,
    )
    y_encoded = _encode_labels(y_true_array, resolved_labels, "y_true")
    pred_encoded = np.vstack(
        [
            _encode_labels(row, resolved_labels, "predictions")
            for row in prediction_array
        ]
    )

    votes = np.stack(
        [(pred_encoded == index).sum(axis=0) for index in range(resolved_labels.size)]
    )
    main_encoded = np.argmax(votes, axis=0)
    disagrees_truth = pred_encoded != y_encoded[None, :]
    disagrees_main = pred_encoded != main_encoded[None, :]

    bias_squared = float(np.mean(main_encoded != y_encoded))
    variance = float(disagrees_main.mean())
    expected_loss = float(disagrees_truth.mean())

    per_class: Dict[str, Dict[str, float]] = {}
    for index, label in enumerate(resolved_labels.tolist()):
        mask = y_encoded == index
        if not np.any(mask):
            continue
        class_bias = float(np.mean(main_encoded[mask] != index))
        per_class[str(label)] = {
            "expected_loss": float(disagrees_truth[:, mask].mean()),
            "bias": class_bias,
            "bias_squared": class_bias,
            "variance": float(disagrees_main[:, mask].mean()),
            "support": int(mask.sum()),
        }

    return {
        "expected_loss": expected_loss,
        "bias": bias_squared,
        "bias_squared": bias_squared,
        "variance": variance,
        "main_prediction": resolved_labels[main_encoded],
        "labels": resolved_labels,
        "per_class": per_class,
    }


def to_jsonable(value: Any) -> Any:
    """Recursively convert NumPy, pathlib, and scalar values for JSON export."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def export_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write JSON with parent-directory creation and NumPy-safe conversion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(dict(payload)), file, indent=2)


def export_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Write row dictionaries to CSV using the first row's field order."""
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)
