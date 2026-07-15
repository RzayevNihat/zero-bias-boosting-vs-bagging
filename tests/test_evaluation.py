import math

import numpy as np

from src.metrics.evaluation import (
    aggregate_scores,
    binary_roc_auc_score,
    brier_score_loss,
    classification_bias_variance,
    classification_metrics,
    confusion_matrix,
    expected_calibration_error,
    log_loss_score,
    paired_t_tests_vs_reference,
    roc_auc_score,
)


def test_classification_metrics_match_hand_checked_binary_case() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array([0.1, 0.6, 0.7, 0.8])

    scores = classification_metrics(y_true, y_pred, y_score)

    assert scores["accuracy"] == 0.75
    np.testing.assert_allclose(scores["macro_precision"], 5.0 / 6.0)
    np.testing.assert_allclose(scores["macro_recall"], 0.75)
    np.testing.assert_allclose(scores["macro_f1"], 11.0 / 15.0)
    assert scores["roc_auc"] == 1.0
    np.testing.assert_allclose(
        scores["log_loss"],
        -np.mean(np.log([0.9, 0.4, 0.7, 0.8])),
    )
    np.testing.assert_allclose(scores["brier_score"], 0.25)
    np.testing.assert_allclose(scores["expected_calibration_error"], 0.3)
    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), [[1, 1], [0, 2]])


def test_binary_roc_auc_handles_tied_scores() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.2, 0.5, 0.5, 0.9])

    assert binary_roc_auc_score(y_true, y_score) == 0.875


def test_multiclass_roc_auc_aligns_score_columns() -> None:
    y_true = np.array(["a", "b", "c", "a", "b", "c"])
    y_score_abc = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.6, 0.2],
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
            [0.1, 0.2, 0.7],
        ]
    )
    y_score_cab = y_score_abc[:, [2, 0, 1]]

    direct = roc_auc_score(y_true, y_score_abc, labels=["a", "b", "c"])
    reordered = roc_auc_score(
        y_true,
        y_score_cab,
        labels=["a", "b", "c"],
        score_labels=["c", "a", "b"],
    )

    np.testing.assert_allclose(direct, 11.0 / 12.0)
    np.testing.assert_allclose(reordered, direct)


def test_calibration_metrics_align_multiclass_score_columns() -> None:
    y_true = np.array(["a", "b", "c", "a"])
    y_score_abc = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1],
        ]
    )
    y_score_cab = y_score_abc[:, [2, 0, 1]]

    direct = log_loss_score(y_true, y_score_abc, labels=["a", "b", "c"])
    reordered = log_loss_score(
        y_true,
        y_score_cab,
        labels=["a", "b", "c"],
        score_labels=["c", "a", "b"],
    )

    np.testing.assert_allclose(reordered, direct)
    assert brier_score_loss(y_true, y_score_abc, labels=["a", "b", "c"]) < 0.4
    assert expected_calibration_error(y_true, y_score_abc, labels=["a", "b", "c"]) < 0.4


def test_aggregate_scores_and_paired_tests_are_reproducible() -> None:
    summary = aggregate_scores({"macro_f1": [0.8, 0.9, float("nan")]})
    assert summary["macro_f1"]["mean"] == 0.8500000000000001
    assert summary["macro_f1"]["n"] == 2.0
    assert 0.0 <= summary["macro_f1"]["ci_low"] <= summary["macro_f1"]["ci_high"] <= 1.0

    tests = paired_t_tests_vs_reference(
        {"AdaBoost": [0.9, 0.8, 0.85], "Tree": [0.7, 0.75, 0.8]},
        reference="AdaBoost",
    )
    assert tests[0]["model"] == "Tree"
    assert tests[0]["n_pairs"] == 3
    assert math.isfinite(tests[0]["p_value_holm"])


def test_classification_bias_variance_uses_modal_main_prediction() -> None:
    predictions = np.array(
        [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 0],
        ]
    )
    y_true = np.array([0, 0, 1, 1])

    result = classification_bias_variance(predictions, y_true)

    assert result["expected_loss"] == 1.0 / 3.0
    assert result["bias_squared"] == 0.25
    assert result["variance"] == 0.25
    np.testing.assert_array_equal(result["main_prediction"], [0, 1, 1, 1])
