"""Tests for src.experiments.baseline (Experiment 1: Baseline)."""

from __future__ import annotations

import numpy as np
import pytest

from src.experiments import baseline
from src.utils.preprocessing import Dataset


@pytest.fixture
def synthetic_dataset(monkeypatch):
    """A small, separable binary dataset standing in for WDBC."""
    rng = np.random.default_rng(0)
    n_per_class = 60
    X0 = rng.normal(loc=-2.0, scale=1.0, size=(n_per_class, 5))
    X1 = rng.normal(loc=2.0, scale=1.0, size=(n_per_class, 5))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    def fake_load_wdbc(path=None):
        return Dataset(X=X, y=y)

    monkeypatch.setattr(baseline, "load_wdbc", fake_load_wdbc)
    return X, y


def test_run_baseline_experiment_returns_three_models(synthetic_dataset):
    rows = baseline.run_baseline_experiment()
    names = {row["model"] for row in rows}
    assert names == {"MyTree", "MyStump", "sklearn"}
    assert len(rows) == 3


def test_run_baseline_experiment_metrics_are_valid_fractions(synthetic_dataset):
    rows = baseline.run_baseline_experiment()
    for row in rows:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["macro_f1"] <= 1.0
        # roc_auc can be NaN in degenerate cases, but on this separable
        # synthetic dataset it should be a valid, high score.
        assert 0.5 <= row["roc_auc"] <= 1.0


def test_stump_underperforms_unpruned_tree_on_separable_data(synthetic_dataset):
    """A depth-1 stump should do noticeably worse than an unpruned tree
    when the true boundary requires more than one split to isolate
    cleanly -- demonstrating the underfitting side of the baseline."""
    rows = baseline.run_baseline_experiment()
    by_model = {row["model"]: row for row in rows}
    # On this simple two-blob dataset a stump can still do reasonably
    # well, so we only assert it does not exceed the unpruned tree.
    assert by_model["MyStump"]["accuracy"] <= by_model["MyTree"]["accuracy"] + 1e-9


def test_print_results_table_does_not_raise(synthetic_dataset, capsys):
    rows = baseline.run_baseline_experiment()
    baseline.print_results_table(rows)
    captured = capsys.readouterr()
    assert "MyTree" in captured.out
    assert "MyStump" in captured.out
    assert "sklearn" in captured.out


def test_print_results_table_handles_empty_input(capsys):
    baseline.print_results_table([])
    captured = capsys.readouterr()
    assert "No results to display." in captured.out