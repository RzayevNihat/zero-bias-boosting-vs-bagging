"""Fast smoke tests for Person 3 experiment code."""

from __future__ import annotations

import numpy as np

from src.experiments.coverage_audit import build_pytest_command, extract_total_coverage
from src.experiments.noise_robustness import run_noise_robustness
from src.experiments.rf_parallel_benchmark import run_parallel_benchmark
from src.experiments.rf_scaling import run_max_depth_sweep, run_n_estimators_sweep
from src.experiments.rf_utils import DatasetBundle


def tiny_bundle() -> DatasetBundle:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(100, 5))
    y = ((X[:, 0] - X[:, 1]) > 0).astype(int)
    return DatasetBundle("tiny", X, y, "small deterministic smoke-test dataset", True)


def test_rf_scaling_sweeps_return_expected_records():
    bundle = tiny_bundle()
    estimator_rows = run_n_estimators_sweep(
        estimator_values=[1, 3],
        n_jobs=1,
        max_depth=4,
        bundles=[bundle],
    )
    depth_rows = run_max_depth_sweep(
        depth_values=[1, 3],
        n_jobs=1,
        bundles=[bundle],
        n_estimators=5,
    )
    assert len(estimator_rows) == 2
    assert len(depth_rows) == 2
    assert {row["n_estimators"] for row in estimator_rows} == {1, 3}
    assert {row["max_depth"] for row in depth_rows} == {1, 3}
    assert all(0.0 <= row["test_accuracy"] <= 1.0 for row in estimator_rows + depth_rows)


def test_noise_experiment_rf_only_smoke():
    rows = run_noise_robustness(
        noise_levels=[0.0, 0.10],
        n_estimators=5,
        max_depth=4,
        n_jobs=1,
        bundles=[tiny_bundle()],
        include_adaboost=False,
    )
    assert len(rows) == 2
    assert {row["noise_fraction"] for row in rows} == {0.0, 0.10}
    assert all(row["model"] == "team_random_forest" for row in rows)
    assert all(row["status"] == "ok" for row in rows)


def test_parallel_benchmark_has_sequential_baseline():
    rows = run_parallel_benchmark(
        worker_values=[1],
        n_estimators=5,
        bundle=tiny_bundle(),
    )
    assert [row["n_jobs"] for row in rows] == [1]
    assert rows[0]["speedup_vs_n_jobs_1"] == 1.0
    assert rows[0]["same_predictions_as_n_jobs_1"]


def test_coverage_helpers():
    command = build_pytest_command(rf_only=True)
    assert "tests/test_random_forest.py" in command
    assert extract_total_coverage("TOTAL 100 10 90%\n") == "90%"
    assert extract_total_coverage("no total here") == "not_found"
