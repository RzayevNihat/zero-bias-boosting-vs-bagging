"""Experiment 3: Random Forest scaling.

Required sweeps:
1. Fix ``max_depth=None`` and vary ``n_estimators`` from 1 to 200.
2. Fix ``n_estimators=100`` and vary ``max_depth`` from 1 to 20.

Run from repository root:

    python -m src.experiments.rf_scaling --fast --n-jobs 1
    python -m src.experiments.rf_scaling --n-jobs 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

from src.bagging.random_forest import RandomForestClassifier
from src.experiments.rf_utils import (
    DatasetBundle,
    FIGURES_DIR,
    RANDOM_STATE,
    ensure_output_dirs,
    evaluate_classifier,
    load_default_bundles,
    prepare_bundle_split,
    save_results_table,
)


def run_n_estimators_sweep(
    estimator_values: Sequence[int] | None = None,
    n_jobs: int = 1,
    max_depth: int | None = None,
    bundles: Sequence[DatasetBundle] | None = None,
) -> list[dict]:
    """Measure test/OOB performance as the number of trees increases."""
    if estimator_values is None:
        estimator_values = [1] + list(range(10, 201, 10))
    if bundles is None:
        bundles = load_default_bundles()

    rows: list[dict] = []
    for bundle in bundles:
        X_train, X_test, y_train, y_test, treatment = prepare_bundle_split(bundle)
        for n_estimators in estimator_values:
            start = time.perf_counter()
            forest = RandomForestClassifier(
                n_estimators=int(n_estimators),
                max_depth=max_depth,
                max_features="sqrt",
                oob_score=True,
                n_jobs=n_jobs,
                random_state=RANDOM_STATE,
            ).fit(X_train, y_train)
            elapsed = time.perf_counter() - start
            metrics = evaluate_classifier(forest, X_test, y_test)
            rows.append(
                {
                    "dataset": bundle.name,
                    "dataset_description": bundle.description,
                    "minority_fraction": bundle.minority_fraction,
                    "imbalance_treatment": treatment,
                    "sweep": "n_estimators",
                    "n_estimators": int(n_estimators),
                    "max_depth": "None",
                    "test_accuracy": metrics["accuracy"],
                    "test_f1_macro": metrics["f1_macro"],
                    "test_auc_roc": metrics["auc_roc"],
                    "minority_recall": metrics["minority_recall"],
                    "oob_accuracy": forest.oob_score_,
                    "fit_seconds": elapsed,
                }
            )
    return rows


def run_max_depth_sweep(
    depth_values: Sequence[int] | None = None,
    n_jobs: int = 1,
    bundles: Sequence[DatasetBundle] | None = None,
    n_estimators: int = 100,
) -> list[dict]:
    """Measure test/OOB performance as maximum tree depth changes."""
    if depth_values is None:
        depth_values = list(range(1, 21))
    if bundles is None:
        bundles = load_default_bundles()

    rows: list[dict] = []
    for bundle in bundles:
        X_train, X_test, y_train, y_test, treatment = prepare_bundle_split(bundle)
        for max_depth in depth_values:
            start = time.perf_counter()
            forest = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=int(max_depth),
                max_features="sqrt",
                oob_score=True,
                n_jobs=n_jobs,
                random_state=RANDOM_STATE,
            ).fit(X_train, y_train)
            elapsed = time.perf_counter() - start
            metrics = evaluate_classifier(forest, X_test, y_test)
            rows.append(
                {
                    "dataset": bundle.name,
                    "dataset_description": bundle.description,
                    "minority_fraction": bundle.minority_fraction,
                    "imbalance_treatment": treatment,
                    "sweep": "max_depth",
                    "n_estimators": n_estimators,
                    "max_depth": int(max_depth),
                    "test_accuracy": metrics["accuracy"],
                    "test_f1_macro": metrics["f1_macro"],
                    "test_auc_roc": metrics["auc_roc"],
                    "minority_recall": metrics["minority_recall"],
                    "oob_accuracy": forest.oob_score_,
                    "fit_seconds": elapsed,
                }
            )
    return rows


def plot_sweep(rows: list[dict], sweep_name: str, x_column: str, filename: str) -> Path:
    """Plot test and OOB accuracy for one scaling sweep."""
    ensure_output_dirs()
    output_path = FIGURES_DIR / filename
    datasets = sorted({row["dataset"] for row in rows if row["sweep"] == sweep_name})

    plt.figure(figsize=(9, 5.5))
    for dataset in datasets:
        subset = [
            row for row in rows
            if row["dataset"] == dataset and row["sweep"] == sweep_name
        ]
        subset.sort(key=lambda item: int(item[x_column]))
        x_values = [int(row[x_column]) for row in subset]
        plt.plot(
            x_values,
            [row["test_accuracy"] for row in subset],
            marker="o",
            label=f"{dataset} test",
        )
        plt.plot(
            x_values,
            [row["oob_accuracy"] for row in subset],
            marker="x",
            linestyle="--",
            label=f"{dataset} OOB",
        )

    plt.xlabel(x_column.replace("_", " ").title())
    plt.ylabel("Accuracy")
    plt.title(f"Random Forest scaling by {x_column.replace('_', ' ')}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="Run a small smoke test.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker processes. Use 1 on Windows/Colab unless benchmarking.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    if args.fast:
        bundles = load_default_bundles()[:1]
        estimator_rows = run_n_estimators_sweep(
            estimator_values=[1, 3, 5],
            n_jobs=args.n_jobs,
            max_depth=4,
            bundles=bundles,
        )
        depth_rows = run_max_depth_sweep(
            depth_values=[1, 3, 5],
            n_jobs=args.n_jobs,
            bundles=bundles,
            n_estimators=10,
        )
    else:
        estimator_rows = run_n_estimators_sweep(n_jobs=args.n_jobs)
        depth_rows = run_max_depth_sweep(n_jobs=args.n_jobs)

    rows = estimator_rows + depth_rows
    csv_path = save_results_table(rows, "rf_scaling_results.csv")
    fig_estimators = plot_sweep(
        rows,
        "n_estimators",
        "n_estimators",
        "rf_scaling_n_estimators.png",
    )
    fig_depth = plot_sweep(rows, "max_depth", "max_depth", "rf_scaling_max_depth.png")
    print(f"Saved results: {csv_path}")
    print(f"Saved figures: {fig_estimators}, {fig_depth}")


if __name__ == "__main__":
    main()
