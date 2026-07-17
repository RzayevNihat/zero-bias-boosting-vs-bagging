"""Sequential versus multiprocessing benchmark for Random Forest training.

This script satisfies the project requirement for a sequential timing baseline.
Use modest worker counts on Windows and Colab to avoid process-spawn memory
pressure.
"""

from __future__ import annotations

import argparse
import time
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from src.bagging.random_forest import RandomForestClassifier
from src.experiments.rf_utils import (
    FIGURES_DIR,
    DatasetBundle,
    ensure_output_dirs,
    evaluate_classifier,
    load_digits_binary_bundle,
    prepare_bundle_split,
    save_results_table,
)


def run_parallel_benchmark(
    worker_values: Sequence[int] = (1, 2),
    n_estimators: int = 100,
    bundle: DatasetBundle | None = None,
) -> list[dict]:
    """Benchmark fit time and confirm prediction quality for each worker count."""
    if bundle is None:
        bundle = load_digits_binary_bundle()
    X_train, X_test, y_train, y_test, treatment = prepare_bundle_split(bundle)

    rows: list[dict] = []
    sequential_time: float | None = None
    reference_predictions = None
    for n_jobs in worker_values:
        start = time.perf_counter()
        forest = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            max_features="sqrt",
            oob_score=True,
            n_jobs=int(n_jobs),
            random_state=42,
        ).fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        predictions = forest.predict(X_test)
        metrics = evaluate_classifier(forest, X_test, y_test)

        if sequential_time is None:
            sequential_time = fit_seconds
            reference_predictions = predictions
        same_predictions = bool((predictions == reference_predictions).all())
        rows.append(
            {
                "dataset": bundle.name,
                "n_estimators": n_estimators,
                "n_jobs": int(n_jobs),
                "fit_seconds": fit_seconds,
                "speedup_vs_n_jobs_1": sequential_time / fit_seconds,
                "accuracy": metrics["accuracy"],
                "oob_accuracy": forest.oob_score_,
                "same_predictions_as_n_jobs_1": same_predictions,
                "imbalance_treatment": treatment,
            }
        )
    return rows


def plot_benchmark(rows: list[dict]) -> str:
    """Save a worker-count versus fit-time plot."""
    ensure_output_dirs()
    frame = pd.DataFrame(rows).sort_values("n_jobs")
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(frame["n_jobs"], frame["fit_seconds"], marker="o")
    plt.xlabel("n_jobs")
    plt.ylabel("Fit time (seconds)")
    plt.title("Random Forest sequential vs parallel training")
    plt.xticks(frame["n_jobs"])
    plt.tight_layout()
    path = FIGURES_DIR / "rf_parallel_benchmark.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Worker counts to benchmark. Avoid -1 on memory-constrained systems.",
    )
    args = parser.parse_args()

    rows = run_parallel_benchmark(
        worker_values=args.workers,
        n_estimators=10 if args.fast else 100,
    )
    csv_path = save_results_table(rows, "rf_parallel_benchmark.csv")
    figure_path = plot_benchmark(rows)
    print(f"Saved results: {csv_path}")
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
