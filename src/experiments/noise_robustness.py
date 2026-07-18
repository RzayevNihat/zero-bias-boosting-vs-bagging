"""Experiment 5: label-noise robustness for Random Forest and AdaBoost.

Training labels are corrupted at 0%, 5%, 10%, and 20%; evaluation always uses
clean test labels. Person 3 owns the Random Forest side. AdaBoost results are
included automatically when the teammate module is importable and functional.

Run from repository root:

    python -m src.experiments.noise_robustness --fast --n-jobs 1
    python -m src.experiments.noise_robustness --n-jobs 1
"""

from __future__ import annotations

import argparse
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.bagging.random_forest import RandomForestClassifier
from src.experiments.rf_utils import (
    DatasetBundle,
    FIGURES_DIR,
    RANDOM_STATE,
    add_label_noise,
    ensure_output_dirs,
    evaluate_classifier,
    load_breast_cancer_bundle,
    load_default_bundles,
    random_oversample_minority,
    save_results_table,
    train_test_scaled_split,
)


def _build_team_adaboost(n_estimators: int, random_state: int):
    """Return the team AdaBoost model, or ``None`` until its integration is ready."""
    try:
        from src.boosting.adaboost import AdaBoostClassifier

        return AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
    except (ImportError, AttributeError):
        return None


def run_noise_robustness(
    noise_levels: Sequence[float] | None = None,
    n_estimators: int = 100,
    max_depth: int | None = 8,
    n_jobs: int = 1,
    bundles: Sequence[DatasetBundle] | None = None,
    include_adaboost: bool = True,
) -> list[dict]:
    """Run the clean-test label-noise experiment on all binary datasets."""
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.20]
    if bundles is None:
        bundles = load_default_bundles()

    rows: list[dict] = []
    for bundle in bundles:
        if not bundle.is_binary:
            continue
        X_train, X_test, y_train, y_test = train_test_scaled_split(bundle.X, bundle.y)

        for noise_fraction in noise_levels:
            noisy_y_train = add_label_noise(
                y_train,
                noise_fraction=float(noise_fraction),
                random_state=RANDOM_STATE + int(float(noise_fraction) * 1000),
            )
            treatment = "none"
            X_model_train = X_train
            y_model_train = noisy_y_train
            if bundle.minority_fraction <= 0.01:
                X_model_train, y_model_train = random_oversample_minority(
                    X_train,
                    noisy_y_train,
                    random_state=RANDOM_STATE,
                )
                treatment = "label_noise_then_random_oversampling_train_only"

            forest = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features="sqrt",
                oob_score=True,
                n_jobs=n_jobs,
                random_state=RANDOM_STATE,
            ).fit(X_model_train, y_model_train)
            rf_metrics = evaluate_classifier(forest, X_test, y_test)
            rows.append(
                {
                    "dataset": bundle.name,
                    "dataset_description": bundle.description,
                    "minority_fraction": bundle.minority_fraction,
                    "imbalance_treatment": treatment,
                    "model": "team_random_forest",
                    "noise_fraction": float(noise_fraction),
                    "accuracy": rf_metrics["accuracy"],
                    "f1_macro": rf_metrics["f1_macro"],
                    "auc_roc": rf_metrics["auc_roc"],
                    "minority_recall": rf_metrics["minority_recall"],
                    "oob_accuracy_on_noisy_train": forest.oob_score_,
                    "status": "ok",
                }
            )

            if not include_adaboost:
                continue
            adaboost = _build_team_adaboost(n_estimators, RANDOM_STATE)
            if adaboost is None:
                rows.append(
                    {
                        "dataset": bundle.name,
                        "dataset_description": bundle.description,
                        "minority_fraction": bundle.minority_fraction,
                        "imbalance_treatment": treatment,
                        "model": "team_adaboost",
                        "noise_fraction": float(noise_fraction),
                        "accuracy": np.nan,
                        "f1_macro": np.nan,
                        "auc_roc": np.nan,
                        "minority_recall": np.nan,
                        "oob_accuracy_on_noisy_train": np.nan,
                        "status": "waiting_for_person_2_adaboost_integration",
                    }
                )
                continue

            try:
                adaboost.fit(X_model_train, y_model_train)
                metrics = evaluate_classifier(adaboost, X_test, y_test)
                rows.append(
                    {
                        "dataset": bundle.name,
                        "dataset_description": bundle.description,
                        "minority_fraction": bundle.minority_fraction,
                        "imbalance_treatment": treatment,
                        "model": "team_adaboost",
                        "noise_fraction": float(noise_fraction),
                        "accuracy": metrics["accuracy"],
                        "f1_macro": metrics["f1_macro"],
                        "auc_roc": metrics["auc_roc"],
                        "minority_recall": metrics["minority_recall"],
                        "oob_accuracy_on_noisy_train": np.nan,
                        "status": "ok",
                    }
                )
            except (ValueError, RuntimeError) as exc:
                rows.append(
                    {
                        "dataset": bundle.name,
                        "dataset_description": bundle.description,
                        "minority_fraction": bundle.minority_fraction,
                        "imbalance_treatment": treatment,
                        "model": "team_adaboost",
                        "noise_fraction": float(noise_fraction),
                        "accuracy": np.nan,
                        "f1_macro": np.nan,
                        "auc_roc": np.nan,
                        "minority_recall": np.nan,
                        "oob_accuracy_on_noisy_train": np.nan,
                        "status": f"failed: {type(exc).__name__}: {exc}",
                    }
                )
    return rows


def plot_noise_results(rows: list[dict]) -> list[str]:
    """Save one clean-test accuracy degradation curve per dataset."""
    ensure_output_dirs()
    frame = pd.DataFrame(rows)
    output_paths: list[str] = []
    for dataset, subset in frame[frame["status"] == "ok"].groupby("dataset"):
        plt.figure(figsize=(7, 4.5))
        for model, model_rows in subset.groupby("model"):
            model_rows = model_rows.sort_values("noise_fraction")
            plt.plot(
                model_rows["noise_fraction"],
                model_rows["accuracy"],
                marker="o",
                label=model,
            )
        plt.xlabel("Training label noise fraction")
        plt.ylabel("Clean test accuracy")
        plt.title(f"Noise robustness: {dataset}")
        plt.legend()
        plt.tight_layout()
        output_path = FIGURES_DIR / f"noise_robustness_{dataset}.png"
        plt.savefig(output_path, dpi=200)
        plt.close()
        output_paths.append(str(output_path))
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="Run a small smoke test.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker processes. Use 1 on Windows/Colab unless benchmarking.",
    )
    parser.add_argument(
        "--rf-only",
        action="store_true",
        help="Skip AdaBoost while Person 2 integration is incomplete.",
    )
    args = parser.parse_args()

    bundles = (
        [load_breast_cancer_bundle()]
        if args.fast
        else load_default_bundles()
    )
    rows = run_noise_robustness(
        noise_levels=[0.0, 0.10] if args.fast else None,
        n_estimators=10 if args.fast else 100,
        max_depth=5 if args.fast else 8,
        n_jobs=args.n_jobs,
        bundles=bundles,
        include_adaboost=not args.rf_only,
    )
    csv_path = save_results_table(rows, "noise_robustness_results.csv")
    figure_paths = plot_noise_results(rows)
    print(f"Saved results: {csv_path}")
    print("Saved figures:")
    for path in figure_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
