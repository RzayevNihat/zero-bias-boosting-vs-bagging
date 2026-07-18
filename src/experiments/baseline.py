"""Baseline experiment for the WDBC dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiments.utils import ensure_results_dir, print_section, timer
from src.metrics.evaluation import evaluate_classifier, export_csv, export_json
from src.trees.decision_tree import DecisionStump, DecisionTree
from src.utils.preprocessing import PreprocessingPipeline, load_wdbc

RANDOM_STATE = 42
TEST_SIZE = 0.2
RESULTS_DIR = ROOT_DIR / "results"
DATA_PATH = ROOT_DIR / "data" / "wdbc.data"


def run_baseline_experiment(
    data_path: Path = DATA_PATH,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> list[dict]:
    """Train the baseline models and return their metrics rows."""
    dataset = load_wdbc(data_path)

    print_section("Splitting data (80/20, stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.y,
    )

    # Fit the scaler on the training split only to avoid data leakage.
    pipeline = PreprocessingPipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    print_section("Training models")
    with timer("Fit MyTree (unpruned)"):
        my_tree = DecisionTree(max_depth=None, criterion="gini", random_state=random_state)
        my_tree.fit(X_train_processed, y_train)

    with timer("Fit MyStump (depth=1)"):
        my_stump = DecisionStump(criterion="gini", random_state=random_state)
        my_stump.fit(X_train_processed, y_train)

    with timer("Fit sklearn reference tree"):
        sk_tree = DecisionTreeClassifier(random_state=random_state)
        sk_tree.fit(X_train_processed, y_train)

    print_section("Evaluating models")
    rows: list[dict] = []
    for model, name in [(my_tree, "MyTree"), (my_stump, "MyStump"), (sk_tree, "sklearn")]:
        metrics = evaluate_classifier(model, X_test_processed, y_test)
        row = {"model": name, "dataset": "wdbc", **metrics}
        rows.append(row)

    return rows


def print_results_table(rows: list[dict]) -> None:
    """Render metrics as a simple aligned table."""
    if not rows:
        print("No results to display.")
        return

    columns = list(rows[0].keys())
    widths = {
        c: max(
            len(c),
            *(len(f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])) for row in rows),
        )
        for c in columns
    }

    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for c in columns:
            value = row[c]
            cells.append((f"{value:.4f}" if isinstance(value, float) else str(value)).ljust(widths[c]))
        print("  ".join(cells))


def main() -> int:
    """Run the WDBC baseline experiment and export the outputs."""
    ensure_results_dir(RESULTS_DIR)

    dataset = load_wdbc(DATA_PATH)
    rows = run_baseline_experiment()
    print_results_table(rows)

    csv_path = RESULTS_DIR / "baseline_results.csv"
    json_path = RESULTS_DIR / "baseline_results.json"
    export_csv(rows, csv_path)
    export_json({"experiment": "baseline", "results": rows}, json_path)

    summary_path = RESULTS_DIR / "baseline_summary.json"
    classes, counts = np.unique(dataset.y, return_counts=True)
    payload = {
        "dataset": "wdbc",
        "n_samples": int(dataset.X.shape[0]),
        "n_features": int(dataset.X.shape[1]),
        "class_counts": {str(c): int(n) for c, n in zip(classes, counts)},
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nSaved results to {csv_path}")
    print(f"Saved results to {json_path}")
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
