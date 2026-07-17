# Experiment 1 - Baseline
# Compares my unpruned tree vs my stump vs sklearn's tree, all on the same
# WDBC data. This is basically the whole point of the module: prove the
# tree actually works before anyone builds boosting/bagging on top of it.
#
# my stuff (trees, preprocessing, metrics) does the actual work, sklearn is
# only here as the "answer key" to check against - never used to train
# anything for real.
#
# run with: python -m src.experiments.baseline
# (needs data/wdbc.data - run download_data.sh first if it's not there)

from __future__ import annotations

import sys
from pathlib import Path

# need this so imports work whether you run the file directly or with -m
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
    # trains all 3 models on the same split and returns a metrics row for each
    dataset = load_wdbc(data_path)

    print_section("Splitting data (80/20, stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.y,
    )

    # fit the scaler on train only! if you fit it on the whole dataset
    # you leak info from the test set and your numbers look better than
    # they should
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
    # just lines everything up nicely in the terminal, nothing fancy
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


def main() -> None:
    ensure_results_dir(RESULTS_DIR)

    rows = run_baseline_experiment()
    print_results_table(rows)

    # save to both formats, csv for opening in excel/pandas, json in case
    # some other script wants to read it back in later
    csv_path = RESULTS_DIR / "baseline_results.csv"
    json_path = RESULTS_DIR / "baseline_results.json"
    export_csv(rows, csv_path)
    export_json({"experiment": "baseline", "results": rows}, json_path)

    print(f"\nSaved results to {csv_path}")
    print(f"Saved results to {json_path}")


if __name__ == "__main__":
    main()