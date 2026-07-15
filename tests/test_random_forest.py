"""Unit and integration tests for the project Random Forest implementation."""

from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.bagging.random_forest import RandomForestClassifier


@pytest.fixture
def toy_binary() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 4))
    y = ((X[:, 0] + 0.8 * X[:, 1]) > 0).astype(int)
    return X, y


@pytest.fixture
def breast_cancer_split():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=42,
        stratify=data.target,
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def test_binary_accuracy_is_reasonable(toy_binary):
    X, y = toy_binary
    model = RandomForestClassifier(
        n_estimators=40,
        max_depth=6,
        random_state=42,
    ).fit(X, y)
    assert np.mean(model.predict(X) == y) >= 0.90


def test_predict_uses_hard_majority_vote(toy_binary):
    X, y = toy_binary
    model = RandomForestClassifier(n_estimators=25, random_state=11).fit(X, y)
    votes = np.vstack([tree.predict(X).astype(int) for tree in model.estimators_])
    expected_encoded = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=model.n_classes_).argmax(),
        axis=0,
        arr=votes,
    )
    expected = model.classes_[expected_encoded]
    np.testing.assert_array_equal(model.predict(X), expected)


def test_predict_proba_has_global_shape_and_sums_to_one(toy_binary):
    X, y = toy_binary
    model = RandomForestClassifier(n_estimators=20, random_state=1).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_proba_aligns_trees_missing_minority_class():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 5))
    y = np.zeros(100, dtype=int)
    y[-1] = 1
    model = RandomForestClassifier(n_estimators=80, random_state=42).fit(X, y)
    assert any(len(tree.classes_) == 1 for tree in model.estimators_)
    proba = model.predict_proba(X[:10])
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-10)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_multiclass_returns_original_labels():
    data = load_digits()
    mask = np.isin(data.target, [1, 4, 7])
    X = data.data[mask][:240]
    # Map original classes to deliberately non-consecutive labels.
    mapping = {1: 10, 4: 20, 7: 30}
    y = np.array([mapping[int(value)] for value in data.target[mask][:240]])
    model = RandomForestClassifier(n_estimators=25, max_depth=7, random_state=42).fit(X, y)
    assert set(np.unique(model.predict(X[:50]))).issubset({10, 20, 30})
    assert model.predict_proba(X[:5]).shape == (5, 3)


def test_feature_importances_are_normalized(toy_binary):
    X, y = toy_binary
    model = RandomForestClassifier(n_estimators=30, random_state=42).fit(X, y)
    assert model.feature_importances_.shape == (X.shape[1],)
    assert np.all(model.feature_importances_ >= 0.0)
    assert np.isclose(model.feature_importances_.sum(), 1.0)


def test_reproducibility_with_fixed_seed(toy_binary):
    X, y = toy_binary
    first = RandomForestClassifier(n_estimators=25, random_state=99).fit(X, y)
    second = RandomForestClassifier(n_estimators=25, random_state=99).fit(X, y)
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X))


@pytest.mark.skipif(
    os.environ.get("RUN_PARALLEL_TESTS") != "1",
    reason="Set RUN_PARALLEL_TESTS=1 to run the multiprocessing integration test.",
)
def test_parallel_matches_sequential(breast_cancer_split):
    X_train, X_test, y_train, _ = breast_cancer_split
    sequential = RandomForestClassifier(
        n_estimators=20,
        max_depth=6,
        n_jobs=1,
        random_state=13,
    ).fit(X_train, y_train)
    parallel = RandomForestClassifier(
        n_estimators=20,
        max_depth=6,
        n_jobs=2,
        random_state=13,
    ).fit(X_train, y_train)
    np.testing.assert_array_equal(sequential.predict(X_test), parallel.predict(X_test))



def test_worker_count_resolution(toy_binary):
    X, y = toy_binary
    model = RandomForestClassifier(n_estimators=5, n_jobs=2, random_state=0)
    assert model._effective_n_jobs() == 2
    model.fit(X, y) if os.environ.get("RUN_PARALLEL_TESTS") == "1" else None


def test_oob_score_in_valid_range(breast_cancer_split):
    X_train, _, y_train, _ = breast_cancer_split
    model = RandomForestClassifier(
        n_estimators=60,
        oob_score=True,
        random_state=42,
    ).fit(X_train, y_train)
    assert 0.0 <= model.oob_score_ <= 1.0


def test_oob_requires_bootstrap():
    with pytest.raises(ValueError):
        RandomForestClassifier(bootstrap=False, oob_score=True)


def test_oob_property_requires_flag(toy_binary):
    X, y = toy_binary
    model = RandomForestClassifier(n_estimators=5, oob_score=False).fit(X, y)
    with pytest.raises(AttributeError):
        _ = model.oob_score_


@pytest.mark.parametrize("max_features", [1, 3, "sqrt", "log2", None])
def test_supported_max_features(toy_binary, max_features):
    X, y = toy_binary
    model = RandomForestClassifier(
        n_estimators=8,
        max_features=max_features,
        random_state=3,
    ).fit(X, y)
    assert model.predict(X[:4]).shape == (4,)


def test_sklearn_accuracy_parity_within_two_percent(breast_cancer_split):
    X_train, X_test, y_train, y_test = breast_cancer_split
    ours = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=42,
    ).fit(X_train, y_train)
    reference = SklearnRandomForest(
        n_estimators=100,
        max_features="sqrt",
        random_state=42,
    ).fit(X_train, y_train)
    ours_accuracy = np.mean(ours.predict(X_test) == y_test)
    reference_accuracy = np.mean(reference.predict(X_test) == y_test)
    assert abs(ours_accuracy - reference_accuracy) <= 0.02


def test_single_feature_and_single_class_edge_cases():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.zeros(20, dtype=int)
    model = RandomForestClassifier(
        n_estimators=5,
        max_features=1,
        random_state=0,
    ).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)
    assert model.predict_proba(X).shape == (20, 1)


def test_input_validation(toy_binary):
    X, y = toy_binary
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=0)
    with pytest.raises(ValueError):
        RandomForestClassifier(max_depth=0)
    with pytest.raises(ValueError):
        RandomForestClassifier(min_samples_split=1)
    with pytest.raises(ValueError):
        RandomForestClassifier(criterion="bad")
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=3, n_jobs=0).fit(X, y)
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=3).fit(X, y[:-1])
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=3).fit(np.empty((0, 2)), np.empty(0))
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=3).fit(X_nan, y)


def test_predict_validation(toy_binary):
    X, y = toy_binary
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    with pytest.raises(RuntimeError):
        model.predict(X)
    model.fit(X, y)
    with pytest.raises(ValueError):
        model.predict(np.zeros((3, X.shape[1] + 1)))
