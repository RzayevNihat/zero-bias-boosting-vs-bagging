from pathlib import Path

import numpy as np
import pytest

from src.boosting.gradient_boosting import GradientBoostingClassifier
from src.utils.preprocessing import load_wdbc


def _make_blobs(n_per_class: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Create two mildly overlapping 2-D Gaussian blobs without sklearn."""
    rng = np.random.default_rng(seed)
    negative = rng.normal(loc=(-2.0, -2.0), scale=1.0, size=(n_per_class, 2))
    positive = rng.normal(loc=(2.0, 2.0), scale=1.0, size=(n_per_class, 2))
    X = np.vstack([negative, positive])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    order = rng.permutation(X.shape[0])
    return X[order], y[order]


def test_training_log_loss_decreases_across_iterations() -> None:
    X, y = _make_blobs()

    model = GradientBoostingClassifier(
        n_estimators=30,
        learning_rate=0.2,
        max_depth=2,
        random_state=0,
    ).fit(X, y)

    losses = model.train_log_loss_
    assert losses.shape == (30,)
    assert losses[-5:].mean() < losses[:5].mean()
    assert losses[-1] < losses[0]


def test_predict_proba_outputs_valid_probabilities() -> None:
    X, y = _make_blobs()

    model = GradientBoostingClassifier(n_estimators=20, random_state=1).fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (X.shape[0], 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_deterministic_with_fixed_random_state() -> None:
    X, y = _make_blobs()

    model_a = GradientBoostingClassifier(n_estimators=15, random_state=42).fit(X, y)
    model_b = GradientBoostingClassifier(n_estimators=15, random_state=42).fit(X, y)

    np.testing.assert_array_equal(model_a.predict(X), model_b.predict(X))
    np.testing.assert_allclose(model_a.predict_proba(X), model_b.predict_proba(X))
    np.testing.assert_allclose(model_a.train_log_loss_, model_b.train_log_loss_)


def test_model_beats_random_guessing() -> None:
    X, y = _make_blobs(n_per_class=80, seed=3)
    split = 120
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.15,
        max_depth=2,
        random_state=5,
    ).fit(X_train, y_train)

    assert model.score(X_test, y_test) > 0.75


def test_handles_very_small_dataset() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = GradientBoostingClassifier(
        n_estimators=10,
        max_depth=1,
        random_state=0,
    ).fit(X, y)

    assert len(model.estimators_) == 10
    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0, atol=1e-8)


def test_staged_predict_length_matches_n_estimators() -> None:
    X, y = _make_blobs(n_per_class=30, seed=4)
    n_estimators = 12

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        random_state=6,
    ).fit(X, y)
    staged = list(model.staged_predict(X))

    assert len(staged) == n_estimators
    np.testing.assert_array_equal(staged[-1], model.predict(X))


def test_rejects_multiclass_targets() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 1, 1, 2, 2])

    with pytest.raises(ValueError):
        GradientBoostingClassifier(n_estimators=5).fit(X, y)


def test_predict_before_fit_raises() -> None:
    model = GradientBoostingClassifier(n_estimators=5)

    with pytest.raises(RuntimeError):
        model.predict(np.array([[0.0], [1.0]]))


def test_gbm_fits_downloaded_wdbc_subset() -> None:
    if not Path("data/wdbc.data").exists():
        pytest.skip("download_data.sh has not been run yet.")

    dataset = load_wdbc()
    indices = np.concatenate(
        [np.flatnonzero(dataset.y == label)[:40] for label in np.unique(dataset.y)]
    )
    X = dataset.X[indices]
    y = dataset.y[indices]

    model = GradientBoostingClassifier(
        n_estimators=20,
        learning_rate=0.2,
        max_depth=2,
        random_state=19,
    ).fit(X, y)

    assert len(model.estimators_) == 20
    assert model.score(X, y) >= 0.8
    assert model.train_log_loss_[-1] < model.train_log_loss_[0]
