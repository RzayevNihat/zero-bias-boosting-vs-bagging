import numpy as np

from src.unsupervised.dbscan import DBSCAN


def test_dbscan_finds_clusters():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.1],
            [5.0, 5.0],
            [5.1, 5.2],
            [4.9, 5.1],
        ]
    )

    model = DBSCAN(
        eps=0.5,
        min_samples=2,
    )

    labels = model.fit_predict(X)

    assert len(np.unique(labels)) == 2
    assert model.n_clusters_ == 2


def test_dbscan_detects_noise():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [5.0, 5.0],
            [20.0, 20.0],
        ]
    )

    model = DBSCAN(
        eps=0.3,
        min_samples=2,
    )

    labels = model.fit_predict(X)

    assert -1 in labels


def test_core_samples_exist():
    X = np.array(
        [
            [0.0, 0.0],
            [0.05, 0.05],
            [0.1, 0.1],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],
        ]
    )

    model = DBSCAN(
        eps=0.2,
        min_samples=2,
    )

    model.fit(X)

    assert len(model.core_sample_indices_) > 0


def test_k_distance_curve():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(50, 2))

    order, distances = DBSCAN.k_distance_curve(
        X,
        k=3,
    )

    assert order.shape == (50,)
    assert distances.shape == (50,)
    assert np.all(np.diff(distances) >= 0)