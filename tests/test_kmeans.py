import numpy as np

from src.unsupervised.kmeans import KMeans


def test_kmeans_finds_expected_number_of_clusters():
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

    model = KMeans(
        n_clusters=2,
        n_init=5,
        random_state=42,
    )

    labels = model.fit_predict(X)

    assert labels.shape == (6,)
    assert len(np.unique(labels)) == 2
    assert model.centroids_.shape == (2, 2)


def test_kmeans_converges_before_max_iterations():
    rng = np.random.default_rng(42)

    cluster_1 = rng.normal(
        loc=[0.0, 0.0],
        scale=0.2,
        size=(40, 2),
    )
    cluster_2 = rng.normal(
        loc=[4.0, 4.0],
        scale=0.2,
        size=(40, 2),
    )

    X = np.vstack([cluster_1, cluster_2])

    model = KMeans(
        n_clusters=2,
        max_iter=100,
        n_init=5,
        random_state=42,
    )

    model.fit(X)

    assert model.n_iter_ is not None
    assert model.n_iter_ <= 100
    assert model.inertia_ is not None
    assert model.inertia_ >= 0


def test_predict_returns_valid_cluster_labels():
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [5.0, 5.0],
            [5.2, 5.1],
        ]
    )

    X_test = np.array(
        [
            [0.1, 0.1],
            [5.1, 5.1],
        ]
    )

    model = KMeans(
        n_clusters=2,
        n_init=5,
        random_state=42,
    )

    model.fit(X_train)
    predictions = model.predict(X_test)

    assert predictions.shape == (2,)
    assert np.all(predictions >= 0)
    assert np.all(predictions < 2)


def test_elbow_curve_returns_decreasing_inertia():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 2))

    cluster_values, inertias = KMeans.elbow_curve(
        X,
        cluster_range=range(1, 5),
        n_init=5,
        random_state=42,
    )

    assert np.array_equal(
        cluster_values,
        np.array([1, 2, 3, 4]),
    )
    assert inertias.shape == (4,)
    assert np.all(np.diff(inertias) <= 1e-10)