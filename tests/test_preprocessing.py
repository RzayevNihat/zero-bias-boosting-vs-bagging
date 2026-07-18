"""Tests for preprocessing utilities and dataset loading."""

import numpy as np
import pytest
import sklearn.datasets as sklearn_datasets

from src.utils.preprocessing import (
    DatasetBundle,
    MeanImputer,
    PreprocessingPipeline,
    StandardScaler,
    _coerce_feature_matrix,
    handle_missing_values,
    load_digits_dataset,
    load_project_datasets,
    train_test_split,
)


def test_handle_missing_values_imputes_column_median():
    X = np.array([[1.0, np.nan], [2.0, 4.0], [3.0, 6.0]])
    X_clean = handle_missing_values(X)
    assert not np.isnan(X_clean).any()
    assert X_clean[0, 1] == 5.0  # median of [4.0, 6.0]


def test_handle_missing_values_does_not_mutate_input():
    X = np.array([[1.0, np.nan], [2.0, 4.0]])
    X_copy = X.copy()
    handle_missing_values(X)
    # original array (with NaN) should be untouched
    assert np.isnan(X[0, 1])
    assert np.array_equal(X, X_copy, equal_nan=True)


def test_handle_missing_values_all_nan_column_raises():
    X = np.array([[np.nan], [np.nan]])
    with pytest.raises(ValueError):
        handle_missing_values(X)


def test_standard_scaler_produces_zero_mean_unit_variance():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-9)


def test_standard_scaler_handles_constant_column_without_nan():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])  # first column constant
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    assert not np.isnan(X_scaled).any()
    assert np.allclose(X_scaled[:, 0], 0.0)


def test_standard_scaler_transform_before_fit_raises():
    scaler = StandardScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(np.array([[1.0, 2.0]]))


def test_standard_scaler_test_set_uses_train_statistics():
    """Reuse discipline: fitting only on train, then transforming test
    with those same statistics -- never refit on test data."""
    X_train = np.array([[1.0], [2.0], [3.0]])
    X_test = np.array([[100.0]])  # far outside training range
    scaler = StandardScaler().fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Should reflect train mean/std, not be re-centered around X_test itself.
    expected = (100.0 - X_train.mean()) / X_train.std()
    assert abs(X_test_scaled[0, 0] - expected) < 1e-9


def test_train_test_split_sizes_and_no_overlap():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    assert len(X_train) + len(X_test) == 10
    assert len(y_train) + len(y_test) == 10
    # No sample should appear in both splits.
    train_rows = {tuple(row) for row in X_train}
    test_rows = {tuple(row) for row in X_test}
    assert train_rows.isdisjoint(test_rows)


def test_train_test_split_stratify_preserves_class_ratio():
    X = np.arange(200).reshape(100, 2).astype(float)
    y = np.array([0] * 90 + [1] * 10)  # imbalanced 90/10
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    # The minority class must appear in the test set (a plain random
    # split on this ratio could easily miss it).
    assert (y_test == 1).sum() > 0
    train_ratio = (y_train == 1).mean()
    test_ratio = (y_test == 1).mean()
    assert abs(train_ratio - test_ratio) < 0.1


def test_standard_scaler_handles_constant_features():
    X = np.array(
        [
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
        ]
    )

    scaler = StandardScaler()
    transformed = scaler.fit_transform(X)

    assert np.allclose(
        transformed[:, 1],
        0.0,
    )

    assert np.isclose(
        scaler.scale_[1],
        1.0,
    )


def test_preprocessing_pipeline_imputes_and_scales():
    X = np.array(
        [
            [1.0, np.nan],
            [2.0, 4.0],
            [3.0, 8.0],
            [4.0, 12.0],
        ]
    )

    pipeline = PreprocessingPipeline()
    transformed = pipeline.fit_transform(X)

    assert transformed.shape == X.shape
    assert not np.isnan(transformed).any()

    assert np.allclose(
        np.mean(transformed, axis=0),
        np.zeros(2),
        atol=1e-12,
    )


def test_pipeline_transform_before_fit_raises_error():
    pipeline = PreprocessingPipeline()

    X = np.array(
        [
            [1.0, 2.0],
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="must be fitted",
    ):
        pipeline.transform(X)


def test_mean_imputer_rejects_all_missing_feature():
    X = np.array(
        [
            [1.0, np.nan],
            [2.0, np.nan],
        ]
    )

    imputer = MeanImputer()

    with pytest.raises(
        ValueError,
        match="only missing values",
    ):
        imputer.fit(X)


def test_scaler_rejects_different_feature_count():
    X_train = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    X_test = np.array(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    with pytest.raises(
        ValueError,
        match="same number of features",
    ):
        scaler.transform(X_test)


def test_mean_imputer_replaces_missing_values():
    X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 8.0]])
    imputer = MeanImputer()
    transformed = imputer.fit_transform(X)
    assert not np.isnan(transformed).any()
    assert np.isclose(transformed[0, 1], 6.0)
    assert np.allclose(imputer.statistics_, np.array([3.0, 6.0]))


def test_mean_imputer_reuses_training_statistics():
    X_train = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 8.0]])
    X_test = np.array([[np.nan, 10.0]])
    imputer = MeanImputer()
    imputer.fit(X_train)
    transformed = imputer.transform(X_test)
    assert np.isclose(transformed[0, 0], 3.0)


def test_train_test_split_is_reproducible_with_same_seed():
    X = np.arange(40).reshape(20, 2).astype(float)
    y = np.array([0, 1] * 10)
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X, y, test_size=0.2, random_state=7)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y, test_size=0.2, random_state=7)
    assert np.array_equal(X_train_a, X_train_b)
    assert np.array_equal(y_train_a, y_train_b)
    assert np.array_equal(X_test_a, X_test_b)
    assert np.array_equal(y_test_a, y_test_b)


def test_load_project_datasets_returns_dataset_bundle(tmp_path):
    dataset_path = tmp_path / "wdbc.data"
    dataset_path.write_text(
        "1,M,1.0,2.0\n2,B,3.0,4.0\n",
        encoding="utf-8",
    )

    bundles = load_project_datasets(
        names=("wdbc",),
        data_dir=tmp_path,
    )

    assert len(bundles) == 1
    assert isinstance(bundles[0], DatasetBundle)
    assert bundles[0].name == "wdbc"
    assert bundles[0].X.shape == (2, 2)
    assert np.array_equal(bundles[0].y, np.array([1, 0]))


def test_load_wdbc_falls_back_to_sklearn_when_local_file_missing(tmp_path):
    dataset = load_project_datasets(
        names=("wdbc",),
        data_dir=tmp_path,
    )[0]

    assert dataset.name == "wdbc"
    assert dataset.X.shape[0] > 0
    assert dataset.y.shape[0] > 0


def test_load_adult_imputes_missing_features(tmp_path):
    dataset_path = tmp_path / "adult.data"
    dataset_path.write_text("?,0\n2,1\n", encoding="utf-8")

    dataset = load_project_datasets(
        names=("adult",),
        data_dir=tmp_path,
    )[0]

    assert dataset.name == "adult"
    assert not np.isnan(dataset.X).any()
    assert np.allclose(dataset.X[0, 0], 2.0)


def test_load_digits_dataset_prefers_local_csv_when_available(tmp_path, monkeypatch):
    dataset_path = tmp_path / "digits.csv"
    dataset_path.write_text("0,1,2\n1,3,4\n", encoding="utf-8")

    def fail_load_digits():
        raise AssertionError("scikit-learn fallback should not be used")

    monkeypatch.setattr(sklearn_datasets, "load_digits", fail_load_digits)

    dataset = load_digits_dataset(path=dataset_path)

    assert dataset.name == "digits"
    assert dataset.X.shape == (2, 2)
    assert np.array_equal(dataset.y, np.array([0, 1]))


def test_coerce_feature_matrix_encodes_categorical_values_consistently():
    values = np.array(
        [[" State-gov", " Bachelors"], ["Self-emp-not-inc", "HS-grad"]],
        dtype=str,
    )

    matrix = _coerce_feature_matrix(values)

    assert matrix.shape == (2, 2)
    assert np.unique(matrix[:, 0]).tolist() == [0.0, 1.0]
    assert np.unique(matrix[:, 1]).tolist() == [0.0, 1.0]
    assert np.all(np.isin(matrix[:, 0], [0.0, 1.0]))
    assert np.all(np.isin(matrix[:, 1], [0.0, 1.0]))
