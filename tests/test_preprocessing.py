import numpy as np
import pytest

from src.utils.preprocessing import (
    MeanImputer,
    PreprocessingPipeline,
    StandardScaler,
)


def test_mean_imputer_replaces_missing_values():
    X = np.array(
        [
            [1.0, np.nan],
            [3.0, 4.0],
            [5.0, 8.0],
        ]
    )

    imputer = MeanImputer()
    transformed = imputer.fit_transform(X)

    assert not np.isnan(transformed).any()
    assert np.isclose(
        transformed[0, 1],
        6.0,
    )
    assert np.allclose(
        imputer.statistics_,
        np.array([3.0, 6.0]),
    )


def test_mean_imputer_reuses_training_statistics():
    X_train = np.array(
        [
            [1.0, 2.0],
            [3.0, np.nan],
            [5.0, 8.0],
        ]
    )

    X_test = np.array(
        [
            [np.nan, 10.0],
        ]
    )

    imputer = MeanImputer()
    imputer.fit(X_train)

    transformed = imputer.transform(X_test)

    assert np.isclose(
        transformed[0, 0],
        3.0,
    )


def test_standard_scaler_produces_zero_mean_and_unit_variance():
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )

    scaler = StandardScaler()
    transformed = scaler.fit_transform(X)

    assert np.allclose(
        np.mean(transformed, axis=0),
        np.zeros(2),
        atol=1e-12,
    )

    assert np.allclose(
        np.std(transformed, axis=0),
        np.ones(2),
        atol=1e-12,
    )


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