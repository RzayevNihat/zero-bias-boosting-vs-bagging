import numpy as np
import pytest
from types import SimpleNamespace

from src.experiments import run_all
from src.experiments import unsupervised_analysis


def test_run_all_includes_baseline_experiment():
    assert any(
        experiment.name == "Baseline"
        and experiment.module == "src.experiments.baseline"
        for experiment in run_all.EXPERIMENTS
    )


def test_unsupervised_load_dataset_uses_shared_loader(monkeypatch):
    calls = {}

    def fake_load_digits_dataset(sample_limit=600):
        calls["sample_limit"] = sample_limit
        return SimpleNamespace(
            X=np.arange(12).reshape(6, 2),
            y=np.array([0, 1, 0, 1, 0, 1]),
        )

    monkeypatch.setattr(
        unsupervised_analysis,
        "load_digits_dataset",
        fake_load_digits_dataset,
    )

    X, y = unsupervised_analysis.load_dataset(
        sample_limit=4,
        random_state=7,
    )

    assert calls["sample_limit"] == 4
    assert X.shape[0] == 4
    assert y.shape[0] == 4


def test_unsupervised_load_dataset_rejects_non_positive_sample_limit():
    with pytest.raises(ValueError, match="sample_limit"):
        unsupervised_analysis.load_dataset(sample_limit=0)
