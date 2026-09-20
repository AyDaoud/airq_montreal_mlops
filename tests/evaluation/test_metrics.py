import numpy as np
import pytest

from src.evaluation.metrics import (
    classification_report_at_threshold,
    interval_coverage,
    mase,
    regression_metrics,
)


def test_regression_metrics_are_correct():
    out = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert out["mae"] == pytest.approx(2.0 / 3.0)
    assert out["rmse"] == pytest.approx(np.sqrt(4.0 / 3.0))


def test_mase_of_persistence_against_itself_is_exactly_one():
    y = np.array([5.0, 7.0, 9.0, 4.0])
    baseline = np.array([4.0, 5.0, 7.0, 9.0])
    assert mase(y, baseline, baseline) == pytest.approx(1.0)


def test_mase_below_one_means_the_model_beat_the_baseline():
    y = np.array([10.0, 10.0, 10.0])
    baseline = np.array([8.0, 8.0, 8.0])  # MAE 2
    model = np.array([9.0, 9.0, 9.0])  # MAE 1
    assert mase(y, model, baseline) == pytest.approx(0.5)


def test_mase_raises_when_the_baseline_is_perfect():
    """A zero-error baseline makes the ratio undefined; fail loudly."""
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="baseline"):
        mase(y, np.array([1.5, 2.5]), y)


def test_classification_report_at_threshold():
    y = [0, 0, 1, 1]
    proba = [0.1, 0.4, 0.6, 0.9]
    out = classification_report_at_threshold(y, proba, threshold=0.5)
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(1.0)
    assert out["n_positive"] == 2
    assert out["n_flagged"] == 2


def test_pr_auc_of_a_random_scorer_approximates_the_base_rate():
    rng = np.random.RandomState(0)
    y = (rng.rand(4000) < 0.05).astype(int)
    out = classification_report_at_threshold(y, rng.rand(4000), threshold=0.5)
    assert out["pr_auc"] == pytest.approx(0.05, abs=0.02)


def test_interval_coverage_counts_containment():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    lower = np.array([0.0, 1.0, 2.0, 0.0])
    upper = np.array([2.0, 3.0, 4.0, 5.0])
    assert interval_coverage(y, lower, upper) == pytest.approx(0.75)
