import numpy as np
import pytest

from src.evaluation.conformal import conformal_quantile, make_interval
from src.evaluation.metrics import interval_coverage


def test_quantile_is_the_empirical_residual_quantile():
    residuals = np.arange(1, 101, dtype=float)  # 1..100
    q = conformal_quantile(residuals, confidence=0.90)
    assert 89.0 <= q <= 92.0, q


def test_interval_is_symmetric_around_the_prediction():
    lower, upper = make_interval(np.array([10.0, 20.0]), q=3.0)
    assert np.allclose(lower, [7.0, 17.0])
    assert np.allclose(upper, [13.0, 23.0])


def test_empirical_coverage_matches_nominal_on_synthetic_data():
    """The whole point of conformal: the interval must mean what it says."""
    rng = np.random.RandomState(0)
    calibration_residuals = np.abs(rng.randn(2000) * 5)
    q = conformal_quantile(calibration_residuals, confidence=0.90)

    truth = rng.randn(4000) * 5 + 50
    predicted = np.full(4000, 50.0)
    lower, upper = make_interval(predicted, q)
    coverage = interval_coverage(truth, lower, upper)
    assert 0.87 <= coverage <= 0.93, coverage


def test_higher_confidence_gives_a_wider_interval():
    residuals = np.abs(np.random.RandomState(1).randn(1000) * 4)
    assert conformal_quantile(residuals, 0.95) > conformal_quantile(residuals, 0.80)


def test_rejects_an_impossible_confidence_level():
    with pytest.raises(ValueError, match="confidence"):
        conformal_quantile(np.array([1.0, 2.0]), confidence=1.5)
