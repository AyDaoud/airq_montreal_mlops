import numpy as np

from src.models.classifier import (
    COST_RATIO,
    build_classifier,
    choose_threshold,
    expected_cost,
)


def test_cost_ratio_is_five_to_one():
    """A missed exceedance costs 5x a false alarm - a public-health call,
    stated rather than buried in a default."""
    assert COST_RATIO == 5.0


def test_expected_cost_weights_misses_more_heavily_than_false_alarms():
    y = np.array([0, 0, 1, 1])
    one_miss = np.array([0, 0, 0, 1])
    one_false_alarm = np.array([1, 0, 1, 1])
    assert expected_cost(y, one_miss) > expected_cost(y, one_false_alarm)


def test_choose_threshold_returns_a_probability():
    rng = np.random.RandomState(0)
    y = (rng.rand(500) < 0.1).astype(int)
    proba = np.clip(y * 0.5 + rng.rand(500) * 0.5, 0, 1)
    threshold = choose_threshold(y, proba)
    assert 0.0 < threshold < 1.0


def test_a_higher_cost_ratio_lowers_the_threshold():
    """Penalising misses harder should make the alarm easier to trip."""
    rng = np.random.RandomState(1)
    y = (rng.rand(800) < 0.1).astype(int)
    proba = np.clip(y * 0.4 + rng.rand(800) * 0.6, 0, 1)
    assert choose_threshold(y, proba, cost_ratio=20.0) <= choose_threshold(
        y, proba, cost_ratio=2.0
    )


def test_build_classifier_is_balanced_for_the_rare_positive_class():
    model = build_classifier()
    assert model.get_params()["class_weight"] == "balanced"
