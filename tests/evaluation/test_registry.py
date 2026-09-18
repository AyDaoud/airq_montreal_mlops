import numpy as np
import pandas as pd

from src.evaluation.registry import CANDIDATES, Candidate, fit_predict


class _AlwaysOne:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(X))


def test_registry_declares_the_measured_winner():
    """Huber on a delta target scored mean MASE 0.899, winning 5/5 folds."""
    assert "huber_delta" in CANDIDATES
    assert CANDIDATES["huber_delta"].target == "delta"


def test_registry_spans_more_than_one_model_family():
    families = {type(c.build()).__name__ for c in CANDIDATES.values()}
    assert len(families) >= 3, f"only {families}"


def test_every_candidate_declares_a_valid_target_framing():
    for name, candidate in CANDIDATES.items():
        assert candidate.target in ("level", "delta"), name


def test_fit_predict_on_a_delta_candidate_recovers_the_level():
    """A delta model must have today's value added back before scoring."""
    rng = np.random.RandomState(0)
    train = pd.DataFrame({"iqa": rng.rand(200) * 10, "lag_1": rng.rand(200) * 10})
    train["target"] = train["iqa"] + 1.0
    test = train.iloc[:20].copy()

    candidate = Candidate(
        name="const_delta", target="delta", build=lambda: _AlwaysOne()
    )
    preds = fit_predict(candidate, train, test, ["iqa", "lag_1"])
    assert np.allclose(preds, test["iqa"] + 1.0)


def test_fit_predict_on_a_level_candidate_returns_the_raw_prediction():
    train = pd.DataFrame({"iqa": [1.0, 2.0], "lag_1": [1.0, 2.0], "target": [5.0, 5.0]})
    candidate = Candidate(
        name="const_level", target="level", build=lambda: _AlwaysOne()
    )
    preds = fit_predict(candidate, train, train, ["iqa", "lag_1"])
    assert np.allclose(preds, 1.0)
