import base64
import importlib.util
import json
from pathlib import Path

import pytest

FUNCTION = Path("terraform/00-guardrails/function/main.py")


def _load():
    spec = importlib.util.spec_from_file_location("killswitch", FUNCTION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event(cost, budget, threshold=None):
    payload = {"costAmount": cost, "budgetAmount": budget}
    if threshold is not None:
        payload["alertThresholdExceeded"] = threshold
    return {"data": base64.b64encode(json.dumps(payload).encode()).decode()}


def test_should_disable_when_spend_reaches_the_trigger():
    """Trigger is $1 on a $5 budget - inside the free tier spend is $0,
    so any real spend is already anomalous."""
    assert _load().should_disable(_event(1.0, 5.0)) is True


def test_should_disable_above_the_trigger():
    assert _load().should_disable(_event(2.4, 5.0)) is True


def test_should_not_disable_below_the_trigger():
    assert _load().should_disable(_event(0.42, 5.0)) is False


def test_zero_spend_is_the_normal_case_and_never_triggers():
    """Normal operation never leaves the free tier, so cost stays at 0."""
    assert _load().should_disable(_event(0.0, 5.0)) is False


def test_trigger_fraction_is_twenty_percent():
    assert _load().TRIGGER_FRACTION == pytest.approx(0.2)


def test_a_malformed_message_does_not_crash_and_does_not_disable():
    """A parsing bug must not take the service down by accident."""
    module = _load()
    assert module.should_disable({"data": "not-base64"}) is False
    assert module.should_disable({}) is False


def test_a_missing_budget_amount_does_not_divide_by_zero():
    assert _load().should_disable(_event(5.0, 0.0)) is False
