"""Disable billing on the project when spend reaches the trigger.

Inside Google Cloud's Always Free tier this project's spend is exactly
zero, so the budget never moves. Any movement means a free allowance has
already been exceeded - the first dollar is the alarm, not the fifth.

The trigger is therefore 20% of a $5 budget. Firing at 100% would spend
the entire tolerance before the switch engaged, and billing data lags by
hours, so the overshoot would then carry past $5.
"""

from __future__ import annotations

import base64
import json
import os

TRIGGER_FRACTION = 0.2


def _decode(event: dict) -> dict | None:
    """Decode a Pub/Sub budget notification, or None if it is unusable."""
    try:
        raw = event.get("data")
        if not raw:
            return None
        return json.loads(base64.b64decode(raw).decode())
    except Exception:  # noqa: BLE001 - a parsing bug must not disable billing
        return None


def should_disable(event: dict) -> bool:
    """True when spend has reached the trigger.

    Returns False on any malformed input: accidentally cutting off a
    working service is worse than a delayed shutdown, and the higher
    budget thresholds still send email.
    """
    payload = _decode(event)
    if not payload:
        return False
    try:
        cost = float(payload.get("costAmount", 0))
        budget = float(payload.get("budgetAmount", 0))
    except (TypeError, ValueError):
        return False
    if budget <= 0:
        return False
    return cost >= budget * TRIGGER_FRACTION


def handle(event, context=None):  # pragma: no cover - needs GCP at runtime
    """Cloud Function entrypoint."""
    if not should_disable(event):
        return "under threshold, no action"

    project = os.environ["GCP_PROJECT_ID"]
    from googleapiclient import discovery

    billing = discovery.build("cloudbilling", "v1")
    name = f"projects/{project}"
    billing.projects().updateBillingInfo(
        name=name, body={"billingAccountName": ""}
    ).execute()
    return f"billing disabled for {project}"
