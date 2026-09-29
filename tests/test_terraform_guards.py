"""The five settings that bound the bill.

Inside the free tier this deployment costs nothing. These settings are
what keep it there, so each gets an assertion rather than a comment.
A silent edit to any of them must fail the build.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

APP = Path("terraform/10-app/main.tf")
GUARDRAILS = Path("terraform/00-guardrails/main.tf")
APP_VARS = Path("terraform/10-app/variables.tf")
WORKFLOW = Path(".github/workflows/deploy-cloudrun.yml")


def test_min_instances_is_zero():
    """Idle instances that are not minimum instances are not charged."""
    assert re.search(r"min_instance_count\s*=\s*0\b", APP.read_text())


def test_max_instances_is_two():
    """Bounds the burn rate to ~$0.19/hour, which is what makes the kill
    switch's hours-long billing lag survivable."""
    assert re.search(r"max_instance_count\s*=\s*2\b", APP.read_text())


def test_region_is_a_free_tier_region():
    """Cloud Run and Cloud Storage free tiers do not apply to
    northamerica-northeast1, however tempting Montreal is."""
    assert "us-central1" in APP_VARS.read_text()


def test_registry_keeps_only_two_versions():
    """The serving image is ~540 MB uncompressed against a 0.5 GB free
    tier, and CI tags every push by commit SHA."""
    text = APP.read_text()
    assert "cleanup_policies" in text
    assert re.search(r"keep_count\s*=\s*2\b", text)


def test_killswitch_fires_at_twenty_percent():
    """Firing at 100% would spend the whole tolerance before the switch
    engaged, and the billing lag would carry it past the ceiling."""
    assert re.search(r"threshold_percent\s*=\s*0\.2\b", GUARDRAILS.read_text())


def test_the_workflow_deploys_with_no_traffic_first():
    """A revision that fails its smoke test must never serve a user."""
    text = WORKFLOW.read_text()
    assert "--no-traffic" in text
    assert text.index("--no-traffic") < text.index("update-traffic")


def test_the_workflow_smoke_tests_predict_not_just_health():
    """Blocker B4: /health passed while /predict raised FileNotFoundError."""
    text = WORKFLOW.read_text()
    assert "/predict" in text
    assert "/health" in text


def test_the_workflow_carries_no_key_material():
    text = WORKFLOW.read_text().lower()
    for forbidden in ("credentials_json", "private_key", "begin private"):
        assert forbidden not in text, forbidden


def test_the_workflow_requests_an_id_token():
    """Without id-token: write, WIF cannot mint a token and the deploy
    falls back to needing a static key."""
    assert re.search(r"id-token:\s*write", WORKFLOW.read_text())


def test_the_wif_provider_is_restricted_to_this_repository():
    """Without the condition, ANY GitHub repository could mint tokens for
    the deploy service account."""
    text = APP.read_text()
    assert "attribute_condition" in text
    assert "assertion.repository ==" in text


def test_no_terraform_state_or_real_tfvars_is_tracked():
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    leaked = [
        f
        for f in tracked
        if f.endswith(".tfstate")
        or f.endswith(".tfstate.backup")
        or (f.endswith(".tfvars") and not f.endswith(".tfvars.example"))
    ]
    assert leaked == [], f"state or real tfvars committed: {leaked}"


def test_provider_lock_files_are_committed():
    """Without the lock, terraform init could resolve a different schema
    than the one these guards were validated against."""
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    assert "terraform/00-guardrails/.terraform.lock.hcl" in tracked
    assert "terraform/10-app/.terraform.lock.hcl" in tracked
