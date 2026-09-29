# Spec D1 — Safe Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A publicly reachable Cloud Run URL serving the model, whose cost is bounded to roughly $2 in the worst realistic case and automatically terminated at $1 of spend.

**Architecture:** Two Terraform root modules with separate state — `00-guardrails` (budget, Pub/Sub, a billing-disable function) applied before `10-app` (Artifact Registry, Cloud Run, Workload Identity Federation). GitHub Actions deploys with no long-lived credentials, publishes a revision with no traffic, smoke-tests it, and only then migrates traffic.

**Tech Stack:** Terraform 1.13.3, Google provider, Cloud Run v2, Artifact Registry, Cloud Functions v2 (Python), Pub/Sub, GitHub Actions with Workload Identity Federation.

**Reference spec:** `docs/superpowers/specs/2026-09-29-spec-d1-safe-deployment-design.md`

---

## Environment notes — read before starting

- Branch `spec-d1/safe-deployment`, **based on `spec-c/observability`, NOT on `main`.** Baseline suite: **190 passed** once Task 1 lands (183 from Spec C plus 7 new).

> **This bit us once.** The branch was originally cut from `main`, which does not yet contain Spec C. The baseline came out at 133 instead of 183, and more importantly `src/serving/app.py` lacked the `/metrics` endpoint and prediction logging entirely — deploying from it would have shipped the pre-observability app and made all of Spec C invisible in production. Caught because a subagent stashed its work and measured the real baseline instead of trusting the number in its brief. Fixed by `git rebase --onto origin/spec-c/observability origin/main`. **Spec C must be merged before, or together with, this branch.**
- Always run python as `.venv/bin/python`.
- **Terraform 1.13.3 is installed. `gcloud` is NOT, and there are NO GCP credentials.**
- Therefore: `terraform fmt`, `terraform validate` and `terraform init -backend=false` are runnable here. **`terraform plan` and `terraform apply` are NOT** — both need credentials. Do not attempt them; the user applies.
- Tests must stay **offline**. No test may contact GCP.
- `terraform.tfvars` holds the user's project ID, billing account ID and alert email. It is **gitignored** and must never be committed.

## The five settings that bound the bill

These are the entire cost story. Task 5 tests that all five are present, and the test must fail if any is changed.

| Setting | Value | Prevents |
|---|---|---|
| `min_instance_count` | `0` | Idle billing — *"idle instances that are not minimum instances are not charged"* |
| `max_instance_count` | `2` | Burn rate above ~$0.19/hour |
| `location` / `region` | `us-central1` | Falling outside the free-tier regions |
| cleanup policy | keep 2 versions | Image storage past the 0.5 GB free tier (the image is ~540 MB uncompressed) |
| kill-switch threshold | `0.2` of a $5 budget = **$1** | Spending the whole tolerance before the switch engages |

## File structure

| File | Responsibility |
|---|---|
| `terraform/00-guardrails/main.tf` | Budget, Pub/Sub topic, billing-disable function, its service account |
| `terraform/00-guardrails/variables.tf` | project_id, billing_account_id, alert_email, budget_amount |
| `terraform/00-guardrails/function/main.py` | The kill switch's decision logic |
| `terraform/10-app/main.tf` | Artifact Registry, Cloud Run, IAM, WIF pool |
| `terraform/10-app/variables.tf` | project_id, region, image, service_name |
| `.github/workflows/deploy-cloudrun.yml` | Build, push, deploy no-traffic, smoke test, migrate |
| `docs/RUNBOOK.md` | Kill-switch recovery, rollback by SHA, teardown |
| `tests/test_terraform_guards.py` | Asserts the five cost settings and that no key material is committed |

---

## Task 1: The kill switch's decision logic

Built and tested **first**, in pure Python, before any Terraform exists. The logic that decides whether to cut off billing deserves tests that do not require a cloud account.

**Files:**
- Create: `terraform/00-guardrails/function/main.py`, `terraform/00-guardrails/function/requirements.txt`
- Test: `tests/test_billing_killswitch.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_billing_killswitch.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/ayman/airq_montreal_mlops
.venv/bin/python -m pytest tests/test_billing_killswitch.py -v
```
Expected: FAIL — the function file does not exist.

- [ ] **Step 3: Implement `terraform/00-guardrails/function/main.py`**

```python
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
```

Create `terraform/00-guardrails/function/requirements.txt`:

```
google-api-python-client==2.149.0
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_billing_killswitch.py -v
```
Expected: 7 passed

Note `handle` is marked `# pragma: no cover` — it needs a live GCP client. The decision logic in `should_disable` is what is tested, and it is the part that can be wrong in a way that costs money or takes down a working service.

- [ ] **Step 5: Lint, format, full suite, commit**

```bash
cd /home/ayman/airq_montreal_mlops
.venv/bin/flake8 src tests scripts terraform; echo "flake8 exit=$?"
.venv/bin/black --check src tests scripts terraform; echo "black exit=$?"
.venv/bin/python -m pytest
git add terraform/00-guardrails/function tests/test_billing_killswitch.py
if git commit -m "feat(guardrails): add the billing kill switch decision logic

Fires at 20% of the budget. Inside the free tier spend is exactly zero,
so any movement means an allowance has already been exceeded - the first
dollar is the alarm, not the fifth.

Returns False on any malformed input: accidentally cutting off a working
service is worse than a delayed shutdown, and the higher thresholds
still send email.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"; then
  echo "COMMIT OK: \$(git log --oneline -1)"
else
  echo "COMMIT FAILED:"; git status --short
fi
```

Use the `if git commit; then` form throughout this plan — pre-commit hooks abort on reformat, and `git commit ...; echo done` reports a success that did not happen.

Expected suite: 190 passed. Do NOT push yet.

---

## Task 2: `terraform/00-guardrails/` — the budget and the kill switch

**Files:**
- Create: `terraform/00-guardrails/{main.tf,variables.tf,outputs.tf,versions.tf}`
- Create: `terraform/00-guardrails/terraform.tfvars.example`
- Modify: `.gitignore`

- [ ] **Step 1: Ignore state and real tfvars before writing any Terraform**

Do this first. Terraform state can contain secrets, and `terraform.tfvars` holds the user's billing account ID.

Append to `.gitignore`:

```
# Terraform
**/.terraform/
*.tfstate
*.tfstate.*
*.tfvars
!*.tfvars.example
crash.log
```

- [ ] **Step 2: Write `terraform/00-guardrails/versions.tf`**

```hcl
terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
```

- [ ] **Step 3: Write `terraform/00-guardrails/variables.tf`**

```hcl
variable "project_id" {
  description = "The dedicated, disposable GCP project for this deployment."
  type        = string
}

variable "project_number" {
  description = "Numeric project number, shown on the project dashboard."
  type        = string
}

variable "billing_account_id" {
  description = "Billing account, format XXXXXX-XXXXXX-XXXXXX."
  type        = string
}

variable "alert_email" {
  description = "Address for budget alert email."
  type        = string
}

variable "region" {
  description = "Free-tier region. Do not change without checking the free tier still applies."
  type        = string
  default     = "us-central1"
}

variable "budget_amount" {
  description = "Outer budget in USD. The kill switch fires at 20% of this."
  type        = number
  default     = 5
}
```

- [ ] **Step 4: Write `terraform/00-guardrails/main.tf`**

```hcl
# Guardrails. Applied BEFORE terraform/10-app, in a separate state, so
# deploying without a cost cap is a deliberate act rather than an oversight.

resource "google_project_service" "required" {
  for_each = toset([
    "cloudbilling.googleapis.com",
    "cloudbudgets.googleapis.com",
    "cloudfunctions.googleapis.com",
    "cloudbuild.googleapis.com",
    "pubsub.googleapis.com",
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "iamcredentials.googleapis.com",
  ])
  service            = each.key
  disable_on_destroy = false
}

resource "google_pubsub_topic" "budget" {
  name       = "airq-budget-alerts"
  depends_on = [google_project_service.required]
}

# The budget is $5, but the function acts on the 20% threshold ($1).
# Inside the free tier spend is exactly $0, so any movement is already
# anomalous. The 50% and 100% thresholds exist only as escalating email.
resource "google_billing_budget" "cap" {
  billing_account = var.billing_account_id
  display_name    = "airq-montreal hard cap"

  budget_filter {
    projects = ["projects/${var.project_number}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.budget_amount)
    }
  }

  threshold_rules {
    threshold_percent = 0.2 # the kill switch acts here
  }
  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 1.0
  }

  all_updates_rule {
    pubsub_topic                     = google_pubsub_topic.budget.id
    schema_version                   = "1.0"
    monitoring_notification_channels = []
    disable_default_iam_recipients   = false
  }
}

resource "google_service_account" "killswitch" {
  account_id   = "airq-killswitch"
  display_name = "Disables billing when the budget trigger is reached"
}

# roles/billing.admin is what lets the function detach billing. It is broad,
# which is precisely why this deployment lives in a dedicated, disposable
# project (decision D1-7).
resource "google_billing_account_iam_member" "killswitch" {
  billing_account_id = var.billing_account_id
  role               = "roles/billing.admin"
  member             = "serviceAccount:${google_service_account.killswitch.email}"
}

resource "google_storage_bucket" "function_source" {
  name                        = "${var.project_id}-killswitch-src"
  location                    = "US"
  uniform_bucket_level_access = true
  force_destroy               = true
}

data "archive_file" "killswitch" {
  type        = "zip"
  source_dir  = "${path.module}/function"
  output_path = "${path.module}/.build/killswitch.zip"
}

resource "google_storage_bucket_object" "killswitch" {
  name   = "killswitch-${data.archive_file.killswitch.output_md5}.zip"
  bucket = google_storage_bucket.function_source.name
  source = data.archive_file.killswitch.output_path
}

resource "google_cloudfunctions2_function" "killswitch" {
  name     = "airq-billing-killswitch"
  location = var.region

  build_config {
    runtime     = "python312"
    entry_point = "handle"
    source {
      storage_source {
        bucket = google_storage_bucket.function_source.name
        object = google_storage_bucket_object.killswitch.name
      }
    }
  }

  service_config {
    max_instance_count    = 1
    available_memory      = "256M"
    timeout_seconds       = 60
    service_account_email = google_service_account.killswitch.email
    environment_variables = {
      GCP_PROJECT_ID = var.project_id
    }
  }

  event_trigger {
    trigger_region = var.region
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.budget.id
    retry_policy   = "RETRY_POLICY_RETRY"
  }

  depends_on = [google_project_service.required]
}
```

Add `archive` to `versions.tf` required_providers:

```hcl
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
```

- [ ] **Step 5: Write `terraform/00-guardrails/outputs.tf`**

```hcl
output "budget_topic" {
  description = "Pub/Sub topic receiving budget notifications."
  value       = google_pubsub_topic.budget.name
}

output "killswitch_service_account" {
  description = "Service account the kill switch runs as."
  value       = google_service_account.killswitch.email
}

output "trigger_amount_usd" {
  description = "Spend at which billing is disabled."
  value       = var.budget_amount * 0.2
}
```

- [ ] **Step 6: Write `terraform/00-guardrails/terraform.tfvars.example`**

```hcl
# Copy to terraform.tfvars and fill in. terraform.tfvars is gitignored.
# Find these in the Cloud console:
#   project_id / project_number : the project dashboard
#   billing_account_id          : Billing > Account management
project_id         = "airq-montreal-XXXXXX"
project_number     = "123456789012"
billing_account_id = "XXXXXX-XXXXXX-XXXXXX"
alert_email        = "you@example.com"
```

- [ ] **Step 7: Validate — this is the limit of what can be checked here**

```bash
cd /home/ayman/airq_montreal_mlops/terraform/00-guardrails
terraform fmt -check -diff .
terraform init -backend=false
terraform validate
```

All three must pass. **Do NOT run `terraform plan` or `apply`** — both need credentials this machine does not have. The user applies.

If `terraform init -backend=false` needs to download providers and the network blocks it, report that; do not skip validation silently.

- [ ] **Step 8: Confirm nothing sensitive is staged, then commit**

```bash
cd /home/ayman/airq_montreal_mlops
git status --short | grep -E "tfstate|\.tfvars$|\.terraform/" && echo "PROBLEM: sensitive file staged" || echo "ok: nothing sensitive"
git add terraform/00-guardrails .gitignore
if git commit -m "feat(guardrails): terraform for the budget and billing kill switch

Separate root module with its own state, applied before terraform/10-app,
so deploying without a cost cap is a deliberate act.

The budget is \$5 but the function acts on the 20% threshold. State files
and terraform.tfvars are gitignored - the latter holds the billing
account id.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"; then
  echo "COMMIT OK: \$(git log --oneline -1)"
else
  echo "COMMIT FAILED:"; git status --short
fi
```

---

## Task 3: `terraform/10-app/` — Artifact Registry, Cloud Run, WIF

**Files:**
- Create: `terraform/10-app/{main.tf,variables.tf,outputs.tf,versions.tf}`
- Create: `terraform/10-app/terraform.tfvars.example`

- [ ] **Step 1: Write `terraform/10-app/versions.tf`**

Same as the guardrails module but without the archive provider:

```hcl
terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
```

- [ ] **Step 2: Write `terraform/10-app/variables.tf`**

```hcl
variable "project_id" {
  type = string
}

variable "project_number" {
  type = string
}

variable "region" {
  description = "MUST stay us-central1: the Cloud Run and Cloud Storage free tiers do not apply elsewhere."
  type        = string
  default     = "us-central1"
}

variable "github_repo" {
  description = "owner/repo allowed to deploy via Workload Identity Federation."
  type        = string
  default     = "AyDaoud/airq_montreal_mlops"
}

variable "image" {
  description = "Image to deploy. CI overrides this with a SHA-tagged reference."
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}
```

- [ ] **Step 3: Write `terraform/10-app/main.tf`**

```hcl
# The application. Apply terraform/00-guardrails FIRST - it holds the cost cap.

# Cleanup policy is load-bearing, not housekeeping: the serving image is
# about 540 MB uncompressed against a 0.5 GB free tier, and CI tags every
# push by commit SHA.
resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = "airq"
  format        = "DOCKER"
  description   = "Serving images, SHA-tagged."

  cleanup_policies {
    id     = "keep-recent-two"
    action = "KEEP"
    most_recent_versions {
      keep_count = 2
    }
  }

  cleanup_policies {
    id     = "delete-others"
    action = "DELETE"
    condition {
      tag_state = "ANY"
    }
  }
}

resource "google_service_account" "runtime" {
  account_id   = "airq-run"
  display_name = "Cloud Run runtime identity for the AirQ API"
}

resource "google_cloud_run_v2_service" "api" {
  name     = "airq-api"
  location = var.region

  template {
    service_account = google_service_account.runtime.email

    scaling {
      # min 0: "idle instances that are not minimum instances are not
      # charged", so an idle service costs nothing at all.
      min_instance_count = 0
      # max 2: bounds the burn rate to ~$0.19/hour, which is what makes
      # the kill switch's hours-long billing lag survivable.
      max_instance_count = 2
    }

    containers {
      image = var.image
      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
      env {
        name  = "MODEL_NAME"
        value = "rf"
      }
      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        failure_threshold     = 10
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  lifecycle {
    # CI deploys new revisions; Terraform must not fight it over the tag.
    ignore_changes = [template[0].containers[0].image, client, client_version]
  }
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# --- Deployment identity: Workload Identity Federation, no static keys ---

resource "google_service_account" "deployer" {
  account_id   = "airq-deployer"
  display_name = "GitHub Actions deploy identity"
}

resource "google_project_iam_member" "deployer_run" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_project_iam_member" "deployer_registry" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_service_account_iam_member" "deployer_actas" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github-pool"
  display_name              = "GitHub Actions"
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }

  # Without this condition ANY GitHub repository could mint tokens for
  # this service account.
  attribute_condition = "assertion.repository == '${var.github_repo}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "wif_binding" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/projects/${var.project_number}/locations/global/workloadIdentityPools/${google_iam_workload_identity_pool.github.workload_identity_pool_id}/attribute.repository/${var.github_repo}"
}
```

- [ ] **Step 4: Write `terraform/10-app/outputs.tf`**

```hcl
output "service_url" {
  description = "Public URL of the deployed API."
  value       = google_cloud_run_v2_service.api.uri
}

output "registry" {
  description = "Docker repository for SHA-tagged images."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "deployer_service_account" {
  value = google_service_account.deployer.email
}

output "workload_identity_provider" {
  description = "Paste into the GitHub workflow's auth step."
  value       = "projects/${var.project_number}/locations/global/workloadIdentityPools/${google_iam_workload_identity_pool.github.workload_identity_pool_id}/providers/${google_iam_workload_identity_pool_provider.github.workload_identity_pool_provider_id}"
}
```

- [ ] **Step 5: Write `terraform/10-app/terraform.tfvars.example`**

```hcl
# Copy to terraform.tfvars and fill in. terraform.tfvars is gitignored.
project_id     = "airq-montreal-XXXXXX"
project_number = "123456789012"
github_repo    = "AyDaoud/airq_montreal_mlops"
```

- [ ] **Step 6: Validate**

```bash
cd /home/ayman/airq_montreal_mlops/terraform/10-app
terraform fmt -check -diff .
terraform init -backend=false
terraform validate
```

Again: **no `plan`, no `apply`.**

- [ ] **Step 7: Commit**

```bash
cd /home/ayman/airq_montreal_mlops
git status --short | grep -E "tfstate|\.tfvars$|\.terraform/" && echo "PROBLEM: sensitive staged" || echo "ok"
git add terraform/10-app
if git commit -m "feat(app): terraform for Artifact Registry, Cloud Run and WIF

min_instance_count 0 and max_instance_count 2 are the two settings that
bound the bill; the registry cleanup policy is load-bearing because the
serving image is ~540 MB against a 0.5 GB free tier and CI tags by SHA.

The WIF provider carries an attribute_condition restricting it to this
repository - without it any GitHub repo could mint tokens for the deploy
service account.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"; then
  echo "COMMIT OK: \$(git log --oneline -1)"
else
  echo "COMMIT FAILED:"; git status --short
fi
```

---

## Task 4: The deploy workflow

**Files:**
- Create: `.github/workflows/deploy-cloudrun.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]
    paths:
      - "src/**"
      - "Dockerfile"
      - "requirements-serving.txt"
      - "scripts/bake_serving_model.py"
      - ".github/workflows/deploy-cloudrun.yml"
  workflow_dispatch:

# Deploying two revisions at once would race over traffic migration.
concurrency:
  group: deploy-cloudrun
  cancel-in-progress: false

env:
  PROJECT_ID: ${{ vars.GCP_PROJECT_ID }}
  REGION: us-central1
  SERVICE: airq-api
  REPO: airq

permissions:
  contents: read
  id-token: write # required to mint the WIF token

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      # The image must contain a model. Blocker B4 in this project was a
      # published image whose artifacts directory was empty: /health passed
      # and /predict raised FileNotFoundError on the first real request.
      - name: Bake the serving model
        run: |
          pip install -r requirements-serving.txt
          python -m scripts.bake_serving_model

      - id: auth
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ vars.GCP_WIF_PROVIDER }}
          service_account: ${{ vars.GCP_DEPLOY_SA }}

      - uses: google-github-actions/setup-gcloud@v2

      - name: Configure Docker for Artifact Registry
        run: gcloud auth configure-docker ${{ env.REGION }}-docker.pkg.dev --quiet

      - name: Build and push, tagged by commit SHA
        run: |
          IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${GITHUB_SHA}"
          echo "IMAGE=$IMAGE" >> "$GITHUB_ENV"
          docker build -t "$IMAGE" .
          docker push "$IMAGE"
          docker image inspect "$IMAGE" --format '{{.Size}}' \
            | awk '{printf "image size: %d MB\n", $1/1024/1024}'

      # Deploy WITHOUT traffic, so a broken revision never serves anyone.
      - name: Deploy a revision with no traffic
        run: |
          REVISION="${SERVICE}-${GITHUB_SHA::7}"
          echo "REVISION=$REVISION" >> "$GITHUB_ENV"
          gcloud run deploy "$SERVICE" \
            --image "$IMAGE" \
            --region "$REGION" \
            --revision-suffix "${GITHUB_SHA::7}" \
            --no-traffic \
            --min-instances 0 \
            --max-instances 2 \
            --cpu 1 --memory 512Mi \
            --allow-unauthenticated \
            --quiet

      - name: Smoke test the new revision before it gets traffic
        run: |
          URL=$(gcloud run revisions describe "$REVISION" \
            --region "$REGION" --format 'value(status.url)')
          if [ -z "$URL" ]; then
            URL=$(gcloud run services describe "$SERVICE" --region "$REGION" \
              --format 'value(status.traffic[0].url)')
          fi
          echo "testing $URL"
          for i in $(seq 1 20); do
            curl -sf "$URL/health" > /dev/null && break || sleep 5
          done
          curl -sf "$URL/health" | grep -q '"ok"'
          PAYLOAD=$(python -c "import json;f=json.load(open('artifacts/rf/feature_names.json'));print(json.dumps({'rows':[{k:1.0 for k in f}]}))")
          RESPONSE=$(curl -sf -X POST "$URL/predict" \
            -H 'Content-Type: application/json' -d "$PAYLOAD")
          echo "$RESPONSE"
          echo "$RESPONSE" | python -c "import json,sys; b=json.load(sys.stdin); assert b['n']==1; assert isinstance(b['preds'][0], float); print('smoke test OK')"

      - name: Migrate traffic to the verified revision
        run: |
          gcloud run services update-traffic "$SERVICE" \
            --region "$REGION" --to-revisions "$REVISION=100" --quiet
          gcloud run services describe "$SERVICE" --region "$REGION" \
            --format 'value(status.url)'

      - name: Roll back if the smoke test failed
        if: failure()
        run: |
          echo "deployment failed; traffic was never migrated to $REVISION"
          gcloud run services describe "$SERVICE" --region "$REGION" \
            --format 'value(status.traffic)' || true
```

Note the workflow reads **repository variables**, not secrets: `GCP_PROJECT_ID`, `GCP_WIF_PROVIDER`, `GCP_DEPLOY_SA`. None is a credential — WIF mints a short-lived token at run time, so no key exists to leak.

- [ ] **Step 2: Validate the YAML parses and contains no key material**

```bash
cd /home/ayman/airq_montreal_mlops
.venv/bin/python -c "
import yaml
w = yaml.safe_load(open('.github/workflows/deploy-cloudrun.yml'))
print('jobs:', list(w['jobs']))
print('permissions:', w['permissions'])
steps = w['jobs']['deploy']['steps']
print('steps:', len(steps))
assert w['permissions']['id-token'] == 'write', 'WIF needs id-token: write'
print('OK')"
grep -niE "credentials_json|private_key|BEGIN PRIVATE" .github/workflows/deploy-cloudrun.yml && echo "PROBLEM: key material" || echo "ok: no key material"
```

- [ ] **Step 3: Commit**

```bash
cd /home/ayman/airq_montreal_mlops
git add .github/workflows/deploy-cloudrun.yml
if git commit -m "feat(ci): deploy to Cloud Run via Workload Identity Federation

No service-account key exists anywhere: WIF mints a short-lived token at
run time and the three repo variables are identifiers, not credentials.

The revision deploys with --no-traffic, is smoke-tested, and only then
receives traffic. A revision that fails never serves a user - the
deployed form of the check that caught blocker B4, where a published
image had an empty artifacts directory and /predict raised
FileNotFoundError on the first real request.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"; then
  echo "COMMIT OK: \$(git log --oneline -1)"
else
  echo "COMMIT FAILED:"; git status --short
fi
```

---

## Task 5: The guard test — the five settings that bound the bill

This is the most important test in the spec. Those five settings are the only thing between the user and a bill, and a comment saying "do not change" would not stop anyone.

**Files:**
- Create: `tests/test_terraform_guards.py`

- [ ] **Step 1: Write the test**

```python
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
    assert 'default     = "us-central1"' in APP_VARS.read_text()


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


def test_the_workflow_carries_no_key_material():
    text = WORKFLOW.read_text().lower()
    for forbidden in ("credentials_json", "private_key", "begin private"):
        assert forbidden not in text, forbidden


def test_the_workflow_requests_an_id_token():
    """Without id-token: write, WIF cannot mint a token and the deploy
    falls back to needing a static key."""
    assert re.search(r"id-token:\s*write", WORKFLOW.read_text())


def test_the_wif_provider_is_restricted_to_this_repository():
    """Without the condition, any GitHub repository could mint tokens for
    the deploy service account."""
    assert "attribute_condition" in APP.read_text()
    assert "assertion.repository ==" in APP.read_text()


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
```

- [ ] **Step 2: Run it**

```bash
cd /home/ayman/airq_montreal_mlops
.venv/bin/python -m pytest tests/test_terraform_guards.py -v
```
Expected: 10 passed

- [ ] **Step 3: Prove each guard test actually catches its regression**

A guard test that would not fail is theatre. Verify three of them:

```bash
cd /home/ayman/airq_montreal_mlops
cp terraform/10-app/main.tf /tmp/main.tf.bak
sed -i 's/max_instance_count = 2/max_instance_count = 50/' terraform/10-app/main.tf
.venv/bin/python -m pytest tests/test_terraform_guards.py -q 2>&1 | tail -3
cp /tmp/main.tf.bak terraform/10-app/main.tf

cp terraform/00-guardrails/main.tf /tmp/g.tf.bak
sed -i 's/threshold_percent = 0.2/threshold_percent = 1.0/' terraform/00-guardrails/main.tf
.venv/bin/python -m pytest tests/test_terraform_guards.py -q 2>&1 | tail -3
cp /tmp/g.tf.bak terraform/00-guardrails/main.tf

.venv/bin/python -m pytest tests/test_terraform_guards.py -q 2>&1 | tail -2
rm -f /tmp/main.tf.bak /tmp/g.tf.bak
```

Each tampering must produce a failure, and the final run must be clean. **Report the output.**

- [ ] **Step 4: Add terraform validation to CI**

Append a job to `.github/workflows/ci.yml`:

```yaml
  terraform:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.9.8
      - name: Validate both modules
        run: |
          for dir in terraform/00-guardrails terraform/10-app; do
            echo "--- $dir ---"
            terraform -chdir=$dir fmt -check -diff
            terraform -chdir=$dir init -backend=false
            terraform -chdir=$dir validate
          done
```

- [ ] **Step 5: Lint, full suite, commit**

```bash
cd /home/ayman/airq_montreal_mlops
.venv/bin/flake8 src tests scripts terraform; echo "flake8 exit=$?"
.venv/bin/black --check src tests scripts terraform; echo "black exit=$?"
.venv/bin/python -m pytest
git add tests/test_terraform_guards.py .github/workflows/ci.yml
if git commit -m "test: assert the five settings that bound the bill

min_instance_count 0, max_instance_count 2, us-central1, a registry
cleanup keeping 2 versions, and a kill-switch threshold of 0.2. These
are the only things between the user and a bill, so each gets an
assertion rather than a comment. Verified each fails when tampered with.

Also asserts the deploy workflow carries no key material, deploys with
--no-traffic before migrating, and that no terraform state or real
tfvars is tracked.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"; then
  echo "COMMIT OK: \$(git log --oneline -1)"
else
  echo "COMMIT FAILED:"; git status --short
fi
```

Expected suite: 200 passed.

---

## Task 6: The runbook and README

**Files:**
- Create: `docs/RUNBOOK.md`
- Modify: `README.md`

- [ ] **Step 1: Write `docs/RUNBOOK.md`**

Cover, with exact commands:

1. **Applying, in order.** `terraform/00-guardrails` first, then `terraform/10-app`. State why: the guardrails hold the cost cap.
2. **Wiring GitHub.** The three repository *variables* from the `10-app` outputs, and that none is a secret.
3. **When the kill switch fires.** Symptom: the URL stops responding and email says billing was disabled. Recovery: re-link the billing account in the console, investigate what spent money, then redeploy. **Warn that leaving billing disabled for ~30 days lets GCP delete resources.**
4. **Rolling back.** `gcloud run services update-traffic airq-api --region us-central1 --to-revisions <sha>=100`.
5. **Tearing it all down.** `terraform destroy` in `10-app`, then `00-guardrails`, then delete the project. Note that deleting the project is the only way to be certain nothing accrues.

- [ ] **Step 2: Add a Deployment section to the README**

State:
- The live URL (filled in after the first deploy).
- **What was applied versus only validated**: the Terraform was `fmt`/`validate`-checked here but applied by the user, because the development machine has no `gcloud` and no credentials. Spec C set this precedent for the compose path; it matters more here, where being wrong costs money.
- The cost design in one table: kill switch at $1, burn bounded to ~$0.19/hour, worst realistic total ~$2 against a $5 ceiling.
- That Google offers **no true hard cap** — budgets notify, they do not stop — and this is the closest achievable.

- [ ] **Step 3: Verify and commit**

```bash
cd /home/ayman/airq_montreal_mlops
.venv/bin/python -m pytest
grep -n "no true hard cap\|\$1\|us-central1" README.md | head -5
git add docs/RUNBOOK.md README.md
if git commit -m "docs: runbook and deployment section

Records that Google offers no true hard spending cap, what the five cost
guards do, and which parts of the infrastructure were applied by the user
versus only validated here.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"; then
  echo "COMMIT OK"; git push
else
  echo "COMMIT FAILED:"; git status --short
fi
```

---

## Final verification

- [ ] **Offline suite**

```bash
mv data /tmp/_parked && .venv/bin/python -m pytest -q 2>&1 | tail -3 ; mv /tmp/_parked data
```

- [ ] **Terraform validates**

```bash
for d in terraform/00-guardrails terraform/10-app; do
  terraform -chdir=$d fmt -check && terraform -chdir=$d init -backend=false -input=false >/dev/null && terraform -chdir=$d validate
done
```

- [ ] **Nothing sensitive is tracked**

```bash
git ls-files | grep -E "\.tfstate|\.tfvars$" && echo "PROBLEM" || echo "clean"
```

- [ ] **CI green**, including the new `terraform` job.

---

## Handover to the user

Everything above is code. These steps need the user's console, and belong in the hand-back:

1. Copy both `terraform.tfvars.example` files to `terraform.tfvars` and fill in project id, project number, billing account id, alert email.
2. `terraform -chdir=terraform/00-guardrails apply` — **the cap, first.**
3. `terraform -chdir=terraform/10-app apply`.
4. Set the three repository variables in GitHub from the `10-app` outputs.
5. Push to `main` to trigger the first deploy.
6. Confirm the live URL, and paste it back so the README can record it.

---

## Plan self-review

Checked against the spec on 2026-09-29:

| Spec section | Covered by |
|---|---|
| §3 guardrails first, separate state | Task 2 |
| §4.1 budget, Pub/Sub, kill function | Tasks 1, 2 |
| §4.2 registry, Cloud Run, WIF | Task 3 |
| §4.3 deploy workflow | Task 4 |
| §4.4 runbook | Task 6 |
| §5 decisions D1-1 … D1-10 | D1-1 T2; D1-2/3/3b T3+T5; D1-4 T3; D1-5 T3; D1-6 T2; D1-7 runbook; D1-8 T4; D1-9 T4; D1-10 T2 |
| §7 testing | Tasks 1, 5 |
| §8 acceptance 1–7 | Final verification + handover |

**Known limit, stated rather than hidden:** no task applies the Terraform. `terraform plan` needs credentials this machine does not have, so the infrastructure is verified by `validate` and by static assertions over the config, never by observing it run. The first genuine proof is the user's own `apply`. This is the same class of limit as Spec A's missing Docker and Spec C's compose path, and the README says so in all three cases.

**Ordering note:** Task 1 (the function's logic) precedes Task 2 (the Terraform that packages it) so the decision that can cost money — or wrongly take down a working service — is tested in plain Python before any cloud resource wraps it.
