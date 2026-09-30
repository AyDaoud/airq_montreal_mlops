# Runbook — Cloud Run Deployment (Spec D1)

Operator reference for deploying, recovering, rolling back and tearing down the Cloud Run
deployment. Written for the moment something is on fire at 2am, not as a tutorial — every
command here is copy-pasteable as written, from the project root.

This document is one half of Spec D1; the other half is `terraform/00-guardrails/`,
`terraform/10-app/` and `.github/workflows/deploy-cloudrun.yml` themselves. Read
[§6](#6-what-was-applied-versus-only-validated) before trusting anything below as "already
proven to work" — it was not run on this machine.

---

## 1. First-time setup, in order

### 1.1 Fill in the tfvars files

Both modules take a `terraform.tfvars` you create locally; the `.example` files are tracked,
the real files are gitignored (`*.tfvars` is ignored, `*.tfvars.example` is not — see
`.gitignore`). Never commit the real files; they are not secrets in the credential sense, but
they do name your specific GCP project.

```bash
cp terraform/00-guardrails/terraform.tfvars.example terraform/00-guardrails/terraform.tfvars
cp terraform/10-app/terraform.tfvars.example terraform/10-app/terraform.tfvars
```

Edit both files:

- `project_id` — the GCP project dashboard.
- `project_number` — same dashboard, next to `project_id`.
- `billing_account_id` — Billing → Account management (00-guardrails only).
- `alert_email` — where budget-threshold notifications go (00-guardrails only).
- `github_repo` — already defaults to `AyDaoud/airq_montreal_mlops` in 10-app; only change
  it if you forked.

### 1.2 Apply the guardrails module first — the cap before the thing it caps

```bash
terraform -chdir=terraform/00-guardrails init
terraform -chdir=terraform/00-guardrails apply
```

This module is the budget, the Pub/Sub topic the budget notifies, the Cloud Function that
disables billing when notified, and the `roles/billing.admin` service account that function
runs as. **Apply it first, always.** If the app module (Cloud Run, the registry, the deploy
service account) went up before the kill switch exists to watch it, any window between the two
applies is a window with no cap on spend at all — exactly the situation this whole spec exists
to prevent. Guardrails has its own Terraform state, entirely separate from the app module, so
this ordering is a procedural rule, not something Terraform enforces for you.

### 1.3 Apply the app module

```bash
terraform -chdir=terraform/10-app init
terraform -chdir=terraform/10-app apply
```

This is the Artifact Registry repository, the Cloud Run service `airq-api` (deployed with a
placeholder image on first apply — the real image arrives via the deploy workflow), the deploy
service account, and the Workload Identity Federation pool/provider restricted by
`attribute_condition` to `AyDaoud/airq_montreal_mlops` (or your fork, per `github_repo` above).

### 1.4 Read the outputs and set the GitHub repository variables

```bash
terraform -chdir=terraform/10-app output
```

Four outputs matter: `service_url`, `registry`, `deployer_service_account`,
`workload_identity_provider`. Take the last two to GitHub: **Settings → Secrets and variables
→ Actions → Variables** (repository variables, not secrets — none of the three is a
credential; WIF is exactly the mechanism that lets the workflow authenticate without one) and
set:

| Variable | Value |
|---|---|
| `GCP_PROJECT_ID` | your `project_id` |
| `GCP_WIF_PROVIDER` | the `workload_identity_provider` output |
| `GCP_DEPLOY_SA` | the `deployer_service_account` output |

### 1.5 Trigger the first real deploy

```bash
git push origin main
```

`.github/workflows/deploy-cloudrun.yml` runs on pushes to `main` that touch `src/**`,
`Dockerfile`, `requirements-serving.txt`, `scripts/bake_serving_model.py` or the workflow file
itself (or on manual `workflow_dispatch`). It bakes the model, authenticates via WIF, builds
and pushes a SHA-tagged image, deploys it with `--no-traffic`, smoke-tests `/health` and
`/predict` against that untrafficked revision, and only then migrates traffic. A revision that
fails its smoke test never serves a user.

Once it succeeds, `terraform -chdir=terraform/10-app output service_url` is the live URL —
copy it into the README placeholder (`## 13. Deployment`).

---

## 2. The cost design

The user's constraint drives every setting here: no free GCP credits, and the single most
important thing is not exceeding **$5**.

| | |
|---|---|
| Budget | $5.00 |
| **Kill switch fires at** | **$1.00 — the 20% threshold** |
| Burn rate ceiling (`max_instance_count = 2`) | ~$0.19/hour |
| Overshoot during a ~6-hour billing lag | ~$1.14 |
| **Worst realistic total** | **≈ $2.15** |

**Why $1 and not $5.** Inside Google Cloud's Always Free tier this project's spend is exactly
**$0** — normal operation never leaves it, so cost stays at zero. Any movement above zero
means a free allowance has already been exceeded; the first dollar is the alarm, not the
fifth. Firing at 100% of a $5 budget would mean the whole tolerance was already spent before
the switch engaged, and the hours-long billing lag would then carry the total past $5 with
nothing left to absorb it.

**The honest caveat: Google offers no true hard spending cap.** Budgets notify; they do not
stop spend on their own. The kill switch (a Cloud Function that disables billing on the
project when notified) is the closest achievable substitute for a hard cap, and it still lags
by hours, because billing data itself lags by hours before a budget notification fires.

### Illustrative: the kill-switch decision function

The table below is the *unit test* of `should_disable()` in
`terraform/00-guardrails/function/main.py` — a **hypothetical spend → decision** table proving
the switch stays engaged (`True`, billing gets disabled) at any spend at or above the trigger.
It is the reassuring case, not a description of what this deployment is expected to spend. In
practice the switch fires at $1, so the larger hypothetical values below are never reached —
burn is capped at ~$0.19/hour and the lag window is hours, not the days it would take to reach
$50 of spend at that ceiling.

| hypothetical spend → decision | disable billing? |
|---|---|
| $0.00 (the normal case) | False |
| $0.42 | False |
| $1.00 (the trigger) | True |
| $2.40 | True |
| $50.00 | True |

### The five guards, each asserted by `tests/test_terraform_guards.py`

| Setting | Value | Prevents |
|---|---|---|
| `min_instance_count` | 0 | Idle billing — Cloud Run does not charge for instances that are not minimum instances while idle |
| `max_instance_count` | 2 | Burn above ~$0.19/hour |
| region | `us-central1` | Falling outside the free-tier regions — the Cloud Run and Cloud Storage free tiers do **not** apply to `northamerica-northeast1`, however tempting deploying near Montréal itself is |
| registry cleanup | keep 2 versions | Image storage past the 0.5 GB Artifact Registry free tier — the serving image is ~540 MB uncompressed and CI tags every push by commit SHA |
| kill-switch threshold | 0.2 (20%) | Spending the whole $5 tolerance before the switch engages |

Each guard was verified to fail when deliberately tampered with during development: setting
`max_instance_count` to 50, moving the threshold to 1.0, and deleting the registry cleanup
policy each produced exactly one test failure in `tests/test_terraform_guards.py` — not a
cascade, not a silent pass.

---

## 3. When the kill switch fires

**Symptom.** The service URL (`terraform -chdir=terraform/10-app output service_url`) stops
responding. An email arrives (sent to `alert_email` from the guardrails tfvars) stating that
billing was disabled for the project.

**Diagnose.** Open the Cloud Billing report for the project and find what actually spent
money. Inside the Always Free tier, spend is $0 by design — so *any* nonzero spend means some
allowance was exceeded (a free-tier limit crossed, a resource created outside the free-tier
regions, more image versions retained than the cleanup policy should have allowed, etc.).
Identify that resource before recovering, or the same thing re-triggers the kill switch
immediately after you re-enable billing.

**Recover.**

1. In the Cloud console, re-link the billing account to the project (Billing → link a billing
   account — this reverses what the kill switch did, which is detach it).
2. Redeploy so a live revision is serving again:
   ```bash
   git push origin main   # or: workflow_dispatch on deploy-cloudrun.yml
   ```
   Cloud Run does not delete the service when billing is disabled, only stops it serving, so
   the existing revisions and Terraform state are intact — this is a redeploy, not a rebuild
   from `terraform apply`.

**Warn explicitly.** Leaving billing disabled for roughly 30 days is understood to let Google
Cloud begin deleting project resources for non-payment. A kill switch that fires and is then
forgotten does not just stop serving traffic — left long enough, it risks losing the project's
resources outright. Treat the recovery email as something to act on, not archive.

---

## 4. Rolling back

Every deployed image is tagged by commit SHA, so a Cloud Run revision maps to an exact commit.
List revisions, pick the one to restore, and move traffic to it:

```bash
gcloud run revisions list --service airq-api --region us-central1
gcloud run services update-traffic airq-api --region us-central1 --to-revisions <REVISION>=100
```

`<REVISION>` is one of the names from the first command (`airq-api-<7-char-sha>`). This moves
100% of traffic to that revision immediately; it does not delete the newer, broken revision,
which stays available in case you need to move forward again.

---

## 5. Tearing it down

```bash
terraform -chdir=terraform/10-app destroy
terraform -chdir=terraform/00-guardrails destroy
```

Destroy the app module first (mirror of the apply order: bring down the thing being capped
before the cap itself, since there is no benefit to reversing that order and every benefit to
keeping the two operations symmetric and easy to reason about).

`terraform destroy` removes what Terraform created, but it is not a guarantee that the project
has zero cost surface left — anything created out-of-band, log storage past its retention
window, or a resource Terraform's state lost track of can still linger. **Deleting the GCP
project itself is the only way to be certain nothing continues to accrue.** Do that in the
Cloud console (IAM & Admin → Settings → Shut down project) after both `destroy` runs succeed.

---

## 6. What was applied versus only validated

This machine has **Terraform 1.13.3** but **no `gcloud` CLI and no GCP credentials**. Every
`.tf` file in `terraform/00-guardrails/` and `terraform/10-app/` was `terraform fmt -check`ed
and `terraform validate`d here, and both modules' provider lock files
(`.terraform.lock.hcl`) are committed so that `terraform init` on your machine resolves the
same `hashicorp/google 6.50.0` schema these guards were checked against.

What did **not** happen on this machine: `terraform plan`, `terraform apply`, or observing the
service actually running. The first genuine proof that this configuration provisions the
infrastructure it claims to is your own `terraform apply` in [§1](#1-first-time-setup-in-order).
This mirrors Specs A and C, where Docker was unavailable here and the serving image and the
`docker-compose.yml` stack were verified in CI rather than run locally — it matters more for
D1 because a wrong assumption here costs money, not just a red CI badge.
