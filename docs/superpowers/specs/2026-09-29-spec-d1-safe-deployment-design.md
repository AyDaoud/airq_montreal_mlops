# Spec D1 — Safe Deployment to Cloud Run

- **Date:** 2026-09-29
- **Status:** Approved (pending spec review)
- **Branch:** `spec-d1/safe-deployment`
- **Predecessors:** Specs A, B, C (A and B merged; C green and pending merge)
- **Successor:** Spec D2 — Postgres, `target_date` wiring, champion/challenger promotion

---

## 1. Context and the one constraint that shapes everything

The project has a working pipeline, a model that beats its baseline, and a live observability stack. None of it is reachable by anyone but its author.

**The author has no free GCP credits.** The 90-day $300 trial expired unused. Deployment therefore has to run on the **Always Free tier**, which is a different and non-expiring allowance, and the design is built around not exceeding it.

### Verified on 2026-09-29 from Google's current free-tier and pricing pages

| Service | Always Free allowance |
|---|---|
| Cloud Run | 2M requests/month · 360,000 GB-s memory · 180,000 vCPU-s · **1 GB** egress from North America |
| Artifact Registry | **0.5 GB** storage |
| Cloud Storage | 5 GB-months, **US regions only** |
| Cloud Build | 2,500 build-minutes/month |

Beyond the free tier, requests cost **$0.40 per million**. And from the Cloud Run pricing page, the line this design leans on:

> *"Idle instances that are not minimum instances are not charged."*

### The honest position on cost

**Google does not offer a true hard spending cap.** Budgets notify; they do not stop. That is a platform limitation, not something this design can engineer away. What it can do is bound the exposure:

- `min_instance_count = 0` removes idle billing entirely — with no traffic, no meter runs.
- `max_instance_count = 3` caps the *burn rate* at roughly **$0.28/hour** even under continuous saturation (3 vCPU at $0.000024/vCPU-s plus 1.5 GiB at $0.0000025/GiB-s).
- The kill switch lags by hours because billing data lags. Bounding the rate is what makes that lag survivable: six hours of total saturation is about **$1.70**.

So: expected cost **$0**; pathological case **single-digit dollars**, then everything stops. The user has explicitly accepted that the service dies at the cap.

### A measured risk specific to this project

The serving image is roughly **540 MB uncompressed**:

```
pyarrow 126.7 · scipy 105.6 · pandas 67.5 · sklearn 45.3 · numpy 38.8  + ~130 base
```

Artifact Registry's free tier is 0.5 GB. Layers are stored compressed (~200 MB), so one image fits — but Spec A tags every push with its commit SHA, and **three or four deploys would exceed the limit**. A cleanup policy is therefore load-bearing, not housekeeping.

---

## 2. Goals and non-goals

### Goals
1. A publicly reachable URL serving the model, which stays up indefinitely at zero cost.
2. Cost exposure bounded and automatically terminated at a $5 budget.
3. Infrastructure described in Terraform, reviewable by `terraform plan`.
4. Deployment from CI with no long-lived credentials in the repository.
5. A deployed revision that is verified working before it receives traffic.

### Non-goals — deferred to Spec D2
- Postgres. The serving container keeps SQLite; `DATABASE_URL` already makes that a one-variable swap.
- Wiring `target_date` through the prediction request.
- Champion/challenger promotion.

### Non-goals — deferred indefinitely
- The evidently 0.4 → 0.7 port, still pinning `numpy<2.1`.
- Any second cloud provider.

---

## 3. Guardrails first, enforced structurally

Two Terraform root modules with **separate state**:

```
terraform/
  00-guardrails/    budget, Pub/Sub topic, billing-disable function
  10-app/           Artifact Registry, Cloud Run, service accounts, WIF
```

Applying `10-app` without `00-guardrails` becomes a deliberate act rather than an oversight. A single module with `depends_on` would order the resources but would not make the omission visible; two states do.

`terraform.tfvars` holds the project ID, billing account ID and alert email. It is **gitignored** — identifiers are not secrets, but they are per-user and do not belong in committed code.

---

## 4. Components

### 4.1 `terraform/00-guardrails/`

- `google_billing_budget` — $5, thresholds at 50%, 90%, 100%, notifications to Pub/Sub and email
- `google_pubsub_topic` — receives budget notifications
- `google_cloudfunctions2_function` — subscribes to the topic; when spend exceeds the budget, calls the Cloud Billing API to **detach the billing account from the project**
- `google_service_account` for the function, granted `roles/billing.admin`

### 4.2 `terraform/10-app/`

- `google_artifact_registry_repository` — Docker format, `us-central1`, with a **cleanup policy keeping the 2 most recent versions**
- `google_cloud_run_v2_service` — `min_instance_count = 0`, `max_instance_count = 3`, 1 vCPU, 512 MiB, `us-central1`
- `google_cloud_run_v2_service_iam_member` — `allUsers` as `roles/run.invoker`, making it public
- `google_iam_workload_identity_pool` + provider, and a deploy service account with only `roles/run.admin`, `roles/artifactregistry.writer`, `roles/iam.serviceAccountUser`

### 4.3 `.github/workflows/deploy-cloudrun.yml`

Triggered on push to `main`. Authenticates via Workload Identity Federation — **no service-account JSON key exists anywhere**. Bakes the serving model, builds the image, pushes it SHA-tagged, deploys a revision with `--no-traffic`, smoke-tests that revision's URL, and only then migrates traffic to it. A revision that fails its smoke test never receives a request.

### 4.4 `docs/RUNBOOK.md`

What to do when the kill switch fires, how to re-enable billing, how to roll back to a previous revision by SHA, and how to tear the whole project down.

---

## 5. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1-1 | Guardrails are a separate Terraform state, applied first | Makes skipping them a deliberate act |
| D1-2 | `min_instance_count = 0` | Idle instances that are not minimum instances are not charged |
| D1-3 | `max_instance_count = 3` | Bounds burn to ~$0.28/hour, which is what makes the kill switch's lag survivable |
| D1-4 | Region `us-central1` | Free tiers for Cloud Run and Cloud Storage do not apply to `northamerica-northeast1` |
| D1-5 | Artifact Registry cleanup, keep 2 versions | The image is ~540 MB uncompressed against a 0.5 GB free tier; SHA tags accumulate |
| D1-6 | Kill switch **disables billing**, and its service account holds `roles/billing.admin` | The user explicitly accepted that the service dies past the cap. The tradeoff: this role can detach billing from any project on the account, which is why D1-7 exists. **Flagged for veto at spec review.** |
| D1-7 | A dedicated, disposable GCP project | Disabling billing kills the whole project; a dedicated one bounds the blast radius to this app and can be deleted entirely afterwards |
| D1-8 | Workload Identity Federation, not a service-account key | No long-lived credential in the repository or in GitHub Secrets |
| D1-9 | Deploy with `--no-traffic`, smoke-test, then migrate | A broken revision never serves a user. This is the deployed form of the check that caught blocker B4 |
| D1-10 | `terraform.tfvars` gitignored | Identifiers are per-user; committed config should be user-agnostic |

---

## 6. What can and cannot be verified here

The development machine has Terraform 1.13.3, **no `gcloud`, and no GCP credentials**. This is the same class of limit as Spec A's missing Docker, and it is stated rather than papered over.

| Verifiable in this session | Requires the user |
|---|---|
| `terraform fmt -check`, `terraform validate` | `terraform apply` |
| Unit tests for the billing function's decision logic | Creating the project and linking billing |
| YAML parsing and linting of the deploy workflow | Configuring the WIF pool binding |
| A test asserting the four cost guards are present in the config | Confirming the live URL responds |

**The README must state which parts were applied and which were only validated.** Spec C set that precedent for the compose path and it applies more strongly here, where being wrong costs money.

---

## 7. Testing

- `terraform validate` on both modules, run in CI.
- A test parsing `10-app/*.tf` and asserting all four cost guards: `min_instance_count = 0`, `max_instance_count = 3`, region `us-central1`, and a cleanup policy retaining 2 versions. These are the settings that bound the bill; a silent edit to any of them should fail the build.
- A test asserting `terraform.tfvars` is gitignored and absent from the index.
- Unit tests for the billing function: it disables billing above the threshold, does nothing below it, and is idempotent when billing is already disabled.
- A test asserting the deploy workflow contains no `credentials_json` and no base64 key blob.

All tests remain offline. No test contacts GCP.

---

## 8. Acceptance criteria

1. `terraform validate` passes on both modules.
2. The guard test passes, and fails if any of the four cost settings is changed.
3. The deploy workflow authenticates by WIF, with no key material anywhere in the repository.
4. A deployed revision is smoke-tested before receiving traffic; a failing revision gets none.
5. `docs/RUNBOOK.md` covers kill-switch recovery, rollback by SHA, and teardown.
6. The README distinguishes what was applied from what was only validated.
7. The full suite still passes offline with `data/` absent.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Google has no true hard cap | Stated plainly in §1. `max_instance_count` bounds the rate; the budget function is a backstop, not a guarantee |
| Kill switch lags hours behind spend | Rate bounded to ~$0.28/hour, so the lag costs single-digit dollars |
| Image exceeds the 0.5 GB registry free tier | Cleanup policy keeps 2 versions; a test asserts it is present |
| `roles/billing.admin` is broad | Confined to a dedicated disposable project (D1-7); flagged for veto (D1-6) |
| Terraform cannot be applied or planned here | Verification limited to `validate` and static assertions; the README says so |
| A bad revision reaches users | `--no-traffic` deploy, smoke test, then traffic migration (D1-9) |
