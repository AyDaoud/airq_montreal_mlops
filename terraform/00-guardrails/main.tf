# Guardrails. Applied BEFORE terraform/10-app, in a separate state, so
# deploying without a cost cap is a deliberate act rather than an oversight.

resource "google_project_service" "required" {
  for_each = toset([
    "cloudbilling.googleapis.com",
    "billingbudgets.googleapis.com",
    "cloudfunctions.googleapis.com",
    "cloudbuild.googleapis.com",
    "pubsub.googleapis.com",
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "iamcredentials.googleapis.com",
    "eventarc.googleapis.com",
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
    pubsub_topic                   = google_pubsub_topic.budget.id
    schema_version                 = "1.0"
    disable_default_iam_recipients = false
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
  depends_on                  = [google_project_service.required]
}

# archive_file reads the live directory, not git, so __pycache__ left by a
# local test run would otherwise be shipped inside the deployed function.
data "archive_file" "killswitch" {
  type        = "zip"
  source_dir  = "${path.module}/function"
  output_path = "${path.module}/.build/killswitch.zip"
  excludes    = ["__pycache__", "__pycache__/*", "*.pyc"]
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
