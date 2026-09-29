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
  name                = "airq-api"
  location            = var.region
  deletion_protection = false

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
