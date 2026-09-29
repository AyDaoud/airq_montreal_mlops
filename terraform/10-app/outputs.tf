output "service_url" {
  description = "Public URL of the deployed API."
  value       = google_cloud_run_v2_service.api.uri
}

output "registry" {
  description = "Docker repository for SHA-tagged images."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "deployer_service_account" {
  description = "Set as the GCP_DEPLOY_SA repository variable in GitHub."
  value       = google_service_account.deployer.email
}

output "workload_identity_provider" {
  description = "Set as the GCP_WIF_PROVIDER repository variable in GitHub."
  value       = "projects/${var.project_number}/locations/global/workloadIdentityPools/${google_iam_workload_identity_pool.github.workload_identity_pool_id}/providers/${google_iam_workload_identity_pool_provider.github.workload_identity_pool_provider_id}"
}
