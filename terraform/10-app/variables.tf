variable "project_id" {
  description = "The dedicated, disposable GCP project for this deployment."
  type        = string
}

variable "project_number" {
  description = "Numeric project number, shown on the project dashboard."
  type        = string
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
