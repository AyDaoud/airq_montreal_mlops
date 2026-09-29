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
