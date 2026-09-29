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
