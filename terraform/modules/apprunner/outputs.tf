# Outputs pour le service API
output "api_service_arn" {
  description = "The ARN of the App Runner API service"
  value       = aws_apprunner_service.api_service.arn
}

output "api_service_id" {
  description = "The ID of the App Runner API service"
  value       = aws_apprunner_service.api_service.service_id
}

output "api_service_url" {
  description = "The URL of the App Runner API service"
  value       = aws_apprunner_service.api_service.service_url
}

output "api_service_status" {
  description = "The status of the App Runner API service"
  value       = aws_apprunner_service.api_service.status
}

# Outputs pour le service UI
output "ui_service_arn" {
  description = "The ARN of the App Runner UI service"
  value       = aws_apprunner_service.ui_service.arn
}

output "ui_service_id" {
  description = "The ID of the App Runner UI service"
  value       = aws_apprunner_service.ui_service.service_id
}

output "ui_service_url" {
  description = "The URL of the App Runner UI service"
  value       = aws_apprunner_service.ui_service.service_url
}

output "ui_service_status" {
  description = "The status of the App Runner UI service"
  value       = aws_apprunner_service.ui_service.status
}
