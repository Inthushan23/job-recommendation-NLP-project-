variable "api_service_name" {
  type        = string
  description = "Nom du service App Runner pour l'API"
}

variable "ui_service_name" {
  type        = string
  description = "Nom du service App Runner pour l'UI"
}

variable "api_ecr_repo_url" {
  type        = string
  description = "URL du repository ECR pour l'API"
}

variable "ui_ecr_repo_url" {
  type        = string
  description = "URL du repository ECR pour l'UI"
}