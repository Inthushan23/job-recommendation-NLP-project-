variable "cluster_name" {
  type        = string
  description = "Nom du cluster ECS"
}

variable "region" {
  type        = string
  description = "Région AWS"
}

variable "vpc_id" {
  type        = string
  description = "ID du VPC par défaut"
}

variable "public_subnet_ids" {
  type        = list(string)
  description = "IDs des subnets publics pour l'ALB"
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "IDs des subnets pour les tâches (on utilisera les publics ici)"
}

variable "ecr_repository_url" {
  type        = string
  description = "URL du dépôt ECR"
}

variable "container_port" {
  type        = number
  default     = 8080
}