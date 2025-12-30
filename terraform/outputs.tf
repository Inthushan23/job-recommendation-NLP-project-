# ECR repository information
output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = module.ecr_G3MG01.repository_url
}

# ECS cluster identifier
output "ecs_cluster_id" {
  description = "ID of the ECS cluster"
  value       = module.ecs_G3MG01.cluster_id
}

# Public entry point for the UI (via the load balancer)
output "ui_alb_url" {
  description = "Public URL of the load balancer to access the UI"
  value       = module.ecs_G3MG01.alb_dns_name
}
