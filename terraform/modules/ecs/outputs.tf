output "cluster_id" {
  value = aws_ecs_cluster.cluster.id
}

output "alb_dns_name" {
  description = "L'URL publique du Load Balancer (TON SITE EST ICI)"
  value       = aws_lb.main.dns_name
}