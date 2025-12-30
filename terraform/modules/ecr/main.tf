# ECR repository to store Docker images
resource "aws_ecr_repository" "repo" {
  name                 = var.repo_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name        = var.repo_name
    Environment = "dev"
    Project     = "MLOps-G3MG01"
  }
}

# Cleanup policy to avoid keeping too many images
resource "aws_ecr_lifecycle_policy" "repo_policy" {
  repository = aws_ecr_repository.repo.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = {
        type = "expire"
      }
    }]
  })
}
