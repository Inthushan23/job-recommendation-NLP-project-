# ------------------------------
# IAM Role pour App Runner API (accès ECR)
# ------------------------------
resource "aws_iam_role" "apprunner_api_ecr_access_role" {
  name = "${var.api_service_name}-ecr-access-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "build.apprunner.amazonaws.com"
      }
    }]
  })

  tags = {
    Name        = "${var.api_service_name}-ecr-access-role"
    Environment = "dev"
    Project     = "MLOps-G3MG01"
  }
}

# IAM Role pour App Runner UI (accès ECR)
resource "aws_iam_role" "apprunner_ui_ecr_access_role" {
  name = "${var.ui_service_name}-ecr-access-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "build.apprunner.amazonaws.com"
      }
    }]
  })

  tags = {
    Name        = "${var.ui_service_name}-ecr-access-role"
    Environment = "dev"
    Project     = "MLOps-G3MG01"
  }
}

# ------------------------------
# IAM Role pour les instances App Runner (API + UI)
# ------------------------------
resource "aws_iam_role" "apprunner_instance_role" {
  name = "apprunner-g3mg01-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "tasks.apprunner.amazonaws.com"
      }
    }]
  })

  tags = {
    Name        = "apprunner-g3mg01-instance-role"
    Environment = "dev"
    Project     = "MLOps-G3MG01"
  }
}

# Policy pour accès S3 et CloudWatch
resource "aws_iam_role_policy" "apprunner_instance_policy" {
  name = "apprunner-g3mg01-instance-policy"
  role = aws_iam_role.apprunner_instance_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}
# ------------------------------
# Service App Runner pour l'API
# ------------------------------
resource "aws_apprunner_service" "api_service" {
  service_name = var.api_service_name

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_api_ecr_access_role.arn
    }

    image_repository {
      image_identifier      = "${var.api_ecr_repo_url}:latest"
      image_repository_type = "ECR"

      # --- CORRECTION ICI : Tout est regroupé au bon endroit ---
      image_configuration {
        port = "8000"
        
        runtime_environment_variables = {
          BUCKET_NAME      = "s3-g3mg01"       
          S3_DATA_FOLDER   = "data/"           
          EXCEL_FILENAME   = "job_data.xlsx"   
        }
      }
      # ---------------------------------------------------------
    }

    auto_deployments_enabled = false
  }

  instance_configuration {
    cpu               = "1024"
    memory            = "2048"
    instance_role_arn = aws_iam_role.apprunner_instance_role.arn
  }

  health_check_configuration {
    protocol            = "HTTP"
    path                = "/recommender/"
    interval            = 10
    timeout             = 20
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  tags = {
    Name        = var.api_service_name
    Environment = "dev"
    Project     = "MLOps-G3MG01"
  }
}
# ------------------------------
# Service App Runner pour l'UI
# ------------------------------
resource "aws_apprunner_service" "ui_service" {
  service_name = var.ui_service_name

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ui_ecr_access_role.arn
    }

    image_repository {
      image_identifier      = "${var.ui_ecr_repo_url}:ui"
      image_repository_type = "ECR"

      image_configuration {
        port = "8501"
      }
    }

    auto_deployments_enabled = false
  }

  instance_configuration {
    cpu               = "1024"
    memory            = "2048"
    instance_role_arn = aws_iam_role.apprunner_instance_role.arn
  }

  health_check_configuration {
    protocol            = "HTTP"
    path                = "/"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  tags = {
    Name        = var.ui_service_name
    Environment = "dev"
    Project     = "MLOps-G3MG01"
  }
}


# ------------------------------
# Attachement des permissions ECR pour l'API
# ------------------------------
resource "aws_iam_role_policy_attachment" "api_ecr_access_policy" {
  role       = aws_iam_role.apprunner_api_ecr_access_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

# ------------------------------
# Attachement des permissions ECR pour l'UI
# ------------------------------
resource "aws_iam_role_policy_attachment" "ui_ecr_access_policy" {
  role       = aws_iam_role.apprunner_ui_ecr_access_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}