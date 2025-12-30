# ECS cluster where everything runs
resource "aws_ecs_cluster" "cluster" {
  name = var.cluster_name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name        = var.cluster_name
    Environment = "dev"
    Project     = "MLOps-G3MG01"
  }
}

# Logs for UI and API containers
resource "aws_cloudwatch_log_group" "ui_logs" {
  name              = "/ecs/${var.cluster_name}"
  retention_in_days = 7
}

# Security group for the load balancer (public entry point)
resource "aws_security_group" "lb_sg" {
  name        = "${var.cluster_name}-lb-sg-v2"
  description = "Controls access to the ALB"
  vpc_id      = var.vpc_id

  ingress {
    protocol    = "tcp"
    from_port   = 80
    to_port     = 80
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Security group for ECS tasks (only ALB can talk to them)
resource "aws_security_group" "ecs_tasks_sg" {
  name        = "${var.cluster_name}-tasks-sg"
  description = "Allow inbound access from the ALB only"
  vpc_id      = var.vpc_id

  ingress {
    protocol        = "tcp"
    from_port       = 8080
    to_port         = 8080
    security_groups = [aws_security_group.lb_sg.id]
  }

  ingress {
    protocol        = "tcp"
    from_port       = 8000
    to_port         = 8000
    security_groups = [aws_security_group.lb_sg.id]
  }

  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${var.cluster_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.lb_sg.id]
  subnets            = var.public_subnet_ids
}

# Target group for the UI service
resource "aws_lb_target_group" "ui" {
  name        = "${var.cluster_name}-ui-tg"
  port        = 8080
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path                = "/"
    healthy_threshold   = 2
    unhealthy_threshold = 10
    timeout             = 60
    interval            = 120
    matcher             = "200-499"
  }

  stickiness {
    type            = "lb_cookie"
    cookie_duration = 86400
    enabled         = true
  }
}

# Target group for the API service
resource "aws_lb_target_group" "api" {
  name        = "${var.cluster_name}-api-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path                = "/recommender/"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200-299"
  }
}

# Main HTTP listener (everything enters here)
resource "aws_lb_listener" "front_end" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ui.arn
  }
}

# Route /recommender/* calls to the API
resource "aws_lb_listener_rule" "api_routing" {
  listener_arn = aws_lb_listener.front_end.arn
  priority     = 10

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }

  condition {
    path_pattern {
      values = ["/recommender/*"]
    }
  }
}

# Task definition for the UI container
resource "aws_ecs_task_definition" "ui" {
  family                   = "${var.cluster_name}-ui-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name      = "ui-container"
      image     = "${var.ecr_repository_url}:ui"
      essential = true
      portMappings = [{ containerPort = 8080, hostPort = 8080 }]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ui_logs.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "ecs-ui"
        }
      }
      environment = [
        { name = "STREAMLIT_SERVER_PORT", value = "8080" },
        { name = "STREAMLIT_SERVER_ADDRESS", value = "0.0.0.0" },
        { name = "STREAMLIT_SERVER_ENABLE_CORS", value = "false" },
        { name = "STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION", value = "false" },
        { name = "API_URL", value = "http://${aws_lb.main.dns_name}" }
      ]
    }
  ])
}

# ECS service for the UI
resource "aws_ecs_service" "ui" {
  name            = "${var.cluster_name}-ui-service"
  cluster         = aws_ecs_cluster.cluster.id
  task_definition = aws_ecs_task_definition.ui.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks_sg.id]
    subnets          = var.private_subnet_ids
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ui.arn
    container_name   = "ui-container"
    container_port   = 8080
  }

  depends_on = [aws_lb_listener.front_end]
}

# Task definition for the API container
resource "aws_ecs_task_definition" "api" {
  family                   = "${var.cluster_name}-api-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name      = "api-container"
      image     = "${var.ecr_repository_url}:api"
      essential = true
      portMappings = [{ containerPort = 8000, hostPort = 8000 }]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ui_logs.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "ecs-api"
        }
      }
    }
  ])
}

# ECS service for the API
resource "aws_ecs_service" "api" {
  name            = "${var.cluster_name}-api-service"
  cluster         = aws_ecs_cluster.cluster.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks_sg.id]
    subnets          = var.private_subnet_ids
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api-container"
    container_port   = 8000
  }
}

# IAM role used by ECS to start containers
resource "aws_iam_role" "ecs_task_execution_role" {
  name = "${var.cluster_name}-task-execution-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

# Standard ECS execution permissions (ECR, logs, etc.)
resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# IAM role assumed by the running containers
resource "aws_iam_role" "ecs_task_role" {
  name = "${var.cluster_name}-task-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

# App-level permissions (S3 + ECR access)
resource "aws_iam_role_policy" "ecs_task_policy" {
  name = "${var.cluster_name}-task-policy"
  role = aws_iam_role.ecs_task_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}
