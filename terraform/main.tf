# Terraform setup and provider versions
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state stored in S3
  backend "s3" {
    bucket  = "s3-g3mg01"
    key     = "g3mg01.tfstate"
    region  = "eu-west-3"
    encrypt = true
  }
}

# AWS provider configuration
provider "aws" {
  region = var.region
}

# Fetch the default VPC
data "aws_vpc" "default" {
  default = true
}

# Fetch all subnets from the default VPC
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ECR module (Docker image registry)
module "ecr_G3MG01" {
  source    = "./modules/ecr"
  repo_name = "ecr-g3mg01"
}

# ECS module (cluster, services, load balancer)
module "ecs_G3MG01" {
  source       = "./modules/ecs"
  cluster_name = "ecs-g3mg01"

  # Network info discovered automatically
  vpc_id            = data.aws_vpc.default.id
  public_subnet_ids = data.aws_subnets.default.ids

  # Cost-saving trick: no NAT Gateway, reuse public subnets
  private_subnet_ids = data.aws_subnets.default.ids

  # ECR repository URL used by ECS to pull images
  ecr_repository_url = module.ecr_G3MG01.repository_url

  region = var.region
}
