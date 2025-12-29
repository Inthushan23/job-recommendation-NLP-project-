terraform {
required_version = ">= 1.5.0"
required_providers {
 aws = {
 source = "hashicorp/aws"
 version = "~> 5.0"
 }
}
# Configuration du backend S3 pour stocker le state
backend "s3" {
 bucket = "s3-g3mg01"
 key = "g3mg01.tfstate"
 region = "eu-west-3"
 encrypt = true
}
}


provider "aws" {
  region = var.region
}

# S3 Bucket Module
# module "s3_G3MG01" {
#   source      = "./modules/s3"
#  bucket_name = "s3-g3mg01"
#}

# ECR Repository Module
module "ecr_G3MG01" {
  source    = "./modules/ecr"
  repo_name = "ecr-g3mg01"
}

# ECS Cluster Module
module "ecs_G3MG01" {
  source       = "./modules/ecs"
  cluster_name = "ecs-g3mg01"
}

# App Runner Service Module
# Décommenté quand vous avez une image Docker dans ECR
# module "apprunner_G0MG00" {
#   source       = "./modules/apprunner"
#   service_name = "apprunner-g3mg01"
#   ecr_repo_url = module.ecr_G0MG00.repository_url
# }
