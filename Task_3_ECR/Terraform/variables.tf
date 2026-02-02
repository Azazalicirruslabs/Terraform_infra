# VARIABLES 

variable "aws_region" {
  description = "AWS region where resources will be created"
  type        = string
  default     = "ap-south-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Name of the project - used as prefix for ECR repos"
  type        = string
  default     = "raia"
}

# SERVICES NAMES- microservices- each will get its own ECR repo

variable "services" {
  default = ["api"] # ← Test only API
}

# variable "services" {
#   default = ["api", "classification", "data_drift", "fairness", "gateway", "mainflow", "regression", "what_if"]
# }


# LIfecycle policy for ECR repositories- How many images to keep

variable "max_image_count" {
  description = "Maximum number of images to keep in each repository"
  type        = number
  default     = 2 # at least 2 images
}

# Docker Path

locals {
  # Path to the project root (where docker-compose.yml is)
  docker_context_path = abspath("${path.module}/..")
}


