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
  description = "List of microservices that need ECR repositories"
  type        = list(string)
  default = [
    "api",            # services/API
    "classification", # services/classification
    "data_drift",     # services/data_drift
    "fairness",       # services/fairness
    "gateway",        # services/gateway
    "mainflow",       # services/mainflow
    "regression",     # services/regression
    "what_if"         # services/what_if
  ]
}


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


