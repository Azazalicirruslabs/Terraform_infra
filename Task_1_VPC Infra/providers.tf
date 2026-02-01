terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.7.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
  required_version = ">= 1.0"
}

provider "aws" {
  region = var.aws_region


  default_tags {
    tags = {
      project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}
