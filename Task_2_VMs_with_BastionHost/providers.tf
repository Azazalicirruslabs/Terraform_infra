terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.7.0"
    }

    #USE AWS RESOURCE FOR KEY PAIR
    tls = { # Generate PEM Key Pair
      source  = "hashicorp/tls"
      version = "4.1.0"
    }
    local = { #Save .pem file locally
      source  = "hashicorp/local"
      version = "2.6.1"
    }
    # random = {
    #   source  = "hashicorp/random"
    #   version = "~> 3.1"
    # }
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
