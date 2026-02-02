terraform {
  required_version = ">= 1.14.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }

    # TLS provider - Generate PEM Key Pair
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }

    # Local provider - Save .pem file locally
    local = {
      source  = "hashicorp/local"
      version = "~> 2.6"
    }
  }
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
