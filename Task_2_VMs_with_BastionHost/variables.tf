
# AWS Region Variable

variable "aws_region" {
  description = "AWS region where resources will be created "
  type        = string
  default     = "ap-south-1"
}


# Environment Variable

variable "environment" {
  description = "Environment name (e.g., dev, prod, Uat)"
  type        = string
  default     = "dev"
}

# Project Name Variable

variable "project_name" {
  description = "Name of the project - used for naming resources"
  type        = string
  default     = "Task_2_Infrastructure_using_VMs_with_BastionHost"
}


# Key Pair Configuration

variable "key_name" {
  description = "Name for the AWS key pair"
  type        = string
  default     = "my-terraform-key"
}


# Instance Configuration

variable "instance_type" {
  description = "EC2 instance type for all VMs"
  type        = string
  default     = "t2.micro"
}

# Availability Zones Variable

variable "availability_zones" {
  description = "List of availability zones to use for subnets"
  type        = list(string)
  default     = ["ap-south-1a"]
}
