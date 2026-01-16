
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
  default     = "xai-explainability"
}

# VPC CIDR Block Variable

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# Subnet CIDR Blocks Variable
# CIDR /25 GIVES 128 IP ADDRESSES,but AWS reserves 5 IPs in each subnet for its own use.    

variable "public_subnet_cidrs" {
  description = "List of CIDR blocks for public subnets"
  type        = list(string)
  default = [
    "10.0.1.0/25",  # Public Subnet 1: 10.0.1.0 - 10.0.1.127 (~123 IPs)
    "10.0.1.128/25" # Public Subnet 2: 10.0.1.128 - 10.0.1.255 (~123 IPs)
  ]
}
variable "private_subnet_cidrs" {
  description = "List of CIDR blocks for private subnets"
  type        = list(string)
  default = [
    "10.0.2.0/25",  # Private Subnet 1: 10.0.2.0 - 10.0.2.127 (~123 IPs)
    "10.0.2.128/25" # Private Subnet 2: 10.0.2.128 - 10.0.2.255 (~123 IPs)
  ]
}

# Availability Zones Variable

variable "availability_zones" {
  description = "List of availability zones to use for subnets"
  type        = list(string)
  default     = ["ap-south-1a", "ap-south-1b"]
}

