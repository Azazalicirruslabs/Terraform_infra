# Variables for AWS Dev Infrastructure

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "ap-south-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name for resource naming and tagging"
  type        = string
  default     = "terraform-dev"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "192.168.0.0/16"
}

variable "public_subnet_cidr" {
  description = "CIDR block for public subnet (~123 usable IPs)"
  type        = string
  default     = "192.168.0.0/25"
}

variable "private_subnet_cidr" {
  description = "CIDR block for private subnet (~123 usable IPs)"
  type        = string
  default     = "192.168.0.128/25"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

# database=name of the postgresql database
variable "db_name" {
  description = "Name of the PostgreSQL database"
  type        = string
  default     = "devdb"
}

# username to login to the postgresql database
variable "db_username" {
  description = "Master username for RDS"
  type        = string
  default     = "dbadmin"
}

variable "db_password" {
  description = "Master password for RDS (sensitive)"
  type        = string
  sensitive   = true
}

variable "db_allocated_storage" {
  description = "Allocated storage for RDS in GB"
  type        = number
  default     = 20 # 20 GB hard drive
}

variable "key_name" {
  description = "Name of the EC2 key pair for SSH access"
  type        = string
  default     = ""
}

# from any ip
variable "allowed_ssh_cidr" {
  description = "CIDR block allowed for SSH access"
  type        = string
  default     = "0.0.0.0/0"
}
