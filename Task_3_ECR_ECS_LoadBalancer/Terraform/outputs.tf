# Show Repo URLs

output "repository_urls" {
  description = "Map of service names to their ECR repository URLs"
  value = {
    for service, repo in aws_ecr_repository.services :
    service => repo.repository_url #map service name to its repo URL
  }
}

# Show Repo ARNs

output "repository_arns" {
  description = "Map of service names to their ECR repository ARNs"
  value = {
    for service, repo in aws_ecr_repository.services :
    service => repo.arn
  }
}

# Individual Repo URL outputs

output "api_repository_url" {
  description = "ECR URL for API service"
  value       = aws_ecr_repository.services["api"].repository_url
}

output "gateway_repository_url" {
  description = "ECR URL for Gateway service"
  value       = aws_ecr_repository.services["gateway"].repository_url
}

output "mainflow_repository_url" {
  description = "ECR URL for Mainflow service"
  value       = aws_ecr_repository.services["mainflow"].repository_url
}
output "classification_repository_url" {
  description = "ECR URL for Classification service"
  value       = aws_ecr_repository.services["classification"].repository_url
}
output "regression_repository_url" {
  description = "ECR URL for Regression service"
  value       = aws_ecr_repository.services["regression"].repository_url
}
output "data_drift_repository_url" {
  description = "ECR URL for Data Drift service"
  value       = aws_ecr_repository.services["data_drift"].repository_url
}
output "fairness_repository_url" {
  description = "ECR URL for Fairness service"
  value       = aws_ecr_repository.services["fairness"].repository_url
}
output "what_if_repository_url" {
  description = "ECR URL for What-If service"
  value       = aws_ecr_repository.services["what_if"].repository_url
}


# AWS Account and Region Info

output "aws_account_id" {
  description = "AWS Account ID"
  value       = data.aws_caller_identity.current.account_id
}

output "aws_region" {
  description = "AWS Region"
  value       = data.aws_region.current.region
}

# ECR login command

output "ecr_login_command" {
  description = "Command to login to ECR manually"
  value       = "aws ecr get-login-password --region ${data.aws_region.current.region} | docker login --username AWS --password-stdin ${data.aws_caller_identity.current.account_id}.dkr.ecr.${data.aws_region.current.region}.amazonaws.com"
}

