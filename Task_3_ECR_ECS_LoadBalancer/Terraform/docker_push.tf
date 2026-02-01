# Build and Push to ECR

#login->build->push for each service

# "null_resource" + "local-exec" = Run local commands

# Login to ECR

resource "null_resource" "ecr_login" {
  provisioner "local-exec" {
    command = "aws ecr get-login-password --region ${data.aws_region.current.region} | docker login --username AWS --password-stdin ${data.aws_caller_identity.current.account_id}.dkr.ecr.${data.aws_region.current.region}.amazonaws.com"
  }

  triggers = {
    always_run = timestamp() # forces re-run each time
  }

  depends_on = [aws_ecr_repository.services] # ensure repos are created first
}


# Build and Push each service

# Mapping of service names to their Dockerfile paths
locals {
  # This maps each service name to its Dockerfile location
  dockerfile_paths = {
    "api"            = "services/API/Dockerfile"
    "classification" = "services/classification/Dockerfile"
    "data_drift"     = "services/data_drift/Dockerfile"
    "fairness"       = "services/fairness/Dockerfile"
    "gateway"        = "services/gateway/Dockerfile"
    "mainflow"       = "services/mainflow/Dockerfile"
    "regression"     = "services/regression/Dockerfile"
    "what_if"        = "services/what_if/Dockerfile"
  }
}

resource "null_resource" "docker_build_push" {
  for_each = toset(var.services)

  provisioner "local-exec" {
    working_dir = local.docker_context_path



    command = <<-EOT
      echo "Building Docker image for ${each.key}..."
      docker build -t ${var.project_name}-${each.key}:latest -f ${local.dockerfile_paths[each.key]} .

      echo "Tagging image for ECR..."
      docker tag ${var.project_name}-${each.key}:latest ${aws_ecr_repository.services[each.key].repository_url}:latest

      echo "Pushing to ECR..."
      docker push ${aws_ecr_repository.services[each.key].repository_url}:latest

      echo "Successfully pushed ${each.key} to ECR!"
    EOT

    interpreter = ["bash", "-lc"]
  }

  # Re-run if Dockerfile content changes
  triggers = {
    dockerfile_hash = filemd5("${local.docker_context_path}/${local.dockerfile_paths[each.key]}")

  }

  depends_on = [
    null_resource.ecr_login,
    aws_ecr_repository.services
  ]
}
