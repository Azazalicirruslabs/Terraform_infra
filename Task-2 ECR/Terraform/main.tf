# get information

data "aws_caller_identity" "current" {} #account info

data "aws_region" "current" {}

# creation of ecer repositories for each microservice

resource "aws_ecr_repository" "services" {

  for_each = toset(var.services)

  name                 = "${var.project_name}-${each.key}" # RAIA-api, RAIA-classification, etc.
  force_delete         = true
  image_tag_mutability = "MUTABLE"

  # SCAN ON PUSH:

  image_scanning_configuration {
    scan_on_push = true
  }

  # ENCRYPTION:
  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Name    = "${var.project_name}-${each.key}"
    Service = each.key
  }
}


# LIFECYCLE POLICY- keep only latest 2 images

resource "aws_ecr_lifecycle_policy" "services" {
  for_each = toset(var.services)

  #  policy to the corresponding repository
  repository = aws_ecr_repository.services[each.key].name

  # policy is in JSON format
  policy = jsonencode({
    rules = [
      {
        # Rule priority (lower = runs first)
        rulePriority = 1

        # only two iages to keep
        description = "Keep only the last ${var.max_image_count} images"


        selection = {

          tagStatus   = "any" # all tagged+untagged images
          countType   = "imageCountMoreThan"
          countNumber = var.max_image_count # 2
        }

        # delete images exceeding the count
        action = {
          type = "expire" # Delete them
        }
      }
    ]
  })
}
