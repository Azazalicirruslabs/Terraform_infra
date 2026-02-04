# S3 Bucket Configuration

# Random suffix for globally unique bucket name
resource "random_id" "bucket_suffix" {
  byte_length = 4 # 8 hex characters-> terraform-dev-bucket-a1b2c3d4
}

# S3 Bucket
resource "aws_s3_bucket" "main" {
  bucket        = "${var.project_name}-bucket-${random_id.bucket_suffix.hex}"
  force_destroy = true # Delete bucket even if it has files (dev only!)

  tags = {
    Name = "${var.project_name}-bucket"
  }
}

# Block Public Access
resource "aws_s3_bucket_public_access_block" "main" {
  bucket = aws_s3_bucket.main.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable Versioning
resource "aws_s3_bucket_versioning" "main" {
  bucket = aws_s3_bucket.main.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Server-Side Encryption
#if someone hacks into AWS storage, they can't read your files!
resource "aws_s3_bucket_server_side_encryption_configuration" "main" {
  bucket = aws_s3_bucket.main.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# Lifecycle 
resource "aws_s3_bucket_lifecycle_configuration" "main" {
  bucket = aws_s3_bucket.main.id

  rule {
    id     = "expire-old-versions"
    status = "Enabled"

    filter {
      prefix = "" # Apply to all files
    }

    noncurrent_version_expiration {
      noncurrent_days = 30 # Delete old versions after 30 days
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7 # Clean up failed uploads after 7 days
    }
  }
}
