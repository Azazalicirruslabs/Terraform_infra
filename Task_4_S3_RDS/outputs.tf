# Output Values

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_id" {
  description = "ID of the public subnet"
  value       = aws_subnet.public.id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = [aws_subnet.private.id, aws_subnet.private_2.id]
}

output "ec2_instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.app.id
}

output "ec2_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.app.public_ip
}

output "ec2_public_dns" {
  description = "Public DNS of the EC2 instance"
  value       = aws_instance.app.public_dns
}

output "rds_endpoint" {
  description = "Endpoint of the RDS PostgreSQL instance"
  value       = aws_db_instance.postgres.endpoint
}

output "rds_address" {
  description = "Address of the RDS PostgreSQL instance"
  value       = aws_db_instance.postgres.address
}

output "rds_port" {
  description = "Port of the RDS PostgreSQL instance"
  value       = aws_db_instance.postgres.port
}

output "rds_database_name" {
  description = "Name of the PostgreSQL database"
  value       = aws_db_instance.postgres.db_name
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.main.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.main.arn
}

output "connection_instructions" {
  description = "Instructions for connecting to the infrastructure"
  value       = <<-EOT
    
    ============================================
    AWS Dev Infrastructure - Connection Details
    ============================================
    
    1. SSH to EC2:
       ssh -i ${var.project_name}-key.pem ec2-user@${aws_instance.app.public_ip}
    
    2. Connect to PostgreSQL from EC2:
       psql -h ${aws_db_instance.postgres.address} -U ${var.db_username} -d ${var.db_name}
    
    3. Test DB connection (from EC2):
       ./test-db-connection.sh <your-db-password>
    
    4. S3 Bucket:
       aws s3 ls s3://${aws_s3_bucket.main.bucket}/
    
    ============================================
  EOT
}

output "ssh_private_key_file" {
  description = "Path to the generated SSH private key file"
  value       = "${var.project_name}-key.pem"
}

output "ssh_command" {
  description = "SSH command to connect to EC2"
  value       = "ssh -i ${var.project_name}-key.pem ec2-user@${aws_instance.app.public_ip}"
}
