# EC2 Instance Configuration

# Get latest Amazon Linux 2023 AMI
data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "root-device-type"
    values = ["ebs"]
  }
}

# IAM Role for EC2 (to access S3)
resource "aws_iam_role" "ec2" {
  name = "${var.project_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com" # EC2 can use this role
        }
      }
    ]
  })

  tags = {
    Name = "${var.project_name}-ec2-role"
  }
}

# IAM Policy for S3 Access -< attach to EC2 Role- >What EC2 Can Do
# No password needed! EC2 automatically has permission.

resource "aws_iam_role_policy" "ec2_s3_access" {
  name = "${var.project_name}-ec2-s3-policy"
  role = aws_iam_role.ec2.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",    #Download files
          "s3:PutObject",    # Upload files
          "s3:DeleteObject", # Delete files
          "s3:ListBucket"    # List files
        ]
        Resource = [
          aws_s3_bucket.main.arn, # Bucket itself
          "${aws_s3_bucket.main.arn}/*"
        ]
      }
    ]
  })
}

# IAM Instance Profile -> Attaches Role to EC2
resource "aws_iam_instance_profile" "ec2" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2.name
}

# EC2 Instance
resource "aws_instance" "app" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.ec2.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2.name
  key_name               = aws_key_pair.generated.key_name

  root_block_device {
    volume_size           = 30  # AMI requires minimum 30GB
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true # Delete when EC2 deleted
  }


  # user data= script that runs AUTOMATICALLY when EC2 starts for the FIRST time.

  # base64encode= Converts text to a format AWS understands
  user_data_base64 = base64encode(<<-EOF
    #!/bin/bash
    # Update system
    dnf update -y # Update all software packages when ec2 starts for the first time
    
    # Install PostgreSQL client
    dnf install -y postgresql15 # Install PostgreSQL client (to connect to RDS)
    
    # Install AWS CLI (already included in AL2023)
    
    # Install useful tools-> 1. htop (system monitor), 2. vim (text editor), 3. git (version control)
    dnf install -y htop vim git
    
    # Create a connection test script-> helper script to test database connection
    cat > /home/ec2-user/test-db-connection.sh << SCRIPT
    #!/bin/bash
    # Test PostgreSQL connection
    # Usage: ./test-db-connection.sh <password>
    export PGHOST="${aws_db_instance.postgres.address}"
    export PGPORT="5432"
    export PGUSER="${var.db_username}"
    export PGDATABASE="${var.db_name}"
    export PGPASSWORD="\$1"
    psql -c "SELECT version();"
    SCRIPT
    
    chmod +x /home/ec2-user/test-db-connection.sh
    chown ec2-user:ec2-user /home/ec2-user/test-db-connection.sh
    
    echo "Setup complete!" > /home/ec2-user/setup-complete.txt
  EOF
  )

  tags = {
    Name = "${var.project_name}-app-server"
  }

  # Ensure the RDS instance is created first
  depends_on = [aws_db_instance.postgres]

  #   Order:
  # 1. Create RDS database     ← First
  # 2. Create EC2 instance     ← Second (needs RDS address)
}
