# RDS PostgreSQL Configuration

# DB Subnet Group
# two private subnets for high availability (if needed in the future)
resource "aws_db_subnet_group" "main" {
  name        = "${var.project_name}-db-subnet-group"
  description = "Database subnet group for ${var.project_name}"
  subnet_ids  = [aws_subnet.private.id, aws_subnet.private_2.id]

  tags = {
    Name = "${var.project_name}-db-subnet-group"
  }
}

# RDS Parameter Group 
resource "aws_db_parameter_group" "postgres" {
  name        = "${var.project_name}-postgres-params"
  family      = "postgres16" # version 16
  description = "Custom parameter group for PostgreSQL 16"

  parameter {
    name  = "log_connections"
    value = "1" # Enable logging -> log when someone connects to the database
  }

  parameter {
    name  = "log_disconnections" # log when someone disconnects from the database
    value = "1"
  }

  tags = {
    Name = "${var.project_name}-postgres-params"
  }
}

# RDS PostgreSQL Instance
resource "aws_db_instance" "postgres" {
  identifier = "${var.project_name}-postgres"

  # Engine Configuration
  engine               = "postgres"
  engine_version       = "16"
  instance_class       = var.db_instance_class # db.t3.micro (small)
  parameter_group_name = aws_db_parameter_group.postgres.name

  # Storage Configuration
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = 100 # Enable autoscaling up to 100GB
  storage_type          = "gp3"
  storage_encrypted     = true

  # Database Configuration
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  port     = 5432

  # Network Configuration
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false
  multi_az               = false # Single AZ for dev (cost saving)

  # Backup Configuration -> automatic backups with a retention period of 7 days
  backup_retention_period = 7                     # Keep backups for 7 days
  backup_window           = "03:00-04:00"         # Backup at 3 AM
  maintenance_window      = "Mon:04:00-Mon:05:00" # Updates on Monday 4 AM

  # Performance Insights

  #   Performance Insights Dashboard          
  # │                                         
  # │ Shows you:                              
  # │ ├── Which queries are slow?             
  # │ ├── How much CPU is used?               
  # │ ├── How many connections?               
  # │ └── Database health  

  performance_insights_enabled          = true
  performance_insights_retention_period = 7



  # Deletion Protection 
  deletion_protection = false # Can delete database
  skip_final_snapshot = true  # Don't backup when deleting
  # When deleted, don't create backup

  # Apply changes immediately in dev
  apply_immediately = true # Changes happen NOW, not next maintenance window

  tags = {
    Name = "${var.project_name}-postgres"
  }
}
