
# VPC resource creation
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.project_name}-${var.environment}-vpc"

  }
}

# Internet Gateway resource creation
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-${var.environment}-igw"

  }
}

# Public Subnets creation
resource "aws_subnet" "public" {
  count                   = length(var.public_subnet_cidrs) # Creates 2 subnets (one per CIDR)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index] # 0, 1, count=2
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-${var.environment}-public-subnet-${count.index + 1}"
    Type = "public"


  }
}

# Private Subnets creation

resource "aws_subnet" "private" {
  count = length(var.private_subnet_cidrs) # Creates 2 subnets

  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]

  # NO public IP - these are private!
  map_public_ip_on_launch = false

  tags = {
    Name = "${var.project_name}-private-subnet-${count.index + 1}"
    Type = "private"
  }
}

# CREATE ROUTE TABLE FOR PUBLIC SUBNETS

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  # Route to Internet: All traffic (0.0.0.0/0) goes to Internet Gateway
  route {
    cidr_block = "0.0.0.0/0"                  # Destination: Anywhere on internet
    gateway_id = aws_internet_gateway.main.id # Via: Internet Gateway
  }

  tags = {
    Name = "${var.project_name}-public-route-table"
  }
}

# Associate public subnets with the public route table

resource "aws_route_table_association" "public" {
  count = length(var.public_subnet_cidrs) # One association per subnet

  subnet_id      = aws_subnet.public[count.index].id # Which subnet
  route_table_id = aws_route_table.public.id         # Which route table
}

# CREATE ROUTE TABLE FOR PRIVATE SUBNETS

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-private-route-table"
  }
}

#  ASSOCIATE PRIVATE SUBNETS WITH PRIVATE ROUTE TABLE

resource "aws_route_table_association" "private" {
  count = length(var.private_subnet_cidrs)

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}
