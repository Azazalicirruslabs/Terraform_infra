
# DATA SOURCES - FETCH LATEST AMIs

# Latest Windows Server 

data "aws_ami" "windows" {
  most_recent = true
  owners      = ["amazon"] #only official Amazon AMIs

  filter {
    name   = "name"
    values = ["Windows_Server-2022-English-Full-Base-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"] #Hardware Virtual Machine (standard)
  }

  filter {
    name   = "architecture"
    values = ["x86_64"] #64-bit architecture
  }
}


# Latest Amazon Linux 

data "aws_ami" "linux" {
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
    name   = "architecture"
    values = ["x86_64"]
  }
}
