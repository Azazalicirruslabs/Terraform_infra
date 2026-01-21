
# EC2 INSTANCES

# Windows Bastion Host (Public Subnet)

resource "aws_instance" "bastion" {
  ami                         = data.aws_ami.windows.id
  instance_type               = var.instance_type
  key_name                    = aws_key_pair.main.key_name
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.bastion.id] #Attaching multiple SGs
  associate_public_ip_address = true

  # Enable detailed monitoring - optional
  monitoring = false

  # Root volume configuration
  root_block_device {
    volume_size           = 30
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true # encrypt the root volume(disk)
  }

  tags = merge(local.common_tags, {
    Name = local.bastion_name
    OS   = "Windows"
    Role = "Bastion"
  })

  # Get Windows password using the key pair
  get_password_data = true # used for decrypting the password with pem key
}

# Windows VM (Private Subnet)
# Accessible only through Bastion Host

resource "aws_instance" "windows_vm" {
  ami                         = data.aws_ami.windows.id
  instance_type               = var.instance_type
  key_name                    = aws_key_pair.main.key_name
  subnet_id                   = aws_subnet.private.id
  vpc_security_group_ids      = [aws_security_group.windows.id]
  associate_public_ip_address = false # Private subnet - no public IP


  monitoring = false

  # Root volume configuration
  root_block_device {
    volume_size           = 30
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  tags = merge(local.common_tags, {
    Name = local.windows_vm_name
    OS   = "Windows"
    Role = "Application"
  })

  # Get Windows password using the key pair
  get_password_data = true
}

# -----------------------------------------------------------------------------
# Linux VM (Public Subnet)
# Has public IP, accessible directly from internet via SSH
# -----------------------------------------------------------------------------
resource "aws_instance" "linux_vm" {
  ami                         = data.aws_ami.linux.id
  instance_type               = var.instance_type
  key_name                    = aws_key_pair.main.key_name
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.linux.id]
  associate_public_ip_address = true # Public subnet - has public IP

  # Enable detailed monitoring (optional)
  monitoring = false

  # Root volume configuration
  root_block_device {
    volume_size           = 30
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  tags = merge(local.common_tags, {
    Name = local.linux_vm_name
    OS   = "Linux"
    Role = "Application"
  })
}
