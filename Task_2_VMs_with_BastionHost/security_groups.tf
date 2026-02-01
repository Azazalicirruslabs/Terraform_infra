
# SECURITY GROUPS - OS-BASED PORT CONFIGURATION

# Bastion Security Group (Windows)
# Allows RDP access from anywhere to connect to bastion-> in public subnet

resource "aws_security_group" "bastion" {
  name        = "${var.project_name}-bastion-sg"
  description = "Security group for Windows Bastion Host - allows RDP from internet"
  vpc_id      = aws_vpc.main.id

  # RDP Access from anywhere
  ingress {
    description = "RDP from anywhere"
    from_port   = local.bastion_ports.rdp # Start port (3389)
    to_port     = local.bastion_ports.rdp # End port (3389)
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # Anywhere
  }

  # Allow all outbound traffic
  egress {
    description = "Allow all outbound traffic"
    from_port   = 0 # All ports("0"-> Any)
    to_port     = 0
    protocol    = "-1" # All protocols("-1"-> Any)
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-bastion-sg"
    OS   = "Windows"
    Role = "Bastion"
  })
}

# Windows Security Group
# Allows RDP and WinRM access ONLY from Bastion

resource "aws_security_group" "windows" {
  name        = "${var.project_name}-windows-sg"
  description = "Security group for Windows VMs - allows RDP/WinRM from Bastion only"
  vpc_id      = aws_vpc.main.id

  # RDP Access from Bastion only
  ingress {
    description     = "RDP from Bastion"
    from_port       = local.windows_ports.rdp #3389
    to_port         = local.windows_ports.rdp
    protocol        = "tcp"
    security_groups = [aws_security_group.bastion.id]
  }

  # WinRM HTTP from Bastion #5985
  ingress {
    description     = "WinRM HTTP from Bastion"
    from_port       = local.windows_ports.winrm_http
    to_port         = local.windows_ports.winrm_http
    protocol        = "tcp"
    security_groups = [aws_security_group.bastion.id]
  }

  # WinRM HTTPS from Bastion # 5986
  ingress {
    description     = "WinRM HTTPS from Bastion"
    from_port       = local.windows_ports.winrm_https
    to_port         = local.windows_ports.winrm_https
    protocol        = "tcp"
    security_groups = [aws_security_group.bastion.id]
  }

  # Allow all outbound traffic 
  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-windows-sg"
    OS   = "Windows"
  })
}

# Linux Security Group
# Allows SSH, HTTP, HTTPS access from internet (public subnet VM)

resource "aws_security_group" "linux" {
  name        = "${var.project_name}-linux-sg"
  description = "Security group for Linux VM - allows SSH, HTTP, HTTPS from internet"
  vpc_id      = aws_vpc.main.id

  # SSH Access from anywhere (for remote login)
  ingress {
    description = "SSH from anywhere"
    from_port   = local.linux_ports.ssh
    to_port     = local.linux_ports.ssh
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTP Access from anywhere (for web server)
  ingress {
    description = "HTTP from anywhere"
    from_port   = local.linux_ports.http
    to_port     = local.linux_ports.http
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTPS Access from anywhere (for secure web server)
  ingress {
    description = "HTTPS from anywhere"
    from_port   = local.linux_ports.https
    to_port     = local.linux_ports.https
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow all outbound traffic
  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-linux-sg"
    OS   = "Linux"
  })
}
