
# OUTPUTS

# VPC Outputs

output "vpc_id" {
  description = "ID of the created VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

# Subnet Outputs

output "public_subnet_id" {
  description = "ID of the public subnet"
  value       = aws_subnet.public.id
}

output "private_subnet_id" {
  description = "ID of the private subnet"
  value       = aws_subnet.private.id
}

# Security Group Outputs

output "bastion_security_group_id" {
  description = "ID of the Bastion security group"
  value       = aws_security_group.bastion.id
}

output "windows_security_group_id" {
  description = "ID of the Windows security group"
  value       = aws_security_group.windows.id
}

output "linux_security_group_id" {
  description = "ID of the Linux security group"
  value       = aws_security_group.linux.id
}

# Instance Outputs

output "bastion_public_ip" {
  description = "Public IP address of the Windows Bastion host"
  value       = aws_instance.bastion.public_ip
}

output "bastion_instance_id" {
  description = "Instance ID of the Windows Bastion host"
  value       = aws_instance.bastion.id
}

output "windows_vm_private_ip" {
  description = "Private IP address of the Windows VM"
  value       = aws_instance.windows_vm.private_ip
}

output "windows_vm_instance_id" {
  description = "Instance ID of the Windows VM"
  value       = aws_instance.windows_vm.id
}

output "linux_vm_public_ip" {
  description = "Public IP address of the Linux VM"
  value       = aws_instance.linux_vm.public_ip
}

output "linux_vm_instance_id" {
  description = "Instance ID of the Linux VM"
  value       = aws_instance.linux_vm.id
}


# Key Pair Outputs

output "key_pair_name" {
  description = "Name of the created key pair"
  value       = aws_key_pair.main.key_name
}

output "private_key_filename" {
  description = "Filename of the saved private key"
  value       = local_file.private_key.filename
}

# AMI Outputs

output "windows_ami_id" {
  description = "AMI ID used for Windows instances"
  value       = data.aws_ami.windows.id
}

output "linux_ami_id" {
  description = "AMI ID used for Linux instance"
  value       = data.aws_ami.linux.id
}

