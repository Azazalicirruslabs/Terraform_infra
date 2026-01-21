
# KEY PAIR - Same PEM KEY FOR ALL INSTANCES

# Generate both Public and TLS Private Key

resource "tls_private_key" "main" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

# AWS Key Pair
# Registers the public key with AWS

resource "aws_key_pair" "main" {
  key_name   = local.pem_key_name
  public_key = tls_private_key.main.public_key_openssh

  tags = merge(local.common_tags, { # Adding pem key name to tags
    Name = local.pem_key_name
  })
}

# Save Private Key to Local File
# The PEM file is saved locally for SSH/RDP access

resource "local_file" "private_key" {
  content         = tls_private_key.main.private_key_pem
  filename        = "${path.module}/${local.pem_key_filename}"
  file_permission = "0400"
}
