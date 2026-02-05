# Terraform AWS Dev Infrastructure

## What This Project Builds

This Terraform project creates a **minimal AWS development environment** with:

- **1 VPC** (Private Network)
- **1 EC2 Instance** (Application Server)
- **1 RDS PostgreSQL Database** (Data Storage)
- **1 S3 Bucket** (File Storage)

---

## Architecture Flow

### How Traffic Flows Through the Infrastructure

**Step 1: User Access**  
You connect from your computer through the internet to your EC2 instance using SSH (port 22). The Internet Gateway allows this traffic to enter the VPC, and the EC2 security group permits SSH connections.

**Step 2: EC2 in Public Subnet**  
The EC2 instance lives in the public subnet (`192.168.0.0/25`), which has a route to the Internet Gateway. This gives EC2 a public IP address that you can connect to directly.

**Step 3: EC2 Connects to RDS**  
From inside EC2, you connect to the RDS PostgreSQL database on port 5432. The database lives in the private subnet (`192.168.0.128/25`), which has NO internet access. Only resources with the EC2 security group attached can connect to RDS.

**Step 4: EC2 Accesses S3**  
EC2 can upload/download files to S3 without any password. This works because EC2 has an IAM Role attached that grants S3 permissions automatically.

**Step 5: RDS Stays Protected**  
The RDS database cannot be accessed directly from the internet. It's in a private subnet with no route to the Internet Gateway, and the security group only allows connections from EC2.

---

## Network Setup

### 1) VPC (Private Network)

- **CIDR:** `192.168.0.0/16` (65,536 IP addresses)
- **Region:** `ap-south-2` (Hyderabad)
- **DNS:** Enabled

### 2) Subnets

| Subnet | CIDR | Type | Purpose |
|--------|------|------|---------|
| Public | `192.168.0.0/25` | Public | EC2 Instance |
| Private 1 | `192.168.0.128/25` | Private | RDS Database |
| Private 2 | `192.168.1.0/25` | Private | RDS (required 2 AZs) |

### 3) Internet Gateway

- Connects public subnet to internet
- Private subnets have NO internet access

### 4) Route Tables

**Public Route Table:**
- `0.0.0.0/0` → Internet Gateway ✅

**Private Route Table:**
- No internet route → Internal only 🔒

---

## Security Groups (Firewall Rules)

### EC2 Security Group

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22 | TCP | 0.0.0.0/0 | SSH Access |
| 80 | TCP | 0.0.0.0/0 | HTTP |
| 443 | TCP | 0.0.0.0/0 | HTTPS |

### RDS Security Group

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 5432 | TCP | EC2 Security Group | PostgreSQL (EC2 only!) |

---

## EC2 Instance

| Property | Value |
|----------|-------|
| **AMI** | Amazon Linux 2023 (latest) |
| **Type** | t3.micro (2 vCPU, 1GB RAM) |
| **Storage** | 30GB gp3 SSD (encrypted) |
| **Subnet** | Public (has public IP) |

**Pre-installed Software:**
- PostgreSQL 15 client
- AWS CLI
- htop, vim, git

---

## RDS PostgreSQL

| Property | Value |
|----------|-------|
| **Engine** | PostgreSQL 16 |
| **Type** | db.t3.micro |
| **Storage** | 20GB gp3 (auto-scale to 100GB) |
| **Encryption** | Enabled |
| **Backup** | 7 days retention |
| **Multi-AZ** | No (dev environment) |

**Access:**
- ❌ Not publicly accessible
- ✅ Only from EC2 via security group

---

## S3 Bucket

| Property | Value |
|----------|-------|
| **Name** | terraform-dev-bucket-xxxxxxxx |
| **Encryption** | AES-256 (SSE-S3) |
| **Versioning** | Enabled |
| **Public Access** | Blocked |
| **Lifecycle** | Delete old versions after 30 days |

**Access:**
- EC2 has IAM Role with S3 permissions
- No password needed from EC2

---

## SSH Key Pair

Terraform automatically generates:
- **Key Name:** `terraform-dev-key`
- **File:** `terraform-dev-key.pem`
- **Algorithm:** RSA 4096-bit

---

## Data Flow

### User → EC2 (SSH)
```
You → Internet → IGW → Public Subnet → EC2 (port 22)
```

### EC2 → RDS (Database)
```
EC2 → Private Subnet → RDS (port 5432)
  ↑
  Security Group allows EC2 only
```

### EC2 → S3 (Files)
```
EC2 → AWS API → S3 Bucket
  ↑
  IAM Role (no password!)
```

---

## How to Connect

### SSH into EC2
```bash
ssh -i terraform-dev-key.pem ec2-user@<EC2_PUBLIC_IP>
```

### Connect to PostgreSQL (from EC2)
```bash
psql -h <RDS_ENDPOINT> -U dbadmin -d devdb
```

### Access S3 (from EC2)
```bash
aws s3 ls s3://terraform-dev-bucket-xxxxx/
aws s3 cp file.txt s3://terraform-dev-bucket-xxxxx/
```

---

## Terraform Commands

```bash
# Initialize providers
terraform init

# Validate configuration
terraform validate

# Preview changes
terraform plan

# Create infrastructure
terraform apply

# Destroy infrastructure
terraform destroy

# View outputs
terraform output
```

---

## File Structure

```
Task_4_S3_RDS/
├── providers.tf          # Terraform & AWS provider config
├── variables.tf          # Input variable definitions
├── terraform.tfvars      # Your actual values (gitignored)
├── vpc.tf                # VPC, subnets, route tables
├── security_groups.tf    # Firewall rules
├── ssh_key.tf            # SSH key generation
├── s3.tf                 # S3 bucket configuration
├── rds.tf                # PostgreSQL database
├── ec2.tf                # EC2 instance
├── outputs.tf            # Output values
├── .gitignore            # Protects sensitive files
└── terraform-dev-key.pem # SSH private key (gitignored)
```

---

## Cost Estimate (Monthly)

| Resource | Type | Est. Cost |
|----------|------|-----------|
| EC2 | t3.micro | ~$8 (free tier eligible) |
| RDS | db.t3.micro | ~$12 (free tier eligible) |
| S3 | Minimal usage | ~$1 |
| **Total** | | **~$21/month** |

*First 12 months may be free under AWS Free Tier*

---

## Resources Used

### 1) Official Documentation

- **Terraform Documentation**  
  https://developer.hashicorp.com/terraform/docs

- **Terraform AWS Provider Docs**  
  https://registry.terraform.io/providers/hashicorp/aws/latest/docs

- **Terraform TLS Provider Docs**  
  https://registry.terraform.io/providers/hashicorp/tls/latest/docs

- **AWS VPC User Guide**  
  https://docs.aws.amazon.com/vpc/latest/userguide/

- **AWS EC2 User Guide**  
  https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/

- **AWS RDS User Guide**  
  https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/

- **AWS S3 User Guide**  
  https://docs.aws.amazon.com/AmazonS3/latest/userguide/

---

### 2) YouTube Channels

- **HashiCorp** - Official Terraform tutorials  
  https://www.youtube.com/@HashiCorp

- **FreeCodeCamp** - Full Terraform courses  
  https://www.youtube.com/@freecodecamp

- **Abhishek Veeramalla** - AWS + Terraform DevOps  
  https://www.youtube.com/@AbhishekVeeramalla

- **TechWorld with Nana** - Terraform & Cloud tutorials  
  https://www.youtube.com/@TechWorldwithNana

- **Cloud With Raj** - AWS Infrastructure  
  https://www.youtube.com/@cloudwithraj

---

### 3) Blogs & Articles

- **Terraform Best Practices**  
  https://www.terraform-best-practices.com/

- **AWS Blog**  
  https://aws.amazon.com/blogs/

- **HashiCorp Blog**  
  https://www.hashicorp.com/blog

- **Medium (Terraform tag)**  
  https://medium.com/tag/terraform

- **Dev.to (Terraform tag)**  
  https://dev.to/t/terraform

---

### 4) AWS Provider Changelog

- **AWS Provider 6.x Changes**  
  https://registry.terraform.io/providers/hashicorp/aws/latest/docs/guides/version-6-upgrade

---

### 5) AI Tools Used

- **Gemini** - Code assistance and debugging

---

## Version Information

| Tool | Version |
|------|---------|
| Terraform | >= 1.14.0 |
| AWS Provider | ~> 6.30 |
| TLS Provider | ~> 4.0 |
| Local Provider | ~> 2.5 |
| Random Provider | ~> 3.6 |

---

## Security Notes

⚠️ **For Development Only:**
- `multi_az = false` (no failover)
- `skip_final_snapshot = true`
- `deletion_protection = false`
- SSH open to `0.0.0.0/0`

🔒 **For Production, change:**
- `multi_az = true`
- `skip_final_snapshot = false`
- `deletion_protection = true`
- Restrict `allowed_ssh_cidr` to your IP

---

## Author

Created with Terraform and AI assistance.  
Date: February 2026
