# Task 1: VPC Infrastructure - Documentation

## 📋 Project Overview

This Terraform project creates a **production-ready VPC (Virtual Private Cloud)** foundation on AWS for the **RAIA** project. The infrastructure follows AWS best practices with multi-AZ deployment for high availability.

---

## 🏗️ Infrastructure Explained

### What is a VPC?

A **Virtual Private Cloud (VPC)** is your own isolated section of the AWS cloud. Think of it as your private data center in the cloud where you have complete control over the network configuration. Nothing can enter or leave your VPC without your permission.

### Our VPC Configuration

We created a VPC with the CIDR block `192.168.0.0/16`, which gives us **65,536 IP addresses** to work with. This VPC is deployed in the **Mumbai region (ap-south-1)** and is named `raia-dev-vpc`.

**Why DNS is enabled:** We enabled both `dns_support` and `dns_hostnames` because many AWS services (like ECS, RDS, and private endpoints) require DNS resolution to work properly. Without this, containers wouldn't be able to discover each other by name.

---

### The Two Types of Subnets

#### Public Subnets (Internet-Facing)

We created **2 public subnets**, one in each Availability Zone for high availability:

| Subnet | CIDR Block | Availability Zone | Usable IPs |
|--------|------------|-------------------|------------|
| Public Subnet 1 | `192.168.1.0/25` | ap-south-1a | 123 |
| Public Subnet 2 | `192.168.1.128/25` | ap-south-1b | 123 |

**What makes them "public"?**
1. They have a route to the **Internet Gateway**
2. Resources launched here automatically get a **public IP address**
3. They can send and receive traffic directly from the internet

**What goes here:** Load Balancers, Bastion hosts (jump servers), NAT Gateways

#### Private Subnets (Isolated)

We created **2 private subnets**, also spanning both Availability Zones:

| Subnet | CIDR Block | Availability Zone | Usable IPs |
|--------|------------|-------------------|------------|
| Private Subnet 1 | `192.168.2.0/25` | ap-south-1a | 123 |
| Private Subnet 2 | `192.168.2.128/25` | ap-south-1b | 123 |

**What makes them "private"?**
1. They have **NO route** to the Internet Gateway
2. Resources here do **NOT get public IP addresses**
3. They cannot be accessed directly from the internet

**What goes here:** Application containers (ECS), databases (RDS), backend services

---

### Internet Gateway (IGW)

The **Internet Gateway** is the door between your VPC and the public internet. It's attached to the VPC and allows resources in public subnets to communicate with the outside world.

**How it works:**
- When a resource in a public subnet wants to reach the internet, traffic goes: `Resource → Route Table → Internet Gateway → Internet`
- When someone from the internet wants to reach your public resource: `Internet → Internet Gateway → Route Table → Resource`

Without an Internet Gateway, your VPC would be completely isolated (which is actually what private subnets are).

---

### Route Tables (The GPS of Your VPC)

Route tables are like GPS navigation for network traffic. They tell AWS where to send packets based on their destination.

#### Public Route Table

| Destination | Target | Meaning |
|-------------|--------|---------|
| `192.168.0.0/16` | local | Traffic within VPC stays inside |
| `0.0.0.0/0` | Internet Gateway | Everything else goes to internet |

The `0.0.0.0/0` route means "any destination not in the VPC" - this is what enables internet access.

#### Private Route Table

| Destination | Target | Meaning |
|-------------|--------|---------|
| `192.168.0.0/16` | local | Traffic within VPC stays inside |

Notice there's **NO `0.0.0.0/0` route** - this is why private subnets cannot reach the internet. To enable internet access for private subnets, you would need to add a **NAT Gateway** (not included in this configuration).

---

### Why Multi-AZ? (High Availability)

We deployed subnets across **two Availability Zones** (ap-south-1a and ap-south-1b). Availability Zones are physically separate data centers within a region.

**Benefits:**
- If one AZ goes down (power outage, natural disaster), your application continues running in the other AZ
- AWS services like Application Load Balancer require subnets in at least 2 AZs
- It's an AWS best practice for production workloads

---

### Traffic Flow Example

**Scenario:** A user wants to access your web application

```
User (Internet)
    ↓
Internet Gateway (raia-dev-igw)
    ↓
Public Route Table (routes to public subnets)
    ↓
Application Load Balancer (in public subnet)
    ↓
Private Route Table (routes to private subnets)
    ↓
ECS Container (in private subnet) → responds back the same path
```

The user never directly touches the private subnet - the ALB acts as a secure gateway.

---

## 📁 File Structure

| File | Purpose |
|------|---------|
| `providers.tf` | AWS provider configuration with default tags |
| `variables.tf` | Input variables for customization |
| `main.tf` | Core infrastructure resources |
| `outputs.tf` | Output values for other modules |
| `RESOURCES.md` | This documentation file |

---

## 🔧 Resources Created

### 1. VPC (Virtual Private Cloud)
```hcl
aws_vpc.main
```
- **CIDR Block**: `192.168.0.0/16` (65,536 IP addresses)
- **DNS Hostnames**: Enabled (required for ECS, RDS)
- **DNS Support**: Enabled
- **Purpose**: Isolated network environment for all RAIA resources

### 2. Internet Gateway (IGW)
```hcl
aws_internet_gateway.main
```
- **Attached to**: VPC
- **Purpose**: Enables internet access for public subnets
- **How it works**: Routes traffic from public subnets to/from the internet

### 3. Public Subnets (2)
```hcl
aws_subnet.public[0]  # 192.168.1.0/25 in ap-south-1a
aws_subnet.public[1]  # 192.168.1.128/25 in ap-south-1b
```
- **Auto-assign Public IP**: Yes
- **Usable IPs per subnet**: 123 (128 - 5 AWS reserved)
- **Use cases**: Load Balancers, Bastion hosts, NAT Gateways

### 4. Private Subnets (2)
```hcl
aws_subnet.private[0]  # 192.168.2.0/25 in ap-south-1a
aws_subnet.private[1]  # 192.168.2.128/25 in ap-south-1b
```
- **Auto-assign Public IP**: No
- **Usable IPs per subnet**: 123
- **Use cases**: ECS containers, RDS databases, internal services

### 5. Route Tables
```hcl
aws_route_table.public   # Routes 0.0.0.0/0 to IGW
aws_route_table.private  # Local routes only
```

### 6. Route Table Associations
```hcl
aws_route_table_association.public[*]   # Links public subnets to public RT
aws_route_table_association.private[*]  # Links private subnets to private RT
```

---

## 📊 CIDR Block Breakdown

### Why /25 for subnets?
| CIDR | Total IPs | AWS Reserved | Usable IPs |
|------|-----------|--------------|------------|
| `/25` | 128 | 5 | **123** |

### AWS Reserved IPs (per subnet):
1. `.0` - Network address
2. `.1` - VPC router
3. `.2` - DNS server
4. `.3` - Reserved for future use
5. `.255` - Broadcast (for /24, varies by CIDR)

---

## 🌐 Networking Concepts Explained

### Public Subnet vs Private Subnet

| Aspect | Public Subnet | Private Subnet |
|--------|--------------|----------------|
| Internet Access | ✅ Direct via IGW | ❌ None (needs NAT) |
| Public IP | ✅ Auto-assigned | ❌ No public IP |
| Inbound from Internet | ✅ Possible | ❌ Not possible |
| Use Cases | ALB, Bastion, NAT GW | ECS, RDS, Lambda |

### Route Table Logic
```
PUBLIC SUBNET ROUTE TABLE:
┌─────────────────┬───────────────────┐
│ Destination     │ Target            │
├─────────────────┼───────────────────┤
│ 192.168.0.0/16  │ local             │  ← VPC internal traffic
│ 0.0.0.0/0       │ igw-xxxxx         │  ← Internet traffic
└─────────────────┴───────────────────┘

PRIVATE SUBNET ROUTE TABLE:
┌─────────────────┬───────────────────┐
│ Destination     │ Target            │
├─────────────────┼───────────────────┤
│ 192.168.0.0/16  │ local             │  ← VPC internal only
└─────────────────┴───────────────────┘
```

---

## 📚 Learning Resources Used

### 🎥 YouTube Channels

- **HashiCorp** (official Terraform tutorials)
- **FreeCodeCamp** (full Terraform courses)
- **Tech Tutorials with Piyush** (Terraform Course)
- **Abhishek Veeramalla** (AWS + Terraform)
- **Rahul Wagh** (AWS + Terraform)

### 📖 Official Documentation

| Resource | Description | Link |
|----------|-------------|------|
| **Terraform AWS Provider** | Official provider docs | [registry.terraform.io/providers/hashicorp/aws](https://registry.terraform.io/providers/hashicorp/aws/latest/docs) |
| **AWS VPC User Guide** | VPC concepts | [docs.aws.amazon.com/vpc](https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html) |
| **Terraform Language Docs** | HCL syntax, functions | [developer.hashicorp.com/terraform](https://developer.hashicorp.com/terraform/language) |
| **AWS Subnet Calculator** | CIDR planning | [cidr.xyz](https://cidr.xyz/) |

### 📝 Blogs & Articles

| Blog | Topic | Link |
|------|-------|------|
| **Spacelift** | Terraform best practices | [spacelift.io/blog](https://spacelift.io/blog/terraform-best-practices) |
| **Medium - Terraform** | Community tutorials | [medium.com/tag/terraform](https://medium.com/tag/terraform) |
| **Dev.to** | DevOps articles | [dev.to/t/terraform](https://dev.to/t/terraform) |
| **AWS Architecture Blog** | Reference architectures | [aws.amazon.com/blogs/architecture](https://aws.amazon.com/blogs/architecture/) |

### 🤖 AI Tools Used

| Tool | Purpose |
|------|---------|
| **Gemini** | Code generation, debugging, explanations |
| **ChatGPT** | Code review, debugging, explanations |

---

## ⚙️ Terraform Commands Reference

```bash
# Initialize Terraform (download providers)
terraform init

# Format code
terraform fmt

# Validate configuration
terraform validate

# Preview changes
terraform plan

# Apply changes
terraform apply

# Destroy infrastructure
terraform destroy
```

---

## 🚀 Future Enhancements

This VPC foundation is ready for:
- [ ] NAT Gateway (for private subnet internet access)
- [ ] Security Groups
- [ ] VPC Endpoints (S3, ECR, CloudWatch)
- [ ] Application Load Balancer
- [ ] ECS Fargate clusters
- [ ] RDS databases

---

## 📋 Tags Applied

All resources are automatically tagged via `default_tags`:
```hcl
project     = "raia"
Environment = "dev"
ManagedBy   = "Terraform"
```

---

*Last Updated: February 1, 2026*
