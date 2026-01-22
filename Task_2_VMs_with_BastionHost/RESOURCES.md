# Terraform AWS Setup (Simple Version)

## What this project builds

This Terraform project creates a **private AWS network** with **3 EC2 virtual machines**:

- **1 Windows Bastion (public)**
- **1 Linux server (public)**
- **1 Windows server (private + secure)**

---

## Network Setup

### 1) VPC (your private network)

- **CIDR:** `192.168.0.0/23`
- This is your **main private network** in AWS.

### 2) Subnets (smaller networks inside the VPC)

✅ **Public Subnet** (`192.168.0.0/25`)

- Has internet access
- Contains:
  - Windows Bastion
  - Linux VM

🔒 **Private Subnet** (`192.168.1.0/25`)

- No direct internet access
- Contains:
  - Private Windows VM

### 3) Internet Gateway

- Connects the **public subnet** to the internet
- The private subnet does **not** use it

### 4) Route Tables (traffic rules)

**Public Route Table**

- Internet traffic goes out through the Internet Gateway

**Private Route Table**

- No internet route → stays internal only

---

## Security (Firewall Rules)

### Bastion Security Group

- Allows **RDP (3389)** from anywhere (so you can connect)

### Linux Security Group

- Allows:
  - **SSH (22)**
  - **HTTP (80)**
  - **HTTPS (443)**

### Private Windows Security Group

- Allows **RDP + WinRM only from Bastion**
- No direct access from the internet

---

## Key Pair (Login Key)

Terraform creates **one `.pem` key**:

- Used for **SSH into Linux**
- Used to **decrypt Windows password**

Saved as:
`my-terraform-key.pem`

---

## EC2 Instances (3 VMs)

### ✅ Windows Bastion (Public)

- Public IP: **Yes**
- Used to access the private Windows VM

### ✅ Linux VM (Public)

- Public IP: **Yes**
- Used for web server / Linux work

### 🔒 Windows VM (Private)

- Public IP: **No**
- Only reachable through Bastion

---

## How you connect

### Linux VM

`You → Internet → Linux (SSH)`

### Bastion

`You → Internet → Bastion (RDP)`

### Private Windows VM

`You → Bastion → Private Windows (RDP)`  
(no direct access from your computer)

---

## Terraform Commands

```bash
terraform init
terraform validate
terraform plan
terraform apply
terraform destroy
```
