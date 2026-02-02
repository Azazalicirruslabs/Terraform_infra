# 📚 RESOURCES - Task_3_ECR Infrastructure Documentation

## 🎯 What This Project Does

This Terraform project automates the creation of AWS ECR (Elastic Container Registry) repositories and pushes Docker images for the RAIA (Responsible AI Analytics) platform microservices.

---

## 🔄 Infrastructure Flow

### Step-by-Step Execution

**1. Provider Initialization**
When you run `terraform init`, Terraform downloads the required AWS provider (v6.7.0) and sets up the backend. The provider is configured to work in the `ap-south-1` (Mumbai) region.

**2. Getting AWS Account Information**
Before creating any resources, Terraform fetches your AWS account ID and region. This information is needed to construct the ECR repository URLs for pushing images.

**3. ECR Repository Creation**
For each microservice defined in the `services` variable, Terraform creates a private ECR repository. Each repository is named with the pattern `raia-{service-name}` (e.g., `raia-api`, `raia-gateway`). The repositories are configured with:
- Image scanning on push (security vulnerability detection)
- AES256 encryption at rest
- Mutable image tags (allows overwriting `latest` tag)
- Force delete enabled (for easy cleanup during development)

**4. Lifecycle Policy Application**
Each repository gets a lifecycle policy that automatically deletes old images, keeping only the latest 2 images. This prevents storage costs from growing indefinitely.

**5. ECR Authentication**
Before pushing any images, Terraform logs into ECR using the AWS CLI. It retrieves a temporary authentication token and passes it to Docker. This token is valid for 12 hours, which is why we regenerate it on every `terraform apply`.

**6. Docker Image Build**
For each service, Terraform runs `docker build` from the project root directory. It uses the Dockerfile located in each service's folder (e.g., `services/API/Dockerfile`). The build context is the entire project root, allowing shared dependencies to be included.

**7. Image Tagging**
After building, the image is tagged with the full ECR repository URL. This tells Docker where to push the image.

**8. Image Push to ECR**
Finally, Docker pushes the tagged image to the AWS ECR repository. The image is now stored in AWS and can be pulled by ECS or other services.

---

## 📁 File Structure & Purpose

| File | Purpose |
|------|---------|
| `providers.tf` | Configures AWS provider, version constraints, default tags |
| `variables.tf` | Defines configurable inputs (region, services list, image count) |
| `main.tf` | Creates ECR repositories and lifecycle policies |
| `docker_push.tf` | Handles ECR login, Docker build, and image push |
| `outputs.tf` | Exports repository URLs, ARNs, and helper commands |

---

## 🛠️ Key Terraform Concepts Used

### terraform_data Resource
We use `terraform_data` (introduced in Terraform 1.4) instead of the older `null_resource`. This is a built-in resource that doesn't require an external provider. It's used to run local shell commands as part of the Terraform workflow.

### for_each Loop
Instead of writing separate resource blocks for each service, we use `for_each = toset(var.services)` to dynamically create resources for all services from a single block.

### triggers_replace
This argument tells Terraform when to re-run a `terraform_data` resource:
- `timestamp()` - Always re-run (used for ECR login to get fresh tokens)
- `filemd5(path)` - Only re-run when file content changes (used for Docker builds)

### local-exec Provisioner
Runs shell commands on your local machine (not on AWS). Used here to execute Docker CLI commands.

### depends_on
Ensures resources are created in the correct order. For example, ECR repositories must exist before we can push images to them.

---

## 🔐 Security Features

1. **Image Scanning**: Every pushed image is automatically scanned for vulnerabilities
2. **Encryption**: All images are encrypted at rest using AES256
3. **Private Repositories**: ECR repos are private by default, requiring authentication
4. **Temporary Credentials**: ECR login tokens expire after 12 hours

---

## 📖 Resources & References

### Official Documentation
- [Terraform AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [AWS ECR Terraform Resource](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/ecr_repository)
- [terraform_data Resource](https://developer.hashicorp.com/terraform/language/resources/terraform-data)
- [AWS ECR User Guide](https://docs.aws.amazon.com/AmazonECR/latest/userguide/what-is-ecr.html)

### Learning Resources
- [HashiCorp Learn - Terraform](https://learn.hashicorp.com/terraform)
- [Spacelift Blog - Terraform Best Practices](https://spacelift.io/blog/terraform-best-practices)

### AI tools used
- **ChatGPT**: for initial code generation and debugging
- **GitHub Copilot**: for code review and best practices guidance

---

###  YouTube (Channels)

- **HashiCorp** (official Terraform tutorials)
- **FreeCodeCamp** (full Terraform courses)
- **Tech Tutorials with Piyush** (Terraform Course)
- **Abhishek Veeramalla** (AWS + Terraform)
- **Rahul Wagh** (AWS + Terraform)

---

## 🚀 How to Use

### First Time Setup
```bash
cd Terraform
terraform init      # Download providers
terraform plan      # Preview changes
terraform apply     # Create resources and push images
```

### Testing Single Service
Edit `variables.tf` to test only one service:
```hcl
variable "services" {
  default = ["api"]  # Test only API
}
```

### Deploy All Services
Restore full service list in `variables.tf`:
```hcl
variable "services" {
  default = ["api", "classification", "data_drift", "fairness", "gateway", "mainflow", "regression", "what_if"]
}
```

### Cleanup
```bash
terraform destroy   # Delete all ECR repositories and images
```

---

## 📊 Services Covered

| Service | Description | ECR Repo Name |
|---------|-------------|---------------|
| API | Authentication & user management | raia-api |
| Classification | ML classification models | raia-classification |
| Data Drift | Data quality monitoring | raia-data_drift |
| Fairness | AI bias detection & analysis | raia-fairness |
| Gateway | API routing & unified docs | raia-gateway |
| Mainflow | Core workflow orchestration | raia-mainflow |
| Regression | Statistical regression analysis | raia-regression |
| What-If | What-if scenario analysis | raia-what_if |

---

## ⚠️ Prerequisites

- AWS CLI configured with appropriate credentials
- Docker installed and running
- Terraform v1.4+ installed
- Git Bash (on Windows) for shell commands

---

*Last Updated: February 2026*
