# Security Configuration for Model Loading

## Overview

The explainability service implements multiple security measures to prevent critical vulnerabilities including arbitrary code execution during model deserialization and Server-Side Request Forgery (SSRF) attacks.

## Security Measures

### 1. Restricted Unpickler

A custom `RestrictedUnpickler` class limits which Python modules can be loaded during deserialization:

- **Allowed Modules**: Only whitelisted ML frameworks (sklearn, numpy, pandas, scipy, xgboost, lightgbm, catboost)
- **Blocked**: All other modules including system modules that could execute arbitrary code
- **Protection**: Prevents malicious pickle files from executing system commands or importing dangerous libraries

### 2. SSRF Protection

All external URLs are validated before making HTTP requests to prevent Server-Side Request Forgery attacks:

- **Scheme Validation**: Only http and https schemes are allowed
- **DNS Resolution**: Hostnames are resolved to IP addresses before the request
- **IP Address Filtering**: Blocks access to:
  - Private networks (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
  - Loopback addresses (127.0.0.0/8, ::1)
  - Link-local addresses (169.254.0.0/16 - includes AWS metadata service at 169.254.169.254)
  - Multicast addresses
  - Reserved IP ranges
- **Protection**: Prevents attackers from using the service to access internal resources, cloud metadata services, or other restricted endpoints

### 3. Source Validation

Model files are validated to ensure they come from trusted sources:

#### Environment Variables

Configure trusted sources using these environment variables:

```bash
# Allowed local directories (comma-separated)
ALLOWED_MODEL_DIRS=/app/models,/data/trusted_models

# Allowed S3 buckets (comma-separated)
ALLOWED_S3_BUCKETS=my-company-ml-models,prod-model-bucket
```

#### Default Behavior

- If `ALLOWED_MODEL_DIRS` is not set or empty, all local paths are allowed
- If `ALLOWED_S3_BUCKETS` is not set or empty, all S3 URLs are allowed
- Pre-signed URLs from other HTTPS sources are allowed (authenticated via application layer)

### 4. File Size Limits

- Maximum model file size: 500 MB
- Prevents memory exhaustion attacks
- Validates both Content-Length header and actual downloaded size

### 5. Network Timeouts

- Download timeout: 30 seconds
- Prevents indefinite hangs on slow/malicious servers

## Configuration Examples

### Docker Environment

Add to your `docker-compose.yml`:

```yaml
services:
  mainflow:
    environment:
      - ALLOWED_MODEL_DIRS=/app/models,/data/ml_models
      - ALLOWED_S3_BUCKETS=prod-ml-models,staging-ml-models
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mainflow-security-config
data:
  ALLOWED_MODEL_DIRS: "/app/models,/data/ml_models"
  ALLOWED_S3_BUCKETS: "prod-ml-models,staging-ml-models"
```

### Local Development

Create a `.env` file:

```bash
ALLOWED_MODEL_DIRS=/home/user/ml_projects/models
ALLOWED_S3_BUCKETS=dev-ml-bucket
```

## Best Practices

1. **Restrict Model Sources**: Always configure `ALLOWED_MODEL_DIRS` and `ALLOWED_S3_BUCKETS` in production
2. **Use Joblib**: Prefer `.joblib` files over `.pkl` when possible (better compression, faster loading)
3. **Model Registry**: Implement a model registry system that vets models before they reach this service
4. **Audit Logging**: Monitor which models are being loaded and from where
5. **Regular Updates**: Keep ML frameworks updated to patch any security vulnerabilities

## Supported Model Formats

- `.joblib` - Recommended for sklearn models
- `.pkl` / `.pickle` - Supported with restricted unpickler

## Supported ML Frameworks

The restricted unpickler allows models from:

- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- NumPy/Pandas/SciPy (supporting libraries)

## Error Messages

### "Loading module 'X' is not allowed"

A pickle file tried to import a non-whitelisted module. This could be:
- A malicious model file attempting code execution
- A model using an unsupported framework
- **Solution**: Add the module to `SAFE_MODULES` if it's a legitimate ML framework

### "Access to [private/loopback/link-local] addresses is not allowed"

A URL resolved to a restricted IP address range. This prevents SSRF attacks.
- **Common causes**:
  - Attempting to access localhost, 127.0.0.1, or internal services
  - Trying to reach cloud metadata endpoints (169.254.169.254)
  - Accessing internal network resources (10.x.x.x, 192.168.x.x, etc.)
- **Solution**: Only use publicly accessible URLs for models and datasets

### "Model source not trusted"

The model path/URL doesn't match configured allowed sources.
- **Solution**: Add the directory or S3 bucket to the environment variables

### "Model file too large"

The model exceeds the 500 MB size limit.
- **Solution**: Optimize your model or increase `MAX_MODEL_SIZE` if necessary (consider memory implications)

## Testing Security

To test the security measures:

```python
# This should fail with "Loading module 'os' is not allowed"
import pickle

class Malicious:
    def __reduce__(self):
        import os
        return (os.system, ('echo hacked',))

with open('malicious.pkl', 'wb') as f:
    pickle.dump(Malicious(), f)

# Attempting to load this will be blocked by RestrictedUnpickler
```

## Additional Recommendations

1. **Network Isolation**: Run model loading in isolated containers
2. **File Scanning**: Use antivirus/malware scanning on uploaded model files
3. **Authentication**: Ensure only authorized users can trigger model loading
4. **Rate Limiting**: Prevent abuse through repeated model loading requests
5. **Audit Trail**: Log all model loading attempts with user context
