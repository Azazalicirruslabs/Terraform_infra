# 🚀 RAIA Platform - API Gateway (Cloud Run Optimized)

## Overview

The RAIA Platform API Gateway is an intelligent FastAPI-based service optimized for **Cloud Run deployment** with **0-min instance scaling**. It provides automatic service discovery, essential production debugging, and streamlined architecture perfect for ML workloads.

### ✨ Key Features

- **🔍 Auto-Discovery**: Automatically finds services via environment variables
- **🏗️ 3-Way Service Management**: Environment variables, fallback defaults, unified REST API
- **🔄 Dynamic Routing**: Routes generated automatically at startup
- **📊 Health Monitoring**: Built-in service health tracking optimized for cold starts
- **📋 Unified OpenAPI**: Merged documentation from all services with auto-discovery
- **☁️ Cloud Run Optimized**: Extended timeouts and thresholds for 0-min instance scaling
- **🔧 Essential Debug Endpoints**: Minimal set of production-critical debugging tools
- **⚡ 7-Endpoint Architecture**: Perfect balance of functionality and simplicity

## 🚀 Quick Start

### 1. Basic Setup (Auto-Discovery)

```bash
# Set service URLs via environment variables
export API_SERVICE_URL="https://your-api-service.com"
export FAIRNESS_SERVICE_URL="https://your-fairness-service.com"
export CLASSIFICATION_SERVICE_URL="https://your-classification-service.com"
export REGRESSION_SERVICE_URL="https://your-regression-service.com"
export DATA_DRIFT_SERVICE_URL="https://your-data-drift-service.com"
export FRONTEND_SERVICE_URL="https://your-frontend-service.com"

# Start gateway
python app/main.py
```

**That's it!** The gateway will:

- ✅ Auto-discover all 6 services
- ✅ Create 7 dynamic routes automatically
- ✅ Generate health/OpenAPI endpoints
- ✅ Provide unified documentation at `/docs`

### 2. Adding a New Service

#### Method A: Environment Variable (Recommended)

```bash
# Add new service - gateway discovers automatically
export ANALYTICS_SERVICE_URL="https://your-analytics-service.com"
# Restart gateway - new service is auto-discovered and routed!
```

#### Method B: Runtime API

```bash
# Add service while gateway is running
curl -X POST "http://localhost:8000/gateway/registry/services/analytics" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-analytics-service.com",
    "prefix": "analytics",
    "has_openapi": true,
    "health_path": "/analytics/health",
    "openapi_path": "/analytics/openapi.json"
  }'
```

## 🛠️ Service Management - 3 Essential Ways

The gateway provides **3 streamlined approaches** for service management, optimized for different scenarios:

### 1. Environment Variables (Production Deployments)

```bash
export API_SERVICE_URL="https://custom-api.com"
export FAIRNESS_SERVICE_URL="https://custom-fairness.com"
```

### 2. Fallback Defaults (Development/Testing)

```bash
export GATEWAY_DISABLE_AUTO_DISCOVERY=true
# Gateway uses built-in fallback URLs for core services
```

### 3. Unified REST API (Runtime Management)

```bash
# Add or update service
curl -X POST "http://localhost:8000/gateway/registry/services/new_service" \
  -d '{
    "url": "https://new-service.com",
    "prefix": "new_service",
    "health_path": "/new_service/health",
    "openapi_path": "/new_service/openapi.json",
    "full_control": true
  }'

# Remove service
curl -X DELETE "http://localhost:8000/gateway/registry/services/old_service"

# Check registry status
curl "http://localhost:8000/gateway/registry/status"
```

## 📊 Service Discovery Conventions

| Pattern | Example | Result |
|---------|---------|--------|
| Environment Variable | `ANALYTICS_SERVICE_URL=https://analytics.com` | Auto-discovered as `analytics` service |
| Route Prefix | Service name `analytics` | Route: `/analytics/{path:path}` |
| Health Endpoint | Service `analytics` | Health: `/analytics/health` |
| OpenAPI Endpoint | Service `analytics` | OpenAPI: `/analytics/openapi.json` |
| Frontend Service | `FRONTEND_SERVICE_URL` | Special handling: `/` and `/{path:path}` |

## 🔧 Gateway Endpoints (7 Total - Optimized Architecture)

### Core Gateway Management (4 Endpoints)

- `GET /gateway/health` - Overall platform health check
- `GET /gateway/registry/status` - **ENHANCED**: Service registry statistics + detailed configurations (URL, prefix, health path, OpenAPI path)
- `GET /gateway/service-management-guide` - Complete service management documentation
- `POST/PUT/DELETE /gateway/registry/services/{name}` - Unified service management API

### Essential Production Debug (3 Endpoints)

- `GET /gateway/debug-health` - **PRODUCTION ESSENTIAL**: Detailed health tracking for SRE/DevOps monitoring
- `GET /gateway/debug-cache` - **PRODUCTION ESSENTIAL**: Cache inspection for OpenAPI troubleshooting
- `POST /gateway/debug-cache/clear` - **PRODUCTION ESSENTIAL**: Emergency cache clearing for recovery

### Auto-Generated Documentation

- `GET /docs` - Unified Swagger UI for all services (auto-discovered endpoints)
- `GET /redoc` - ReDoc documentation
- `GET /openapi.json` - Combined OpenAPI specification with auto-discovery

> **Note**: Debug endpoints are specifically designed for **Cloud Run 0-min instance deployments** to handle cold starts, cache management, and incident response.

## 🏗️ Architecture

### Before: Manual Maintenance (❌ High Complexity)

```python
# 53+ lines of manual maintenance required!

# Manual service URLs (8 lines)
SERVICES = {
    "api": "https://api-service.com",
    "fairness": "https://fairness-service.com",
    # ... manual additions for each service
}

# Manual route functions (30+ lines)
@app.api_route("/api/{path:path}")
async def route_to_api(path: str, request: Request):
    # ... manual routing code for each service

# Manual debug endpoints (15+ lines)
@app.get("/gateway/debug-urls")
@app.get("/gateway/debug-service-specs")
@app.get("/gateway/debug-endpoints")
@app.get("/gateway/debug-ml-performance")
# ... 5+ development-only debug endpoints
```

### After: Auto-Discovery + Essential Debug (✅ Optimized)

```python
# 3 lines achieve everything!

service_registry = ServiceRegistry()  # Auto-discovers services
create_dynamic_routes()               # Creates routes automatically

# Only 3 essential debug endpoints for production:
# /gateway/debug-health - SRE monitoring
# /gateway/debug-cache - OpenAPI troubleshooting
# /gateway/debug-cache/clear - Emergency recovery
```

**Result**: **53+ lines reduced to 3 lines** (95% reduction) + **7 perfectly optimized endpoints**

## 🎯 Service Management: Current vs Optimized

### Before: Manual Process (❌ Complex)

1. ❌ Add service URL to SERVICES dict
2. ❌ Add health endpoint to health_endpoints dict
3. ❌ Add OpenAPI endpoint to openapi_endpoints dict
4. ❌ Create @app.api_route function for service
5. ❌ Add multiple debug endpoints for development

**Total**: 5+ manual steps + code changes + redeployment

### After: Auto-Discovery + Essential Debug (✅ Streamlined)

1. ✅ Set environment variable: `export NEW_SERVICE_URL="https://new.com"`
2. ✅ 3 essential debug endpoints handle all production scenarios

**Total**: 1 environment variable + optimal debugging!

## ☁️ Cloud Run Deployment (0-Min Instance Optimized)

### Current Cloud Run Configuration

**Gateway**: 0-5 instances (High availability)
**Services**: 0-2 instances (Cost-optimized)

### Docker Environment Variables

```dockerfile
ENV API_SERVICE_URL=https://api-prod.company.com
ENV FAIRNESS_SERVICE_URL=https://fairness-prod.company.com
ENV CLASSIFICATION_SERVICE_URL=https://classification-prod.company.com
ENV REGRESSION_SERVICE_URL=https://regression-prod.company.com
ENV DATA_DRIFT_SERVICE_URL=https://datadrift-prod.company.com
ENV FRONTEND_SERVICE_URL=https://frontend-prod.company.com
```

### Cloud Run Deployment Command

```bash
gcloud run deploy gateway \
  --set-env-vars="API_SERVICE_URL=https://api-service-hash.run.app" \
  --set-env-vars="FAIRNESS_SERVICE_URL=https://fairness-service-hash.run.app" \
  --set-env-vars="CLASSIFICATION_SERVICE_URL=https://classification-service-hash.run.app" \
  --min-instances=0 \
  --max-instances=5 \
  --cpu=2 \
  --memory=4Gi
```

### Cloud Run Performance Expectations

- **Cold Start Times**: 10-30 seconds for ML services (model loading)
- **Warm Response**: 1-5 seconds for active instances
- **Idle Timeout**: ~15 minutes before scaling to zero
- **Debug Endpoints**: Essential for monitoring cold start impact and cache state

## 🔍 Monitoring & Health Tracking

### Gateway Health Check

```bash
curl http://localhost:8000/gateway/health
```

Response includes **Cloud Run optimized** health metrics:

```json
{
  "gateway": {
    "status": "healthy",
    "version": "1.0.0",
    "check_duration_ms": 245.8
  },
  "services": {
    "total": 6,
    "healthy": 5,
    "unhealthy": 1,
    "details": [
      {
        "service": "api",
        "status": "healthy",
        "response_time_ms": 1234.5
      }
    ]
  }
}
```

### Essential Debug Endpoints for Cloud Run

```bash
# Detailed health tracking for SRE/DevOps monitoring
curl http://localhost:8000/gateway/debug-health

# Cache inspection for OpenAPI troubleshooting
curl http://localhost:8000/gateway/debug-cache

# Emergency cache clearing for recovery
curl -X POST http://localhost:8000/gateway/debug-cache/clear
```

### Service Registry Status

```bash
curl http://localhost:8000/gateway/registry/status
```

Response shows the **3-Way Service Management** system with **detailed service configurations**:

```json
{
  "status": "active",
  "management_system": "3_essential_ways",
  "service_counts": {
    "total_services": 6,
    "auto_discovered": 6,
    "manually_managed": 0
  },
  "services": [
    {
      "name": "api",
      "url": "https://raia-api-dev-1020052770212.europe-west1.run.app",
      "prefix": "api",
      "health_path": "/api/health",
      "openapi_path": "/api/openapi.json",
      "has_openapi": true,
      "discovery_method": "auto-discovery"
    },
    {
      "name": "fairness",
      "url": "https://raia-fairness-dev-1020052770212.europe-west1.run.app",
      "prefix": "fairness",
      "health_path": "/fairness/health",
      "openapi_path": "/fairness/openapi.json",
      "has_openapi": true,
      "discovery_method": "auto-discovery"
    }
  ],
  "management_methods": {
    "1_environment_variables": "export *_SERVICE_URL for deployment automation",
    "2_fallback_defaults": "automatic core services when no env vars found",
    "3_unified_rest_api": "POST/PUT /gateway/registry/services/{name} with full control"
  }
}
```

## 🧪 Testing & Debugging

### Test Auto-Discovery

```bash
# Set test environment variable
export TEST_SERVICE_URL="https://test-service.com"

# Restart gateway to auto-discover
python app/main.py

# Verify discovery
curl http://localhost:8000/gateway/registry/status
```

### Test Runtime Service Management

```python
from app.main import service_registry

# Test manual addition with full control
service_registry.add_manual_service(
    name="test",
    url="https://test.com",
    prefix="test",
    health_path="/test/health",
    openapi_path="/test/openapi.json",
    full_control=True
)

# Test URL update
service_registry.update_service_url("test", "https://new-test.com")

# Test removal
service_registry.remove_service("test")
```

### Essential Debug Testing

```bash
# Test health tracking (Cloud Run optimized)
curl http://localhost:8000/gateway/debug-health

# Test cache management (critical for 0-min instances)
curl http://localhost:8000/gateway/debug-cache

# Test emergency recovery
curl -X POST http://localhost:8000/gateway/debug-cache/clear
```

## 🛡️ Cloud Run Resilience & Error Handling

### Cold Start Optimization

- **Extended Timeouts**: 45-60s+ for ML model loading on first request
- **Progressive Retries**: 3 attempts with exponential backoff for cold starts
- **Fallback Specs**: Automatic fallback OpenAPI specs when services unavailable
- **Health Thresholds**: 70% failure rate threshold (accounts for cold start timeouts)

### Cache Management (Critical for 0-Min Instances)

- **OpenAPI Caching**: 5-minute TTL, automatic invalidation on failures
- **Cache Loss Recovery**: When instances scale to 0, cache rebuilds automatically
- **Emergency Cache Clear**: `/gateway/debug-cache/clear` for incident response
- **Cache Efficiency Monitoring**: `/gateway/debug-cache` tracks cache performance

### Circuit Breaking & Recovery

- **Health Tracking**: Real-time monitoring optimized for Cloud Run scaling patterns
- **Automatic Recovery**: Service health status updates when instances come online
- **Failure Detection**: Distinguishes between cold starts and actual service failures

## 📈 Performance Optimization

### Connection Management
- **Connection Pooling**: 20 keepalive connections
- **High Concurrency**: 100 max connections
- **Connection Reuse**: 30-second keepalive expiry

### ML Workload Optimization
- **Model Loading Timeouts**: 2+ minutes for cold starts
- **Large File Support**: Extended upload/download timeouts
- **Concurrent Processing**: Non-blocking async operations

## 🔧 Local Development

### Development Setup

```bash
# Use local services for development
export API_SERVICE_URL="http://localhost:8001"
export FAIRNESS_SERVICE_URL="http://localhost:8002"
export CLASSIFICATION_SERVICE_URL="http://localhost:8003"
export REGRESSION_SERVICE_URL="http://localhost:8004"
export DATA_DRIFT_SERVICE_URL="http://localhost:8005"
export FRONTEND_SERVICE_URL="http://localhost:3000"

# Start gateway
python app/main.py
```

### Essential Debug Mode

```bash
# Check health tracking for local services
curl http://localhost:8000/gateway/debug-health

# Monitor cache for local development
curl http://localhost:8000/gateway/debug-cache

# Clear cache during development iterations
curl -X POST http://localhost:8000/gateway/debug-cache/clear
```

### Service Registry Management

```bash
# View current service configuration
curl http://localhost:8000/gateway/registry/status

# Add a local test service
curl -X POST "http://localhost:8000/gateway/registry/services/test" \
  -d '{
    "url": "http://localhost:9000",
    "prefix": "test",
    "has_openapi": true,
    "health_path": "/test/health",
    "openapi_path": "/test/openapi.json"
  }'
```

## 📋 Migration Guide

### From Manual to Auto-Discovery

1. **Backup Current Configuration**
   ```bash
   # Save current SERVICES dict
   cp app/main.py app/main.py.backup
   ```

2. **Set Environment Variables**
   ```bash
   export API_SERVICE_URL="$CURRENT_API_URL"
   export FAIRNESS_SERVICE_URL="$CURRENT_FAIRNESS_URL"
   # ... etc
   ```

3. **Deploy New Version**
   ```bash
   # Deploy with auto-discovery enabled
   python app/main.py
   ```

4. **Verify Services**
   ```bash
   curl http://localhost:8000/gateway/registry/status
   ```

5. **Remove Manual Code** (after verification)
   - Delete manual SERVICES dict
   - Remove manual route functions
   - Remove manual endpoint mappings

## 🚀 Future Roadmap

- **Service Mesh Integration**: Istio/Linkerd discovery
- **Load Balancing**: Multiple instance discovery per service
- **Circuit Breaker Patterns**: Automatic failure isolation
- **Metrics Integration**: Prometheus/Grafana dashboards
- **Configuration UI**: Web interface for service management

## 📞 Support

For issues, feature requests, or questions:
- **Repository**: [XAI-Explainability](https://github.com/cirruslabs-io/XAI-Explainability)
- **Issues**: GitHub Issues tracker
- **Documentation**: `/docs` endpoint when gateway is running

---

**🎉 Zero Manual Service Registration Required!**
The gateway automatically discovers and routes all services via environment variables while preserving complete manual control for edge cases.

## 🎯 Active Service Routes (Auto-Discovered)

| Service | Path Pattern | Purpose | Health Check |
|---------|-------------|---------|--------------|
| **Frontend** | `/` (default) | React/Vue UI application | N/A (UI service) |
| **API** | `/api/*` | User management, auth, core API | `/api/health` |
| **Fairness** | `/fairness/*` | Fairness analysis and metrics | `/fairness/health` |
| **Classification** | `/classification/*` | ML classification models | `/classification/health` |
| **Regression** | `/regression/*` | ML regression models | `/regression/health` |
| **Data Drift** | `/data_drift/*` | Data drift detection | `/data_drift/health` |
| **Gateway** | `/gateway/*` | Gateway management + 3 essential debug endpoints | Built-in |

> **Note**: All routes are **auto-generated** from environment variables. No manual route maintenance required!

## 🔧 Complete Service Management Guide

This guide covers both **adding new services** and **removing deprecated services**. The gateway uses **auto-discovery** but you can still manually add/remove services when needed.

### 🌟 **Method 1: Environment Variable (Recommended - 90% of cases)**

#### Adding a New Service

```bash
# Step 1: Set environment variable
export YOUR_NEW_SERVICE_URL="https://your-new-service.run.app"

# Step 2: Restart gateway (auto-discovers and routes automatically)
python app/main.py

# Step 3: Verify (optional)
curl http://localhost:8000/gateway/registry/status
curl http://localhost:8000/your_new_service/health
```

#### Removing a Service

```bash
# Step 1: Remove environment variable
unset OLD_SERVICE_URL

# Step 2: Restart gateway
python app/main.py

# Service is automatically removed from routing
```

### 🛠️ **Method 2: Runtime API (Advanced scenarios)**

#### Adding a Service at Runtime

```bash
curl -X POST "http://localhost:8000/gateway/registry/services/new_service" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://new-service.run.app",
    "prefix": "new_service",
    "has_openapi": true,
    "health_path": "/new_service/health",
    "openapi_path": "/new_service/openapi.json",
    "full_control": true
  }'
```

#### Removing a Service at Runtime

```bash
curl -X DELETE "http://localhost:8000/gateway/registry/services/old_service"
```

### 📋 **Method 3: Environment Variables (For permanent changes)**

For **permanent service additions** that need to be committed to the repository:

#### Adding Environment Variables in Production

Update deployment workflows in `.github/workflows/gcp-gateway-service.yml`:

```yaml
# Add to both DEV and PRD environments:
env:
  API_SERVICE_URL: https://raia-api-dev-1020052770212.europe-west1.run.app
  FAIRNESS_SERVICE_URL: https://raia-fairness-dev-1020052770212.europe-west1.run.app
  CLASSIFICATION_SERVICE_URL: https://raia-classification-dev-1020052770212.europe-west1.run.app
  REGRESSION_SERVICE_URL: https://raia-regression-dev-1020052770212.europe-west1.run.app
  DATA_DRIFT_SERVICE_URL: https://raia-datadrift-dev-1020052770212.europe-west1.run.app
  YOUR_SERVICE_URL: https://raia-your-service-dev-1020052770212.europe-west1.run.app  # 👈 ADD HERE
  FRONTEND_SERVICE_URL: https://raia-frontend-dev-1020052770212.europe-west1.run.app
```

> **Note**: Gateway uses **auto-discovery** - just add the environment variable and restart!

#### Removing Environment Variables

Simply remove the environment variable from deployment workflows:

```yaml
# Remove the service URL from both DEV and PRD environments
# Gateway will auto-remove the service after restart
```

> **Note**: With auto-discovery, removal is as simple as removing the environment variable!
## 🧪 **Service Requirements & Testing**

#### Requirements for New Services

- **Service must expose OpenAPI** at `/{service-name}/openapi.json`
- **Service must have health endpoint** at `/{service-name}/health`
- **Follow naming convention**: `raia-{service-name}-{env}-{hash}.europe-west1.run.app`
- **Auto-discovery handles routing** - no manual route functions needed
- **Service will automatically appear** in unified documentation at `/docs`
- **Health monitoring** will automatically include the new service

#### Testing Your Changes

After adding or removing a service, use these **essential debug endpoints**:

```bash
# Essential health tracking for Cloud Run deployments
curl http://localhost:8000/gateway/debug-health

# Cache inspection for OpenAPI troubleshooting
curl http://localhost:8000/gateway/debug-cache

# Emergency cache clearing for recovery
curl -X POST http://localhost:8000/gateway/debug-cache/clear

# Verify service registry status (includes detailed service configurations)
curl http://localhost:8000/gateway/registry/status

# Test unified documentation
curl http://localhost:8000/docs

# Verify your service's health endpoint
curl http://localhost:8000/your_service/health
```
    "classification": "/classification/openapi.json",
    "regression": "/regression/openapi.json",
    "data_drift": "/data_drift/openapi.json",
    # "what_if": "/what_if/openapi.json",  # 👈 DELETE THIS LINE
}
```

##### 🎯 **Location 3: Service Descriptions (Line ~439-445)**
**❌ REMOVE the service description:**
```python
service_descriptions = {
    "api": "Core API - User management, authentication, file handling, and project operations",
    "fairness": "AI Fairness Analysis - Bias detection, mitigation strategies, and fairness metrics",
    "classification": "Classification Analysis - Model explainability, feature importance, and decision insights",
    "regression": "Regression Analysis - Model interpretation, feature relationships, and prediction explanations",
    "data_drift": "Data Drift Detection - Model performance monitoring, statistical analysis, and data quality checks",
    # "what_if": "What-If Analysis - Hypothetical scenario analysis",  # 👈 DELETE THIS LINE
}
```

##### 🎯 **Location 4: Health Endpoints (Line ~880-887)**
**❌ REMOVE the health check:**
```python
health_endpoints = {
    "api": "/api/health",
    "fairness": "/fairness/health",
    "classification": "/classification/health",
    "regression": "/regression/health",
    "data_drift": "/data_drift/health",
    # "what_if": "/what_if/health",  # 👈 DELETE THIS LINE
    "frontend": "/health"
}
```

##### 🎯 **Location 5: Route Handler (Around Line ~1150)**
**❌ REMOVE the entire route handler block:**
```python
# DELETE THIS ENTIRE BLOCK:
# @app.api_route("/what_if/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
# async def route_to_what_if(path: str, request: Request):
#     target_url = f"{SERVICES['what_if']}/what_if/{path}"
#     return await proxy_request(target_url, request, "what_if")
```

##### 🎯 **Location 6: Frontend Exclusions (Line ~1165)**
**❌ REMOVE from the exclusion pattern:**
```python
# Remove 'what_if/' from the startswith tuple
if path.startswith(('api/', 'fairness/', 'classification/', 'regression/', 'data_drift/', 'gateway/', 'docs', 'redoc', 'openapi.json')):
    #                                                          👆 'what_if/' REMOVED
    raise HTTPException(status_code=404, detail="Not found")
```

#### Step 2: Update `.github/workflows/gcp-gateway-service.yml` ❌

##### 🎯 **Location: Environment Variables (Line 122-129)**
**❌ REMOVE from BOTH environments:**
```yaml
env:
  API_SERVICE_URL: https://raia-api-dev-1020052770212.europe-west1.run.app
  FAIRNESS_SERVICE_URL: https://raia-fairness-dev-1020052770212.europe-west1.run.app
  CLASSIFICATION_SERVICE_URL: https://raia-classification-dev-1020052770212.europe-west1.run.app
  REGRESSION_SERVICE_URL: https://raia-regression-dev-1020052770212.europe-west1.run.app
  DATA_DRIFT_SERVICE_URL: https://raia-datadrift-dev-1020052770212.europe-west1.run.app
  # WHAT_IF_SERVICE_URL: https://raia-what-if-dev-1020052770212.europe-west1.run.app  # 👈 DELETE THIS LINE
  FRONTEND_SERVICE_URL: https://raia-frontend-dev-1020052770212.europe-west1.run.app
```

#### Step 3: Update this `README.md` file ❌

##### 🎯 **Location 1: Service Routes Table**
**❌ REMOVE the service row from the table**

##### 🎯 **Location 2: Environment Variables Section**
**❌ REMOVE the export statement**

### **Requirements for New Services**

- **Service must expose OpenAPI** at `/{service-name}/openapi.json`
- **Service must have health endpoint** at `/{service-name}/health`
- **Follow naming convention**: `raia-{service-name}-{env}-{hash}.europe-west1.run.app`
- **Update all 4 files**: main.py, workflow.yml, README.md, and architecture docs
- **Service will automatically appear** in unified documentation at `/docs`
- **Health monitoring** will automatically include the new service
- **Caching and request tracking** will be automatically enabled

### **Requirements for Deprecated Services**

- **Remove from all 4 files** (main.py, workflow, README, architecture)
- **Test gateway functionality** after removal using debug endpoints
- **Update documentation** to reflect changes
- **Verify health monitoring** no longer includes the removed service

### **🔍 Testing Your Changes**

After adding or removing a service, use these debug endpoints to verify:

```bash
# Check health monitoring for all services
curl http://localhost:8000/gateway/debug-health

# Inspect the gateway cache status
curl http://localhost:8000/gateway/debug-cache

# Clear the gateway cache if needed
curl -X POST http://localhost:8000/gateway/debug-cache/clear

# Test the unified documentation
curl http://localhost:8000/docs

# Verify your service's health endpoint works through gateway
curl http://localhost:8000/your_service/health
```

## 🤖 **ML Workload Capabilities**

### **🚀 Enterprise ML Performance Features:**

#### **💪 High Concurrency Support:**
- **100 concurrent connections** with 20 keepalive connections
- **Connection pooling** optimized for ML service patterns
- **2-minute timeouts** for inference and model loading
- **Async request handling** for parallel ML operations

#### **📈 ML-Optimized Configuration:**
```python
# Gateway automatically configures:
Connection Limits: 100 max, 20 keepalive
Timeouts: 120s read, 30s write, 30s connect
Health Checks: 30s buffer for cold starts
Failure Threshold: 70% (accounts for cold starts)
```

#### **🎯 Large Request Handling:**
- **Extended upload timeouts** for large datasets
- **Streaming response support** for large ML outputs
- **Request size optimization** for model inputs
- **Memory-efficient proxying** for large payloads

### **📊 ML Performance Monitoring:**

#### **Real-time ML Analytics:**
```bash
# Get comprehensive ML performance analysis
curl http://localhost:8000/gateway/debug-ml-performance

# Response includes:
# - Per-service performance categories (Excellent/Good/Needs Optimization)
# - ML-specific recommendations for each service
# - Platform readiness score (Production Ready/Development/Needs Work)
# - Concurrent request capacity assessment
# - Cold start impact analysis
```

#### **Performance Categories:**
- **🟢 Excellent**: <3s response, <20% failure rate
- **🟡 Good**: <6s response, <40% failure rate
- **🟠 Acceptable**: <10s response, <60% failure rate (Cold start impact)
- **🔴 Needs Optimization**: >10s response, >60% failure rate

#### **Smart Recommendations:**
- **Model optimization** suggestions for slow services
- **Cloud Run resource** allocation guidance
- **Cold start mitigation** strategies
- **Instance scaling** recommendations based on usage

### **⚡ Performance Benchmarks:**

#### **Current Capacity (0-2 instance scaling):**
- **Concurrent Requests**: ~50-100 simultaneous ML operations
- **Large File Uploads**: Up to 2GB with extended timeouts
- **Model Inference**: 2-minute max processing time
- **Cold Start Recovery**: 30-60 seconds for ML service warmup

#### **Recommended Scaling for High Volume:**
```bash
# For >500 requests/hour per service:
Min Instances: 1 (reduce cold starts)
Max Instances: 5 (higher throughput)
CPU: 2-4 vCPUs per ML service
Memory: 4-8GB per ML service
```

### **🔧 Production ML Optimizations:**

#### **Already Implemented:**
- ✅ **Extended timeouts** for ML model processing
- ✅ **Connection pooling** for service communication
- ✅ **Health tracking** with ML-aware failure thresholds
- ✅ **Cold start resilience** with progressive retry logic
- ✅ **Performance analytics** with ML-specific insights

#### **Recommended for Scale:**
- 🔄 **Request queuing** for high-volume batch processing
- 🔄 **Rate limiting** per user/tenant for fair resource allocation
- 🔄 **Caching layer** for frequently requested ML predictions
- 🔄 **Service mesh** for advanced load balancing
- 🔄 **Monitoring alerts** for performance degradation

### **🚀 Quick Reference Checklist**

#### **✅ Adding New Service Checklist:**

- [ ] 1. Add to SERVICES dictionary (main.py line ~155)
- [ ] 2. Add to openapi_endpoints (main.py line ~168)
- [ ] 3. Add to service_descriptions (main.py line ~439)
- [ ] 4. Add to health_endpoints (main.py line ~880)
- [ ] 5. Add route handler (main.py line ~1150)
- [ ] 6. Update frontend exclusions (main.py line ~1165)
- [ ] 7. Add to DEV env vars (workflow.yml line ~122)
- [ ] 8. Add to PRD env vars (workflow.yml line ~150)
- [ ] 9. Add to Service Routes table (README.md line ~37)
- [ ] 10. Add to Environment Variables (README.md line ~345)
- [ ] 11. Test locally: `curl http://localhost:8000/your_service/health`
- [ ] 12. Test unified docs: `curl http://localhost:8000/docs`
- [ ] 13. Verify debug endpoints: `curl http://localhost:8000/gateway/debug-endpoints`

#### **❌ Removing Service Checklist:**

- [ ] 1. Remove from SERVICES dictionary (main.py line ~155)
- [ ] 2. Remove from openapi_endpoints (main.py line ~168)
- [ ] 3. Remove from service_descriptions (main.py line ~439)
- [ ] 4. Remove from health_endpoints (main.py line ~880)
- [ ] 5. Remove route handler (main.py line ~1150)
- [ ] 6. Remove from frontend exclusions (main.py line ~1165)
- [ ] 7. Remove from DEV env vars (workflow.yml line ~122)
- [ ] 8. Remove from PRD env vars (workflow.yml line ~150)
- [ ] 9. Remove from Service Routes table (README.md)
- [ ] 10. Remove from Environment Variables (README.md)
- [ ] 11. Test gateway health: `curl /gateway/health`
- [ ] 12. Verify debug endpoints: `curl http://localhost:8000/gateway/debug-endpoints`
- [ ] 2. Add to openapi_endpoints (main.py line ~43)
- [ ] 3. Add to health_endpoints (main.py line ~356)
- [ ] 4. Add route handler (main.py line ~558)
- [ ] 5. Update frontend exclusions (main.py line ~572)
- [ ] 6. Add to DEV env vars (workflow.yml line ~122)
- [ ] 7. Add to PRD env vars (workflow.yml line ~150)
- [ ] 8. Add to Service Routes table (README.md line ~37)
- [ ] 9. Add to Environment Variables (README.md line ~280)
- [ ] 10. Test locally: `curl http://localhost:8000/your_service/health`
- [ ] 11. Test unified docs: `curl http://localhost:8000/docs`

#### **❌ Removing Service Checklist:**

- [ ] 1. Remove from SERVICES dictionary (main.py line ~31)
- [ ] 2. Remove from openapi_endpoints (main.py line ~43)
- [ ] 3. Remove from health_endpoints (main.py line ~356)
- [ ] 4. Remove route handler (main.py line ~558)
- [ ] 5. Remove from frontend exclusions (main.py line ~572)
- [ ] 6. Remove from DEV env vars (workflow.yml line ~122)
- [ ] 7. Remove from PRD env vars (workflow.yml line ~150)
- [ ] 8. Remove from Service Routes table (README.md)
- [ ] 9. Remove from Environment Variables (README.md)
- [ ] 10. Test gateway health: `curl /gateway/health`

## 🌍 Environment Variables

Set these environment variables for local development or production:

```bash
# Required for each service
export API_SERVICE_URL=https://your-api-service.run.app
export FAIRNESS_SERVICE_URL=https://your-fairness-service.run.app
export CLASSIFICATION_SERVICE_URL=https://your-classification-service.run.app
export REGRESSION_SERVICE_URL=https://your-regression-service.run.app
export DATA_DRIFT_SERVICE_URL=https://your-data-drift-service.run.app
export FRONTEND_SERVICE_URL=https://your-frontend-service.run.app

# Optional: Add new services here
# export YOUR_NEW_SERVICE_URL=https://your-new-service.run.app
```

## 📱 Usage Examples

```bash
# Health monitoring
curl https://your-gateway-url.run.app/gateway/health

# API access via gateway (recommended)
curl https://your-gateway-url.run.app/api/users
curl https://your-gateway-url.run.app/fairness/analyze
curl https://your-gateway-url.run.app/classification/predict

# Frontend loads by default
curl https://your-gateway-url.run.app/
```

## 🚀 Deployment

Automatic deployment via GitHub Actions when changes are pushed to `services/gateway/`.

**URLs:**
- **Dev**: `https://raia-dev-1020052770212.europe-west1.run.app`
- **Prod**: `https://raia-prd-1020052770212.europe-west1.run.app`

## 🎯 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export API_SERVICE_URL=https://your-api-service.run.app
export FAIRNESS_SERVICE_URL=https://your-fairness-service.run.app
export CLASSIFICATION_SERVICE_URL=https://your-classification-service.run.app
export REGRESSION_SERVICE_URL=https://your-regression-service.run.app
export DATA_DRIFT_SERVICE_URL=https://your-data-drift-service.run.app
export FRONTEND_SERVICE_URL=https://your-frontend-service.run.app

# Run the gateway
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🔍 System Health

The gateway provides comprehensive health monitoring optimized for Cloud Run auto-scaling:

```bash
# Check gateway and all services health
curl https://your-gateway-url.run.app/gateway/health

# Response includes:
# - Gateway status
# - Individual service health
# - Response times (includes cold start times)
# - Error details
# - Overall system health summary
```

### **☁️ Cloud Run Performance Notes:**

**Current Configuration:**
- **Gateway**: 0-5 instances (High availability)
- **All Services**: 0-2 instances (Cost-optimized)

**Expected Behavior:**
- **Cold Start Times**: 10-30 seconds for ML services (model loading)
- **Warm Response**: 1-5 seconds for active instances
- **Idle Timeout**: ~15 minutes before scaling to zero

**Health Metrics Interpretation:**
- **High failure rates (>50%)** often indicate cold start timeouts
- **Long response times (>6s)** include model initialization
- **Gateway health tracking** is optimized for auto-scaling patterns

## 🛠️ Troubleshooting

### Service Not Responding

1. Check `/gateway/health` to see which services are down
2. **Cold Start Check**: Wait 30-60 seconds for ML services to initialize
3. Verify service URLs in environment variables
4. Use debug endpoints: `curl /gateway/debug-health` for detailed metrics

### Documentation Not Showing

1. Verify services expose `/openapi.json` endpoints
2. Check service URLs are correct
3. **Cold Start Consideration**: OpenAPI fetch may timeout during cold starts
4. Check cache status: `curl /gateway/debug-cache`

### Routing Issues

1. Ensure path patterns match service expectations
2. **Extended Timeouts**: Gateway uses 2-minute timeouts for ML cold starts
3. Check CORS settings if browser requests fail
4. Verify service authentication if required

## 📝 Notes

- **Frontend Service**: UI application (no OpenAPI specs)
- **Service Health**: All API services must implement health endpoints
- **OpenAPI Integration**: Services must expose OpenAPI specs at `/{service}/openapi.json`
- **Route Priority**: Specific routes (e.g., `/api/`) take precedence over catch-all routes

---

**Last Updated**: October 2025
**Gateway Version**: v2.1 (Optimized)
**Active Services**: 6 (API, Fairness, Classification, Regression, Data Drift, Frontend)
