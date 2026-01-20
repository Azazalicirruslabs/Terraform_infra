# Model Drift Detection Backend

## 🎯 Project Overview

This backend provides a unified API for both **Data Drift** and **Model Drift** detection functionality. The codebase is strategically segregated to allow parallel development while maintaining a cohesive production system.

## 📁 Project Structure

```
backend/
├── src/
│   ├── data_drift/          # 🔵 Data Drift Module (Teammate's Domain)
│   │   ├── routes/          # API endpoints for data drift
│   │   ├── services/        # Core data drift logic
│   │   ├── models/          # Data drift data models
│   │   ├── utils/           # Data drift utilities
│   │   └── README.md        # Detailed data drift guide
│   │
│   ├── model_drift/         # 🟢 Model Drift Module (Your Domain)
│   │   ├── routes/          # API endpoints for model drift
│   │   ├── services/        # Core model drift logic
│   │   ├── models/          # Model drift data models
│   │   ├── utils/           # Model drift utilities
│   │   └── README.md        # Detailed model drift guide
│   │
│   ├── shared/              # 🟡 Shared Components
│   │   ├── middleware/      # Authentication, CORS, validation
│   │   ├── database/        # Database connection & models
│   │   ├── upload/          # Unified file upload system
│   │   ├── config/          # Configuration management
│   │   └── utils/           # Common utilities
│   │
│   ├── main.py              # Main FastAPI app wiring all routers
│
├── uploads/                 # File storage directory
├── tests/                   # Test suites (unit & integration)
├── config/                  # Environment configurations
├── docs/                    # API documentation
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or 3.11
- (Optional) Node.js for any shared utilities
- MongoDB or PostgreSQL (TBD based on requirements)

### Installation
```bash
cd backend
pip install -r requirements.txt  # Install core Python dependencies in your venv (FastAPI, pandas, scipy, etc.)
```

### Environment Setup
```
python -m venv venv
# Configure your environment variables
```

## 🏃‍♂️ Running the Server

- Start the unified FastAPI backend:

```bash
cd backend
uvicorn src.main:app --reload --port 8001  # Starts FastAPI server (this changes according to how YOU or me will modify data drift or Model drift, so there is no CONFUSION !!!!)
```

- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## 🔌 API Architecture

### Base URL Structure
```
/api/v1/data-drift/*     # All data drift endpoints (FastAPI)
/api/v1/model-drift/*    # All model drift endpoints (FastAPI)
/api/v1/upload/*         # Unified upload system (FastAPI)
```

### Shared Endpoints
- `POST /api/v1/upload/dataset` - Upload reference/current datasets
- `GET /api/v1/config/thresholds` - Get drift thresholds
- `PUT /api/v1/config/thresholds` - Update drift thresholds

## 🔧 Configuration Management

### Threshold Configuration
```json
{
  "data_drift": {
    "low": 0.05,
    "medium": 0.15,
    "high": 0.25
  },
  "model_drift": {
    "performance_degradation": {
      "low": 0.02,
      "medium": 0.05,
      "high": 0.10
    }
  }
}
```

## 📊 Data Flow

```
Frontend Upload → Unified Upload Service → Module Router → Specific Module Processing → Response
```

## 🧪 Testing Strategy

### Unit Tests
- Each module has isolated unit tests
- Test data drift and model drift logic independently
- Shared utilities have comprehensive test coverage

### Integration Tests
- End-to-end API testing
- Cross-module functionality testing
- Frontend-backend integration tests

## 📚 Documentation

- **API Documentation**: Available at `/docs` when server is running
- **Module-Specific Guides**: See individual README files

## 🚨 Important Notes

### For Team Coordination:
1. **Shared Components**: Discuss changes to `src/shared/` with both team members
2. **API Contracts**: Don't modify response formats without frontend consultation
3. **Database Schema**: Coordinate any schema changes
4. **Dependencies**: Update requirements when adding new packages

### Development Best Practices:
- Use consistent error handling patterns
- Follow established API response formats
- Write comprehensive tests for new features
- Document complex algorithms and business logic

## 🔄 Deployment Pipeline

```
Development → Testing → Staging → Production
```

Both modules will be deployed as a single unified backend service.

## 📞 Support & Contact

- **Data Drift Issues**: Check `src/data_drift/README.md`
- **Model Drift Issues**: Check `src/model_drift/README.md`
- **Shared Components**: Discuss with both team members
- **Infrastructure**: Contact DevOps team
