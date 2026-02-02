# Regression Service

## 📖 Overview

This is the regression microservice for the XAI-Explainability platform.

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Docker & Docker Compose

### Local Development

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the service:**
   ```bash
   python main.py
   ```

4. **Access the service:**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs

### Docker Development

1. **Build and run:**
   ```bash
   docker-compose up regression
   ```

2. **Build only:**
   ```bash
   docker build -t regression -f services/regression/Dockerfile .
   ```

## 📁 Project Structure

```tree
services/regression/
├── app/
│   ├── config/
│   │   └── settings.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ai_explanation_service.py
│   │   ├── analysis_service.py
│   │   ├── base_model_service.py
│   │   ├── classification_service.py
│   │   ├── dependence_service.py
│   │   ├── feature_service.py
│   │   ├── interaction_service.py
│   │   ├── model_service.py
│   │   ├── prediction_service.py
│   │   ├── README.md
│   │   └── tree_service.py
│   ├── routers/
│   │   ├── logic.py
│   │   └── regression.py
│   ├── schemas/
│   │   └── regression_schema.py
│   ├── utils/
│   │   └── error_handler.py
│   └── main.py
├── .dockerignore
├── __init__.py
├── Dockerfile
├── pyproject.toml
├── README.md
└── requirements.txt
```

## 🔧 Configuration

Add your configuration details here.

## 📋 API Endpoints

Document your API endpoints here:

- `GET /health` - Health check
- Add your endpoints...

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=regression
```

## 🚀 Deployment

This service is automatically deployed via GitHub Actions when changes are pushed to the main branches.

## 📚 Additional Documentation

Add links to additional documentation here..
