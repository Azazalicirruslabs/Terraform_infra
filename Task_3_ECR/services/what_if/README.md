# What_If Service

## 📖 Overview

This is the what_if microservice for the XAI-Explainability platform.

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
   docker-compose up what_if
   ```

2. **Build only:**
   ```bash
   docker build -t what_if -f services/what_if/Dockerfile .
   ```

## 📁 Project Structure

```tree
services/what_if/
├── .dockerignore
├── .gitignore
├── Dockerfile
└── README.md
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
pytest --cov=what_if
```

## 🚀 Deployment

This service is automatically deployed via GitHub Actions when changes are pushed to the main branches.

## 📚 Additional Documentation

Add links to additional documentation here.
