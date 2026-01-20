# 🚀 RAIA (Responsible AI Analytics) Platform

## 📚 Quick Access - API Documentation

🌟 **Unified API Documentation**: Access all microservices in one place!

- **📖 Interactive Swagger UI**: `/docs` - Test all endpoints directly in your browser
- **📋 OpenAPI Specification**: `/openapi.json` - Complete API specification
- **📗 Alternative Docs**: `/redoc` - Clean, organized documentation view
- **🔍 Gateway Health**: `/gateway/health` - Monitor all services status
- **🐛 Debug Endpoints**: `/gateway/debug-endpoints` - Live endpoint counts per service

> **💡 Pro Tip**: The unified documentation automatically updates when services add new endpoints - no manual maintenance needed!

For detailed gateway configuration and advanced features, see: [**Gateway Service README**](services/gateway/README.md)

---

## 🛠️ Setup Instructions

### 1. Create Virtual Environment

```bash
python3.13.3 -m venv venv

# On Windows
.\venv\Scripts\activate

# On Unix/MacOS
source venv/bin/activate
```

### 2. 📦 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. ⚙️ Environment Configuration

Create `.env` file in the root folder with following credentials:

```env
# Authentication
SECRET_KEY = "XAI"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database Configuration
DB_NAME = "XAI"
DB_USERNAME = "postgres"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
```

### 4. 🗃️ Database Migration

- alembic upgrade head

### 5. 🚀 Run Server

```bash
# Development mode
uvicorn app.main:app --reload

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. 💾 Database

PostgreSQL is used as the primary database.

## 🏗️ Architecture Overview

The RAIA platform follows a **microservices architecture** with the following services:

| Service | Purpose | Key Endpoints |
|---------|---------|---------------|
| **🌐 Gateway** | API routing & unified documentation | `/docs`, `/gateway/health` |
| **🔐 API** | Authentication & user management | `/api/auth/login`, `/api/users/*` |
| **⚖️ Fairness** | AI bias detection & analysis | `/fairness/analyze`, `/fairness/metrics` |
| **🧠 Classification** | ML classification models | `/classification/predict`, `/classification/explain` |
| **📈 Regression** | Statistical regression analysis | `/regression/predict`, `/regression/explain` |
| **📊 Data Drift** | Data quality monitoring | `/data_drift/detect`, `/data_drift/report` |

**🎯 All services are accessible through the unified gateway at `/docs`**

---

## 📁 Project Structure

```tree
XAI EXPLAINABILITY/
├── app/
│   ├── __init__.py
│   ├── main.py                 # 🎯 Entry point of the app
│   ├── api/                    # 🌐 API route definitions
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── routes.py       # API endpoints
│   ├── core/                   # ⚙️ App configuration
│   │   ├── __init__.py
│   │   └── config.py          # Settings using Pydantic
│   ├── models/                 # 📊 Data models
│   │   ├── __init__.py
│   │   └── user.py            # User model
│   ├── schemas/                # 📋 Data schemas
│   │   ├── __init__.py
│   │   └── user_schema.py     # User schema
│   ├── services/              # 🔧 Business logic
│   │   ├── __init__.py
│   │   └── user_service.py    # User services
│   ├── database/              # 🗄️ Database utilities
│   │   ├── __init__.py
│   │   └── connections.py     # DB connection handler
│   └── utils/                 # 🔨 Utility functions
│       ├── __init__.py
│       └── token.py           # Token utilities
├── tests/                     # 🧪 Test cases
│   └── test_user.py
├── .env                       # 🔐 Environment variables
├── requirements.txt           # 📦 Dependencies
└── README.md                  # 📖 Documentation
```

## 🤝 Contributing

Feel free to contribute to this project by creating issues or submitting pull requests.
