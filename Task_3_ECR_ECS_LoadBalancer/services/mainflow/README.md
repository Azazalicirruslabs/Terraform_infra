# 🚀 XAI Explainability Project Setup

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

+ alembic upgrade head

### 5. 🚀 Run Server

```bash
# Development mode
uvicorn services.mainflow.app.main:app --reload

# Production mode
uvicorn services.mainflow.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. 💾 Database

PostgreSQL is used as the primary database.

---

## 📁 Project Structure
XAI-EXPLAINABILITY
│
├── app
│   ├── main.py
│   │
│   ├── core
│   │   ├── __init__.py
│   │   ├── xai_service.py   # SHAP, LIME
│   │   ├── ml_service.py    # ML logic
│   │   ├── logging.py       # logging setup
│   │   ├── dependencies.py  # Depends() common deps
│   │   └── events.py       # other logic
│   │
│   ├── routers
│   │   ├── __init__.py
│   │   ├── health.py
│   │   ├── explainability.py
│   │   ├── upload.py
│   │   └── analysis.py
│   │
│   ├── schemas
│   │   ├── __init__.py
│   │   ├── request.py # create one file per API to request and response
│   │   └── response.py
│   │
│   ├── database
│   │   ├── __init__.py
│   │   ├── base.py        # DB session
│   │   ├── connection.py
│   │
│   ├── utils
│   │   ├── __init__.py
│   │   ├── file_utils.py
│   │   ├── csv_utils.py
│   │   ├── time_utils.py
│   │   └── validation.py
│   │
│   └── config
│       └── config.py
│
├── tests
│   ├── __init__.py
│   ├── test_health.py
│   └── test_explainability.py
│
├── requirements.txt
├── README.md
└── dockerfile


## 🤝 Contributing

Feel free to contribute to this project by creating issues or submitting pull requests..

Put in core/ if:
1. It uses FastAPI concepts (Depends, Request, Response etc)
2. It’s part of app lifecycle
3. It’s logging setup
4. It’s shared across multiple routers

Put in utils/ if:
1. Pure Python function
2. No FastAPI dependency
3. Stateless
4. Easily reusable in scripts or tests
