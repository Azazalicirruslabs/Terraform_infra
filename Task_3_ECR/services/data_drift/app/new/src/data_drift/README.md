# Data Drift Module - Developer Guide

## 🎯 Welcome, Data Drift Developer!

This guide provides everything you need to implement the data drift detection functionality that powers the beautiful frontend we've built.

## 📁 Your Working Directory Structure

```
src/data_drift/
├── routes/              # Your FastAPI endpoints
│   ├── __init__.py      # Package initialization
│   ├── dashboard.py     # Drift dashboard endpoints
│   ├── upload.py        # Simple upload for testing
│   ├── feature.py       # Feature analysis endpoints
│   └── statistical.py   # Statistical reports endpoints
│
├── services/            # Your core ML/Math logic (Python)
│   ├── __init__.py      # Package initialization
│   ├── drift_detector.py    # Main drift detection algorithms
│   ├── feature_analyzer.py  # Feature-level drift analysis
│   ├── statistical_analyzer.py # Statistical test implementations
│   └── threshold_manager.py # Threshold management logic
│
├── models/              # Pydantic models & schemas
│   ├── __init__.py      # Package initialization
│   ├── drift_result.py  # Drift detection result schemas
│   ├── dataset.py       # Dataset metadata schemas
│   └── feature.py       # Feature analysis schemas
│
├── utils/               # Utility functions (Python)
│   ├── __init__.py      # Package initialization
│   ├── data_validator.py # Input data validation
│   ├── statistical_tests.py # Statistical test utilities (scipy, etc.)
│   └── visualization_helper.py # Data prep for charts
│
└── README.md           # This file
```

## 🐍 Python Dependencies & Setup

### Core ML/Statistical Libraries You'll Need u may need more, work accordingly:
```bash
pip install pandas numpy scipy scikit-learn
pip install fastapi uvicorn python-multipart
pip install pydantic typing-extensions
pip install matplotlib seaborn plotly  # For visualization data prep
```

### Key Python Modules for Data Drift:
- **pandas/numpy**: Data manipulation and numerical computations
- **scipy.stats**: Statistical tests (KS test, Chi-square, Mann-Whitney, etc.)
- **scikit-learn**: Preprocessing, feature analysis, drift detection algorithms
- **fastapi**: REST API framework
- **pydantic**: Data validation and serialization

## 🎨 Frontend Integration Points

### The frontend expects 4 main tabs of data drift functionality:

1. **📊 Drift Analysis Dashboard** - Main overview with key metrics and visualizations
2. **🔍 Feature Deep Dive** - Detailed analysis of individual features showing drift patterns
3. **⚖️ Class Imbalance Analysis** - Analysis of class distribution changes and imbalance metrics
4. **� Statistical Reports** - Comprehensive statistical analysis and reporting

## 🚀 Quick Setup for Local Development

### 1. Simple Upload Functionality (For Testing)

Create basic endpoints to upload reference and current datasets:

**File**: `routes/upload.py`
```python
# FastAPI endpoints for file upload
from fastapi import UploadFile, File

@router.post("/upload/reference")
async def upload_reference_dataset(file: UploadFile = File(...)):
    """POST /api/v1/data-drift/upload/reference"""
    pass

@router.post("/upload/current")
async def upload_current_dataset(file: UploadFile = File(...)):
    """POST /api/v1/data-drift/upload/current"""
    pass
```

**Expected Response**:
```json
{
  "status": "success",
  "message": "Dataset uploaded successfully",
  "data": {
    "dataset_id": "uuid",
    "filename": "reference_data.csv",
    "rows": 1000,
    "columns": 15,
    "upload_time": "2025-09-02T10:30:00Z"
  }
}
```

### 2. Threshold Configuration

**File**: `routes/upload.py` (or create `routes/config.py`)
```python
# FastAPI endpoints for threshold management
from pydantic import BaseModel

class ThresholdConfig(BaseModel):
    low: float = 0.05
    medium: float = 0.15
    high: float = 0.25

@router.get("/thresholds", response_model=ThresholdConfig)
async def get_thresholds():
    """GET /api/v1/data-drift/thresholds"""
    pass

@router.put("/thresholds")
async def update_thresholds(thresholds: ThresholdConfig):
    """PUT /api/v1/data-drift/thresholds"""
    pass
```

## 📋 API Endpoints You Need to Implement

### 1. Drift Analysis Dashboard (`routes/dashboard.py`)

```python
# FastAPI endpoint for main dashboard
@router.get("/dashboard")
async def get_drift_dashboard():
    """GET /api/v1/data-drift/dashboard"""
    pass
```

**Frontend expects this response structure**:
```json
{
  "status": "success",
  "data": {
    "overall_drift_score": 0.23,
    "drift_level": "medium", // "low", "medium", "high"
    "features_analyzed": 15,
    "features_with_drift": 8,
    "last_analysis": "2025-09-02T10:30:00Z",
    "summary_metrics": {
      "statistical_distance": 0.15,
      "population_stability_index": 0.18,
      "jensen_shannon_divergence": 0.12
    },
    "feature_summary": [
      {
        "feature_name": "age",
        "drift_score": 0.08,
        "drift_level": "low",
        "change_type": "distribution_shift"
      }
      // ... more features
    ]
  }
}
```

### 2. Feature Deep Dive (`routes/feature.py`)

```python
# FastAPI endpoints in routes/feature.py
@router.get("/features/analysis")
async def get_feature_analysis():
    """GET /api/v1/data-drift/features/analysis"""
    pass

@router.get("/features/{feature_name}/details")
async def get_feature_details(feature_name: str):
    """GET /api/v1/data-drift/features/{feature_name}/details"""
    pass
```

**Response for feature analysis**:
```json
{
  "status": "success",
  "data": {
    "features": [
      {
        "name": "age",
        "type": "numerical",
        "drift_score": 0.08,
        "drift_level": "low",
        "statistical_tests": {
          "ks_test": {
            "statistic": 0.05,
            "p_value": 0.23,
            "significant": false
          },
          "chi_square": {
            "statistic": 12.5,
            "p_value": 0.08,
            "significant": false
          }
        },
        "distribution_comparison": {
          "reference_mean": 35.2,
          "current_mean": 36.8,
          "reference_std": 12.1,
          "current_std": 13.5
        },
        "visualization_data": {
          "histogram": [
            {"bin": "20-25", "reference": 150, "current": 120},
            {"bin": "25-30", "reference": 200, "current": 180}
            // ... more bins
          ]
        }
      }
      // ... more features
    ]
  }
}
```

### 3. Statistical Reports (`routes/statistical.py`)

```python
# FastAPI endpoints in routes/statistical.py
@router.get("/statistical/reports")
async def get_statistical_reports():
    """GET /api/v1/data-drift/statistical/reports"""
    pass

@router.get("/statistical/tests/{test_type}")
async def get_statistical_test(test_type: str):
    """GET /api/v1/data-drift/statistical/tests/{test_type}"""
    pass
```

**Response structure**:
```json
{
  "status": "success",
  "data": {
    "overall_statistics": {
      "total_features": 15,
      "features_with_significant_drift": 5,
      "overall_p_value": 0.001,
      "confidence_level": 0.95
    },
    "statistical_tests": [
      {
        "test_name": "Kolmogorov-Smirnov",
        "features_tested": ["age", "income", "score"],
        "significant_results": 2,
        "overall_conclusion": "significant_drift_detected"
      }
      // ... more tests
    ],
    "detailed_results": [
      {
        "feature": "age",
        "tests": {
          "ks_test": {"statistic": 0.15, "p_value": 0.001},
          "mann_whitney": {"statistic": 1250, "p_value": 0.003}
        }
      }
      // ... more features
    ]
  }
}
```

### 4. Class Imbalance Analysis (`routes/dashboard.py`)

```python
# FastAPI endpoints in routes/dashboard.py
@router.get("/class-imbalance/analysis")
async def get_class_imbalance_analysis():
    """GET /api/v1/data-drift/class-imbalance/analysis"""
    pass

@router.get("/class-imbalance/metrics")
async def get_class_imbalance_metrics():
    """GET /api/v1/data-drift/class-imbalance/metrics"""
    pass
```

**Response structure**:
```json
{
  "status": "success",
  "data": {
    "drift_impact": {
      "severity": "medium",
      "affected_features": ["age", "income", "location"],
      "business_impact": "moderate",
      "recommended_actions": [
        "Retrain model with recent data",
        "Monitor feature 'age' closely",
        "Consider feature engineering for 'location'"
      ]
    },
    "feature_importance": [
      {
        "feature": "age",
        "importance": 0.25,
        "drift_score": 0.15,
        "impact_score": 0.18
      }
      // ... more features
    ],
    "trend_analysis": {
      "drift_trend": "increasing",
      "time_series": [
        {"date": "2025-08-01", "drift_score": 0.10},
        {"date": "2025-08-15", "drift_score": 0.15},
        {"date": "2025-09-01", "drift_score": 0.23}
      ]
    }
  }
}
```


### Quick Test Commands
```bash
# Upload reference dataset
curl -X POST http://localhost:5000/api/v1/data-drift/upload/reference \
  -F "file=@test_data/reference_dataset.csv"

# Upload current dataset
curl -X POST http://localhost:5000/api/v1/data-drift/upload/current \
  -F "file=@test_data/current_dataset.csv"

# Get drift dashboard
curl http://localhost:5000/api/v1/data-drift/dashboard
```



### Validation Rules
- Datasets must have matching column schemas
- Minimum 100 rows for statistical significance
- Support CSV, JSON, and Parquet formats
- File size limit: 100MB

### Statistical Tests to Implement
- **Numerical features**: Kolmogorov-Smirnov, Mann-Whitney U
- **Categorical features**: Chi-square, Cramér's V
- **Distribution distance**: Jensen-Shannon divergence, Wasserstein distance
- **Population Stability Index (PSI)**

### 4. Thresholds Interpretation
```javascript
{
  "low": 0.05,     // PSI < 0.05: No significant drift
  "medium": 0.15,  // 0.05 <= PSI < 0.15: Some drift, monitor
  "high": 0.25     // PSI >= 0.15: Significant drift, action needed
}
```

## 📊 Data Processing Pipeline

```
CSV Upload → Data Validation → Statistical Analysis → Drift Detection → Response Formatting
```

### Key Processing Steps:
1. **Data Ingestion**: Parse CSV, validate schema
2. **Feature Analysis**: Identify numerical vs categorical
3. **Statistical Testing**: Run appropriate tests per feature type
4. **Threshold Comparison**: Classify drift levels
5. **Visualization Prep**: Format data for frontend charts

## 🚨 Important Notes

### Frontend Dependencies:
- The frontend expects specific response formats (follow examples above)
- Chart data must be formatted for Recharts library
- Timestamps should be ISO 8601 format
- All numeric values should be rounded to 3 decimal places

### Performance Considerations:
- Cache drift calculations for 5 minutes
- Use async processing for large datasets
- Implement pagination for large feature lists
- Store processed results to avoid re-computation

### Security:
- Validate all file uploads
- Sanitize filename inputs
- Implement rate limiting
- Log all upload activities

## 🧪 Testing Your API Endpoints with FastAPI Swagger

### FastAPI Swagger Documentation Setup
The backend uses FastAPI which automatically generates interactive API documentation. This is perfect for testing your endpoints!

### 1. Start Your FastAPI Server
```bash
cd backend/src/data_drift
uvicorn main:app --reload --port 8001
```

### 2. Access Swagger UI
- **URL**: `http://localhost:8001/docs`
- **Alternative ReDoc**: `http://localhost:8001/redoc`

### 3. Testing Workflow for Each Tab

#### 📊 Drift Analysis Dashboard Testing
1. **Endpoint**: `GET /api/v1/data-drift/dashboard`
2. **In Swagger UI**:
   - Click on the dashboard endpoint
   - Click "Try it out"
   - Click "Execute"
   - Verify response matches frontend expectations
