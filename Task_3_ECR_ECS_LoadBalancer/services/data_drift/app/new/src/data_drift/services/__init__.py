# Data Drift Business Logic Services
"""
Core business logic and ML algorithms for data drift detection.

This package contains:
- Statistical drift detection algorithms
- Feature analysis and comparison logic
- Threshold management
- Data preprocessing utilities
"""

from .drift_service import run_data_drift

__all__ = ["run_data_drift"]
