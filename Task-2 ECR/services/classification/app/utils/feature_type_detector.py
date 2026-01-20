"""
Robust feature type detection for handling encoded categorical variables.
Combines multiple heuristics to detect categorical vs numerical features.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


class FeatureTypeDetector:
    """
    Advanced feature type detection that goes beyond pandas dtypes.
    Designed to identify encoded categorical features (e.g., label encoded integers).

    Args:
        cardinality_threshold: Max unique values to consider categorical (default: 20)
        cardinality_ratio_threshold: Max ratio of unique/total to consider categorical (default: 0.05 = 5%)
        min_samples_for_ratio: Minimum sample size to apply ratio check (default: 100)
    """

    def __init__(
        self,
        cardinality_threshold: int = 20,
        cardinality_ratio_threshold: float = 0.05,
        min_samples_for_ratio: int = 100,
    ):
        self.cardinality_threshold = cardinality_threshold
        self.cardinality_ratio_threshold = cardinality_ratio_threshold
        self.min_samples_for_ratio = min_samples_for_ratio

    def detect_feature_type(self, series: pd.Series) -> str:
        """
        Detect if a feature is 'numerical' or 'categorical' using multiple heuristics.

        Args:
            series: pandas Series to analyze

        Returns:
            'numerical' or 'categorical'
        """
        # Handle empty series
        if len(series) == 0:
            return "categorical"

        # Drop NaN for analysis
        series_clean = series.dropna()
        if len(series_clean) == 0:
            return "categorical"

        total_count = len(series_clean)
        unique_count = series_clean.nunique()

        # Rule 0: Domain-specific identifier patterns (ZipCode, ID, SSN, etc.)
        column_name = series.name.lower() if series.name else ""
        if self._is_identifier_column(column_name, series_clean):
            return "categorical"

        # Rule 1: Explicitly string/object dtype → categorical
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            return "categorical"

        # Rule 2: Boolean dtype → categorical
        if pd.api.types.is_bool_dtype(series):
            return "categorical"

        # Rule 3: Datetime dtype → treat as categorical (can't correlate dates directly)
        if pd.api.types.is_datetime64_any_dtype(series):
            return "categorical"

        # Rule 4: Low cardinality (absolute threshold)
        if unique_count <= self.cardinality_threshold:
            # Additional check: verify if values look like encoded categories
            if self._is_likely_encoded_categorical(series_clean):
                return "categorical"

        # Rule 5: Cardinality ratio check (for larger datasets)
        if total_count >= self.min_samples_for_ratio:
            cardinality_ratio = unique_count / total_count
            if cardinality_ratio <= self.cardinality_ratio_threshold:
                # Low ratio suggests categorical
                if self._is_likely_encoded_categorical(series_clean):
                    return "categorical"

        # Rule 6: Check for integer-only values in float columns (encoded categories)
        if pd.api.types.is_float_dtype(series):
            if self._is_integer_valued_float(series_clean):
                # Float column with only integer values - check cardinality
                if unique_count <= self.cardinality_threshold * 2:  # More lenient for floats
                    return "categorical"

        # Default: numerical
        return "numerical"

    def _is_identifier_column(self, column_name: str, series: pd.Series) -> bool:
        """
        Detect if a column is an identifier (ID, ZipCode, SSN, etc.) based on name and pattern.
        Identifiers should be treated as categorical even with high cardinality.
        """
        # Check column name patterns
        identifier_patterns = [
            "id",
            "_id",
            "code",
            "zip",
            "postal",
            "ssn",
            "account",
            "customer",
            "user",
            "product",
            "order",
            "transaction",
            "reference",
            "key",
            "uuid",
            "guid",
        ]

        for pattern in identifier_patterns:
            if pattern in column_name:
                # Additional validation: high cardinality (most values are unique)
                unique_ratio = series.nunique() / len(series)
                if unique_ratio > 0.7:  # More than 70% unique values
                    return True
                # Or moderate cardinality with integer-only values
                if self._is_integer_valued(series) and unique_ratio > 0.3:
                    return True

        return False

    def _is_likely_encoded_categorical(self, series: pd.Series) -> bool:
        """
        Check if numeric values look like encoded categories.
        Returns True if values are:
        - Integers (or float representation of integers)
        - Sequential or have regular gaps
        - Start from 0 or 1 (common encoding patterns)
        """
        # Check if all values are effectively integers
        if not self._is_integer_valued(series):
            return False

        # Get unique integer values
        unique_vals = sorted(series.unique())

        # Pattern 1: Sequential starting from 0 or 1 (classic label encoding)
        min_val = unique_vals[0]
        max_val = unique_vals[-1]
        expected_range = max_val - min_val + 1

        if min_val in [0, 1] and len(unique_vals) == expected_range:
            return True  # Sequential encoding detected

        # Pattern 2: Small set of distinct integer values (e.g., 1, 2, 3, 5, 7)
        if len(unique_vals) <= self.cardinality_threshold:
            return True

        return False

    def _is_integer_valued(self, series: pd.Series) -> bool:
        """Check if all values are integers (even if stored as float)."""
        try:
            # For numeric types, check if values equal their integer conversion
            if pd.api.types.is_numeric_dtype(series):
                return np.allclose(series, series.astype(int), rtol=0, atol=1e-10)
            return False
        except (ValueError, TypeError, OverflowError):
            return False

    def _is_integer_valued_float(self, series: pd.Series) -> bool:
        """Specifically check if a float column contains only integer values."""
        try:
            return (series == series.astype(int)).all()
        except (ValueError, TypeError, OverflowError):
            return False

    def detect_all_features(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect types for all features in a DataFrame.

        Args:
            df: pandas DataFrame

        Returns:
            Dictionary mapping feature names to types ('numerical' or 'categorical')
        """
        return {col: self.detect_feature_type(df[col]) for col in df.columns}

    def get_numerical_features(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of numerical feature names.

        Args:
            df: pandas DataFrame

        Returns:
            List of numerical feature names
        """
        return [col for col, ftype in self.detect_all_features(df).items() if ftype == "numerical"]

    def get_categorical_features(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of categorical feature names.

        Args:
            df: pandas DataFrame

        Returns:
            List of categorical feature names
        """
        return [col for col, ftype in self.detect_all_features(df).items() if ftype == "categorical"]

    def get_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed statistics for feature type detection.

        Args:
            df: pandas DataFrame

        Returns:
            DataFrame with statistics for each feature
        """
        stats = []
        for col in df.columns:
            series = df[col].dropna()
            total = len(series)
            unique = series.nunique()

            stats.append(
                {
                    "feature": col,
                    "dtype": str(df[col].dtype),
                    "detected_type": self.detect_feature_type(df[col]),
                    "total_values": total,
                    "unique_values": unique,
                    "cardinality_ratio": unique / total if total > 0 else 0,
                    "is_integer_valued": (
                        self._is_integer_valued(series) if pd.api.types.is_numeric_dtype(series) else False
                    ),
                    "missing_count": df[col].isnull().sum(),
                }
            )

        return pd.DataFrame(stats)


# Singleton instance with default thresholds
default_detector = FeatureTypeDetector(cardinality_threshold=20, cardinality_ratio_threshold=0.05)


def detect_numerical_features(df: pd.DataFrame, cardinality_threshold: int = 20) -> List[str]:
    """
    Quick function to detect numerical features.

    Args:
        df: pandas DataFrame
        cardinality_threshold: Maximum unique values to consider categorical

    Returns:
        List of numerical feature names
    """
    detector = FeatureTypeDetector(cardinality_threshold=cardinality_threshold)
    return detector.get_numerical_features(df)


def detect_categorical_features(df: pd.DataFrame, cardinality_threshold: int = 20) -> List[str]:
    """
    Quick function to detect categorical features.

    Args:
        df: pandas DataFrame
        cardinality_threshold: Maximum unique values to consider categorical

    Returns:
        List of categorical feature names
    """
    detector = FeatureTypeDetector(cardinality_threshold=cardinality_threshold)
    return detector.get_categorical_features(df)
