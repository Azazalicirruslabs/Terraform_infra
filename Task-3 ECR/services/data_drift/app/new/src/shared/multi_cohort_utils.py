"""
Multi-Cohort Utilities for Data Drift Analysis
Handles loading, parsing, and validation of multiple cohort datasets
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .models import CohortMetadata, FileInfo
from .s3_utils import load_s3_csv

logger = logging.getLogger(__name__)


@dataclass
class CohortInfo:
    """Information parsed from filename"""

    cohort_name: str
    cohort_type: str  # "baseline" or "current"
    original_filename: str
    is_baseline: bool


class ValidationResult:
    """Result of cohort compatibility validation"""

    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.common_features: List[str] = []
        self.missing_features: Dict[str, List[str]] = {}
        self.data_type_mismatches: List[Dict[str, Any]] = []

    def add_error(self, error: str):
        self.is_valid = False
        self.errors.append(error)
        logger.error(f"Validation error: {error}")

    def add_warning(self, warning: str):
        self.warnings.append(warning)
        logger.warning(f"Validation warning: {warning}")


def parse_cohort_from_filename(file_name: str) -> CohortInfo:
    """
    Parse cohort information from filename.

    Naming convention:
    - ref_* or ref_*.csv → baseline (e.g., ref_cohort_1.csv)
    - cur_* or cohort_* or cur_*.csv → current cohort (e.g., cur_baseline.csv, cohort_2.csv)

    Args:
        file_name: Name of the file (e.g., "ref_cohort_1.csv")

    Returns:
        CohortInfo with parsed information
    """
    file_name_lower = file_name.lower()
    base_name = file_name_lower.replace(".csv", "")

    # Determine if baseline or current
    if file_name_lower.startswith("ref_"):
        cohort_type = "baseline"
        is_baseline = True
        # Extract cohort name: ref_cohort_1 → baseline
        cohort_name = "baseline"
        logger.info(f"Identified baseline file: {file_name}")
    elif file_name_lower.startswith("cur_") or file_name_lower.startswith("cohort_"):
        cohort_type = "current"
        is_baseline = False

        # Extract cohort number from filename
        # Examples: cur_baseline.csv → cohort_1, cohort_2.csv → cohort_2, cohort_3.csv → cohort_3
        match = re.search(r"cohort[_\s]*(\d+)", base_name)
        if match:
            cohort_num = match.group(1)
            cohort_name = f"cohort_{cohort_num}"
        elif "baseline" in base_name:
            # cur_baseline.csv is treated as cohort_1 (first current cohort)
            cohort_name = "cohort_1"
        else:
            # Fallback: use filename without prefix
            cohort_name = base_name.replace("cur_", "").replace("cohort_", "")
            if not cohort_name:
                cohort_name = "cohort_unknown"

        logger.info(f"Identified current cohort file: {file_name} → {cohort_name}")
    else:
        # Default: treat as current cohort
        logger.warning(f"File {file_name} doesn't match expected pattern. Treating as current cohort.")
        cohort_type = "current"
        is_baseline = False
        cohort_name = base_name or "cohort_unknown"

    return CohortInfo(
        cohort_name=cohort_name, cohort_type=cohort_type, original_filename=file_name, is_baseline=is_baseline
    )


def load_multiple_cohorts(files: List[FileInfo], max_workers: int = 4) -> Dict[str, Tuple[pd.DataFrame, CohortInfo]]:
    """
    Load multiple cohort files from S3 in parallel.

    Args:
        files: List of FileInfo objects containing S3 URLs
        max_workers: Maximum number of parallel download threads

    Returns:
        Dictionary mapping cohort_name to (DataFrame, CohortInfo)
        Example: {"baseline": (df, info), "cohort_1": (df, info)}
    """
    cohorts = {}

    def load_single_cohort(file_info: FileInfo) -> Tuple[str, pd.DataFrame, CohortInfo]:
        """Load a single cohort file"""
        try:
            logger.info(f"Loading file: {file_info.file_name} from {file_info.url[:50]}...")
            df = load_s3_csv(file_info.url)
            cohort_info = parse_cohort_from_filename(file_info.file_name)
            logger.info(f"Successfully loaded {file_info.file_name}: {df.shape[0]} rows, {df.shape[1]} columns")
            return (cohort_info.cohort_name, df, cohort_info)
        except Exception as e:
            logger.error(f"Failed to load {file_info.file_name}: {e}")
            raise

    # Load files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(load_single_cohort, file_info): file_info for file_info in files}

        for future in as_completed(future_to_file):
            file_info = future_to_file[future]
            try:
                cohort_name, df, cohort_info = future.result()
                cohorts[cohort_name] = (df, cohort_info)
            except Exception as e:
                logger.error(f"Error processing {file_info.file_name}: {e}")
                raise

    # Ensure we have exactly one baseline
    baseline_count = sum(1 for _, (_, info) in cohorts.items() if info.is_baseline)
    if baseline_count == 0:
        raise ValueError("No baseline cohort found. Ensure one file starts with 'ref_'")
    if baseline_count > 1:
        raise ValueError(f"Multiple baseline cohorts found ({baseline_count}). Only one baseline allowed.")

    logger.info(f"Loaded {len(cohorts)} cohorts: {list(cohorts.keys())}")
    return cohorts


def validate_cohort_compatibility(cohorts: Dict[str, Tuple[pd.DataFrame, CohortInfo]]) -> ValidationResult:
    """
    Validate that cohorts are compatible for drift analysis.

    Checks:
    - All cohorts have at least some common features
    - Data types are consistent across cohorts
    - No cohort is empty

    Args:
        cohorts: Dictionary of cohort_name -> (DataFrame, CohortInfo)

    Returns:
        ValidationResult with errors, warnings, and compatibility info
    """
    result = ValidationResult()

    if len(cohorts) < 2:
        result.add_error("At least 2 cohorts required (1 baseline + 1 current)")
        return result

    # Extract DataFrames and feature sets
    dfs = {name: df for name, (df, _) in cohorts.items()}
    feature_sets = {name: set(df.columns) for name, df in dfs.items()}

    # Check for empty DataFrames
    for name, df in dfs.items():
        if df.empty:
            result.add_error(f"Cohort '{name}' is empty")
        if df.shape[0] < 2:
            result.add_warning(f"Cohort '{name}' has only {df.shape[0]} row(s). Statistical tests may be unreliable.")
        if df.shape[1] == 0:
            result.add_error(f"Cohort '{name}' has no features")

    # Find common features
    common_features = set.intersection(*feature_sets.values())
    result.common_features = sorted(list(common_features))

    if not common_features:
        result.add_error("No common features found across all cohorts")
        return result

    logger.info(f"Found {len(common_features)} common features across all cohorts")

    # Check for missing features in some cohorts
    all_features = set.union(*feature_sets.values())
    for name, features in feature_sets.items():
        missing = all_features - features
        if missing:
            result.missing_features[name] = sorted(list(missing))
            result.add_warning(f"Cohort '{name}' is missing {len(missing)} features: {list(missing)[:5]}...")

    # Check data type consistency for common features
    baseline_name = next(name for name, (_, info) in cohorts.items() if info.is_baseline)
    baseline_df = dfs[baseline_name]

    for feature in common_features:
        baseline_dtype = baseline_df[feature].dtype

        for name, df in dfs.items():
            if name == baseline_name:
                continue

            cohort_dtype = df[feature].dtype

            # Check for type mismatches
            if baseline_dtype != cohort_dtype:
                # Allow numeric type variations (int64 vs float64)
                is_numeric_mismatch = pd.api.types.is_numeric_dtype(baseline_dtype) and pd.api.types.is_numeric_dtype(
                    cohort_dtype
                )

                if not is_numeric_mismatch:
                    mismatch = {
                        "feature": feature,
                        "baseline_type": str(baseline_dtype),
                        "cohort": name,
                        "cohort_type": str(cohort_dtype),
                    }
                    result.data_type_mismatches.append(mismatch)
                    result.add_warning(
                        f"Data type mismatch for feature '{feature}': "
                        f"baseline={baseline_dtype}, {name}={cohort_dtype}"
                    )

    return result


def compute_cohort_metadata(cohort_name: str, df: pd.DataFrame, url: str, file_name: str) -> CohortMetadata:
    """
    Compute metadata for a single cohort.

    Args:
        cohort_name: Name of the cohort
        df: DataFrame containing the cohort data
        url: S3 URL of the file
        file_name: Original filename

    Returns:
        CohortMetadata object
    """
    cohort_info = parse_cohort_from_filename(file_name)

    # Compute duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Compute feature-level metadata
    features_metadata = []
    for col in df.columns:
        feature_info = {
            "feature_name": col,
            "data_type": "numerical" if pd.api.types.is_numeric_dtype(df[col]) else "categorical",
            "missing_percent": float((df[col].isna().sum() / len(df)) * 100),
            "unique_count": int(df[col].nunique()),
        }
        features_metadata.append(feature_info)

    metadata = CohortMetadata(
        cohort_name=cohort_name,
        cohort_type=cohort_info.cohort_type,
        file_name=file_name,
        url=url,
        shape=[int(df.shape[0]), int(df.shape[1])],
        total_features=int(df.shape[1]),
        total_rows=int(df.shape[0]),
        duplicate_rows=duplicate_rows,
        features=features_metadata,
    )

    return metadata


def check_data_quality(cohorts: Dict[str, Tuple[pd.DataFrame, CohortInfo]]) -> Dict[str, Any]:
    """
    Check data quality across all cohorts.

    Args:
        cohorts: Dictionary of cohort_name -> (DataFrame, CohortInfo)

    Returns:
        Dictionary with quality metrics and warnings
    """
    quality_report = {"overall_quality": "good", "warnings": [], "metrics": {}}

    for name, (df, info) in cohorts.items():
        cohort_metrics = {
            "missing_value_percent": float((df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100),
            "duplicate_percent": float((df.duplicated().sum() / len(df)) * 100),
            "features_with_high_missing": 0,
            "features_with_no_variance": 0,
        }

        # Check for features with high missing values (>50%)
        high_missing_features = []
        for col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > 50:
                high_missing_features.append(col)

        cohort_metrics["features_with_high_missing"] = len(high_missing_features)

        if high_missing_features:
            quality_report["warnings"].append(
                f"Cohort '{name}' has {len(high_missing_features)} feature(s) with >50% missing values"
            )

        # Check for features with no variance
        no_variance_features = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].std() == 0:
                no_variance_features.append(col)

        cohort_metrics["features_with_no_variance"] = len(no_variance_features)

        if no_variance_features:
            quality_report["warnings"].append(
                f"Cohort '{name}' has {len(no_variance_features)} feature(s) with no variance"
            )

        quality_report["metrics"][name] = cohort_metrics

    # Set overall quality
    if len(quality_report["warnings"]) > 5:
        quality_report["overall_quality"] = "poor"
    elif len(quality_report["warnings"]) > 2:
        quality_report["overall_quality"] = "fair"

    return quality_report
