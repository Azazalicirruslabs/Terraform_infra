import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, entropy, ks_2samp, mannwhitneyu, wasserstein_distance

from shared.auth import get_current_user

from ...shared.ai_explanation_service import ai_explanation_service
from ...shared.models import AnalysisRequest, MultiCohortAnalysisRequest
from ...shared.multi_cohort_utils import load_multiple_cohorts, validate_cohort_compatibility
from ...shared.s3_utils import load_s3_csv, validate_dataframe
from ..services.multi_cohort_drift_service import multi_cohort_analyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


router = APIRouter(prefix="/data_drift", tags=["Data Drift - Feature Analysis"])


@dataclass
class DriftThresholds:
    """Configuration class for drift detection thresholds"""

    # Statistical significance levels
    HIGH_DRIFT_P_VALUE = 0.01
    MEDIUM_DRIFT_P_VALUE = 0.05

    # PSI thresholds (industry standard)
    PSI_LOW = 0.1
    PSI_MEDIUM = 0.25
    PSI_HIGH = 0.5

    # KL divergence thresholds
    KL_LOW = 0.1
    KL_MEDIUM = 0.3
    KL_HIGH = 0.5

    # Wasserstein distance thresholds (normalized)
    WASSERSTEIN_LOW = 0.1
    WASSERSTEIN_MEDIUM = 0.3
    WASSERSTEIN_HIGH = 0.5

    # Missing value change thresholds (percentage)
    MISSING_VALUE_CHANGE_LOW = 5
    MISSING_VALUE_CHANGE_MEDIUM = 15
    MISSING_VALUE_CHANGE_HIGH = 30


class AdvancedDriftDetector:
    """Enhanced drift detection with multiple statistical methods"""

    def __init__(self, thresholds: DriftThresholds = None):
        self.thresholds = thresholds or DriftThresholds()

    def detect_numerical_drift(self, ref_data: pd.Series, curr_data: pd.Series) -> Dict[str, Any]:
        """Comprehensive numerical drift detection using multiple methods"""
        try:
            # Remove NaN values
            ref_clean = ref_data.dropna()
            curr_clean = curr_data.dropna()

            if len(ref_clean) == 0 or len(curr_clean) == 0:
                return self._create_empty_numerical_result()

            # Statistical tests
            ks_stat, ks_p_value = ks_2samp(ref_clean, curr_clean)
            mw_stat, mw_p_value = mannwhitneyu(ref_clean, curr_clean, alternative="two-sided")

            # Distance measures
            wasserstein_dist = wasserstein_distance(ref_clean, curr_clean)

            # Normalize Wasserstein distance by data range
            data_range = max(ref_clean.max() - ref_clean.min(), curr_clean.max() - curr_clean.min())
            normalized_wasserstein = wasserstein_dist / max(data_range, 1e-10)

            # PSI calculation
            psi_score = self.calculate_psi(ref_clean, curr_clean, feature_type="numerical")

            # KL divergence
            kl_divergence = self.calculate_kl_divergence(ref_clean, curr_clean, feature_type="numerical")

            # Statistical moments comparison
            ref_mean, curr_mean = float(ref_clean.mean()), float(curr_clean.mean())
            ref_std, curr_std = ref_clean.std(), curr_clean.std()
            ref_skew, curr_skew = ref_clean.skew(), curr_clean.skew()
            ref_kurt, curr_kurt = ref_clean.kurtosis(), curr_clean.kurtosis()

            # Calculate percentage changes
            mean_change = self.calculate_percentage_change(ref_mean, curr_mean)
            std_change = self.calculate_percentage_change(ref_std, curr_std)

            # Primary drift score (weighted combination)
            drift_score = self._calculate_combined_drift_score(
                ks_stat, psi_score, normalized_wasserstein, kl_divergence
            )

            # Drift severity determination
            drift_severity, confidence_level = self._determine_drift_severity_numerical(
                ks_p_value, mw_p_value, psi_score, kl_divergence, normalized_wasserstein
            )

            # Create histogram for visualization
            histogram_data = self._create_histogram_data(ref_clean, curr_clean)

            return {
                "drift_score": round(drift_score, 4),
                "drift_detected": drift_severity != "Low",
                "drift_severity": drift_severity,
                "confidence_level": confidence_level,
                "statistical_tests": {
                    "kolmogorov_smirnov": {"statistic": round(ks_stat, 4), "p_value": round(ks_p_value, 6)},
                    "mann_whitney_u": {"statistic": round(float(mw_stat), 4), "p_value": round(mw_p_value, 6)},
                },
                "distance_measures": {
                    "psi": round(psi_score, 4),
                    "kl_divergence": round(kl_divergence, 4),
                    "wasserstein_distance": round(wasserstein_dist, 4),
                    "normalized_wasserstein": round(normalized_wasserstein, 4),
                },
                "descriptive_stats": {
                    "ref_mean": round(float(ref_mean), 4),
                    "curr_mean": round(float(curr_mean), 4),
                    "ref_std": round(float(ref_std), 4),
                    "curr_std": round(float(curr_std), 4),
                    "ref_skewness": round(float(ref_skew), 4),
                    "curr_skewness": round(float(curr_skew), 4),
                    "ref_kurtosis": round(float(ref_kurt), 4),
                    "curr_kurtosis": round(float(curr_kurt), 4),
                    "mean_change_percent": round(mean_change, 2),
                    "std_change_percent": round(std_change, 2),
                },
                "visualization_data": histogram_data,
            }

        except Exception as e:
            logger.warning(f"Error in numerical drift detection: {e}")
            return self._create_empty_numerical_result()

    def detect_categorical_drift(self, ref_data: pd.Series, curr_data: pd.Series) -> Dict[str, Any]:
        """Comprehensive categorical drift detection"""
        try:
            # Remove NaN values
            ref_clean = ref_data.dropna()
            curr_clean = curr_data.dropna()

            if len(ref_clean) == 0 or len(curr_clean) == 0:
                return self._create_empty_categorical_result()

            # Get value counts
            ref_counts = ref_clean.value_counts()
            curr_counts = curr_clean.value_counts()

            # Get all unique categories
            all_categories = sorted(set(ref_counts.index) | set(curr_counts.index))

            if len(all_categories) <= 1:
                return self._create_empty_categorical_result()

            # Align counts for all categories
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]

            # Chi-square test
            try:
                chi2_stat, chi2_p_value, dof, expected = chi2_contingency([ref_aligned, curr_aligned])
                cramers_v = self._calculate_cramers_v(
                    chi2_stat, sum(ref_aligned) + sum(curr_aligned), len(all_categories)
                )
            except (ValueError, ZeroDivisionError):
                chi2_stat, chi2_p_value, cramers_v = 0.0, 1.0, 0.0

            # Convert to probability distributions
            ref_total = sum(ref_aligned)
            curr_total = sum(curr_aligned)

            ref_probs = [count / ref_total for count in ref_aligned]
            curr_probs = [count / curr_total for count in curr_aligned]

            # Jensen-Shannon divergence
            js_divergence = jensenshannon(ref_probs, curr_probs) ** 2

            # PSI calculation
            psi_score = self.calculate_psi_categorical(ref_probs, curr_probs)

            # KL divergence
            kl_divergence = self.calculate_kl_divergence_categorical(ref_probs, curr_probs)

            # Calculate drift score
            drift_score = self._calculate_combined_categorical_drift_score(
                chi2_stat, psi_score, js_divergence, kl_divergence, ref_total + curr_total
            )

            # Determine drift severity
            drift_severity, confidence_level = self._determine_drift_severity_categorical(
                chi2_p_value, psi_score, js_divergence, cramers_v
            )

            # Category analysis
            category_analysis = self._analyze_category_changes(
                all_categories, ref_counts, curr_counts, ref_total, curr_total
            )

            return {
                "drift_score": round(drift_score, 4),
                "drift_detected": drift_severity != "Low",
                "drift_severity": drift_severity,
                "confidence_level": confidence_level,
                "statistical_tests": {
                    "chi_square": {"statistic": round(chi2_stat, 4), "p_value": round(chi2_p_value, 6)},
                    "cramers_v": round(cramers_v, 4),
                },
                "distance_measures": {
                    "psi": round(psi_score, 4),
                    "kl_divergence": round(kl_divergence, 4),
                    "jensen_shannon_divergence": round(js_divergence, 4),
                },
                "category_analysis": category_analysis,
                "category_distributions": {
                    "reference": {str(k): float(v) for k, v in zip(all_categories, ref_probs)},
                    "current": {str(k): float(v) for k, v in zip(all_categories, curr_probs)},
                    "reference_counts": {str(k): int(v) for k, v in zip(all_categories, ref_aligned)},
                    "current_counts": {str(k): int(v) for k, v in zip(all_categories, curr_aligned)},
                },
            }

        except Exception as e:
            logger.warning(f"Error in categorical drift detection: {e}")
            return self._create_empty_categorical_result()

    def calculate_psi(
        self, ref_data: pd.Series, curr_data: pd.Series, feature_type: str = "numerical", bins: int = 10
    ) -> float:
        """Calculate Population Stability Index"""
        try:
            if feature_type == "numerical":
                # Create bins based on reference data
                _, bin_edges = np.histogram(ref_data, bins=bins)
                ref_counts, _ = np.histogram(ref_data, bins=bin_edges)
                curr_counts, _ = np.histogram(curr_data, bins=bin_edges)
            else:
                # For categorical data, use value counts
                ref_counts = ref_data.value_counts()
                curr_counts = curr_data.value_counts()
                all_cats = ref_counts.index.union(curr_counts.index)
                ref_counts = [ref_counts.get(cat, 0) for cat in all_cats]
                curr_counts = [curr_counts.get(cat, 0) for cat in all_cats]

            # Convert to percentages
            ref_pct = np.array(ref_counts) / max(sum(ref_counts), 1)
            curr_pct = np.array(curr_counts) / max(sum(curr_counts), 1)

            # Calculate PSI
            psi = 0
            for i in range(len(ref_pct)):
                if ref_pct[i] > 0 and curr_pct[i] > 0:
                    psi += (ref_pct[i] - curr_pct[i]) * np.log(ref_pct[i] / curr_pct[i])
                elif ref_pct[i] == 0 and curr_pct[i] > 0:
                    psi += curr_pct[i] * 1  # Penalty for new categories
                elif ref_pct[i] > 0 and curr_pct[i] == 0:
                    psi += ref_pct[i] * 1  # Penalty for missing categories

            return abs(psi)

        except Exception as e:
            logger.warning(f"Error calculating PSI: {e}")
            return 0.0

    def calculate_kl_divergence(
        self, ref_data: pd.Series, curr_data: pd.Series, feature_type: str = "numerical", bins: int = 10
    ) -> float:
        """Calculate Kullback-Leibler divergence"""
        try:
            if feature_type == "numerical":
                # Create bins and get distributions
                _, bin_edges = np.histogram(ref_data, bins=bins)
                ref_counts, _ = np.histogram(ref_data, bins=bin_edges)
                curr_counts, _ = np.histogram(curr_data, bins=bin_edges)

                # Convert to probabilities with smoothing
                ref_probs = (ref_counts + 1e-10) / (sum(ref_counts) + bins * 1e-10)
                curr_probs = (curr_counts + 1e-10) / (sum(curr_counts) + bins * 1e-10)
            else:
                ref_counts = ref_data.value_counts()
                curr_counts = curr_data.value_counts()
                all_cats = ref_counts.index.union(curr_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_cats]
                curr_aligned = [curr_counts.get(cat, 0) for cat in all_cats]

                # Convert to probabilities with smoothing
                ref_probs = np.array(ref_aligned) + 1e-10
                curr_probs = np.array(curr_aligned) + 1e-10
                ref_probs = ref_probs / sum(ref_probs)
                curr_probs = curr_probs / sum(curr_probs)

            # Calculate symmetric KL divergence
            kl_div = 0.5 * (entropy(ref_probs, curr_probs) + entropy(curr_probs, ref_probs))
            return kl_div

        except Exception as e:
            logger.warning(f"Error calculating KL divergence: {e}")
            return 0.0

    def calculate_psi_categorical(self, ref_probs: List[float], curr_probs: List[float]) -> float:
        """Calculate PSI for categorical data"""
        try:
            psi = 0
            for ref_p, curr_p in zip(ref_probs, curr_probs):
                if ref_p > 0 and curr_p > 0:
                    psi += (ref_p - curr_p) * np.log(ref_p / curr_p)
                elif ref_p == 0 and curr_p > 0:
                    psi += curr_p * 1
                elif ref_p > 0 and curr_p == 0:
                    psi += ref_p * 1
            return abs(psi)
        except Exception as e:
            logger.warning(f"Error calculating categorical PSI: {e}")
            return 0.0

    def calculate_kl_divergence_categorical(self, ref_probs: List[float], curr_probs: List[float]) -> float:
        """Calculate KL divergence for categorical data"""
        try:
            # Add smoothing to avoid division by zero
            ref_smooth = np.array(ref_probs) + 1e-10
            curr_smooth = np.array(curr_probs) + 1e-10
            ref_smooth = ref_smooth / sum(ref_smooth)
            curr_smooth = curr_smooth / sum(curr_smooth)

            # Symmetric KL divergence
            kl_div = 0.5 * (entropy(ref_smooth, curr_smooth) + entropy(curr_smooth, ref_smooth))
            return kl_div
        except Exception as e:
            logger.warning(f"Error calculating categorical KL divergence: {e}")
            return 0.0

    def calculate_percentage_change(self, ref_val: float, curr_val: float) -> float:
        """Calculate percentage change with proper handling of edge cases"""
        if abs(ref_val) < 1e-10:  # Reference is effectively zero
            return 0.0 if abs(curr_val) < 1e-10 else 100.0
        return ((curr_val - ref_val) / abs(ref_val)) * 100

    def _calculate_combined_drift_score(self, ks_stat: float, psi: float, wasserstein: float, kl_div: float) -> float:
        """Calculate weighted combination drift score for numerical features"""
        # Normalize each component to 0-1 scale
        ks_normalized = min(ks_stat * 2, 1.0)  # KS stat is typically 0-0.5
        psi_normalized = min(psi / self.thresholds.PSI_HIGH, 1.0)
        wasserstein_normalized = min(wasserstein / self.thresholds.WASSERSTEIN_HIGH, 1.0)
        kl_normalized = min(kl_div / self.thresholds.KL_HIGH, 1.0)

        # Weighted combination (KS and PSI are most reliable)
        combined_score = 0.3 * ks_normalized + 0.3 * psi_normalized + 0.2 * wasserstein_normalized + 0.2 * kl_normalized

        return combined_score

    def _calculate_combined_categorical_drift_score(
        self, chi2_stat: float, psi: float, js_div: float, kl_div: float, total_samples: int
    ) -> float:
        """Calculate weighted combination drift score for categorical features"""
        # Normalize chi-square by degrees of freedom and sample size
        chi2_normalized = min(chi2_stat / max(total_samples, 100), 1.0)
        psi_normalized = min(psi / self.thresholds.PSI_HIGH, 1.0)
        js_normalized = min(js_div, 1.0)  # JS divergence is already 0-1
        kl_normalized = min(kl_div / self.thresholds.KL_HIGH, 1.0)

        # Weighted combination
        combined_score = 0.4 * chi2_normalized + 0.3 * psi_normalized + 0.2 * js_normalized + 0.1 * kl_normalized

        return combined_score

    def _determine_drift_severity_numerical(
        self, ks_p: float, mw_p: float, psi: float, kl_div: float, wasserstein: float
    ) -> Tuple[str, str]:
        """Determine drift severity and confidence for numerical features"""
        # Count strong indicators
        strong_indicators = 0
        medium_indicators = 0

        # Statistical significance
        if min(ks_p, mw_p) < self.thresholds.HIGH_DRIFT_P_VALUE:
            strong_indicators += 2
        elif min(ks_p, mw_p) < self.thresholds.MEDIUM_DRIFT_P_VALUE:
            medium_indicators += 1

        # PSI thresholds
        if psi > self.thresholds.PSI_HIGH:
            strong_indicators += 1
        elif psi > self.thresholds.PSI_MEDIUM:
            medium_indicators += 1

        # KL divergence
        if kl_div > self.thresholds.KL_HIGH:
            strong_indicators += 1
        elif kl_div > self.thresholds.KL_MEDIUM:
            medium_indicators += 1

        # Wasserstein distance
        if wasserstein > self.thresholds.WASSERSTEIN_HIGH:
            strong_indicators += 1
        elif wasserstein > self.thresholds.WASSERSTEIN_MEDIUM:
            medium_indicators += 1

        # Determine severity and confidence
        if strong_indicators >= 2:
            return "Critical", "High"
        elif strong_indicators >= 1 or medium_indicators >= 3:
            return "High", "High" if strong_indicators >= 1 else "Medium"
        elif medium_indicators >= 1:
            return "Medium", "Medium"
        else:
            return "Low", "High"

    def _determine_drift_severity_categorical(
        self, chi2_p: float, psi: float, js_div: float, cramers_v: float
    ) -> Tuple[str, str]:
        """Determine drift severity and confidence for categorical features"""
        strong_indicators = 0
        medium_indicators = 0

        # Statistical significance
        if chi2_p < self.thresholds.HIGH_DRIFT_P_VALUE:
            strong_indicators += 2
        elif chi2_p < self.thresholds.MEDIUM_DRIFT_P_VALUE:
            medium_indicators += 1

        # PSI thresholds
        if psi > self.thresholds.PSI_HIGH:
            strong_indicators += 1
        elif psi > self.thresholds.PSI_MEDIUM:
            medium_indicators += 1

        # Jensen-Shannon divergence
        if js_div > 0.5:
            strong_indicators += 1
        elif js_div > 0.3:
            medium_indicators += 1

        # Cramer's V (effect size)
        if cramers_v > 0.5:
            strong_indicators += 1
        elif cramers_v > 0.3:
            medium_indicators += 1

        # Determine severity and confidence
        if strong_indicators >= 2:
            return "Critical", "High"
        elif strong_indicators >= 1 or medium_indicators >= 3:
            return "High", "High" if strong_indicators >= 1 else "Medium"
        elif medium_indicators >= 1:
            return "Medium", "Medium"
        else:
            return "Low", "High"

    def _calculate_cramers_v(self, chi2_stat: float, n: int, k: int) -> float:
        """Calculate Cramer's V (effect size for chi-square)"""
        try:
            if n == 0 or k <= 1:
                return 0.0
            return np.sqrt(chi2_stat / (n * (k - 1)))
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _create_histogram_data(self, ref_data: pd.Series, curr_data: pd.Series, bins: int = 20) -> Dict[str, Any]:
        """Create histogram data for visualization"""
        try:
            # Use common bins for both datasets
            all_data = pd.concat([ref_data, curr_data])
            bin_edges = np.histogram_bin_edges(all_data, bins=bins)

            ref_hist, _ = np.histogram(ref_data, bins=bin_edges)
            curr_hist, _ = np.histogram(curr_data, bins=bin_edges)

            return {
                "reference_histogram": ref_hist.tolist(),
                "current_histogram": curr_hist.tolist(),
                "bin_edges": bin_edges.tolist(),
                "bin_centers": ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist(),
            }
        except Exception as e:
            logger.warning(f"Error creating histogram data: {e}")
            return {"reference_histogram": [], "current_histogram": [], "bin_edges": [], "bin_centers": []}

    def _analyze_category_changes(
        self, categories: List, ref_counts: pd.Series, curr_counts: pd.Series, ref_total: int, curr_total: int
    ) -> Dict[str, Any]:
        """Analyze changes in categorical distribution"""
        try:
            category_changes = []
            new_categories = []
            disappeared_categories = []

            for cat in categories:
                ref_count = ref_counts.get(cat, 0)
                curr_count = curr_counts.get(cat, 0)

                ref_pct = (ref_count / ref_total) * 100 if ref_total > 0 else 0
                curr_pct = (curr_count / curr_total) * 100 if curr_total > 0 else 0

                change_pct = curr_pct - ref_pct

                if ref_count == 0 and curr_count > 0:
                    new_categories.append(
                        {"category": str(cat), "frequency": int(curr_count), "percentage": float(curr_pct)}
                    )
                elif ref_count > 0 and curr_count == 0:
                    disappeared_categories.append(
                        {"category": str(cat), "frequency": int(ref_count), "percentage": float(ref_pct)}
                    )
                elif abs(change_pct) > 1.0:  # Significant change threshold
                    category_changes.append(
                        {
                            "category": str(cat),
                            "ref_count": int(ref_count),
                            "curr_count": int(curr_count),
                            "ref_percentage": round(float(ref_pct), 2),
                            "curr_percentage": round(float(curr_pct), 2),
                            "change_percentage": round(float(change_pct), 2),
                        }
                    )

            return {
                "significant_changes": sorted(
                    category_changes, key=lambda x: abs(x["change_percentage"]), reverse=True
                )[:10],
                "new_categories": new_categories,
                "disappeared_categories": disappeared_categories,
                "total_categories_ref": int(len([c for c in categories if ref_counts.get(c, 0) > 0])),
                "total_categories_curr": int(len([c for c in categories if curr_counts.get(c, 0) > 0])),
            }
        except Exception as e:
            logger.warning(f"Error analyzing category changes: {e}")
            return {
                "significant_changes": [],
                "new_categories": [],
                "disappeared_categories": [],
                "total_categories_ref": 0,
                "total_categories_curr": 0,
            }

    def _create_empty_numerical_result(self) -> Dict[str, Any]:
        """Create empty result for failed numerical analysis"""
        return {
            "drift_score": 0.0,
            "drift_detected": False,
            "drift_severity": "Unknown",
            "confidence_level": "Low",
            "statistical_tests": {
                "kolmogorov_smirnov": {"statistic": 0.0, "p_value": 1.0},
                "mann_whitney_u": {"statistic": 0.0, "p_value": 1.0},
            },
            "distance_measures": {
                "psi": 0.0,
                "kl_divergence": 0.0,
                "wasserstein_distance": 0.0,
                "normalized_wasserstein": 0.0,
            },
            "descriptive_stats": {
                "ref_mean": 0.0,
                "curr_mean": 0.0,
                "ref_std": 0.0,
                "curr_std": 0.0,
                "ref_skewness": 0.0,
                "curr_skewness": 0.0,
                "ref_kurtosis": 0.0,
                "curr_kurtosis": 0.0,
                "mean_change_percent": 0.0,
                "std_change_percent": 0.0,
            },
            "visualization_data": {
                "reference_histogram": [],
                "current_histogram": [],
                "bin_edges": [],
                "bin_centers": [],
            },
        }

    def _create_empty_categorical_result(self) -> Dict[str, Any]:
        """Create empty result for failed categorical analysis"""
        return {
            "drift_score": 0.0,
            "drift_detected": False,
            "drift_severity": "Unknown",
            "confidence_level": "Low",
            "statistical_tests": {"chi_square": {"statistic": 0.0, "p_value": 1.0}, "cramers_v": 0.0},
            "distance_measures": {"psi": 0.0, "kl_divergence": 0.0, "jensen_shannon_divergence": 0.0},
            "category_analysis": {
                "significant_changes": [],
                "new_categories": [],
                "disappeared_categories": [],
                "total_categories_ref": 0,
                "total_categories_curr": 0,
            },
            "category_distributions": {"reference": {}, "current": {}, "reference_counts": {}, "current_counts": {}},
        }


def create_ai_summary_for_feature_analysis(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Summarizes the detailed feature analysis into a compact format suitable for an LLM prompt"""
    try:
        data = analysis_data.get("data", {})
        executive_summary = data.get("executive_summary", {})
        insights = data.get("insights", {})

        summary = {
            "total_features_analyzed": executive_summary.get("total_features_analyzed", 0),
            "features_with_drift": executive_summary.get("features_with_drift", 0),
            "drift_percentage": executive_summary.get("drift_percentage", 0),
            "overall_drift_score": executive_summary.get("overall_drift_score", 0),
            "highest_drift_feature": executive_summary.get("highest_drift_feature"),
            "risk_level": insights.get("risk_level", "Unknown"),
            "summary_text": insights.get("summary_text", ""),
            "recommendations": insights.get("recommendations", []),
        }

        # Get top 10 most drifted features with enhanced details
        feature_analysis = data.get("feature_analysis", [])
        sorted_features = sorted(feature_analysis, key=lambda x: x.get("drift_score", 0), reverse=True)

        top_features = []
        for feature in sorted_features[:10]:
            feature_summary = {
                "feature_name": feature.get("feature_name"),
                "feature_type": feature.get("feature_type"),
                "drift_score": round(feature.get("drift_score", 0), 3),
                "drift_severity": feature.get("drift_severity"),
                "confidence_level": feature.get("confidence_level"),
                "drift_detected": feature.get("drift_detected", False),
            }

            # Add specific metrics based on feature type
            if feature.get("feature_type") == "numerical":
                stats = feature.get("descriptive_stats", {})
                feature_summary.update(
                    {
                        "mean_change_percent": stats.get("mean_change_percent", 0),
                        "std_change_percent": stats.get("std_change_percent", 0),
                    }
                )
            elif feature.get("feature_type") == "categorical":
                cat_analysis = feature.get("category_analysis", {})
                feature_summary.update(
                    {
                        "new_categories_count": len(cat_analysis.get("new_categories", [])),
                        "disappeared_categories_count": len(cat_analysis.get("disappeared_categories", [])),
                    }
                )

            top_features.append(feature_summary)

        summary["top_drifted_features"] = top_features
        return summary

    except Exception as e:
        logger.warning(f"Error creating AI summary: {e}")
        return {"error": "Failed to create AI summary"}


def generate_enhanced_insights(
    feature_results: List[Dict[str, Any]], overall_drift_score: float, analysis_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate comprehensive human-readable insights for feature analysis"""
    try:
        # Categorize features by drift severity
        critical_features = [f for f in feature_results if f.get("drift_severity") == "Critical"]
        high_drift_features = [f for f in feature_results if f.get("drift_severity") == "High"]
        medium_drift_features = [f for f in feature_results if f.get("drift_severity") == "Medium"]
        low_drift_features = [f for f in feature_results if f.get("drift_severity") == "Low"]

        total_features = len(feature_results)
        drifted_features = critical_features + high_drift_features + medium_drift_features

        if len(drifted_features) == 0:
            return {
                "summary": "Excellent data stability detected across the dataset. All features maintain stable distributions with no significant drift patterns.",
                "risk_assessment": "Minimal Risk",
                "business_impact": "No immediate impact expected on model performance.",
                "recommendations": [
                    "Continue regular monitoring with current thresholds",
                    "Model should perform reliably on new data",
                    "Consider extending monitoring intervals due to stability",
                    "Focus monitoring resources on other data quality aspects",
                ],
                "technical_insights": [
                    "Statistical tests show no significant distribution changes",
                    "PSI scores remain within acceptable ranges",
                    "Feature relationships appear stable",
                ],
            }

        # Calculate percentages
        critical_pct = (len(critical_features) / total_features) * 100
        high_pct = (len(high_drift_features) / total_features) * 100
        drift_pct = (len(drifted_features) / total_features) * 100

        # Determine overall risk level
        if critical_pct > 20 or overall_drift_score > 0.8:
            risk_level = "Critical Risk"
            risk_color = "red"
        elif critical_pct > 10 or high_pct > 30 or overall_drift_score > 0.6:
            risk_level = "High Risk"
            risk_color = "orange"
        elif high_pct > 15 or drift_pct > 40 or overall_drift_score > 0.4:
            risk_level = "Moderate Risk"
            risk_color = "yellow"
        elif drift_pct > 20 or overall_drift_score > 0.2:
            risk_level = "Low Risk"
            risk_color = "blue"
        else:
            risk_level = "Minimal Risk"
            risk_color = "green"

        # Generate summary text
        summary_parts = []
        summary_parts.append(
            f"Data drift analysis identified {len(drifted_features)} out of {total_features} features ({drift_pct:.1f}%) showing significant distribution changes."
        )

        if critical_features:
            critical_names = [f"'{f['feature_name']}'" for f in critical_features[:3]]
            summary_parts.append(
                f"Critical drift detected in {len(critical_features)} features: {', '.join(critical_names)}{'...' if len(critical_features) > 3 else ''}."
            )

        if high_drift_features:
            summary_parts.append(f"High-severity drift affects {len(high_drift_features)} additional features.")

        summary_parts.append(
            f"Overall drift score: {overall_drift_score:.3f} indicates {risk_level.lower()} to model performance."
        )

        # Business impact assessment
        if risk_level == "Critical Risk":
            business_impact = "Immediate action required. Model performance likely severely degraded. Prediction reliability compromised."
        elif risk_level == "High Risk":
            business_impact = "Significant model performance degradation expected. Business decisions may be affected by reduced prediction accuracy."
        elif risk_level == "Moderate Risk":
            business_impact = "Moderate impact on model performance. Monitor business metrics closely and prepare mitigation strategies."
        elif risk_level == "Low Risk":
            business_impact = (
                "Minor impact expected. Current model should perform adequately with increased monitoring."
            )
        else:
            business_impact = "No immediate impact expected on model performance or business operations."

        # Generate recommendations
        recommendations = []
        technical_insights = []

        if risk_level == "Critical Risk":
            recommendations.extend(
                [
                    "Immediate model retraining required with recent data",
                    "Implement emergency model rollback procedures if available",
                    "Investigate root causes of critical feature changes",
                    "Establish real-time drift monitoring and alerting",
                    "Review data collection and preprocessing pipelines",
                    "Consider ensemble methods to improve robustness",
                ]
            )
            technical_insights.extend(
                [
                    "Multiple features show severe distribution shifts",
                    "Statistical significance tests indicate systematic changes",
                    "PSI scores exceed critical thresholds",
                    "Feature relationships likely altered",
                ]
            )
        elif risk_level == "High Risk":
            recommendations.extend(
                [
                    "Schedule model retraining within 1-2 weeks",
                    "Increase monitoring frequency for affected features",
                    "Implement gradient-based model updates if possible",
                    "Validate model performance on recent data",
                    "Investigate data source changes or external factors",
                    "Consider feature engineering adjustments",
                ]
            )
            technical_insights.extend(
                [
                    "Significant drift patterns detected across multiple features",
                    "Distribution shifts exceed normal variation ranges",
                    "Model assumptions likely violated",
                ]
            )
        elif risk_level == "Moderate Risk":
            recommendations.extend(
                [
                    "Plan model refresh within 4-6 weeks",
                    "Monitor model performance metrics daily",
                    "Investigate drift patterns in most affected features",
                    "Consider incremental learning approaches",
                    "Update feature importance rankings",
                    "Enhance data quality monitoring",
                ]
            )
            technical_insights.extend(
                [
                    "Moderate distribution changes detected",
                    "Some statistical tests indicate significance",
                    "Feature stability shows concerning trends",
                ]
            )
        elif risk_level == "Low Risk":
            recommendations.extend(
                [
                    "Continue standard monitoring protocols",
                    "Schedule model review in 2-3 months",
                    "Monitor drifted features more closely",
                    "Document observed patterns for trend analysis",
                    "Consider threshold adjustments if needed",
                ]
            )
            technical_insights.extend(
                [
                    "Limited but measurable distribution changes",
                    "Most features remain within acceptable ranges",
                    "Drift patterns are manageable",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Maintain current monitoring schedule",
                    "Continue with existing model deployment",
                    "Consider reducing monitoring frequency",
                    "Focus resources on other data quality initiatives",
                ]
            )
            technical_insights.extend(
                [
                    "All features show excellent stability",
                    "Distribution patterns remain consistent",
                    "Model assumptions well maintained",
                ]
            )

        # Feature type insights
        numerical_features = [f for f in feature_results if f.get("feature_type") == "numerical"]
        categorical_features = [f for f in feature_results if f.get("feature_type") == "categorical"]

        if numerical_features:
            num_drifted = len([f for f in numerical_features if f.get("drift_detected")])
            technical_insights.append(f"Numerical features: {num_drifted}/{len(numerical_features)} showing drift")

        if categorical_features:
            cat_drifted = len([f for f in categorical_features if f.get("drift_detected")])
            technical_insights.append(f"Categorical features: {cat_drifted}/{len(categorical_features)} showing drift")

        return {
            "summary": " ".join(summary_parts),
            "risk_assessment": risk_level,
            "risk_color": risk_color,
            "business_impact": business_impact,
            "recommendations": recommendations,
            "technical_insights": technical_insights,
            "feature_breakdown": {
                "critical": len(critical_features),
                "high": len(high_drift_features),
                "medium": len(medium_drift_features),
                "low": len(low_drift_features),
                "stable": total_features - len(drifted_features),
            },
        }

    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        return {
            "summary": "Feature analysis completed but insights generation failed.",
            "risk_assessment": "Unknown",
            "recommendations": ["Review analysis results manually"],
            "technical_insights": ["Detailed analysis available in feature_analysis section"],
        }


def calculate_feature_business_impact(
    feature_result: Dict[str, Any], feature_importance: Dict[str, float] = None
) -> float:
    """Calculate enhanced business impact score considering feature importance and drift characteristics"""
    try:
        base_score = feature_result.get("drift_score", 0)
        feature_name = feature_result.get("feature_name", "")

        # Statistical significance multiplier
        primary_test = feature_result.get("statistical_tests", {})
        if feature_result.get("feature_type") == "numerical":
            p_value = primary_test.get("kolmogorov_smirnov", {}).get("p_value", 1.0)
        else:
            p_value = primary_test.get("chi_square", {}).get("p_value", 1.0)

        if p_value < 0.001:
            significance_multiplier = 1.4
        elif p_value < 0.01:
            significance_multiplier = 1.2
        elif p_value < 0.05:
            significance_multiplier = 1.1
        else:
            significance_multiplier = 0.9

        # Confidence level multiplier
        confidence = feature_result.get("confidence_level", "Medium")
        confidence_multiplier = {"High": 1.2, "Medium": 1.0, "Low": 0.8}.get(confidence, 1.0)

        # Feature type adjustment
        feature_type = feature_result.get("feature_type", "numerical")
        type_multiplier = 1.1 if feature_type == "categorical" else 1.0

        # Feature importance (if available)
        importance_multiplier = 1.0
        if feature_importance and feature_name in feature_importance:
            # Normalize importance to 0.8-1.4 range
            max_importance = max(feature_importance.values()) if feature_importance else 1.0
            normalized_importance = feature_importance[feature_name] / max_importance
            importance_multiplier = 0.8 + (normalized_importance * 0.6)

        # PSI-based adjustment
        distance_measures = feature_result.get("distance_measures", {})
        psi_score = distance_measures.get("psi", 0)
        psi_multiplier = 1.0
        if psi_score > 0.25:
            psi_multiplier = 1.3
        elif psi_score > 0.1:
            psi_multiplier = 1.1

        # Calculate final impact score
        impact_score = (
            base_score
            * significance_multiplier
            * confidence_multiplier
            * type_multiplier
            * importance_multiplier
            * psi_multiplier
        )

        # Cap at reasonable maximum
        return min(2.0, max(0.0, impact_score))

    except Exception as e:
        logger.warning(f"Error calculating business impact: {e}")
        return feature_result.get("drift_score", 0.0)


async def _handle_legacy_feature_analysis(request: AnalysisRequest, user: dict):
    """
    Handle legacy single reference vs single current feature analysis

    Args:
        request: AnalysisRequest containing single reference and current URLs
        user: Authenticated user

    Returns:
        Comprehensive feature analysis for single dataset pair
    """
    try:
        # Load data from S3 URLs
        reference_df = load_s3_csv(request.reference_url)
        current_df = load_s3_csv(request.current_url)

        # Validate datasets
        validate_dataframe(reference_df, "Reference")
        validate_dataframe(current_df, "Current")

        # Get common columns
        common_columns = list(set(reference_df.columns) & set(current_df.columns))

        if not common_columns:
            raise HTTPException(status_code=400, detail="No common columns found between datasets")

        # Initialize drift detector
        drift_detector = AdvancedDriftDetector()

        # Track analysis results
        feature_analysis_list = []
        failed_features = []
        total_drift_score = 0.0

        logger.info(f"Starting analysis of {len(common_columns)} features")

        # Analyze each feature
        for col in common_columns:
            try:
                logger.debug(f"Analyzing feature: {col}")

                # Extract series
                ref_series = reference_df[col]
                curr_series = current_df[col]

                # Calculate missing values
                missing_ref = int(ref_series.isna().sum())
                missing_curr = int(curr_series.isna().sum())
                missing_change_pct = drift_detector.calculate_percentage_change(
                    missing_ref / len(ref_series) * 100, missing_curr / len(curr_series) * 100
                )

                # Determine feature type
                feature_type = (
                    "numerical" if ref_series.dtype in ["int64", "float64", "float32", "int32"] else "categorical"
                )

                # Perform drift detection
                if feature_type == "numerical":
                    drift_results = drift_detector.detect_numerical_drift(ref_series, curr_series)
                else:
                    drift_results = drift_detector.detect_categorical_drift(ref_series, curr_series)

                # Skip if analysis failed
                if drift_results.get("drift_severity") == "Unknown":
                    failed_features.append(col)
                    continue

                # Calculate business impact
                business_impact = calculate_feature_business_impact(drift_results)

                # Construct comprehensive feature analysis
                feature_analysis = {
                    "feature_name": col,
                    "feature_type": feature_type,
                    "drift_score": drift_results["drift_score"],
                    "drift_detected": drift_results["drift_detected"],
                    "drift_severity": drift_results["drift_severity"],
                    "confidence_level": drift_results["confidence_level"],
                    "business_impact_score": round(business_impact, 4),
                    # Missing value analysis
                    "missing_values": {
                        "reference_count": missing_ref,
                        "current_count": missing_curr,
                        "reference_percentage": round((missing_ref / len(ref_series)) * 100, 2),
                        "current_percentage": round((missing_curr / len(curr_series)) * 100, 2),
                        "change_percentage": round(missing_change_pct, 2),
                    },
                    # Statistical tests and distance measures
                    "statistical_tests": drift_results["statistical_tests"],
                    "distance_measures": drift_results["distance_measures"],
                }

                # Add type-specific analysis
                if feature_type == "numerical":
                    feature_analysis.update(
                        {
                            "descriptive_statistics": drift_results["descriptive_stats"],
                            "visualization_data": drift_results["visualization_data"],
                        }
                    )
                else:
                    feature_analysis.update(
                        {
                            "category_analysis": drift_results["category_analysis"],
                            "category_distributions": drift_results["category_distributions"],
                        }
                    )

                feature_analysis_list.append(feature_analysis)
                total_drift_score += drift_results["drift_score"]

            except Exception as e:
                logger.warning(f"Failed to analyze feature {col}: {e}")
                failed_features.append(col)
                continue

        if len(feature_analysis_list) == 0:
            raise HTTPException(status_code=400, detail="No features could be successfully analyzed")

        # Calculate overall metrics
        total_features = len(feature_analysis_list)
        overall_drift_score = total_drift_score / total_features

        # Categorize features by severity
        critical_features = [f for f in feature_analysis_list if f["drift_severity"] == "Critical"]
        high_features = [f for f in feature_analysis_list if f["drift_severity"] == "High"]
        medium_features = [f for f in feature_analysis_list if f["drift_severity"] == "Medium"]
        low_features = [f for f in feature_analysis_list if f["drift_severity"] == "Low"]

        # Determine overall status
        critical_pct = (len(critical_features) / total_features) * 100
        high_pct = (len(high_features) / total_features) * 100
        drifted_pct = ((len(critical_features) + len(high_features) + len(medium_features)) / total_features) * 100

        if critical_pct > 20 or overall_drift_score > 0.8:
            overall_status = "critical"
        elif critical_pct > 10 or high_pct > 30 or overall_drift_score > 0.6:
            overall_status = "high"
        elif high_pct > 15 or drifted_pct > 40 or overall_drift_score > 0.4:
            overall_status = "medium"
        else:
            overall_status = "low"

        # Analysis metadata with proper type conversion
        analysis_metadata = {
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_dataset_size": int(len(reference_df)),  # Ensure Python int
            "current_dataset_size": int(len(current_df)),  # Ensure Python int
            "total_features": int(total_features),  # Ensure Python int
            "failed_features": int(len(failed_features)),  # Ensure Python int
            "failed_feature_names": failed_features,
            "common_features_analyzed": int(len(common_columns)),  # Ensure Python int
            "feature_breakdown": {
                "critical_drift": int(len(critical_features)),  # Ensure Python int
                "high_drift": int(len(high_features)),  # Ensure Python int
                "medium_drift": int(len(medium_features)),  # Ensure Python int
                "low_drift": int(len(low_features)),  # Ensure Python int
            },
            "dataset_info": {
                "reference_filename": request.reference_url.split("/")[-1] if request.reference_url else "",
                "current_filename": request.current_url.split("/")[-1] if request.current_url else "",
                # Convert shape tuples to ensure Python ints
                "reference_shape": tuple(int(x) for x in reference_df.shape),
                "current_shape": tuple(int(x) for x in current_df.shape),
            },
        }

        # Generate enhanced insights
        insights = generate_enhanced_insights(feature_analysis_list, overall_drift_score, analysis_metadata)

        # Find highest drift feature
        highest_drift_feature = (
            max(feature_analysis_list, key=lambda x: x["drift_score"])["feature_name"]
            if feature_analysis_list
            else None
        )

        # Construct final result
        result = {
            "status": "success",
            "data": {
                # Executive Summary
                "executive_summary": {
                    "total_features_analyzed": total_features,
                    "features_with_drift": len(critical_features) + len(high_features) + len(medium_features),
                    "drift_percentage": round(drifted_pct, 1),
                    "overall_drift_score": round(overall_drift_score, 4),
                    "overall_status": overall_status,
                    "highest_drift_feature": highest_drift_feature,
                    "business_risk_level": insights["risk_assessment"],
                },
                # Detailed feature analysis (sorted by drift score)
                "feature_analysis": sorted(feature_analysis_list, key=lambda x: x["drift_score"], reverse=True),
                # Enhanced insights and recommendations
                "insights": insights,
                # Analysis metadata
                "analysis_metadata": analysis_metadata,
            },
        }

        # Generate AI explanation
        try:
            ai_summary_payload = create_ai_summary_for_feature_analysis(result)
            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=ai_summary_payload, analysis_type="feature_analysis"
            )
            result["llm_response"] = ai_explanation
            logger.info("AI explanation generated successfully")
        except Exception as e:
            logger.warning(f"AI explanation failed: {e}")
            result["llm_response"] = {
                "summary": f"Comprehensive feature analysis completed for {total_features} features.",
                "detailed_explanation": f"Analysis identified {len(critical_features + high_features)} features with significant drift patterns. {insights['risk_assessment']} detected for model performance.",
                "key_takeaways": [
                    f"Overall drift score: {overall_drift_score:.3f}",
                    f"Risk level: {insights['risk_assessment']}",
                    "Detailed feature-level analysis available",
                    "Recommendations provided for remediation",
                ],
                "recommendations": insights.get("recommendations", [])[:3],
            }

        # Convert numpy types to native Python types for JSON serialization
        result = convert_numpy_types(result)

        logger.info(f"Feature analysis completed successfully")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Legacy feature analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature analysis failed: {str(e)}")


async def _handle_multi_cohort_feature_analysis(request: MultiCohortAnalysisRequest, user: dict):
    """
    Handle multi-cohort feature analysis (1 baseline + up to 4 current cohorts)

    Args:
        request: MultiCohortAnalysisRequest containing multiple file URLs
        user: Authenticated user

    Returns:
        Multi-cohort feature analysis with cross-cohort aggregation
    """
    try:
        logger.info(f"Starting multi-cohort feature analysis with {len(request.files)} files")

        # Load all cohorts in parallel
        cohorts = load_multiple_cohorts(request.files)

        # Validate cohort compatibility
        validation_result = validate_cohort_compatibility(cohorts)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400, detail=f"Cohort validation failed: {', '.join(validation_result.errors)}"
            )

        # Get common features from validation result
        common_features = validation_result.common_features
        logger.info(f"Found {len(common_features)} common features")

        # Identify baseline and current cohorts
        baseline_cohort = None
        baseline_name = None
        current_cohorts = {}

        for cohort_name, (df, info) in cohorts.items():
            if info.is_baseline:
                baseline_cohort = df
                baseline_name = cohort_name
            else:
                current_cohorts[cohort_name] = df

        if baseline_cohort is None or baseline_name is None:
            raise HTTPException(status_code=400, detail="No baseline cohort found. Ensure one file starts with 'ref_'")

        if not current_cohorts:
            raise HTTPException(
                status_code=400,
                detail="No current cohorts found. Ensure at least one file starts with 'cur_' or 'cohort_'",
            )

        logger.info(f"Baseline: {baseline_name}, Current cohorts: {list(current_cohorts.keys())}")

        logger.info(f"Analyzing {len(common_features)} common features across cohorts")

        # Filter features if specified
        if request.selected_features:
            common_features = [f for f in common_features if f in request.selected_features]
            logger.info(f"Filtered to {len(common_features)} selected features")

        # Track results per cohort pair
        cohort_feature_results = {}  # {cohort_name: [PairwiseDriftResult]}
        cohort_feature_details = {}  # {cohort_name: [detailed_feature_dict]}

        # Analyze each current cohort against baseline
        for cohort_name, cohort_df in current_cohorts.items():
            logger.info(f"Analyzing cohort: {cohort_name}")

            # Compute pairwise drift for all features
            pairwise_results = []
            for feature in common_features:
                try:
                    result = multi_cohort_analyzer.compute_pairwise_drift(
                        baseline_cohort, cohort_df, feature, baseline_name, cohort_name
                    )
                    pairwise_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to analyze feature {feature} for cohort {cohort_name}: {e}")
                    continue

            # Store pairwise results for aggregation
            cohort_feature_results[cohort_name] = pairwise_results

            # Build detailed feature analysis for API response
            feature_details_list = []
            for result in pairwise_results:
                # Calculate missing values
                baseline_series = baseline_cohort[result.feature_name]
                cohort_series = cohort_df[result.feature_name]
                missing_baseline = int(baseline_series.isna().sum())
                missing_cohort = int(cohort_series.isna().sum())

                # Map overall_status to drift_severity
                status_to_severity_map = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}
                drift_severity = status_to_severity_map.get(result.overall_status, "Unknown")

                # Build detailed feature dict
                feature_detail = {
                    "feature_name": result.feature_name,
                    "feature_type": result.feature_type,
                    "cohort_name": cohort_name,
                    "drift_score": result.overall_drift_score,
                    "drift_detected": result.overall_status in ["critical", "high", "medium"],
                    "drift_severity": drift_severity,
                    "overall_status": result.overall_status,
                    # Missing value analysis
                    "missing_values": {
                        "baseline_count": missing_baseline,
                        "cohort_count": missing_cohort,
                        "baseline_percentage": round((missing_baseline / len(baseline_series)) * 100, 2),
                        "cohort_percentage": round((missing_cohort / len(cohort_series)) * 100, 2),
                    },
                    # Statistical tests (extract from tests dict)
                    "statistical_tests": {
                        test_name: {
                            "test_name": test_result.test_name,
                            "statistic": test_result.statistic,
                            "p_value": test_result.p_value,
                            "threshold": test_result.threshold,
                            "status": test_result.status,  # "stable", "moderate", "severe"
                            "confidence": test_result.confidence,  # "high", "medium", "low"
                        }
                        for test_name, test_result in result.tests.items()
                    },
                    # Descriptive stats
                    "descriptive_statistics": result.descriptive_stats,
                    # Plot data (contains visualization data for all tests)
                    "plot_data": result.plot_data,
                }

                feature_details_list.append(feature_detail)

            cohort_feature_details[cohort_name] = feature_details_list

        # Aggregate results across all cohorts (flatten all pairwise results)
        all_pairwise_results = []
        for results_list in cohort_feature_results.values():
            all_pairwise_results.extend(results_list)

        aggregated_results = multi_cohort_analyzer.aggregate_drift_across_features(all_pairwise_results)

        # Build cohort summaries
        cohort_summaries = []
        for cohort_name, pairwise_results in cohort_feature_results.items():
            if not pairwise_results:
                continue

            total_features = len(pairwise_results)
            total_drift_score = sum(result.overall_drift_score for result in pairwise_results)
            avg_drift_score = total_drift_score / total_features if total_features > 0 else 0

            critical_count = len([r for r in pairwise_results if r.overall_status == "critical"])
            high_count = len([r for r in pairwise_results if r.overall_status == "high"])
            medium_count = len([r for r in pairwise_results if r.overall_status == "medium"])
            low_count = len([r for r in pairwise_results if r.overall_status == "low"])

            drifted_count = critical_count + high_count + medium_count
            drift_percentage = (drifted_count / total_features) * 100 if total_features > 0 else 0

            # Determine cohort status
            if critical_count > total_features * 0.2 or avg_drift_score > 0.8:
                cohort_status = "critical"
            elif critical_count > total_features * 0.1 or high_count > total_features * 0.3 or avg_drift_score > 0.6:
                cohort_status = "high"
            elif high_count > total_features * 0.15 or drift_percentage > 40 or avg_drift_score > 0.4:
                cohort_status = "medium"
            else:
                cohort_status = "low"

            # Get top drifted features (from detailed results)
            detailed_features = cohort_feature_details.get(cohort_name, [])
            top_drifted = sorted(detailed_features, key=lambda x: x["drift_score"], reverse=True)[:5]

            cohort_summaries.append(
                {
                    "cohort_name": cohort_name,
                    "total_features": total_features,
                    "drifted_features": drifted_count,
                    "drift_percentage": round(drift_percentage, 1),
                    "average_drift_score": round(avg_drift_score, 4),
                    "status": cohort_status,
                    "severity_breakdown": {
                        "critical": critical_count,
                        "high": high_count,
                        "medium": medium_count,
                        "low": low_count,
                    },
                    "top_drifted_features": top_drifted,
                }
            )

        # Calculate overall metrics across all cohorts
        all_drift_scores = []
        all_severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for pairwise_results in cohort_feature_results.values():
            for result in pairwise_results:
                all_drift_scores.append(result.overall_drift_score)
                if result.overall_status in all_severity_counts:
                    all_severity_counts[result.overall_status] += 1

        overall_drift_score = sum(all_drift_scores) / len(all_drift_scores) if all_drift_scores else 0
        total_analyses = len(all_drift_scores)

        # Determine overall status
        critical_pct = (all_severity_counts["critical"] / total_analyses) * 100 if total_analyses > 0 else 0
        high_pct = (all_severity_counts["high"] / total_analyses) * 100 if total_analyses > 0 else 0
        drifted_pct = (
            (
                (all_severity_counts["critical"] + all_severity_counts["high"] + all_severity_counts["medium"])
                / total_analyses
            )
            * 100
            if total_analyses > 0
            else 0
        )

        if critical_pct > 20 or overall_drift_score > 0.8:
            overall_status = "critical"
        elif critical_pct > 10 or high_pct > 30 or overall_drift_score > 0.6:
            overall_status = "high"
        elif high_pct > 15 or drifted_pct > 40 or overall_drift_score > 0.4:
            overall_status = "medium"
        else:
            overall_status = "low"

        # Analysis metadata
        analysis_metadata = {
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline_cohort": baseline_name,
            "baseline_dataset_size": int(len(baseline_cohort)),
            "num_current_cohorts": len(current_cohorts),
            "current_cohorts": list(current_cohorts.keys()),
            "total_features_analyzed": len(common_features),
            "total_feature_cohort_pairs": total_analyses,
            "cohort_sizes": {name: int(len(df)) for name, df in current_cohorts.items()},
        }

        # Construct final result
        result = {
            "status": "success",
            "analysis_type": "multi_cohort",
            "data": {
                # Executive Summary
                "executive_summary": {
                    "baseline_cohort": baseline_name,
                    "num_cohorts_analyzed": len(current_cohorts),
                    "total_features_analyzed": len(common_features),
                    "total_drift_analyses": total_analyses,
                    "overall_drift_score": round(overall_drift_score, 4),
                    "overall_status": overall_status,
                    "drift_percentage_across_all_pairs": round(drifted_pct, 1),
                    "most_drifted_cohort": (
                        max(cohort_summaries, key=lambda x: x["average_drift_score"])["cohort_name"]
                        if cohort_summaries
                        else None
                    ),
                    "least_drifted_cohort": (
                        min(cohort_summaries, key=lambda x: x["average_drift_score"])["cohort_name"]
                        if cohort_summaries
                        else None
                    ),
                },
                # Cohort-level summaries
                "cohort_summaries": sorted(cohort_summaries, key=lambda x: x["average_drift_score"], reverse=True),
                # Per-cohort detailed feature analysis
                "cohort_feature_analysis": {
                    cohort_name: sorted(features, key=lambda x: x["drift_score"], reverse=True)
                    for cohort_name, features in cohort_feature_details.items()
                },
                # Aggregated cross-cohort insights
                "aggregated_insights": aggregated_results,
                # Analysis metadata
                "analysis_metadata": analysis_metadata,
            },
        }

        # Generate AI explanation for multi-cohort analysis
        try:
            ai_summary_payload = {
                "baseline_cohort": baseline_name,
                "num_cohorts": len(current_cohorts),
                "cohort_names": list(current_cohorts.keys()),
                "total_features": len(common_features),
                "overall_drift_score": round(overall_drift_score, 2),
                "overall_status": overall_status,
                "drift_percentage": round(drifted_pct, 1),
                "cohort_summaries": cohort_summaries,
                "top_consistently_drifted_features": (
                    aggregated_results.get("consistently_drifted_features", [])[:5] if aggregated_results else []
                ),
            }

            ai_explanation = ai_explanation_service.generate_explanation(
                analysis_data=ai_summary_payload, analysis_type="multi_cohort_feature_analysis"
            )
            result["llm_response"] = ai_explanation
            logger.info("Multi-cohort AI explanation generated successfully")
        except Exception as e:
            logger.warning(f"Multi-cohort AI explanation failed: {e}")
            result["llm_response"] = {
                "summary": f"Multi-cohort feature analysis completed for {len(current_cohorts)} cohorts against baseline '{baseline_name}'.",
                "detailed_explanation": f"Overall drift score: {overall_drift_score:.3f}. Status: {overall_status}",
                "key_takeaways": [
                    f"Analyzed {len(common_features)} features across {len(current_cohorts)} cohorts",
                    (
                        f"Most drifted cohort: {max(cohort_summaries, key=lambda x: x['average_drift_score'])['cohort_name']}"
                        if cohort_summaries
                        else "N/A"
                    ),
                    f"Overall drift percentage: {drifted_pct:.1f}%",
                    "Detailed cohort-level analysis available",
                ],
                "recommendations": [
                    "Review cohort-specific drift patterns",
                    "Investigate consistently drifted features across cohorts",
                    "Consider cohort-specific model retraining if needed",
                ],
            }

        # Convert numpy types to native Python types for JSON serialization
        result = convert_numpy_types(result)

        logger.info(f"Multi-cohort feature analysis completed successfully")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-cohort feature analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multi-cohort feature analysis failed: {str(e)}")


@router.post("/feature-analysis")
async def get_feature_analysis(
    request: Union[AnalysisRequest, MultiCohortAnalysisRequest], user: dict = Depends(get_current_user)
):
    """
    Get comprehensive feature drift analysis for datasets loaded from S3

    Supports two formats:

    **1. Legacy Format (AnalysisRequest):** Single reference vs single current comparison
    ```json
    {
        "reference_url": "https://s3.amazonaws.com/...",
        "current_url": "https://s3.amazonaws.com/...",
        "target_column": "optional",
        "config": {}
    }
    ```

    **2. Multi-Cohort Format (MultiCohortAnalysisRequest):** 1 baseline vs up to 4 cohorts
    ```json
    {
        "files": [
            {"file_name": "ref_cohort_1.csv", "folder": "files", "url": "https://..."},
            {"file_name": "cur_baseline.csv", "folder": "files", "url": "https://..."},
            {"file_name": "cohort_2.csv", "folder": "files", "url": "https://..."}
        ],
        "target_column": "optional",
        "config": {},
        "selected_features": ["feature1", "feature2"],  // optional
        "selected_tests": ["KS", "PSI", "KL"]  // optional
    }
    ```

    **File Naming Convention:**
    - `ref_*` or `ref_*.csv` → Baseline/Reference cohort (exactly 1 required)
    - `cur_*` or `cohort_*` → Current cohorts (1-4 allowed)

    Args:
        request: Either AnalysisRequest (legacy) or MultiCohortAnalysisRequest (new)
        user: Authenticated user

    Returns:
        Feature analysis results with multiple drift detection methods, insights and recommendations
        Format depends on request type (single-pair or multi-cohort)
    """
    try:
        # Detect request format and route to appropriate handler
        if hasattr(request, "files"):
            # New multi-cohort format
            logger.info(f"Feature analysis: Multi-cohort format detected with {len(request.files)} files")
            return await _handle_multi_cohort_feature_analysis(request, user)
        else:
            # Legacy single-pair format
            logger.info("Feature analysis: Legacy format detected (single reference vs current)")
            return await _handle_legacy_feature_analysis(request, user)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Feature analysis failed: {str(e)}")
