"""
Multi-Cohort Drift Analysis Service
Core engine for computing drift across multiple cohorts using 5 statistical tests
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

logger = logging.getLogger(__name__)


# Drift thresholds (industry standard)
class DriftThresholds:
    """Standard thresholds for drift detection"""

    # P-value thresholds
    HIGH_DRIFT_P_VALUE = 0.01
    MEDIUM_DRIFT_P_VALUE = 0.05

    # PSI thresholds
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

    # Jensen-Shannon divergence thresholds
    JS_LOW = 0.1
    JS_MEDIUM = 0.3
    JS_HIGH = 0.5


@dataclass
class TestResult:
    """Result of a single statistical test"""

    test_name: str
    statistic: float
    p_value: Optional[float]
    threshold: float
    status: str  # "stable", "moderate", "severe"
    confidence: str  # "high", "medium", "low"
    plot_data: Optional[Dict[str, Any]] = None  # Data for frontend visualization


@dataclass
class PairwiseDriftResult:
    """Result of drift analysis between two cohorts for a single feature"""

    feature_name: str
    feature_type: str  # "numerical" or "categorical"
    baseline_cohort: str
    comparison_cohort: str
    tests: Dict[str, TestResult]
    overall_drift_score: float
    overall_status: str
    descriptive_stats: Dict[str, Any]
    plot_data: Optional[Dict[str, Any]] = None  # Aggregated plot data for all tests


# ===== Plot Data Helper Functions =====


def compute_cdf_data(baseline_series: pd.Series, cohort_series: pd.Series, n_points: int = 100) -> Dict[str, List]:
    """
    Compute CDF data for KS and Wasserstein plots.

    Returns:
        Dictionary with x_values, baseline_cdf, current_cdf
    """
    # Combine and sort all unique values
    all_values = np.concatenate([baseline_series.values, cohort_series.values])
    x_values = np.linspace(all_values.min(), all_values.max(), n_points)

    # Compute CDF for both distributions
    baseline_cdf = np.array([np.mean(baseline_series <= x) for x in x_values])
    current_cdf = np.array([np.mean(cohort_series <= x) for x in x_values])

    return {"x_values": x_values.tolist(), "baseline_cdf": baseline_cdf.tolist(), "current_cdf": current_cdf.tolist()}


def compute_psi_bucket_data(baseline_series: pd.Series, cohort_series: pd.Series, n_bins: int = 10) -> Dict[str, List]:
    """
    Compute PSI bucket data for bar chart visualization.

    Returns:
        Dictionary with bucket_labels, baseline_proportions, current_proportions, psi_per_bucket
    """
    # Create bins based on baseline quantiles
    try:
        bin_edges = np.percentile(baseline_series, np.linspace(0, 100, n_bins + 1))
        # Ensure unique bin edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            # Fallback if data has no variance
            bin_edges = np.array([baseline_series.min() - 1, baseline_series.max() + 1])
    except:
        bin_edges = np.linspace(baseline_series.min(), baseline_series.max(), n_bins + 1)

    # Compute histograms
    baseline_hist, _ = np.histogram(baseline_series, bins=bin_edges)
    current_hist, _ = np.histogram(cohort_series, bins=bin_edges)

    # Convert to proportions
    baseline_prop = baseline_hist / max(baseline_hist.sum(), 1)
    current_prop = current_hist / max(current_hist.sum(), 1)

    # Avoid log(0) by adding small epsilon
    baseline_prop = np.where(baseline_prop == 0, 0.0001, baseline_prop)
    current_prop = np.where(current_prop == 0, 0.0001, current_prop)

    # Compute PSI per bucket
    psi_per_bucket = (baseline_prop - current_prop) * np.log(baseline_prop / current_prop)

    # Create bucket labels
    bucket_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges) - 1)]

    return {
        "bucket_labels": bucket_labels,
        "bucket_edges": bin_edges.tolist(),
        "baseline_proportions": baseline_prop.tolist(),
        "current_proportions": current_prop.tolist(),
        "psi_per_bucket": psi_per_bucket.tolist(),
    }


def compute_density_data(baseline_series: pd.Series, cohort_series: pd.Series, n_bins: int = 30) -> Dict[str, List]:
    """
    Compute probability density data for KL and JS divergence plots.

    Returns:
        Dictionary with bin_centers, bin_edges, baseline_density, current_density
    """
    # Determine common bin range
    min_val = min(baseline_series.min(), cohort_series.min())
    max_val = max(baseline_series.max(), cohort_series.max())
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)

    # Compute histograms
    baseline_hist, _ = np.histogram(baseline_series, bins=bin_edges, density=True)
    current_hist, _ = np.histogram(cohort_series, bins=bin_edges, density=True)

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {
        "bin_centers": bin_centers.tolist(),
        "bin_edges": bin_edges.tolist(),
        "baseline_density": baseline_hist.tolist(),
        "current_density": current_hist.tolist(),
    }


def compute_categorical_frequency_data(baseline_series: pd.Series, cohort_series: pd.Series) -> Dict[str, List]:
    """
    Compute categorical frequency data for Chi-square visualization.

    Returns:
        Dictionary with categories, baseline_frequencies, current_frequencies, baseline_proportions, current_proportions
    """
    # Get all unique categories from both series
    baseline_counts = baseline_series.value_counts()
    current_counts = cohort_series.value_counts()

    # Align categories
    all_categories = baseline_counts.index.union(current_counts.index).tolist()
    baseline_aligned = baseline_counts.reindex(all_categories, fill_value=0)
    current_aligned = current_counts.reindex(all_categories, fill_value=0)

    # Compute proportions
    baseline_prop = (baseline_aligned / len(baseline_series)).tolist()
    current_prop = (current_aligned / len(cohort_series)).tolist()

    return {
        "categories": [str(cat) for cat in all_categories],
        "baseline_frequencies": baseline_aligned.tolist(),
        "current_frequencies": current_aligned.tolist(),
        "baseline_proportions": baseline_prop,
        "current_proportions": current_prop,
    }


class MultiCohortDriftAnalyzer:
    """Core engine for multi-cohort drift analysis"""

    def __init__(self, thresholds: DriftThresholds = None):
        self.thresholds = thresholds or DriftThresholds()

    def compute_pairwise_drift(
        self,
        baseline_df: pd.DataFrame,
        cohort_df: pd.DataFrame,
        feature_name: str,
        baseline_name: str = "baseline",
        cohort_name: str = "cohort",
    ) -> PairwiseDriftResult:
        """
        Compute drift between two cohorts for a single feature.

        Args:
            baseline_df: Reference/baseline DataFrame
            cohort_df: Comparison cohort DataFrame
            feature_name: Name of the feature to analyze
            baseline_name: Name of baseline cohort
            cohort_name: Name of comparison cohort

        Returns:
            PairwiseDriftResult with all 5 test results
        """
        baseline_series = baseline_df[feature_name].dropna()
        cohort_series = cohort_df[feature_name].dropna()

        # Determine feature type
        is_numerical = pd.api.types.is_numeric_dtype(baseline_series)
        feature_type = "numerical" if is_numerical else "categorical"

        logger.info(f"Computing drift for feature '{feature_name}' ({feature_type}): {baseline_name} vs {cohort_name}")

        # Compute appropriate tests based on feature type
        if is_numerical:
            test_results = self._compute_numerical_tests(baseline_series, cohort_series)
            descriptive_stats = self._compute_numerical_descriptive_stats(
                baseline_series, cohort_series, baseline_name, cohort_name
            )
        else:
            test_results = self._compute_categorical_tests(baseline_series, cohort_series)
            descriptive_stats = self._compute_categorical_descriptive_stats(
                baseline_series, cohort_series, baseline_name, cohort_name
            )

        # Determine overall drift score and status
        overall_score, overall_status = self._determine_overall_drift(test_results)

        return PairwiseDriftResult(
            feature_name=feature_name,
            feature_type=feature_type,
            baseline_cohort=baseline_name,
            comparison_cohort=cohort_name,
            tests=test_results,
            overall_drift_score=overall_score,
            overall_status=overall_status,
            descriptive_stats=descriptive_stats,
        )

    def _compute_numerical_tests(self, baseline: pd.Series, current: pd.Series) -> Dict[str, TestResult]:
        """Compute all 5 statistical tests for numerical features with plot data"""
        tests = {}

        # Pre-compute plot data for efficiency
        try:
            cdf_data = compute_cdf_data(baseline, current)
        except Exception as e:
            logger.warning(f"Failed to compute CDF data: {e}")
            cdf_data = None

        try:
            psi_bucket_data = compute_psi_bucket_data(baseline, current)
        except Exception as e:
            logger.warning(f"Failed to compute PSI bucket data: {e}")
            psi_bucket_data = None

        try:
            density_data = compute_density_data(baseline, current)
        except Exception as e:
            logger.warning(f"Failed to compute density data: {e}")
            density_data = None

        # 1. Kolmogorov-Smirnov Test
        try:
            ks_stat, ks_pval = ks_2samp(baseline, current)
            ks_status = self._classify_status_by_pvalue(ks_pval)
            ks_confidence = "high" if ks_pval < 0.001 else "medium" if ks_pval < 0.05 else "low"

            tests["KS"] = TestResult(
                test_name="Kolmogorov-Smirnov",
                statistic=float(ks_stat),
                p_value=float(ks_pval),
                threshold=self.thresholds.MEDIUM_DRIFT_P_VALUE,
                status=ks_status,
                confidence=ks_confidence,
                plot_data=cdf_data,  # CDF curves for KS test
            )
        except Exception as e:
            logger.warning(f"KS test failed: {e}")
            tests["KS"] = self._create_failed_test("KS")

        # 2. Population Stability Index (PSI)
        try:
            psi_value = self._calculate_psi(baseline, current)
            psi_status = self._classify_status_by_threshold(
                psi_value, self.thresholds.PSI_LOW, self.thresholds.PSI_MEDIUM
            )

            tests["PSI"] = TestResult(
                test_name="Population Stability Index",
                statistic=float(psi_value),
                p_value=None,
                threshold=self.thresholds.PSI_MEDIUM,
                status=psi_status,
                confidence="high" if psi_value > self.thresholds.PSI_HIGH else "medium",
                plot_data=psi_bucket_data,  # Bucket-wise data for PSI
            )
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            tests["PSI"] = self._create_failed_test("PSI")

        # 3. Kullback-Leibler Divergence
        try:
            kl_value = self._calculate_kl_divergence(baseline, current)
            kl_status = self._classify_status_by_threshold(kl_value, self.thresholds.KL_LOW, self.thresholds.KL_MEDIUM)

            tests["KL"] = TestResult(
                test_name="Kullback-Leibler Divergence",
                statistic=float(kl_value),
                p_value=None,
                threshold=self.thresholds.KL_MEDIUM,
                status=kl_status,
                confidence="high" if kl_value > self.thresholds.KL_HIGH else "medium",
                plot_data=density_data,  # PDF/density curves for KL
            )
        except Exception as e:
            logger.warning(f"KL divergence calculation failed: {e}")
            tests["KL"] = self._create_failed_test("KL")

        # 4. Jensen-Shannon Divergence
        try:
            js_value = self._calculate_js_divergence(baseline, current)
            js_status = self._classify_status_by_threshold(js_value, self.thresholds.JS_LOW, self.thresholds.JS_MEDIUM)

            tests["JS"] = TestResult(
                test_name="Jensen-Shannon Divergence",
                statistic=float(js_value),
                p_value=None,
                threshold=self.thresholds.JS_MEDIUM,
                status=js_status,
                confidence="high" if js_value > self.thresholds.JS_HIGH else "medium",
                plot_data=density_data,  # PDF/density curves for JS (same as KL)
            )
        except Exception as e:
            logger.warning(f"JS divergence calculation failed: {e}")
            tests["JS"] = self._create_failed_test("JS")

        # 5. Wasserstein Distance
        try:
            wass_dist = wasserstein_distance(baseline, current)
            # Normalize by data range
            data_range = max(baseline.max(), current.max()) - min(baseline.min(), current.min())
            if data_range > 0:
                wass_normalized = wass_dist / data_range
            else:
                wass_normalized = 0.0

            wass_status = self._classify_status_by_threshold(
                wass_normalized, self.thresholds.WASSERSTEIN_LOW, self.thresholds.WASSERSTEIN_MEDIUM
            )

            tests["Wasserstein"] = TestResult(
                test_name="Wasserstein Distance",
                statistic=float(wass_normalized),
                p_value=None,
                threshold=self.thresholds.WASSERSTEIN_MEDIUM,
                status=wass_status,
                confidence="high" if wass_normalized > self.thresholds.WASSERSTEIN_HIGH else "medium",
                plot_data=cdf_data,  # CDF curves for Wasserstein (same as KS)
            )
        except Exception as e:
            logger.warning(f"Wasserstein distance calculation failed: {e}")
            tests["Wasserstein"] = self._create_failed_test("Wasserstein")

        return tests

    def _compute_categorical_tests(self, baseline: pd.Series, current: pd.Series) -> Dict[str, TestResult]:
        """Compute statistical tests for categorical features with plot data"""
        tests = {}

        # Pre-compute plot data for categorical features
        try:
            categorical_freq_data = compute_categorical_frequency_data(baseline, current)
        except Exception as e:
            logger.warning(f"Failed to compute categorical frequency data: {e}")
            categorical_freq_data = None

        # 1. Chi-Square Test
        try:
            # Create contingency table
            baseline_counts = baseline.value_counts()
            current_counts = current.value_counts()

            # Get all unique categories
            all_categories = sorted(set(baseline_counts.index) | set(current_counts.index))

            # Build contingency table
            contingency = []
            for cat in all_categories:
                contingency.append([baseline_counts.get(cat, 0), current_counts.get(cat, 0)])

            contingency = np.array(contingency)

            if contingency.sum() > 0 and len(all_categories) > 1:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                chi2_status = self._classify_status_by_pvalue(p_value)
                chi2_confidence = "high" if p_value < 0.001 else "medium" if p_value < 0.05 else "low"

                tests["Chi-square"] = TestResult(
                    test_name="Chi-Square Test",
                    statistic=float(chi2),
                    p_value=float(p_value),
                    threshold=self.thresholds.MEDIUM_DRIFT_P_VALUE,
                    status=chi2_status,
                    confidence=chi2_confidence,
                    plot_data=categorical_freq_data,  # Categorical frequency data
                )
            else:
                tests["Chi-square"] = self._create_failed_test("Chi-square")
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
            tests["Chi-square"] = self._create_failed_test("Chi-square")

        # 2. Population Stability Index (PSI) for categorical
        try:
            psi_value = self._calculate_psi_categorical(baseline, current)
            psi_status = self._classify_status_by_threshold(
                psi_value, self.thresholds.PSI_LOW, self.thresholds.PSI_MEDIUM
            )

            tests["PSI"] = TestResult(
                test_name="Population Stability Index",
                statistic=float(psi_value),
                p_value=None,
                threshold=self.thresholds.PSI_MEDIUM,
                status=psi_status,
                confidence="high" if psi_value > self.thresholds.PSI_HIGH else "medium",
                plot_data=categorical_freq_data,  # Same categorical data for PSI
            )
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            tests["PSI"] = self._create_failed_test("PSI")

        # 3. KL Divergence for categorical
        try:
            kl_value = self._calculate_kl_divergence_categorical(baseline, current)
            kl_status = self._classify_status_by_threshold(kl_value, self.thresholds.KL_LOW, self.thresholds.KL_MEDIUM)

            tests["KL"] = TestResult(
                test_name="Kullback-Leibler Divergence",
                statistic=float(kl_value),
                p_value=None,
                threshold=self.thresholds.KL_MEDIUM,
                status=kl_status,
                confidence="high" if kl_value > self.thresholds.KL_HIGH else "medium",
                plot_data=categorical_freq_data,  # Same categorical data for KL
            )
        except Exception as e:
            logger.warning(f"KL divergence calculation failed: {e}")
            tests["KL"] = self._create_failed_test("KL")

        # 4. Jensen-Shannon Divergence for categorical
        try:
            js_value = self._calculate_js_divergence_categorical(baseline, current)
            js_status = self._classify_status_by_threshold(js_value, self.thresholds.JS_LOW, self.thresholds.JS_MEDIUM)

            tests["JS"] = TestResult(
                test_name="Jensen-Shannon Divergence",
                statistic=float(js_value),
                p_value=None,
                threshold=self.thresholds.JS_MEDIUM,
                status=js_status,
                confidence="high" if js_value > self.thresholds.JS_HIGH else "medium",
                plot_data=categorical_freq_data,  # Same categorical data for JS
            )
        except Exception as e:
            logger.warning(f"JS divergence calculation failed: {e}")
            tests["JS"] = self._create_failed_test("JS")

        # 5. Wasserstein - Not applicable for categorical, mark as N/A
        tests["Wasserstein"] = TestResult(
            test_name="Wasserstein Distance",
            statistic=0.0,
            p_value=None,
            threshold=0.0,
            status="N/A",
            confidence="N/A",
            plot_data=None,  # N/A for categorical features
        )

        return tests

    def _calculate_psi(self, baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index for numerical data"""
        try:
            # Create bins based on baseline distribution
            _, bin_edges = np.histogram(baseline, bins=bins)

            # Get distributions
            baseline_dist, _ = np.histogram(baseline, bins=bin_edges)
            current_dist, _ = np.histogram(current, bins=bin_edges)

            # Convert to percentages
            baseline_pct = baseline_dist / len(baseline) + 1e-10  # Add small epsilon
            current_pct = current_dist / len(current) + 1e-10

            # Calculate PSI
            psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

            return float(psi)
        except Exception as e:
            logger.warning(f"PSI calculation error: {e}")
            return 0.0

    def _calculate_psi_categorical(self, baseline: pd.Series, current: pd.Series) -> float:
        """Calculate PSI for categorical data"""
        try:
            baseline_counts = baseline.value_counts(normalize=True)
            current_counts = current.value_counts(normalize=True)

            all_categories = set(baseline_counts.index) | set(current_counts.index)

            psi = 0.0
            for cat in all_categories:
                baseline_pct = baseline_counts.get(cat, 1e-10)
                current_pct = current_counts.get(cat, 1e-10)
                psi += (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)

            return float(psi)
        except Exception as e:
            logger.warning(f"PSI categorical calculation error: {e}")
            return 0.0

    def _calculate_kl_divergence(self, baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate KL divergence for numerical data"""
        try:
            _, bin_edges = np.histogram(baseline, bins=bins)

            baseline_dist, _ = np.histogram(baseline, bins=bin_edges)
            current_dist, _ = np.histogram(current, bins=bin_edges)

            baseline_prob = (baseline_dist + 1e-10) / (baseline_dist.sum() + bins * 1e-10)
            current_prob = (current_dist + 1e-10) / (current_dist.sum() + bins * 1e-10)

            kl_div = np.sum(current_prob * np.log(current_prob / baseline_prob))

            return float(kl_div)
        except Exception as e:
            logger.warning(f"KL divergence calculation error: {e}")
            return 0.0

    def _calculate_kl_divergence_categorical(self, baseline: pd.Series, current: pd.Series) -> float:
        """Calculate KL divergence for categorical data"""
        try:
            baseline_counts = baseline.value_counts()
            current_counts = current.value_counts()

            all_categories = set(baseline_counts.index) | set(current_counts.index)

            baseline_prob = np.array([baseline_counts.get(cat, 1e-10) for cat in all_categories])
            current_prob = np.array([current_counts.get(cat, 1e-10) for cat in all_categories])

            baseline_prob = baseline_prob / baseline_prob.sum()
            current_prob = current_prob / current_prob.sum()

            kl_div = np.sum(current_prob * np.log((current_prob + 1e-10) / (baseline_prob + 1e-10)))

            return float(kl_div)
        except Exception as e:
            logger.warning(f"KL divergence categorical calculation error: {e}")
            return 0.0

    def _calculate_js_divergence(self, baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Jensen-Shannon divergence for numerical data"""
        try:
            _, bin_edges = np.histogram(baseline, bins=bins)

            baseline_dist, _ = np.histogram(baseline, bins=bin_edges)
            current_dist, _ = np.histogram(current, bins=bin_edges)

            baseline_prob = (baseline_dist + 1e-10) / (baseline_dist.sum() + bins * 1e-10)
            current_prob = (current_dist + 1e-10) / (current_dist.sum() + bins * 1e-10)

            js_div = jensenshannon(baseline_prob, current_prob)

            return float(js_div)
        except Exception as e:
            logger.warning(f"JS divergence calculation error: {e}")
            return 0.0

    def _calculate_js_divergence_categorical(self, baseline: pd.Series, current: pd.Series) -> float:
        """Calculate Jensen-Shannon divergence for categorical data"""
        try:
            baseline_counts = baseline.value_counts()
            current_counts = current.value_counts()

            all_categories = set(baseline_counts.index) | set(current_counts.index)

            baseline_prob = np.array([baseline_counts.get(cat, 1e-10) for cat in all_categories])
            current_prob = np.array([current_counts.get(cat, 1e-10) for cat in all_categories])

            baseline_prob = baseline_prob / baseline_prob.sum()
            current_prob = current_prob / current_prob.sum()

            js_div = jensenshannon(baseline_prob, current_prob)

            return float(js_div)
        except Exception as e:
            logger.warning(f"JS divergence categorical calculation error: {e}")
            return 0.0

    def _compute_numerical_descriptive_stats(
        self, baseline: pd.Series, current: pd.Series, baseline_name: str, current_name: str
    ) -> Dict[str, Any]:
        """Compute descriptive statistics for numerical features"""
        stats = {
            f"{baseline_name}_mean": float(baseline.mean()),
            f"{baseline_name}_median": float(baseline.median()),
            f"{baseline_name}_std": float(baseline.std()),
            f"{baseline_name}_min": float(baseline.min()),
            f"{baseline_name}_max": float(baseline.max()),
            f"{current_name}_mean": float(current.mean()),
            f"{current_name}_median": float(current.median()),
            f"{current_name}_std": float(current.std()),
            f"{current_name}_min": float(current.min()),
            f"{current_name}_max": float(current.max()),
            "mean_change_percent": (
                float(((current.mean() - baseline.mean()) / baseline.mean()) * 100) if baseline.mean() != 0 else 0.0
            ),
            "std_change_percent": (
                float(((current.std() - baseline.std()) / baseline.std()) * 100) if baseline.std() != 0 else 0.0
            ),
        }
        return stats

    def _compute_categorical_descriptive_stats(
        self, baseline: pd.Series, current: pd.Series, baseline_name: str, current_name: str
    ) -> Dict[str, Any]:
        """Compute descriptive statistics for categorical features"""
        baseline_counts = baseline.value_counts()
        current_counts = current.value_counts()

        all_categories = set(baseline_counts.index) | set(current_counts.index)
        new_categories = list(set(current_counts.index) - set(baseline_counts.index))
        disappeared_categories = list(set(baseline_counts.index) - set(current_counts.index))

        stats = {
            f"{baseline_name}_unique_count": int(baseline.nunique()),
            f"{baseline_name}_mode": str(baseline.mode()[0]) if len(baseline.mode()) > 0 else None,
            f"{current_name}_unique_count": int(current.nunique()),
            f"{current_name}_mode": str(current.mode()[0]) if len(current.mode()) > 0 else None,
            "total_categories": len(all_categories),
            "new_categories": new_categories,
            "disappeared_categories": disappeared_categories,
            "category_change_count": len(new_categories) + len(disappeared_categories),
        }
        return stats

    def _classify_status_by_pvalue(self, p_value: float) -> str:
        """Classify drift status based on p-value"""
        if p_value < self.thresholds.HIGH_DRIFT_P_VALUE:
            return "severe"
        elif p_value < self.thresholds.MEDIUM_DRIFT_P_VALUE:
            return "moderate"
        else:
            return "stable"

    def _classify_status_by_threshold(self, value: float, low_threshold: float, medium_threshold: float) -> str:
        """Classify drift status based on threshold values"""
        if value >= medium_threshold * 2:  # Severe is 2x medium threshold
            return "severe"
        elif value >= medium_threshold:
            return "moderate"
        elif value >= low_threshold:
            return "moderate"
        else:
            return "stable"

    def _determine_overall_drift(self, tests: Dict[str, TestResult]) -> Tuple[float, str]:
        """Determine overall drift score and status from all tests"""
        # Count severe, moderate, stable tests
        severe_count = sum(1 for test in tests.values() if test.status == "severe")
        moderate_count = sum(1 for test in tests.values() if test.status == "moderate")
        stable_count = sum(1 for test in tests.values() if test.status == "stable")

        total_valid_tests = severe_count + moderate_count + stable_count

        if total_valid_tests == 0:
            return 0.0, "unknown"

        # Weighted scoring: severe=1.0, moderate=0.5, stable=0.0
        overall_score = (severe_count * 1.0 + moderate_count * 0.5) / total_valid_tests

        # Determine overall status
        if severe_count >= 2 or (severe_count >= 1 and moderate_count >= 2):
            overall_status = "severe"
        elif moderate_count >= 2 or severe_count >= 1:
            overall_status = "moderate"
        else:
            overall_status = "stable"

        return overall_score, overall_status

    def _create_failed_test(self, test_name: str) -> TestResult:
        """Create a failed test result placeholder"""
        return TestResult(
            test_name=test_name, statistic=0.0, p_value=None, threshold=0.0, status="error", confidence="N/A"
        )

    def aggregate_drift_across_features(self, pairwise_results: List[PairwiseDriftResult]) -> Dict[str, Any]:
        """
        Aggregate drift results across all features for a cohort pair.

        Args:
            pairwise_results: List of drift results for all features

        Returns:
            Dictionary with aggregated metrics
        """
        if not pairwise_results:
            return {
                "total_features": 0,
                "drifted_features": 0,
                "drift_percentage": 0.0,
                "overall_drift_score": 0.0,
                "overall_status": "unknown",
            }

        total_features = len(pairwise_results)
        drifted_features = sum(1 for r in pairwise_results if r.overall_status in ["moderate", "severe"])
        drift_percentage = (drifted_features / total_features) * 100

        # Calculate average drift score
        avg_drift_score = np.mean([r.overall_drift_score for r in pairwise_results])

        # Aggregate test scores
        test_names = ["KS", "PSI", "KL", "JS", "Wasserstein", "Chi-square"]
        tests_summary = {}

        for test_name in test_names:
            test_scores = []
            test_statuses = []

            for result in pairwise_results:
                if test_name in result.tests and result.tests[test_name].status != "N/A":
                    test_scores.append(result.tests[test_name].statistic)
                    test_statuses.append(result.tests[test_name].status)

            if test_scores:
                avg_score = float(np.mean(test_scores))
                drifted_count = sum(1 for s in test_statuses if s in ["moderate", "severe"])

                # Determine test status
                if drifted_count / len(test_statuses) > 0.5:
                    test_status = "moderate"
                elif drifted_count / len(test_statuses) > 0.3:
                    test_status = "moderate"
                else:
                    test_status = "stable"

                tests_summary[test_name] = {
                    "avg_score": avg_score,
                    "status": test_status,
                    "drifted_count": drifted_count,
                    "total_tested": len(test_scores),
                }

        # Severity breakdown
        severity_counts = {"stable": 0, "moderate": 0, "severe": 0}
        for result in pairwise_results:
            severity_counts[result.overall_status] = severity_counts.get(result.overall_status, 0) + 1

        # Determine overall status
        severe_pct = severity_counts["severe"] / total_features
        moderate_pct = severity_counts["moderate"] / total_features

        if severe_pct > 0.3 or (severe_pct > 0.1 and moderate_pct > 0.4):
            overall_status = "severe"
        elif moderate_pct > 0.3 or severe_pct > 0.1:
            overall_status = "moderate"
        else:
            overall_status = "stable"

        return {
            "total_features": total_features,
            "drifted_features": drifted_features,
            "drift_percentage": float(drift_percentage),
            "overall_drift_score": float(avg_drift_score),
            "overall_status": overall_status,
            "tests_summary": tests_summary,
            "severity_breakdown": severity_counts,
        }

    def create_drift_matrix(
        self, cohorts: Dict[str, pd.DataFrame], pairs: List[Tuple[str, str]], common_features: List[str]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Create drift matrix for multiple cohort pairs.

        Args:
            cohorts: Dictionary of cohort_name -> DataFrame
            pairs: List of (baseline_name, cohort_name) tuples
            common_features: List of features to analyze

        Returns:
            Dictionary mapping (baseline, cohort) -> aggregated drift metrics
        """
        drift_matrix = {}

        for baseline_name, cohort_name in pairs:
            logger.info(f"Computing drift matrix for pair: {baseline_name} vs {cohort_name}")

            baseline_df = cohorts[baseline_name]
            cohort_df = cohorts[cohort_name]

            # Compute drift for all features
            pairwise_results = []
            for feature in common_features:
                try:
                    result = self.compute_pairwise_drift(baseline_df, cohort_df, feature, baseline_name, cohort_name)
                    pairwise_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to compute drift for feature {feature}: {e}")

            # Aggregate results
            aggregated = self.aggregate_drift_across_features(pairwise_results)
            aggregated["feature_results"] = pairwise_results

            drift_matrix[(baseline_name, cohort_name)] = aggregated

        return drift_matrix


# Convenience instance
multi_cohort_analyzer = MultiCohortDriftAnalyzer()
