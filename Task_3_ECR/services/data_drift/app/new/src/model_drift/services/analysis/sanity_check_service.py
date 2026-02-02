"""
Sanity Check Service - Pre-Flight Diagnostic Analysis
Estimates expected model performance on new data by analyzing distributional similarity
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

warnings.filterwarnings("ignore")


class SanityCheckService:
    """Service for sanity check and short-term performance estimation"""

    def __init__(self):
        # Thresholds for similarity classification
        self.similarity_thresholds = {
            "stable": 0.90,  # >0.90 = stable
            "low": 0.70,  # 0.70-0.90 = low drift
            "moderate": 0.50,  # 0.50-0.70 = moderate drift
            "high": 0.30,  # 0.30-0.50 = high drift
            # <0.30 = critical drift
        }

        # Performance drop estimation ranges
        self.performance_ranges = {
            "stable": (0.00, 0.02),  # 0-2% drop
            "low": (0.02, 0.05),  # 2-5% drop
            "moderate": (0.05, 0.10),  # 5-10% drop
            "high": (0.10, 0.20),  # 10-20% drop
            "critical": (0.20, 0.40),  # 20-40% drop
        }

    def analyze_sanity_check(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        target_column: Optional[str] = None,
        model: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        model_type: str = "classification",
    ) -> Dict[str, Any]:
        """
        Comprehensive sanity check analysis

        Args:
            reference_df: Reference dataset
            current_df: Current dataset
            target_column: Target column name (optional)
            model: Trained model (optional, for feature importance)
            feature_names: List of feature names
            model_type: "classification" or "regression"

        Returns:
            Dictionary with sanity check results
        """
        try:
            # 1. Identify features (exclude target if provided)
            if feature_names is None:
                feature_names = list(reference_df.columns)
                if target_column and target_column in feature_names:
                    feature_names.remove(target_column)

            # 2. Get common features between datasets
            common_features = list(set(feature_names) & set(current_df.columns))

            if len(common_features) == 0:
                return {"error": "No common features found between reference and current datasets"}

            # 3. Extract feature importances if model available
            feature_importances = self._extract_feature_importances(model, common_features)

            # 4. Analyze each feature
            scatter_points = []
            distribution_comparisons = []

            for feature in common_features:
                feature_analysis = self._analyze_single_feature(
                    reference_df[feature],
                    current_df[feature],
                    feature,
                    feature_importances.get(feature) if feature_importances else None,
                )

                if feature_analysis:
                    scatter_points.append(feature_analysis["scatter_point"])
                    if feature_analysis.get("distribution_comparison"):
                        distribution_comparisons.append(feature_analysis["distribution_comparison"])

            # 5. Sort by expected drop (highest first)
            scatter_points.sort(key=lambda x: x["expected_drop"], reverse=True)

            # 6. Get top 3 drifted features for detailed comparison
            top_drifted = (
                distribution_comparisons[:3] if len(distribution_comparisons) >= 3 else distribution_comparisons
            )

            # 7. Calculate overall assessment
            overall_assessment = self._calculate_overall_assessment(scatter_points, model_type)

            # 8. Generate recommendations
            recommendations = self._generate_recommendations(scatter_points, overall_assessment)

            return {
                "scatter_points": scatter_points,
                "distribution_comparisons": top_drifted,
                "sanity_check_summary": overall_assessment,
                "recommendations": recommendations,
                "warnings": [],
            }

        except Exception as e:
            return {
                "error": f"Sanity check analysis failed: {str(e)}",
                "scatter_points": [],
                "distribution_comparisons": [],
                "recommendations": [],
            }

    def _analyze_single_feature(
        self, ref_values: pd.Series, curr_values: pd.Series, feature_name: str, importance: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single feature for drift"""
        try:
            # Determine feature type
            is_numerical = pd.api.types.is_numeric_dtype(ref_values)

            if is_numerical:
                return self._analyze_numerical_feature(ref_values, curr_values, feature_name, importance)
            else:
                return self._analyze_categorical_feature(ref_values, curr_values, feature_name, importance)

        except Exception as e:
            print(f"Error analyzing feature {feature_name}: {str(e)}")
            return None

    def _analyze_numerical_feature(
        self, ref_values: pd.Series, curr_values: pd.Series, feature_name: str, importance: Optional[float] = None
    ) -> Dict[str, Any]:
        """Analyze numerical feature using KS test"""
        # Remove NaN values
        ref_clean = ref_values.dropna()
        curr_clean = curr_values.dropna()

        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return None

        # KS test
        ks_stat, p_value = ks_2samp(ref_clean, curr_clean)

        # Convert KS statistic to similarity (0 = identical, 1 = completely different)
        # Similarity = 1 - ks_stat
        similarity = 1.0 - ks_stat
        similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

        # Classify severity
        severity = self._classify_severity(similarity)

        # Estimate performance drop
        expected_drop = self._estimate_performance_drop(similarity, importance)

        # Calculate statistics
        ref_stats = {
            "mean": float(ref_clean.mean()),
            "std": float(ref_clean.std()),
            "min": float(ref_clean.min()),
            "max": float(ref_clean.max()),
            "median": float(ref_clean.median()),
        }

        curr_stats = {
            "mean": float(curr_clean.mean()),
            "std": float(curr_clean.std()),
            "min": float(curr_clean.min()),
            "max": float(curr_clean.max()),
            "median": float(curr_clean.median()),
        }

        return {
            "scatter_point": {
                "feature": feature_name,
                "similarity": similarity,
                "expected_drop": expected_drop,
                "importance": importance,
                "severity": severity,
                "drift_type": "numerical",
                "test_statistic": float(ks_stat),
                "p_value": float(p_value),
            },
            "distribution_comparison": {
                "feature": feature_name,
                "reference_stats": ref_stats,
                "current_stats": curr_stats,
                "test_statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": p_value < 0.05,
            },
        }

    def _analyze_categorical_feature(
        self, ref_values: pd.Series, curr_values: pd.Series, feature_name: str, importance: Optional[float] = None
    ) -> Dict[str, Any]:
        """Analyze categorical feature using chi-square and distribution comparison"""
        # Remove NaN values
        ref_clean = ref_values.dropna()
        curr_clean = curr_values.dropna()

        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return None

        # Get value counts
        ref_dist = ref_clean.value_counts(normalize=True).sort_index()
        curr_dist = curr_clean.value_counts(normalize=True).sort_index()

        # Align distributions (add missing categories with 0 frequency)
        all_categories = set(ref_dist.index) | set(curr_dist.index)
        ref_aligned = pd.Series([ref_dist.get(cat, 0) for cat in all_categories], index=list(all_categories))
        curr_aligned = pd.Series([curr_dist.get(cat, 0) for cat in all_categories], index=list(all_categories))

        # Calculate Jensen-Shannon divergence (0 = identical, 1 = completely different)
        js_divergence = jensenshannon(ref_aligned, curr_aligned)
        if np.isnan(js_divergence):
            js_divergence = 0.0

        # Convert to similarity
        similarity = 1.0 - js_divergence
        similarity = max(0.0, min(1.0, similarity))

        # Chi-square test
        try:
            # Create contingency table
            ref_counts = ref_clean.value_counts()
            curr_counts = curr_clean.value_counts()

            # Align categories
            all_cats = sorted(set(ref_counts.index) | set(curr_counts.index))
            ref_freq = [ref_counts.get(cat, 0) for cat in all_cats]
            curr_freq = [curr_counts.get(cat, 0) for cat in all_cats]

            contingency = np.array([ref_freq, curr_freq])
            chi2, p_value, dof, expected = chi2_contingency(contingency)
        except Exception:
            chi2 = 0.0
            p_value = 1.0

        # Classify severity
        severity = self._classify_severity(similarity)

        # Estimate performance drop
        expected_drop = self._estimate_performance_drop(similarity, importance)

        # Calculate statistics
        ref_stats = {
            "unique_values": int(ref_clean.nunique()),
            "top_category": str(ref_clean.mode()[0]) if len(ref_clean.mode()) > 0 else "N/A",
            "top_frequency": float(ref_clean.value_counts(normalize=True).iloc[0]) if len(ref_clean) > 0 else 0.0,
            "distribution": {str(k): float(v) for k, v in ref_dist.head(5).items()},
        }

        curr_stats = {
            "unique_values": int(curr_clean.nunique()),
            "top_category": str(curr_clean.mode()[0]) if len(curr_clean.mode()) > 0 else "N/A",
            "top_frequency": float(curr_clean.value_counts(normalize=True).iloc[0]) if len(curr_clean) > 0 else 0.0,
            "distribution": {str(k): float(v) for k, v in curr_dist.head(5).items()},
        }

        return {
            "scatter_point": {
                "feature": feature_name,
                "similarity": similarity,
                "expected_drop": expected_drop,
                "importance": importance,
                "severity": severity,
                "drift_type": "categorical",
                "test_statistic": float(chi2) if not np.isnan(chi2) else 0.0,
                "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
            },
            "distribution_comparison": {
                "feature": feature_name,
                "reference_stats": ref_stats,
                "current_stats": curr_stats,
                "test_statistic": float(chi2) if not np.isnan(chi2) else 0.0,
                "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
                "drift_detected": p_value < 0.05,
            },
        }

    def _classify_severity(self, similarity: float) -> str:
        """Classify drift severity based on similarity score"""
        if similarity >= self.similarity_thresholds["stable"]:
            return "stable"
        elif similarity >= self.similarity_thresholds["low"]:
            return "low"
        elif similarity >= self.similarity_thresholds["moderate"]:
            return "moderate"
        elif similarity >= self.similarity_thresholds["high"]:
            return "high"
        else:
            return "critical"

    def _estimate_performance_drop(self, similarity: float, importance: Optional[float] = None) -> float:
        """
        Estimate expected performance drop based on similarity

        Heuristic approach:
        - Higher similarity = lower expected drop
        - Weight by feature importance if available
        """
        # Classify severity
        severity = self._classify_severity(similarity)

        # Get performance range for this severity
        min_drop, max_drop = self.performance_ranges[severity]

        # Calculate base drop (within range, inversely proportional to similarity)
        # Lower similarity within range = higher drop
        if severity == "stable":
            normalized = (self.similarity_thresholds["stable"] - similarity) / (
                1.0 - self.similarity_thresholds["stable"] + 1e-6
            )
        elif severity == "low":
            normalized = (self.similarity_thresholds["low"] - similarity) / (
                self.similarity_thresholds["stable"] - self.similarity_thresholds["low"] + 1e-6
            )
        elif severity == "moderate":
            normalized = (self.similarity_thresholds["moderate"] - similarity) / (
                self.similarity_thresholds["low"] - self.similarity_thresholds["moderate"] + 1e-6
            )
        elif severity == "high":
            normalized = (self.similarity_thresholds["high"] - similarity) / (
                self.similarity_thresholds["moderate"] - self.similarity_thresholds["high"] + 1e-6
            )
        else:  # critical
            normalized = min(
                1.0, (self.similarity_thresholds["high"] - similarity) / self.similarity_thresholds["high"]
            )

        normalized = max(0.0, min(1.0, normalized))

        # Calculate drop within range
        base_drop = min_drop + (max_drop - min_drop) * normalized

        # Weight by feature importance if available
        if importance is not None:
            weighted_drop = base_drop * importance
        else:
            # If no importance, assume moderate importance (0.5)
            weighted_drop = base_drop * 0.5

        return float(weighted_drop)

    def _extract_feature_importances(
        self, model: Optional[Any], feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """Extract and normalize feature importances from model"""
        if model is None:
            return None

        try:
            # Try to get feature importances
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                # For linear models, use absolute coefficients
                importances = np.abs(model.coef_)
                if len(importances.shape) > 1:
                    importances = importances.mean(axis=0)
            else:
                return None

            # Normalize to sum to 1
            if importances.sum() > 0:
                importances = importances / importances.sum()

            # Create dictionary
            importance_dict = {feature: float(imp) for feature, imp in zip(feature_names, importances)}

            return importance_dict

        except Exception as e:
            print(f"Could not extract feature importances: {str(e)}")
            return None

    def _calculate_overall_assessment(self, scatter_points: List[Dict[str, Any]], model_type: str) -> Dict[str, Any]:
        """Calculate overall sanity check assessment"""
        if len(scatter_points) == 0:
            return {
                "expected_metric": "accuracy" if model_type == "classification" else "r2",
                "predicted_value": 0.0,
                "confidence": 0.0,
                "drift_type": "unknown",
                "severity": "critical",
                "alert_level": "🔴 Warning",
                "overall_similarity_score": 0.0,
                "num_features_analyzed": 0,
                "num_drifted_features": 0,
            }

        # Calculate overall similarity (weighted average if importances available)
        total_similarity = 0.0
        total_weight = 0.0

        for point in scatter_points:
            importance = point.get("importance", 0.5)  # Default to 0.5 if not available
            if importance is None:
                importance = 0.5

            total_similarity += point["similarity"] * importance
            total_weight += importance

        overall_similarity = total_similarity / total_weight if total_weight > 0 else 0.0

        # Calculate expected performance drop (weighted average)
        total_drop = 0.0
        for point in scatter_points:
            importance = point.get("importance", 0.5)
            if importance is None:
                importance = 0.5
            total_drop += point["expected_drop"] * importance

        expected_drop = total_drop / total_weight if total_weight > 0 else 0.0

        # Classify overall severity
        overall_severity = self._classify_severity(overall_similarity)

        # Determine alert level
        if overall_severity in ["stable", "low"]:
            alert_level = "🟢 Stable"
        elif overall_severity == "moderate":
            alert_level = "🟡 Monitor"
        else:
            alert_level = "🔴 Warning"

        # Count drifted features (p_value < 0.05)
        num_drifted = sum(1 for point in scatter_points if point.get("p_value", 1.0) < 0.05)

        # Estimate expected metric value (assuming baseline of 0.85 for classification, 0.75 for regression)
        baseline = 0.85 if model_type == "classification" else 0.75
        predicted_value = max(0.0, baseline - expected_drop)

        # Calculate confidence (higher similarity = higher confidence)
        confidence = overall_similarity

        # Infer drift type (simplified heuristic)
        if overall_severity in ["stable", "low"]:
            drift_type = "stable"
        elif num_drifted / len(scatter_points) > 0.5:
            drift_type = "data"  # More than half features drifted = data drift
        else:
            drift_type = "data"  # Conservative assumption

        return {
            "expected_metric": "f1" if model_type == "classification" else "r2",
            "predicted_value": float(predicted_value),
            "confidence": float(confidence),
            "drift_type": drift_type,
            "severity": overall_severity,
            "alert_level": alert_level,
            "overall_similarity_score": float(overall_similarity),
            "num_features_analyzed": len(scatter_points),
            "num_drifted_features": num_drifted,
        }

    def _generate_recommendations(
        self, scatter_points: List[Dict[str, Any]], overall_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Overall recommendation based on severity
        severity = overall_assessment["severity"]

        if severity == "stable":
            recommendations.append("✓ Data quality is stable. Model should perform well on new data.")
        elif severity == "low":
            recommendations.append(
                "⚠ Minor drift detected. Monitor model performance but no immediate action required."
            )
        elif severity == "moderate":
            recommendations.append("⚠ Moderate drift detected. Consider investigating data collection process.")
            recommendations.append(
                f"→ Monitor model performance closely. Expected drop: {overall_assessment['expected_metric']} may decrease to ~{overall_assessment['predicted_value']:.2f}"
            )
        elif severity == "high":
            recommendations.append("🔴 High drift detected. Review data sources and feature engineering.")
            recommendations.append(
                f"→ Model performance likely to degrade. Expected {overall_assessment['expected_metric']}: ~{overall_assessment['predicted_value']:.2f}"
            )
            recommendations.append("→ Consider retraining the model with recent data.")
        else:  # critical
            recommendations.append("🔴 CRITICAL drift detected. Immediate investigation required!")
            recommendations.append("→ DO NOT deploy model on this data without thorough validation.")
            recommendations.append("→ Verify data collection pipeline and feature extraction process.")
            recommendations.append("→ Model retraining strongly recommended.")

        # Feature-specific recommendations (top 3 most drifted)
        high_drift_features = [p for p in scatter_points if p["severity"] in ["high", "critical"]][:3]

        if high_drift_features:
            recommendations.append("\n📊 Features requiring attention:")
            for feature_data in high_drift_features:
                feature = feature_data["feature"]
                severity = feature_data["severity"]
                recommendations.append(
                    f"  • '{feature}' shows {severity} drift - investigate data collection for this feature"
                )

        # Data quality checks
        if overall_assessment["num_drifted_features"] > overall_assessment["num_features_analyzed"] * 0.5:
            recommendations.append("\n🔍 Data Quality Check: More than 50% of features show significant drift.")
            recommendations.append("  → Verify that the current dataset represents the same problem domain.")
            recommendations.append("  → Check for data preprocessing inconsistencies.")

        return recommendations


# Create singleton instance
sanity_check_service = SanityCheckService()
