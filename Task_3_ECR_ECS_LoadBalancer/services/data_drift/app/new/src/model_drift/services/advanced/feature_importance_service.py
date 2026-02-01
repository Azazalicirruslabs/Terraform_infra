"""
Feature Importance Service - Implements both Permutation and SHAP importance analysis
Following research specifications for feature importance drift detection
"""

import warnings
from typing import Any, Dict, List

import numpy as np
from sklearn.base import is_classifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# Optional SHAP import - will handle gracefully if not available
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class FeatureImportanceService:
    """Service for calculating and comparing feature importance using multiple methods"""

    def __init__(self):
        self.shap_available = SHAP_AVAILABLE

    def permutation_importance_analysis(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Calculate permutation importance for features

        Permutation importance measures feature importance by calculating the increase in
        model error when a feature's values are randomly shuffled

        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            feature_names: Names of features (optional)
            n_repeats: Number of permutation repeats
            random_state: Random state for reproducibility

        Returns:
            Dictionary with permutation importance results
        """
        try:
            # Determine scoring metric based on model type
            if is_classifier(model):
                scoring = "accuracy"
            else:
                scoring = "neg_mean_squared_error"

            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring
            )

            # Prepare feature names
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            # Create results
            importance_data = []
            for i, feature in enumerate(feature_names):
                importance_data.append(
                    {
                        "feature": feature,
                        "importance_mean": float(perm_importance.importances_mean[i]),
                        "importance_std": float(perm_importance.importances_std[i]),
                        "importance_values": perm_importance.importances[i].tolist(),
                    }
                )

            # Sort by importance
            importance_data.sort(key=lambda x: x["importance_mean"], reverse=True)

            return {
                "method": "permutation_importance",
                "n_features": len(feature_names),
                "n_repeats": n_repeats,
                "scoring": scoring,
                "feature_importances": importance_data,
                "total_importance": float(np.sum(perm_importance.importances_mean)),
            }

        except Exception as e:
            return {"error": f"Permutation importance calculation failed: {str(e)}"}

    def shap_importance_analysis(
        self, model, X: np.ndarray, feature_names: List[str] = None, max_evals: int = 1000
    ) -> Dict[str, Any]:
        """
        Calculate SHAP importance for features

        SHAP values provide a unified measure of feature importance based on
        cooperative game theory

        Args:
            model: Trained model
            X: Feature matrix (sample for SHAP calculation)
            feature_names: Names of features (optional)
            max_evals: Maximum evaluations for SHAP explainer

        Returns:
            Dictionary with SHAP importance results
        """
        try:
            if not self.shap_available:
                return {"error": "SHAP library not available. Install with: pip install shap"}

            # Prepare feature names
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            # Create appropriate SHAP explainer
            try:
                # Try TreeExplainer first (works for tree-based models)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            except:
                try:
                    # Fall back to Permutation explainer
                    explainer = shap.explainers.Permutation(model.predict, X[:100])
                    shap_values = explainer(X[:max_evals])
                    if hasattr(shap_values, "values"):
                        shap_values = shap_values.values
                except:
                    # Final fallback to KernelExplainer (slower but universal)
                    background = shap.sample(X, min(100, len(X)))
                    explainer = shap.KernelExplainer(model.predict, background)
                    shap_values = explainer.shap_values(X[: min(max_evals, len(X))])

            # Handle multi-class classification
            if isinstance(shap_values, list):
                # For multi-class, use the mean absolute SHAP values across classes
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)

            # Calculate feature importance as mean absolute SHAP values
            if len(shap_values.shape) == 2:
                feature_importance = np.mean(np.abs(shap_values), axis=0)
            else:
                feature_importance = np.abs(shap_values)

            # Create results
            importance_data = []
            for i, feature in enumerate(feature_names):
                importance_data.append(
                    {
                        "feature": feature,
                        "importance": float(feature_importance[i]),
                        "importance_normalized": float(feature_importance[i] / np.sum(feature_importance)),
                    }
                )

            # Sort by importance
            importance_data.sort(key=lambda x: x["importance"], reverse=True)

            return {
                "method": "shap",
                "n_features": len(feature_names),
                "n_samples": len(X),
                "feature_importances": importance_data,
                "total_importance": float(np.sum(feature_importance)),
            }

        except Exception as e:
            return {"error": f"SHAP importance calculation failed: {str(e)}"}

    def compare_feature_importance(
        self, importance_ref: Dict[str, Any], importance_curr: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare feature importance between reference and current models

        Args:
            importance_ref: Reference model importance results
            importance_curr: Current model importance results

        Returns:
            Dictionary with importance comparison and drift analysis
        """
        try:
            if "error" in importance_ref or "error" in importance_curr:
                return {"error": "Cannot compare importance with errors in input data"}

            # Extract feature importance data
            ref_features = {item["feature"]: item for item in importance_ref["feature_importances"]}
            curr_features = {item["feature"]: item for item in importance_curr["feature_importances"]}

            # Find common features
            common_features = set(ref_features.keys()) & set(curr_features.keys())

            if not common_features:
                return {"error": "No common features found between models"}

            # Calculate importance changes
            feature_drifts = []
            for feature in common_features:
                ref_imp = ref_features[feature].get("importance_mean", ref_features[feature].get("importance", 0))
                curr_imp = curr_features[feature].get("importance_mean", curr_features[feature].get("importance", 0))

                absolute_change = curr_imp - ref_imp
                relative_change = (absolute_change / ref_imp * 100) if ref_imp != 0 else 0

                feature_drifts.append(
                    {
                        "feature": feature,
                        "reference_importance": float(ref_imp),
                        "current_importance": float(curr_imp),
                        "absolute_change": float(absolute_change),
                        "relative_change": float(relative_change),
                        "drift_magnitude": float(abs(relative_change)),
                    }
                )

            # Sort by drift magnitude
            feature_drifts.sort(key=lambda x: x["drift_magnitude"], reverse=True)

            # Calculate drift statistics
            drift_magnitudes = [fd["drift_magnitude"] for fd in feature_drifts]
            avg_drift = np.mean(drift_magnitudes)
            max_drift = np.max(drift_magnitudes)

            # Categorize drift severity
            if max_drift >= 50:  # 50% change
                drift_severity = "High"
            elif max_drift >= 25:  # 25% change
                drift_severity = "Medium"
            elif max_drift >= 10:  # 10% change
                drift_severity = "Low"
            else:
                drift_severity = "Minimal"

            # Identify most impacted features
            high_drift_features = [fd for fd in feature_drifts if fd["drift_magnitude"] >= 25]
            increased_importance = [fd for fd in feature_drifts if fd["relative_change"] > 10]
            decreased_importance = [fd for fd in feature_drifts if fd["relative_change"] < -10]

            return {
                "comparison_summary": {
                    "method": importance_ref.get("method", "unknown"),
                    "common_features": len(common_features),
                    "average_drift": float(avg_drift),
                    "maximum_drift": float(max_drift),
                    "drift_severity": drift_severity,
                },
                "feature_drift_analysis": feature_drifts,
                "drift_categories": {
                    "high_drift_features": high_drift_features,
                    "increased_importance": increased_importance,
                    "decreased_importance": decreased_importance,
                },
                "insights": {
                    "most_increased": (
                        feature_drifts[0]["feature"]
                        if feature_drifts and feature_drifts[0]["relative_change"] > 0
                        else None
                    ),
                    "most_decreased": (
                        min(feature_drifts, key=lambda x: x["relative_change"])["feature"] if feature_drifts else None
                    ),
                    "stability_score": float(100 - avg_drift),  # Higher is more stable
                },
            }

        except Exception as e:
            return {"error": f"Feature importance comparison failed: {str(e)}"}

    def generate_impact_assessment(self, comparison_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate human-readable impact assessment based on feature importance drift

        Args:
            comparison_result: Results from compare_feature_importance

        Returns:
            Dictionary with impact assessment text and recommendations
        """
        try:
            if "error" in comparison_result:
                return comparison_result

            summary = comparison_result["comparison_summary"]
            categories = comparison_result["drift_categories"]
            insights = comparison_result["insights"]

            # Generate assessment text
            severity = summary["drift_severity"]
            max_drift = summary["maximum_drift"]

            assessment_parts = []

            # Overall assessment
            if severity == "High":
                assessment_parts.append(
                    f"Significant feature importance shift detected with {max_drift:.1f}% maximum change."
                )
            elif severity == "Medium":
                assessment_parts.append(
                    f"Moderate feature importance changes observed with {max_drift:.1f}% maximum change."
                )
            elif severity == "Low":
                assessment_parts.append(
                    f"Minor feature importance drift detected with {max_drift:.1f}% maximum change."
                )
            else:
                assessment_parts.append("Feature importance remains relatively stable between models.")

            # Feature-specific insights
            if insights["most_increased"]:
                assessment_parts.append(
                    f"Feature '{insights['most_increased']}' shows the highest increase in importance."
                )

            if insights["most_decreased"]:
                most_decreased = min(comparison_result["feature_drift_analysis"], key=lambda x: x["relative_change"])
                assessment_parts.append(
                    f"Feature '{most_decreased['feature']}' shows the largest decrease in importance ({most_decreased['relative_change']:.1f}% change)."
                )

            # High drift features
            if categories["high_drift_features"]:
                high_drift_names = [f["feature"] for f in categories["high_drift_features"][:3]]
                assessment_parts.append(f"Features with significant drift: {', '.join(high_drift_names)}.")

            # Stability assessment
            stability_score = insights["stability_score"]
            if stability_score >= 90:
                assessment_parts.append("Model feature dependencies remain highly stable.")
            elif stability_score >= 75:
                assessment_parts.append("Model shows moderate stability in feature usage.")
            else:
                assessment_parts.append("Model exhibits significant changes in feature reliance patterns.")

            # Generate recommendations
            recommendations = []

            if severity in ["High", "Medium"]:
                recommendations.append("Monitor model performance closely due to feature importance changes")
                recommendations.append("Investigate potential data distribution shifts")

                if len(categories["decreased_importance"]) > 2:
                    recommendations.append("Review data quality for features with decreased importance")

                if len(categories["increased_importance"]) > 2:
                    recommendations.append("Validate the reliability of features with increased importance")

            if stability_score < 75:
                recommendations.append("Consider model retraining with recent data")
                recommendations.append("Review feature engineering pipeline for consistency")

            return {
                "impact_assessment": " ".join(assessment_parts),
                "recommendations": recommendations,
                "stability_metrics": {
                    "stability_score": stability_score,
                    "drift_severity": severity,
                    "features_with_high_drift": len(categories["high_drift_features"]),
                },
            }

        except Exception as e:
            return {"error": f"Impact assessment generation failed: {str(e)}"}

    def comprehensive_importance_analysis(
        self,
        model_ref,
        model_curr,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        X_curr: np.ndarray,
        y_curr: np.ndarray,
        feature_names: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive feature importance analysis using both methods

        Args:
            model_ref: Reference model
            model_curr: Current model
            X_ref: Reference features
            y_ref: Reference targets
            X_curr: Current features
            y_curr: Current targets
            feature_names: Feature names

        Returns:
            Dictionary with complete importance analysis
        """
        try:
            results = {"analysis_type": "comprehensive_feature_importance", "methods_used": []}

            # Permutation importance analysis
            perm_ref = self.permutation_importance_analysis(model_ref, X_ref, y_ref, feature_names)
            perm_curr = self.permutation_importance_analysis(model_curr, X_curr, y_curr, feature_names)

            if "error" not in perm_ref and "error" not in perm_curr:
                perm_comparison = self.compare_feature_importance(perm_ref, perm_curr)
                perm_impact = self.generate_impact_assessment(perm_comparison)

                results["permutation_analysis"] = {
                    "reference": perm_ref,
                    "current": perm_curr,
                    "comparison": perm_comparison,
                    "impact_assessment": perm_impact,
                }
                results["methods_used"].append("permutation_importance")

            # SHAP analysis (if available)
            if self.shap_available:
                # Use sample for SHAP to avoid computational overhead
                sample_size = min(500, len(X_ref), len(X_curr))
                X_ref_sample = X_ref[:sample_size]
                X_curr_sample = X_curr[:sample_size]

                shap_ref = self.shap_importance_analysis(model_ref, X_ref_sample, feature_names)
                shap_curr = self.shap_importance_analysis(model_curr, X_curr_sample, feature_names)

                if "error" not in shap_ref and "error" not in shap_curr:
                    shap_comparison = self.compare_feature_importance(shap_ref, shap_curr)
                    shap_impact = self.generate_impact_assessment(shap_comparison)

                    results["shap_analysis"] = {
                        "reference": shap_ref,
                        "current": shap_curr,
                        "comparison": shap_comparison,
                        "impact_assessment": shap_impact,
                    }
                    results["methods_used"].append("shap")

            # Overall conclusion
            if len(results["methods_used"]) > 0:
                results["status"] = "success"
                results["available_methods"] = results["methods_used"]
            else:
                results["status"] = "failed"
                results["error"] = "No feature importance methods succeeded"

            return results

        except Exception as e:
            return {"error": f"Comprehensive importance analysis failed: {str(e)}"}


# Service instance
feature_importance_service = FeatureImportanceService()
