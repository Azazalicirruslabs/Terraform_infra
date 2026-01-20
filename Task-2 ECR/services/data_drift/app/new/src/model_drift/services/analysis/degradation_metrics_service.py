"""
Degradation Metrics Service - Tab 2 Analysis
Orchestrates all three sub-tabs: Model Disagreement, Confidence Analysis, Feature Importance Drift
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from services.data_drift.app.new.src.model_drift.services.advanced.feature_importance_service import (
    feature_importance_service,
)
from services.data_drift.app.new.src.model_drift.services.advanced.model_disagreement_service import (
    model_disagreement_service,
)
from services.data_drift.app.new.src.model_drift.services.core.calibration_service import calibration_service

warnings.filterwarnings("ignore")


class DegradationMetricsService:
    """Service for degradation metrics analysis with three sub-tabs (Tab 2)"""

    def __init__(self):
        pass

    def _detect_model_type(self, model_ref, model_curr, pred_ref_proba, pred_curr_proba) -> str:
        """
        Detect if models are regression or classification

        Args:
            model_ref: Reference model
            model_curr: Current model
            pred_ref_proba: Reference probabilities (None for regression)
            pred_curr_proba: Current probabilities (None for regression)

        Returns:
            "regression" or "classification"
        """
        # Primary check: if probabilities are None, likely regression
        if pred_ref_proba is None and pred_curr_proba is None:
            # Verify models don't have predict_proba
            ref_has_proba = hasattr(model_ref, "predict_proba") if model_ref is not None else False
            curr_has_proba = hasattr(model_curr, "predict_proba") if model_curr is not None else False
            if not ref_has_proba and not curr_has_proba:
                return "regression"
        return "classification"

    def analyze_degradation_metrics(
        self,
        y_true: np.ndarray,
        pred_ref: np.ndarray,
        pred_curr: np.ndarray,
        pred_ref_proba: np.ndarray,
        pred_curr_proba: np.ndarray,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        X_curr: np.ndarray,
        y_curr: np.ndarray,
        model_ref,
        model_curr,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive degradation metrics analysis

        Args:
            y_true: True labels for test set
            pred_ref: Reference model predictions (binary)
            pred_curr: Current model predictions (binary)
            pred_ref_proba: Reference model probabilities
            pred_curr_proba: Current model probabilities
            X_ref: Reference training features
            y_ref: Reference training targets
            X_curr: Current training features
            y_curr: Current training targets
            model_ref: Reference model object
            model_curr: Current model object
            feature_names: Feature names (optional)

        Returns:
            Dictionary with all three sub-tab analyses
        """
        try:
            # Enhanced input validation with detailed logging
            validation_errors = []

            # Log input shapes for debugging
            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"Degradation metrics analysis starting...")
            logger.info(f"Input validation - y_true: {y_true.shape if y_true is not None else None}")
            logger.info(f"Input validation - pred_ref: {pred_ref.shape if pred_ref is not None else None}")
            logger.info(f"Input validation - pred_curr: {pred_curr.shape if pred_curr is not None else None}")

            # Detect model type (regression vs classification)
            model_type = self._detect_model_type(model_ref, model_curr, pred_ref_proba, pred_curr_proba)
            logger.info(f"Detected model type: {model_type}")

            # Check required arrays
            if pred_ref is None or len(pred_ref) == 0:
                validation_errors.append("Reference predictions are empty or None")
            if pred_curr is None or len(pred_curr) == 0:
                validation_errors.append("Current predictions are empty or None")

            # Check probability arrays (only required for classification)
            prob_analysis_possible = True
            if model_type == "regression":
                # For regression, probabilities don't exist
                prob_analysis_possible = False
                logger.info("Regression model detected - skipping probability-based analyses")
            elif pred_ref_proba is not None and pred_curr_proba is not None:
                if len(pred_ref_proba) == 0:
                    validation_errors.append("Reference probabilities array is empty")
                    prob_analysis_possible = False
                elif len(pred_curr_proba) == 0:
                    validation_errors.append("Current probabilities array is empty")
                    prob_analysis_possible = False
                else:
                    logger.info(f"Probability shapes: ref={pred_ref_proba.shape}, curr={pred_curr_proba.shape}")
                    logger.info(
                        f"Probability ranges: ref=[{np.min(pred_ref_proba):.6f}, {np.max(pred_ref_proba):.6f}], curr=[{np.min(pred_curr_proba):.6f}, {np.max(pred_curr_proba):.6f}]"
                    )
            else:
                if model_type == "classification":
                    logger.warning(
                        "Prediction probabilities not available for classification - some analyses will be limited"
                    )
                prob_analysis_possible = False

            # Check array length consistency
            if pred_ref is not None and pred_curr is not None:
                if len(pred_ref) != len(pred_curr):
                    validation_errors.append(
                        f"Prediction array length mismatch: ref={len(pred_ref)}, curr={len(pred_curr)}"
                    )

            if pred_ref_proba is not None and pred_curr_proba is not None:
                if len(pred_ref_proba) != len(pred_curr_proba):
                    validation_errors.append(
                        f"Probability array length mismatch: ref={len(pred_ref_proba)}, curr={len(pred_curr_proba)}"
                    )

            # If we have validation errors, log them but still try to run what we can
            if validation_errors:
                logger.warning(f"Validation warnings: {'; '.join(validation_errors)}")

            analysis_results = {
                "analysis_type": "degradation_metrics",
                "model_type": model_type,
                "sub_tabs": {"model_disagreement": {}, "confidence_analysis": {}, "feature_importance_drift": {}},
                "overall_degradation_assessment": {},
            }

            # Sub-tab 1: Model Disagreement Analysis (only for classification with probabilities)
            if prob_analysis_possible:
                disagreement_analysis = self._analyze_model_disagreement(pred_ref_proba, pred_curr_proba)
                analysis_results["sub_tabs"]["model_disagreement"] = disagreement_analysis
            else:
                if model_type == "regression":
                    analysis_results["sub_tabs"]["model_disagreement"] = {
                        "info": "Model disagreement analysis not applicable for regression models",
                        "message": "Regression models do not produce probability distributions. Use residual analysis instead.",
                        "model_type": "regression",
                    }
                else:
                    analysis_results["sub_tabs"]["model_disagreement"] = {
                        "error": "Model disagreement analysis requires prediction probabilities",
                        "message": "Enable predict_proba on your classification model to get disagreement analysis",
                        "available_data": "binary_predictions_only",
                    }

            # Sub-tab 2: Confidence Analysis (only for classification with probabilities)
            if prob_analysis_possible and y_true is not None and len(y_true) > 0:
                confidence_analysis = self._analyze_confidence(y_true, pred_ref_proba, pred_curr_proba)
                analysis_results["sub_tabs"]["confidence_analysis"] = confidence_analysis
            else:
                if model_type == "regression":
                    analysis_results["sub_tabs"]["confidence_analysis"] = {
                        "info": "Confidence/calibration analysis not applicable for regression models",
                        "message": "Regression models do not produce confidence scores. Use prediction interval analysis instead.",
                        "model_type": "regression",
                    }
                else:
                    analysis_results["sub_tabs"]["confidence_analysis"] = {
                        "error": "Confidence analysis requires prediction probabilities and ground truth labels",
                        "message": "Ensure your model has predict_proba method and ground truth is available",
                        "available_data": f"probabilities_available={prob_analysis_possible}, ground_truth_available={y_true is not None and len(y_true) > 0 if y_true is not None else False}",
                    }

            # Sub-tab 3: Feature Importance Drift (try even without probabilities)
            try:
                feature_importance_analysis = self._analyze_feature_importance_drift(
                    model_ref, model_curr, X_ref, y_ref, X_curr, y_curr, feature_names
                )
                analysis_results["sub_tabs"]["feature_importance_drift"] = feature_importance_analysis
            except Exception as e:
                logger.error(f"Feature importance analysis failed: {e}")
                analysis_results["sub_tabs"]["feature_importance_drift"] = {
                    "error": f"Feature importance analysis failed: {str(e)}",
                    "message": "Feature importance analysis requires models with feature_importances_ attribute or similar",
                }

            # Overall degradation assessment
            overall_assessment = self._generate_overall_degradation_assessment(
                analysis_results["sub_tabs"]["model_disagreement"],
                analysis_results["sub_tabs"]["confidence_analysis"],
                analysis_results["sub_tabs"]["feature_importance_drift"],
            )
            analysis_results["overall_degradation_assessment"] = overall_assessment

            logger.info("Degradation metrics analysis completed successfully")
            return analysis_results

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Degradation metrics analysis failed: {str(e)}")
            return {
                "analysis_type": "degradation_metrics",
                "error": f"Degradation metrics analysis failed: {str(e)}",
                "sub_tabs": {
                    "model_disagreement": {"error": str(e)},
                    "confidence_analysis": {"error": str(e)},
                    "feature_importance_drift": {"error": str(e)},
                },
                "overall_degradation_assessment": {"error": str(e)},
            }

    def _analyze_model_disagreement(self, pred_ref_proba: np.ndarray, pred_curr_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze model disagreement (Sub-tab 1) with enhanced validation"""

        try:
            # Validate inputs
            if pred_ref_proba is None or pred_curr_proba is None:
                return {"error": "Prediction probabilities are required for model disagreement analysis"}

            if len(pred_ref_proba) == 0 or len(pred_curr_proba) == 0:
                return {"error": "Empty prediction arrays provided"}

            if len(pred_ref_proba) != len(pred_curr_proba):
                return {"error": f"Prediction array length mismatch: {len(pred_ref_proba)} vs {len(pred_curr_proba)}"}

            # Check for valid probability ranges
            ref_min, ref_max = np.min(pred_ref_proba), np.max(pred_ref_proba)
            curr_min, curr_max = np.min(pred_curr_proba), np.max(pred_curr_proba)

            if ref_min < 0 or ref_max > 1 or curr_min < 0 or curr_max > 1:
                return {
                    "error": f"Invalid probability values detected. Reference: [{ref_min:.6f}, {ref_max:.6f}], Current: [{curr_min:.6f}, {curr_max:.6f}]"
                }

            # Get comprehensive disagreement analysis
            disagreement_result = model_disagreement_service.comprehensive_disagreement_analysis(
                pred_ref_proba, pred_curr_proba, sample_size=1000
            )

            if "error" in disagreement_result:
                return {"error": f"Model disagreement analysis failed: {disagreement_result['error']}"}

            # Validate required fields in disagreement_result
            required_fields = ["disagreement_statistics", "scatter_plot", "threshold_analysis"]
            missing_fields = [field for field in required_fields if field not in disagreement_result]
            if missing_fields:
                return {"error": f"Model disagreement service missing required fields: {missing_fields}"}

            # Validate disagreement_statistics has required metrics
            stats = disagreement_result["disagreement_statistics"]
            if "error" in stats:
                return {"error": f"Disagreement statistics calculation failed: {stats['error']}"}

            required_stats = [
                "mean_absolute_difference",
                "maximum_difference",
                "standard_deviation",
                "pearson_correlation",
            ]
            missing_stats = [stat for stat in required_stats if stat not in stats]
            if missing_stats:
                return {"error": f"Model disagreement statistics missing required metrics: {missing_stats}"}

            # Format for frontend consumption
            formatted_result = {
                "prediction_scatter_plot": disagreement_result["scatter_plot"],
                "disagreement_analysis": {
                    "mean_absolute_difference": disagreement_result["disagreement_statistics"][
                        "mean_absolute_difference"
                    ],
                    "maximum_difference": disagreement_result["disagreement_statistics"]["maximum_difference"],
                    "standard_deviation": disagreement_result["disagreement_statistics"]["standard_deviation"],
                    "pearson_correlation": disagreement_result["disagreement_statistics"]["pearson_correlation"],
                },
                "decision_threshold_analysis": disagreement_result["threshold_analysis"]["threshold_analysis"],
                "summary": {
                    "disagreement_level": disagreement_result["summary"]["disagreement_level"],
                    "correlation_level": disagreement_result["summary"]["correlation_level"],
                    "total_predictions": disagreement_result["summary"]["total_predictions"],
                },
            }

            return formatted_result

        except Exception as e:
            return {"error": f"Model disagreement analysis failed: {str(e)}"}

    def _analyze_confidence(
        self, y_true: np.ndarray, pred_ref_proba: np.ndarray, pred_curr_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze confidence and calibration (Sub-tab 2) with enhanced validation"""

        try:
            # Validate inputs
            if pred_ref_proba is None or pred_curr_proba is None:
                return {"error": "Prediction probabilities are required for confidence analysis"}

            if y_true is None or len(y_true) == 0:
                return {"error": "Ground truth labels are required for confidence analysis"}

            if len(pred_ref_proba) != len(y_true) or len(pred_curr_proba) != len(y_true):
                return {
                    "error": f"Array length mismatch. Ground truth: {len(y_true)}, Ref proba: {len(pred_ref_proba)}, Curr proba: {len(pred_curr_proba)}"
                }

            # Check for valid probability and label ranges
            if np.min(pred_ref_proba) < 0 or np.max(pred_ref_proba) > 1:
                return {
                    "error": f"Invalid reference probability values: [{np.min(pred_ref_proba):.6f}, {np.max(pred_ref_proba):.6f}]"
                }

            if np.min(pred_curr_proba) < 0 or np.max(pred_curr_proba) > 1:
                return {
                    "error": f"Invalid current probability values: [{np.min(pred_curr_proba):.6f}, {np.max(pred_curr_proba):.6f}]"
                }

            # Get comprehensive calibration analysis for both models
            calibration_comparison = calibration_service.compare_calibrations(y_true, pred_ref_proba, pred_curr_proba)

            if "error" in calibration_comparison:
                return {"error": f"Calibration comparison failed: {calibration_comparison['error']}"}

            # Validate calibration comparison structure
            required_keys = ["reference_analysis", "current_analysis", "metric_changes"]
            missing_keys = [key for key in required_keys if key not in calibration_comparison]
            if missing_keys:
                return {"error": f"Calibration comparison missing required keys: {missing_keys}"}

            ref_analysis = calibration_comparison["reference_analysis"]
            curr_analysis = calibration_comparison["current_analysis"]

            # Validate reference and current analysis structure
            for analysis_name, analysis in [("reference", ref_analysis), ("current", curr_analysis)]:
                if "error" in analysis:
                    return {"error": f"{analysis_name.capitalize()} analysis failed: {analysis['error']}"}
                if "metrics" not in analysis:
                    return {"error": f"{analysis_name.capitalize()} analysis missing metrics"}
                if "detailed_results" not in analysis:
                    return {"error": f"{analysis_name.capitalize()} analysis missing detailed_results"}

            # Format confidence score distribution for bar chart with enhanced validation
            ref_distribution = ref_analysis["detailed_results"]["confidence_distribution"]
            curr_distribution = curr_analysis["detailed_results"]["confidence_distribution"]

            # Validate distribution structure
            if "bin_labels" not in ref_distribution or "bin_counts" not in ref_distribution:
                return {"error": "Reference distribution missing required fields"}
            if "bin_labels" not in curr_distribution or "bin_counts" not in curr_distribution:
                return {"error": "Current distribution missing required fields"}

            # Handle potentially different bin structures (adaptive binning)
            confidence_distribution = []

            # Use reference distribution structure as base
            ref_labels = ref_distribution["bin_labels"]
            ref_counts = ref_distribution["bin_counts"]
            curr_counts = curr_distribution["bin_counts"]

            # Ensure arrays have the same length
            min_length = min(len(ref_labels), len(ref_counts), len(curr_counts))

            for i in range(min_length):
                confidence_distribution.append(
                    {
                        "bin": ref_labels[i],
                        "reference_count": ref_counts[i],
                        "current_count": curr_counts[i] if i < len(curr_counts) else 0,
                    }
                )

            # If current distribution has more bins, add them with 0 reference count
            if len(curr_counts) > min_length:
                curr_labels = curr_distribution["bin_labels"]
                for i in range(min_length, len(curr_counts)):
                    confidence_distribution.append(
                        {
                            "bin": curr_labels[i] if i < len(curr_labels) else f"bin_{i}",
                            "reference_count": 0,
                            "current_count": curr_counts[i],
                        }
                    )

            # Format calibration curves for line chart
            ref_curve = ref_analysis["detailed_results"]["calibration_curve"]
            curr_curve = curr_analysis["detailed_results"]["calibration_curve"]

            calibration_curves = {
                "reference_curve": {
                    "mean_predicted_probability": ref_curve["mean_predicted_probability"],
                    "fraction_of_positives": ref_curve["fraction_of_positives"],
                },
                "current_curve": {
                    "mean_predicted_probability": curr_curve["mean_predicted_probability"],
                    "fraction_of_positives": curr_curve["fraction_of_positives"],
                },
                "perfect_calibration": ref_curve["perfect_calibration"],
            }

            # Format confidence metrics cards
            confidence_metrics = {
                "brier_score": {
                    "reference": ref_analysis["metrics"]["brier_score"],
                    "current": curr_analysis["metrics"]["brier_score"],
                    "delta": calibration_comparison["metric_changes"]["brier_score_change"]["absolute_change"],
                },
                "expected_calibration_error": {
                    "reference": ref_analysis["metrics"]["expected_calibration_error"],
                    "current": curr_analysis["metrics"]["expected_calibration_error"],
                    "delta": calibration_comparison["metric_changes"]["expected_calibration_error_change"][
                        "absolute_change"
                    ],
                },
                "maximum_calibration_error": {
                    "reference": ref_analysis["metrics"]["maximum_calibration_error"],
                    "current": curr_analysis["metrics"]["maximum_calibration_error"],
                    "delta": calibration_comparison["metric_changes"]["maximum_calibration_error_change"][
                        "absolute_change"
                    ],
                },
                "confidence_entropy": {
                    "reference": ref_analysis["metrics"]["confidence_entropy"],
                    "current": curr_analysis["metrics"]["confidence_entropy"],
                    "delta": calibration_comparison["metric_changes"]["confidence_entropy_change"]["absolute_change"],
                },
            }

            formatted_result = {
                "confidence_score_distribution": confidence_distribution,
                "calibration_curves": calibration_curves,
                "confidence_metrics": confidence_metrics,
                "calibration_assessment": {
                    "reference_quality": ref_analysis["calibration_assessment"]["quality"],
                    "current_quality": curr_analysis["calibration_assessment"]["quality"],
                    "drift_severity": calibration_comparison["calibration_drift"]["severity"],
                },
            }

            return formatted_result

        except Exception as e:
            return {"error": f"Confidence analysis failed: {str(e)}"}

    def _analyze_feature_importance_drift(
        self,
        model_ref,
        model_curr,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        X_curr: np.ndarray,
        y_curr: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze feature importance drift (Sub-tab 3)"""

        try:
            # Get comprehensive feature importance analysis
            importance_analysis = feature_importance_service.comprehensive_importance_analysis(
                model_ref, model_curr, X_ref, y_ref, X_curr, y_curr, feature_names
            )

            if "error" in importance_analysis:
                return importance_analysis

            # Use permutation importance results (more reliable than SHAP for this use case)
            if "permutation_analysis" in importance_analysis:
                perm_analysis = importance_analysis["permutation_analysis"]
                comparison = perm_analysis["comparison"]
                impact_assessment = perm_analysis["impact_assessment"]

                # Format feature importance comparison for bar chart
                feature_importance_comparison = []
                for feature_data in comparison["feature_drift_analysis"]:
                    feature_importance_comparison.append(
                        {
                            "feature": feature_data["feature"],
                            "reference_importance": feature_data["reference_importance"],
                            "current_importance": feature_data["current_importance"],
                        }
                    )

                # Format feature importance drift table
                feature_importance_drift = []
                for feature_data in comparison["feature_drift_analysis"][:10]:  # Top 10 features
                    feature_importance_drift.append(
                        {
                            "feature": feature_data["feature"],
                            "delta": feature_data["absolute_change"],
                            "percentage_change": feature_data["relative_change"],
                        }
                    )

                # Generate impact assessment text
                impact_text = impact_assessment.get(
                    "impact_assessment", "No significant feature importance changes detected."
                )

                # Placeholder for historical trend (would need historical data)
                feature_importance_trend = {
                    "message": "Historical trend analysis requires stored model versions",
                    "top_features": [item["feature"] for item in feature_importance_comparison[:5]],
                }

                formatted_result = {
                    "feature_importance_comparison": feature_importance_comparison,
                    "feature_importance_drift": feature_importance_drift,
                    "impact_assessment": impact_text,
                    "feature_importance_trend": feature_importance_trend,
                    "drift_summary": {
                        "drift_severity": comparison["comparison_summary"]["drift_severity"],
                        "stability_score": comparison["insights"]["stability_score"],
                        "features_analyzed": comparison["comparison_summary"]["common_features"],
                    },
                }

            else:
                formatted_result = {"error": "Feature importance analysis not available"}

            return formatted_result

        except Exception as e:
            return {"error": f"Feature importance drift analysis failed: {str(e)}"}

    def _generate_overall_degradation_assessment(
        self, disagreement_analysis: Dict, confidence_analysis: Dict, feature_importance_analysis: Dict
    ) -> Dict[str, Any]:
        """Generate overall degradation assessment across all sub-tabs"""

        try:
            assessment = {
                "overall_degradation_level": "Minimal",
                "key_degradation_indicators": [],
                "recommendations": [],
                "sub_tab_summaries": {},
            }

            degradation_score = 0

            # Assess model disagreement
            if "error" not in disagreement_analysis:
                disagreement_level = disagreement_analysis["summary"]["disagreement_level"]
                assessment["sub_tab_summaries"]["model_disagreement"] = disagreement_level

                if disagreement_level == "High":
                    degradation_score += 3
                    assessment["key_degradation_indicators"].append("High model prediction disagreement")
                elif disagreement_level == "Medium":
                    degradation_score += 2
                    assessment["key_degradation_indicators"].append("Medium model prediction disagreement")
                elif disagreement_level == "Low":
                    degradation_score += 1

            # Assess calibration drift
            if "error" not in confidence_analysis:
                calibration_drift = confidence_analysis["calibration_assessment"]["drift_severity"]
                assessment["sub_tab_summaries"]["confidence_analysis"] = calibration_drift

                if calibration_drift == "High":
                    degradation_score += 3
                    assessment["key_degradation_indicators"].append("High calibration drift detected")
                elif calibration_drift == "Medium":
                    degradation_score += 2
                    assessment["key_degradation_indicators"].append("Medium calibration drift detected")
                elif calibration_drift == "Low":
                    degradation_score += 1

            # Assess feature importance drift
            if "error" not in feature_importance_analysis and "drift_summary" in feature_importance_analysis:
                feature_drift = feature_importance_analysis["drift_summary"]["drift_severity"]
                assessment["sub_tab_summaries"]["feature_importance_drift"] = feature_drift

                if feature_drift == "High":
                    degradation_score += 3
                    assessment["key_degradation_indicators"].append("High feature importance drift")
                elif feature_drift == "Medium":
                    degradation_score += 2
                    assessment["key_degradation_indicators"].append("Medium feature importance drift")
                elif feature_drift == "Low":
                    degradation_score += 1

            # Determine overall degradation level
            if degradation_score >= 7:
                assessment["overall_degradation_level"] = "Critical"
            elif degradation_score >= 5:
                assessment["overall_degradation_level"] = "High"
            elif degradation_score >= 3:
                assessment["overall_degradation_level"] = "Medium"
            elif degradation_score >= 1:
                assessment["overall_degradation_level"] = "Low"

            # Generate recommendations
            if assessment["overall_degradation_level"] in ["Critical", "High"]:
                assessment["recommendations"].extend(
                    ["Immediate investigation required", "Consider model retraining", "Review data pipeline for issues"]
                )
            elif assessment["overall_degradation_level"] == "Medium":
                assessment["recommendations"].extend(
                    ["Monitor closely", "Prepare for potential retraining", "Investigate root causes"]
                )
            else:
                assessment["recommendations"].append("Continue regular monitoring")

            if not assessment["key_degradation_indicators"]:
                assessment["key_degradation_indicators"].append("No significant degradation detected")

            assessment["degradation_score"] = degradation_score

            return assessment

        except Exception as e:
            return {"error": f"Overall degradation assessment failed: {str(e)}"}


# Service instance
degradation_metrics_service = DegradationMetricsService()
