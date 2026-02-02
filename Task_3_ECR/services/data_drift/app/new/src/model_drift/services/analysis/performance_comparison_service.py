"""
Performance Comparison Service - Tab 1 Analysis
Orchestrates comprehensive performance metrics comparison between reference and current models
"""

import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np

from ..advanced.psi_service import psi_service
from ..core.effect_size_service import effect_size_service
from ..core.metrics_calculation_service import metrics_calculation_service
from ..core.statistical_tests_service import statistical_tests_service

warnings.filterwarnings("ignore")

# Configure logger
logger = logging.getLogger(__name__)


class PerformanceComparisonService:
    """Service for comprehensive performance comparison analysis (Tab 1)"""

    def __init__(self):
        pass

    def _validate_model_type_consistency(
        self,
        is_regression: bool,
        y_true: np.ndarray,
        pred_ref: np.ndarray,
        pred_curr: np.ndarray,
        pred_ref_proba: Optional[np.ndarray],
        pred_curr_proba: Optional[np.ndarray],
    ) -> tuple[bool, str]:
        """
        Validate that the detected model type is consistent with the data

        Args:
            is_regression: Detected model type (True for regression)
            y_true: True labels/values
            pred_ref: Reference predictions
            pred_curr: Current predictions
            pred_ref_proba: Reference probabilities
            pred_curr_proba: Current probabilities

        Returns:
            Tuple of (is_valid, error_message)
        """
        unique_ratio = len(np.unique(y_true)) / len(y_true) if len(y_true) > 0 else 0

        if is_regression:
            # For regression, probabilities should be None
            if pred_ref_proba is not None or pred_curr_proba is not None:
                return (
                    False,
                    f"Regression model detected but probabilities provided (ref_proba: {pred_ref_proba is not None}, curr_proba: {pred_curr_proba is not None})",
                )

            # For regression, predictions should be continuous
            pred_ref_unique_ratio = len(np.unique(pred_ref)) / len(pred_ref) if len(pred_ref) > 0 else 0
            pred_curr_unique_ratio = len(np.unique(pred_curr)) / len(pred_curr) if len(pred_curr) > 0 else 0

            if pred_ref_unique_ratio < 0.05 or pred_curr_unique_ratio < 0.05:
                return (
                    False,
                    f"Regression detected but predictions look discrete (ref unique ratio: {pred_ref_unique_ratio:.4f}, curr: {pred_curr_unique_ratio:.4f})",
                )
        else:
            # For classification, predictions should be discrete
            if unique_ratio > 0.5:
                logger.warning(
                    f"Classification detected but y_true has high unique ratio ({unique_ratio:.4f}). This might be a regression problem."
                )

        return True, ""

    def _detect_regression_model(self, model_ref, model_curr, y_true, pred_ref_proba, pred_curr_proba) -> bool:
        """
        Detect if the models are regression models

        Strategy:
        1. Check if predict_proba is None (regression models don't have predict_proba)
        2. Check if models have predict_proba method
        3. Check y_true data type (continuous vs discrete)

        Args:
            model_ref: Reference model object
            model_curr: Current model object
            y_true: True labels/values
            pred_ref_proba: Reference model probabilities (None for regression)
            pred_curr_proba: Current model probabilities (None for regression)

        Returns:
            Boolean indicating if models are regression models
        """
        logger.info("=" * 80)
        logger.info("DETECTING MODEL TYPE")
        logger.info(f"pred_ref_proba is None: {pred_ref_proba is None}")
        logger.info(f"pred_curr_proba is None: {pred_curr_proba is None}")
        logger.info(f"y_true shape: {y_true.shape if hasattr(y_true, 'shape') else len(y_true)}")
        logger.info(f"y_true dtype: {y_true.dtype if hasattr(y_true, 'dtype') else type(y_true)}")
        logger.info(f"Unique values in y_true: {len(np.unique(y_true))}")
        logger.info(f"y_true min: {np.min(y_true)}, max: {np.max(y_true)}")

        # Primary check: predict_proba is None for regression
        if pred_ref_proba is None and pred_curr_proba is None:
            logger.info("Primary check: Both pred_proba are None")

            # Secondary check: verify models don't have predict_proba method
            ref_has_proba = hasattr(model_ref, "predict_proba") if model_ref is not None else False
            curr_has_proba = hasattr(model_curr, "predict_proba") if model_curr is not None else False

            logger.info(f"model_ref has predict_proba: {ref_has_proba}")
            logger.info(f"model_curr has predict_proba: {curr_has_proba}")

            if not ref_has_proba and not curr_has_proba:
                logger.info("✓ DETECTED AS REGRESSION: No predict_proba method found")
                logger.info("=" * 80)
                return True

            # Tertiary check: analyze y_true data type
            # If y_true has many unique values and is continuous, likely regression
            unique_ratio = len(np.unique(y_true)) / len(y_true) if len(y_true) > 0 else 0
            logger.info(f"Unique ratio (unique values / total): {unique_ratio:.4f}")

            if unique_ratio > 0.1:  # More than 10% unique values suggests regression
                logger.info(f"✓ DETECTED AS REGRESSION: High unique ratio ({unique_ratio:.4f} > 0.1)")
                logger.info("=" * 80)
                return True

        logger.info("✗ DETECTED AS CLASSIFICATION")
        logger.info("=" * 80)
        return False

    def _is_higher_better(self, metric_name: str) -> bool:
        """
        Determine if higher values are better for a given metric

        Args:
            metric_name: Name of the metric

        Returns:
            Boolean indicating if higher is better
        """
        # Metrics where lower is better
        lower_is_better_terms = [
            "error",
            "loss",
            "mse",
            "rmse",
            "mae",
            "mape",
            "max_error",
            "mean_squared_error",
            "mean_absolute_error",
            "log_loss",
        ]

        metric_lower = metric_name.lower()
        for term in lower_is_better_terms:
            if term in metric_lower:
                return False
        return True

    def _calculate_delta_metrics(self, ref_metrics: Dict[str, Any], curr_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate clear delta values for all metrics

        Args:
            ref_metrics: Reference model metrics
            curr_metrics: Current model metrics

        Returns:
            Dictionary with delta values and improvement indicators
        """
        delta_metrics = {}

        # Skip non-numeric keys
        skip_keys = {
            "confusion_matrix",
            "classification_report",
            "error",
            "precision_per_class",
            "recall_per_class",
            "f1_score_per_class",
        }

        for metric_name in ref_metrics:
            if metric_name in skip_keys or metric_name not in curr_metrics:
                continue

            if isinstance(ref_metrics[metric_name], (int, float)) and isinstance(
                curr_metrics[metric_name], (int, float)
            ):
                ref_value = float(ref_metrics[metric_name])
                curr_value = float(curr_metrics[metric_name])

                # Calculate delta values
                absolute_change = curr_value - ref_value
                percentage_change = (absolute_change / ref_value * 100) if ref_value != 0 else 0.0

                # Determine if this is an improvement
                higher_is_better = self._is_higher_better(metric_name)
                improved = absolute_change > 0 if higher_is_better else absolute_change < 0

                # Determine change magnitude
                abs_percent_change = abs(percentage_change)
                if abs_percent_change >= 20:
                    magnitude = "Large"
                elif abs_percent_change >= 10:
                    magnitude = "Medium"
                elif abs_percent_change >= 5:
                    magnitude = "Small"
                else:
                    magnitude = "Minimal"

                delta_metrics[metric_name] = {
                    "reference_value": ref_value,
                    "current_value": curr_value,
                    "absolute_change": round(absolute_change, 6),
                    "percentage_change": round(percentage_change, 2),
                    "improved": improved,
                    "magnitude": magnitude,
                    "direction": (
                        "increase" if absolute_change > 0 else "decrease" if absolute_change < 0 else "no_change"
                    ),
                }

        return delta_metrics

    def _create_detailed_metric_comparison(
        self, ref_metrics: Dict[str, Any], curr_metrics: Dict[str, Any], delta_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a detailed metric comparison table suitable for frontend display

        Args:
            ref_metrics: Reference model metrics
            curr_metrics: Current model metrics
            delta_metrics: Calculated delta metrics

        Returns:
            Dictionary with detailed comparison data
        """
        detailed_comparison = {
            "table_data": [],
            "summary": {
                "total_metrics": len(delta_metrics),
                "improved_metrics": 0,
                "degraded_metrics": 0,
                "unchanged_metrics": 0,
            },
        }

        for metric_name, delta_data in delta_metrics.items():
            # Count improvements/degradations
            if delta_data["improved"]:
                detailed_comparison["summary"]["improved_metrics"] += 1
            elif delta_data["direction"] == "no_change":
                detailed_comparison["summary"]["unchanged_metrics"] += 1
            else:
                detailed_comparison["summary"]["degraded_metrics"] += 1

            # Create table row
            table_row = {
                "metric": metric_name.replace("_", " ").title(),
                "reference_value": delta_data["reference_value"],
                "current_value": delta_data["current_value"],
                "absolute_change": delta_data["absolute_change"],
                "percentage_change": delta_data["percentage_change"],
                "change_direction": delta_data["direction"],
                "improvement_status": (
                    "Improved"
                    if delta_data["improved"]
                    else "Degraded" if delta_data["direction"] != "no_change" else "Unchanged"
                ),
                "change_magnitude": delta_data["magnitude"],
                "higher_is_better": self._is_higher_better(metric_name),
            }

            detailed_comparison["table_data"].append(table_row)

        # Sort by absolute percentage change (largest changes first)
        detailed_comparison["table_data"].sort(key=lambda x: abs(x["percentage_change"]), reverse=True)

        return detailed_comparison

    def analyze_performance_comparison(
        self,
        y_true: np.ndarray,
        pred_ref: np.ndarray,
        pred_curr: np.ndarray,
        pred_ref_proba: Optional[np.ndarray] = None,
        pred_curr_proba: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        model_ref=None,
        model_curr=None,
        y_true_ref: Optional[np.ndarray] = None,
        y_true_curr: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive performance comparison analysis

        Args:
            y_true: True labels (DEPRECATED - use y_true_ref and y_true_curr instead)
            pred_ref: Reference model predictions
            pred_curr: Current model predictions
            pred_ref_proba: Reference model probabilities (optional)
            pred_curr_proba: Current model probabilities (optional)
            X: Feature matrix (optional, for advanced tests)
            model_ref: Reference model object (optional)
            model_curr: Current model object (optional)
            y_true_ref: Reference dataset true labels (preferred over y_true)
            y_true_curr: Current dataset true labels (preferred over y_true)

        Returns:
            Dictionary with complete performance comparison analysis including delta values
        """
        # Handle backward compatibility: if y_true_ref/y_true_curr not provided, use y_true
        if y_true_ref is None:
            y_true_ref = y_true
            logger.warning("Using deprecated y_true parameter for reference. Please use y_true_ref instead.")
        if y_true_curr is None:
            y_true_curr = y_true
            logger.warning("Using deprecated y_true parameter for current. Please use y_true_curr instead.")

        # Log array sizes for validation
        logger.info(
            f"Array sizes - y_true_ref: {y_true_ref.shape}, y_true_curr: {y_true_curr.shape}, pred_ref: {pred_ref.shape}, pred_curr: {pred_curr.shape}"
        )

        # Validate array size consistency
        if y_true_ref.shape[0] != pred_ref.shape[0]:
            error_msg = f"Size mismatch for reference: y_true_ref has {y_true_ref.shape[0]} samples but pred_ref has {pred_ref.shape[0]}"
            logger.error(error_msg)
            return {"error": "Array size mismatch", "details": error_msg}

        if y_true_curr.shape[0] != pred_curr.shape[0]:
            error_msg = f"Size mismatch for current: y_true_curr has {y_true_curr.shape[0]} samples but pred_curr has {pred_curr.shape[0]}"
            logger.error(error_msg)
            return {"error": "Array size mismatch", "details": error_msg}

        try:
            analysis_results = {
                "analysis_type": "performance_comparison",
                "summary": {},
                "metrics_comparison": {},
                "delta_metrics": {},  # New section for clear delta values
                "detailed_metric_comparison": {},  # Enhanced detailed comparison
                "statistical_tests": {},
                "effect_size_analysis": {},
                "prediction_drift": {},
                "overall_assessment": {},
            }

            # 1. Detect model type (regression vs classification)
            # Use y_true_ref for model type detection (both datasets should have same target type)
            is_regression = self._detect_regression_model(
                model_ref, model_curr, y_true_ref, pred_ref_proba, pred_curr_proba
            )

            # 1.5 Validate model type consistency
            is_valid, error_msg = self._validate_model_type_consistency(
                is_regression, y_true_ref, pred_ref, pred_curr, pred_ref_proba, pred_curr_proba
            )
            if not is_valid:
                logger.error(f"Model type validation failed: {error_msg}")
                return {
                    "error": "Model type validation failed",
                    "details": error_msg,
                    "detected_type": "regression" if is_regression else "classification",
                }

            logger.info("=" * 80)
            logger.info("CALCULATING METRICS")
            logger.info(f"Model Type Detected: {'REGRESSION' if is_regression else 'CLASSIFICATION'}")
            logger.info(f"y_true_ref: {y_true_ref.shape}, y_true_curr: {y_true_curr.shape}")
            logger.info(f"pred_ref: {pred_ref.shape}, pred_curr: {pred_curr.shape}")

            # 2. Calculate comprehensive metrics for both models (type-aware)
            # CRITICAL: Use y_true_ref with pred_ref and y_true_curr with pred_curr
            if is_regression:
                logger.info("Calling regression_metrics() for reference model...")
                ref_metrics = metrics_calculation_service.regression_metrics(y_true_ref, pred_ref)
                logger.info(f"Reference metrics calculated: {list(ref_metrics.keys())}")

                logger.info("Calling regression_metrics() for current model...")
                curr_metrics = metrics_calculation_service.regression_metrics(y_true_curr, pred_curr)
                logger.info(f"Current metrics calculated: {list(curr_metrics.keys())}")

                analysis_results["model_type"] = "regression"
                logger.info("✓ Successfully calculated REGRESSION metrics")
            else:
                logger.info("Calling classification_metrics() for reference model...")
                ref_metrics = metrics_calculation_service.classification_metrics(y_true_ref, pred_ref, pred_ref_proba)
                logger.info(f"Reference metrics calculated: {list(ref_metrics.keys())}")

                logger.info("Calling classification_metrics() for current model...")
                curr_metrics = metrics_calculation_service.classification_metrics(
                    y_true_curr, pred_curr, pred_curr_proba
                )
                logger.info(f"Current metrics calculated: {list(curr_metrics.keys())}")

                analysis_results["model_type"] = "classification"
                logger.info("✓ Successfully calculated CLASSIFICATION metrics")

            logger.info("=" * 80)

            # Validate metrics calculation
            if "error" in ref_metrics or "error" in curr_metrics:
                error_msg = (
                    f"Failed to calculate basic metrics. Model type: {analysis_results.get('model_type', 'unknown')}. "
                )
                if "error" in ref_metrics:
                    error_msg += f"Reference error: {ref_metrics['error']}. "
                if "error" in curr_metrics:
                    error_msg += f"Current error: {curr_metrics['error']}."
                logger.error(error_msg)
                return {
                    "error": "Failed to calculate basic metrics",
                    "details": error_msg,
                    "model_type": analysis_results.get("model_type", "unknown"),
                    "ref_metrics_error": ref_metrics.get("error"),
                    "curr_metrics_error": curr_metrics.get("error"),
                }

            # 3. Calculate clear delta metrics with improvement indicators
            delta_metrics = self._calculate_delta_metrics(ref_metrics, curr_metrics)

            # 4. Create detailed metric comparison for frontend display
            detailed_metric_comparison = self._create_detailed_metric_comparison(
                ref_metrics, curr_metrics, delta_metrics
            )

            # 5. Calculate traditional metrics differences and drift analysis (for backward compatibility)
            metrics_diff = metrics_calculation_service.calculate_metrics_difference(ref_metrics, curr_metrics)

            # 6. Performance degradation analysis
            degradation_analysis = metrics_calculation_service.performance_degradation_analysis(
                ref_metrics, curr_metrics
            )

            # 7. Statistical significance testing
            # CRITICAL: Statistical tests require both models to predict on the SAME dataset
            # If reference and current datasets have different sizes, skip statistical tests
            # as we're comparing performance on different data distributions
            statistical_results = {}
            if y_true_ref.shape[0] == y_true_curr.shape[0]:
                logger.info("Datasets have same size - running statistical significance tests")
                statistical_results = statistical_tests_service.run_all_tests(
                    y_true_curr, pred_ref, pred_curr, pred_ref_proba, pred_curr_proba, X, model_ref, model_curr
                )
            else:
                logger.warning(
                    f"Skipping statistical tests: Different dataset sizes (ref: {y_true_ref.shape[0]}, curr: {y_true_curr.shape[0]})"
                )
                logger.warning("Statistical tests require both models to predict on the same test set")
                statistical_results = {
                    "skipped": True,
                    "reason": "Different dataset sizes - statistical comparison not applicable",
                    "reference_size": int(y_true_ref.shape[0]),
                    "current_size": int(y_true_curr.shape[0]),
                    "note": "Performance metrics comparison is still valid and available in metrics_comparison section",
                }

            # 8. Effect size analysis (use appropriate primary metric)
            if is_regression:
                # For regression, use R2 or RMSE as primary metric
                primary_metric = ref_metrics.get("r2", ref_metrics.get("rmse", 0.0))
                ref_primary_array = np.array([primary_metric])
                curr_primary_metric = curr_metrics.get("r2", curr_metrics.get("rmse", 0.0))
                curr_primary_array = np.array([curr_primary_metric])
            else:
                # For classification, use accuracy
                ref_primary_array = np.array([ref_metrics["accuracy"]])
                curr_primary_array = np.array([curr_metrics["accuracy"]])

            effect_analysis = effect_size_service.comprehensive_effect_analysis(
                ref_primary_array, curr_primary_array, "Reference Model", "Current Model"
            )

            # 9. Prediction distribution drift (PSI) - only for classification
            prediction_drift = {}
            if not is_regression and pred_ref_proba is not None and pred_curr_proba is not None:
                prediction_drift = psi_service.calculate_prediction_psi(pred_ref_proba, pred_curr_proba)

            # 10. Overall performance assessment (enhanced to use delta metrics)
            overall_assessment = self._generate_overall_assessment(
                metrics_diff,
                degradation_analysis,
                statistical_results,
                effect_analysis,
                prediction_drift,
                delta_metrics,
            )

            # Compile results with enhanced structure
            analysis_results.update(
                {
                    "summary": {
                        "reference_samples": len(y_true_ref),
                        "current_samples": len(y_true_curr),
                        "metrics_analyzed": len(delta_metrics),
                        "metrics_improved": detailed_metric_comparison["summary"]["improved_metrics"],
                        "metrics_degraded": detailed_metric_comparison["summary"]["degraded_metrics"],
                        "metrics_unchanged": detailed_metric_comparison["summary"]["unchanged_metrics"],
                    },
                    "metrics_comparison": {
                        "reference_metrics": ref_metrics,
                        "current_metrics": curr_metrics,
                        "metrics_differences": metrics_diff,  # Legacy format for backward compatibility
                        "degradation_analysis": degradation_analysis,
                    },
                    "delta_metrics": delta_metrics,  # New clear delta values section
                    "detailed_metric_comparison": detailed_metric_comparison,  # Enhanced detailed comparison
                    "statistical_tests": statistical_results,
                    "effect_size_analysis": effect_analysis,
                    "prediction_drift": prediction_drift,
                    "overall_assessment": overall_assessment,
                }
            )

            return analysis_results

        except Exception as e:
            error_msg = f"Performance comparison analysis failed: {str(e)}"
            logger.error("=" * 80)
            logger.error(f"CRITICAL ERROR: {error_msg}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            logger.error("=" * 80)
            import traceback

            logger.error(traceback.format_exc())
            return {
                "error": "Performance comparison analysis failed",
                "details": error_msg,
                "error_type": type(e).__name__,
            }

    def _generate_overall_assessment(
        self,
        metrics_diff: Dict,
        degradation_analysis: Dict,
        statistical_results: Dict,
        effect_analysis: Dict,
        prediction_drift: Dict,
        delta_metrics: Dict,
    ) -> Dict[str, Any]:
        """Generate overall assessment of model performance comparison"""

        try:
            assessment = {
                "drift_detected": False,
                "drift_severity": "Minimal",
                "key_findings": [],
                "recommendations": [],
                "risk_level": "Low",
                "delta_summary": {"significant_changes": [], "largest_improvement": None, "largest_degradation": None},
            }

            # Analyze delta metrics for key insights
            significant_deltas = []
            improvements = []
            degradations = []

            for metric_name, delta_data in delta_metrics.items():
                if abs(delta_data["percentage_change"]) >= 5:  # 5% threshold
                    significant_deltas.append(
                        {
                            "metric": metric_name.replace("_", " ").title(),
                            "change": f"{delta_data['percentage_change']:+.1f}%",
                            "improved": delta_data["improved"],
                        }
                    )

                    if delta_data["improved"]:
                        improvements.append((metric_name, delta_data["percentage_change"]))
                    else:
                        degradations.append((metric_name, abs(delta_data["percentage_change"])))

            assessment["delta_summary"]["significant_changes"] = significant_deltas[:5]  # Top 5

            if improvements:
                best_improvement = max(improvements, key=lambda x: abs(x[1]))
                assessment["delta_summary"]["largest_improvement"] = {
                    "metric": best_improvement[0].replace("_", " ").title(),
                    "improvement": f"{best_improvement[1]:+.1f}%",
                }

            if degradations:
                worst_degradation = max(degradations, key=lambda x: x[1])
                assessment["delta_summary"]["largest_degradation"] = {
                    "metric": worst_degradation[0].replace("_", " ").title(),
                    "degradation": f"{worst_degradation[1]:.1f}%",
                }

            # Check for drift indicators (existing logic)
            drift_indicators = 0

            # 1. Metrics degradation check
            if "drift_summary" in metrics_diff:
                drift_severity = metrics_diff["drift_summary"]["drift_severity"]
                if drift_severity in ["High", "Medium"]:
                    drift_indicators += 1
                    assessment["key_findings"].append(f"Performance metrics show {drift_severity.lower()} drift")

            # 2. Statistical significance check
            if "summary" in statistical_results:
                significant_tests = statistical_results["summary"]["significant_tests"]
                if significant_tests > 0:
                    drift_indicators += 1
                    assessment["key_findings"].append(
                        f"{significant_tests} statistical tests show significant differences"
                    )

            # 3. Effect size check
            if "overall_assessment" in effect_analysis:
                effect_magnitude = effect_analysis["overall_assessment"]["magnitude"]
                if effect_magnitude in ["Medium", "Large"]:
                    drift_indicators += 1
                    assessment["key_findings"].append(f"{effect_magnitude} practical effect size detected")

            # 4. Prediction drift check
            if "psi" in prediction_drift:
                psi_level = prediction_drift["interpretation"]["level"]
                if psi_level in ["High", "Medium"]:
                    drift_indicators += 1
                    assessment["key_findings"].append(f"{psi_level} prediction distribution drift (PSI)")

            # 5. Critical degradation check
            if "summary" in degradation_analysis:
                critical_degradations = degradation_analysis["summary"]["critical_degradations"]
                if critical_degradations > 0:
                    drift_indicators += 2  # Weight this more heavily
                    assessment["key_findings"].append(f"{critical_degradations} critical performance degradations")

            # Determine overall drift status
            if drift_indicators >= 3:
                assessment["drift_detected"] = True
                assessment["drift_severity"] = "High"
                assessment["risk_level"] = "High"
            elif drift_indicators >= 2:
                assessment["drift_detected"] = True
                assessment["drift_severity"] = "Medium"
                assessment["risk_level"] = "Medium"
            elif drift_indicators >= 1:
                assessment["drift_detected"] = True
                assessment["drift_severity"] = "Low"
                assessment["risk_level"] = "Low"

            # Generate recommendations
            if assessment["drift_detected"]:
                if assessment["drift_severity"] == "High":
                    assessment["recommendations"].extend(
                        [
                            "Immediate model retraining recommended",
                            "Investigate root causes of performance degradation",
                            "Consider emergency rollback to previous model version",
                        ]
                    )
                elif assessment["drift_severity"] == "Medium":
                    assessment["recommendations"].extend(
                        [
                            "Schedule model retraining within next cycle",
                            "Monitor performance closely",
                            "Investigate data quality and distribution changes",
                        ]
                    )
                else:  # Low
                    assessment["recommendations"].extend(
                        [
                            "Continue monitoring model performance",
                            "Prepare for potential retraining",
                            "Document observed changes for trend analysis",
                        ]
                    )
            else:
                assessment["recommendations"].append("Model performance remains stable - continue regular monitoring")

            # Add no findings message if applicable
            if not assessment["key_findings"]:
                assessment["key_findings"].append("No significant performance changes detected")

            return assessment

        except Exception as e:
            return {"error": f"Overall assessment generation failed: {str(e)}"}

    def generate_executive_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary for performance comparison"""

        try:
            if "error" in analysis_results:
                return analysis_results

            overall_assessment = analysis_results["overall_assessment"]
            delta_metrics = analysis_results.get("delta_metrics", {})

            # Key performance changes using new delta metrics
            key_changes = []
            for metric_name, delta_data in delta_metrics.items():
                if abs(delta_data["percentage_change"]) >= 5:  # 5% threshold
                    key_changes.append(
                        {
                            "metric": metric_name.replace("_", " ").title(),
                            "change_percentage": f"{delta_data['percentage_change']:+.1f}%",
                            "direction": "improved" if delta_data["improved"] else "degraded",
                            "magnitude": delta_data["magnitude"],
                        }
                    )

            # Sort by magnitude of change
            key_changes.sort(
                key=lambda x: abs(float(x["change_percentage"].replace("%", "").replace("+", ""))), reverse=True
            )

            summary = {
                "model_drift_detected": overall_assessment["drift_detected"],
                "drift_severity": overall_assessment["drift_severity"],
                "risk_level": overall_assessment["risk_level"],
                "key_performance_changes": key_changes[:5],  # Top 5 changes
                "primary_concerns": overall_assessment["key_findings"][:3],  # Top 3 findings
                "immediate_actions": overall_assessment["recommendations"][:2],  # Top 2 recommendations
                "delta_summary": overall_assessment.get("delta_summary", {}),  # New delta summary
                "detailed_analysis_available": True,
            }

            return summary

        except Exception as e:
            return {"error": f"Executive summary generation failed: {str(e)}"}


# Service instance
performance_comparison_service = PerformanceComparisonService()
