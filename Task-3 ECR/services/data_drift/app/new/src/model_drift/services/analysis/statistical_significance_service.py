"""
Statistical Significance Service - Tab 3 Analysis
Orchestrates comprehensive statistical testing and hypothesis validation
"""

import warnings
from typing import Any, Dict, Optional

import numpy as np

from ..core.effect_size_service import effect_size_service
from ..core.metrics_calculation_service import metrics_calculation_service
from ..core.statistical_tests_service import statistical_tests_service

warnings.filterwarnings("ignore")


class StatisticalSignificanceService:
    """Service for statistical significance analysis and hypothesis testing (Tab 3)"""

    def __init__(self):
        self.alpha = 0.05  # Default significance level
        self.bonferroni_correction = True

    def _detect_model_type(self, pred_ref_proba, pred_curr_proba, model_ref, model_curr) -> str:
        """
        Detect if models are regression or classification

        Args:
            pred_ref_proba: Reference probabilities (None for regression)
            pred_curr_proba: Current probabilities (None for regression)
            model_ref: Reference model
            model_curr: Current model

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

    def analyze_statistical_significance(
        self,
        y_true: np.ndarray,
        pred_ref: np.ndarray,
        pred_curr: np.ndarray,
        pred_ref_proba: Optional[np.ndarray] = None,
        pred_curr_proba: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        model_ref=None,
        model_curr=None,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical significance analysis

        Args:
            y_true: True labels
            pred_ref: Reference model predictions
            pred_curr: Current model predictions
            pred_ref_proba: Reference model probabilities (optional)
            pred_curr_proba: Current model probabilities (optional)
            X: Feature matrix (optional)
            model_ref: Reference model object (optional)
            model_curr: Current model object (optional)
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary with comprehensive statistical analysis
        """
        try:
            self.alpha = alpha

            # Detect model type
            model_type = self._detect_model_type(pred_ref_proba, pred_curr_proba, model_ref, model_curr)

            analysis_results = {
                "analysis_type": "statistical_significance",
                "model_type": model_type,
                "hypothesis_testing": {},
                "effect_size_analysis": {},
                "power_analysis": {},
                "multiple_comparisons": {},
                "interpretation": {},
                "recommendations": {},
            }

            # 1. Comprehensive hypothesis testing
            hypothesis_results = self._perform_hypothesis_testing(
                y_true, pred_ref, pred_curr, pred_ref_proba, pred_curr_proba, X, model_ref, model_curr
            )
            analysis_results["hypothesis_testing"] = hypothesis_results

            # 2. Effect size analysis (type-aware)
            effect_size_results = self._perform_effect_size_analysis(y_true, pred_ref, pred_curr, model_type)
            analysis_results["effect_size_analysis"] = effect_size_results

            # 3. Power analysis
            power_results = self._perform_power_analysis(y_true, pred_ref, pred_curr)
            analysis_results["power_analysis"] = power_results

            # 4. Multiple comparisons correction
            multiple_comp_results = self._apply_multiple_comparisons_correction(hypothesis_results)
            analysis_results["multiple_comparisons"] = multiple_comp_results

            # 5. Statistical interpretation
            interpretation = self._generate_statistical_interpretation(
                hypothesis_results, effect_size_results, multiple_comp_results
            )
            analysis_results["interpretation"] = interpretation

            # 6. Recommendations
            recommendations = self._generate_statistical_recommendations(
                interpretation, effect_size_results, power_results
            )
            analysis_results["recommendations"] = recommendations

            return analysis_results

        except Exception as e:
            return {"error": f"Statistical significance analysis failed: {str(e)}"}

    def _perform_hypothesis_testing(
        self,
        y_true: np.ndarray,
        pred_ref: np.ndarray,
        pred_curr: np.ndarray,
        pred_ref_proba: Optional[np.ndarray],
        pred_curr_proba: Optional[np.ndarray],
        X: Optional[np.ndarray],
        model_ref,
        model_curr,
    ) -> Dict[str, Any]:
        """Perform comprehensive hypothesis testing"""

        try:
            # Get all statistical tests
            all_tests = statistical_tests_service.run_all_tests(
                y_true, pred_ref, pred_curr, pred_ref_proba, pred_curr_proba, X, model_ref, model_curr
            )

            if "error" in all_tests:
                return all_tests

            # Format test results for frontend with proper null handling
            formatted_tests = {}

            for test_name, test_result in all_tests["tests"].items():
                if "error" not in test_result:
                    # Get test statistic with proper fallback to null instead of "N/A"
                    test_statistic = test_result.get("test_statistic")
                    if test_statistic is None:
                        test_statistic = test_result.get("statistic")  # McNemar uses "statistic"
                    if test_statistic is None:
                        test_statistic = test_result.get("z_score")
                    if test_statistic is None:
                        test_statistic = test_result.get("f_statistic")
                    if test_statistic is None:
                        test_statistic = test_result.get("t_statistic")  # t-test uses "t_statistic"

                    formatted_tests[test_name] = {
                        "test_name": test_result["test_name"],
                        "p_value": test_result["p_value"],
                        "significant": test_result["significant"],
                        "interpretation": test_result["interpretation"],
                        "test_statistic": test_statistic,  # Now properly null instead of "N/A"
                        "method_description": self._get_test_description(test_name),
                    }

            # Summary statistics
            total_tests = len(formatted_tests)
            significant_tests = sum(1 for test in formatted_tests.values() if test["significant"])

            hypothesis_summary = {
                "total_tests_conducted": total_tests,
                "significant_tests": significant_tests,
                "significance_rate": significant_tests / total_tests if total_tests > 0 else 0,
                "alpha_level": self.alpha,
                "null_hypothesis": "No significant difference between model performances",
                "alternative_hypothesis": "Significant difference exists between model performances",
            }

            return {"individual_tests": formatted_tests, "summary": hypothesis_summary, "raw_results": all_tests}

        except Exception as e:
            return {"error": f"Hypothesis testing failed: {str(e)}"}

    def _perform_effect_size_analysis(
        self, y_true: np.ndarray, pred_ref: np.ndarray, pred_curr: np.ndarray, model_type: str
    ) -> Dict[str, Any]:
        """Perform comprehensive effect size analysis (type-aware)"""

        try:
            # Calculate metrics based on model type
            if model_type == "regression":
                ref_metrics = metrics_calculation_service.regression_metrics(y_true, pred_ref)
                curr_metrics = metrics_calculation_service.regression_metrics(y_true, pred_curr)
                key_metrics = ["r2", "rmse", "mae"]
                primary_metric_name = "r2"
            else:
                ref_metrics = metrics_calculation_service.classification_metrics(y_true, pred_ref)
                curr_metrics = metrics_calculation_service.classification_metrics(y_true, pred_curr)
                key_metrics = ["precision", "recall", "f1_score"]
                primary_metric_name = "accuracy"

            if "error" in ref_metrics or "error" in curr_metrics:
                return {"error": "Could not calculate metrics for effect size analysis"}

            # Effect size for primary metric (accuracy for classification, r2 for regression)
            if primary_metric_name in ref_metrics and primary_metric_name in curr_metrics:
                ref_primary = np.array([ref_metrics[primary_metric_name]])
                curr_primary = np.array([curr_metrics[primary_metric_name]])

                primary_effect = effect_size_service.comprehensive_effect_analysis(
                    ref_primary, curr_primary, "Reference Model", "Current Model"
                )
            else:
                primary_effect = {"error": f"Primary metric '{primary_metric_name}' not available"}

            # Effect size for other key metrics
            effect_sizes = {}

            for metric in key_metrics:
                if metric in ref_metrics and metric in curr_metrics:
                    ref_val = np.array([ref_metrics[metric]])
                    curr_val = np.array([curr_metrics[metric]])

                    metric_effect = effect_size_service.cohens_d(ref_val, curr_val)
                    effect_sizes[metric] = {
                        "cohens_d": metric_effect["cohens_d"],
                        "interpretation": metric_effect["interpretation"],
                        "practical_significance": metric_effect["interpretation"] in ["Medium", "Large"],
                    }

            # Overall effect size summary
            all_effect_sizes = []
            if "error" not in primary_effect and primary_effect.get("overall_assessment", {}).get(
                "average_effect_size"
            ):
                all_effect_sizes.append(primary_effect["overall_assessment"]["average_effect_size"])
            all_effect_sizes.extend([es["cohens_d"] for es in effect_sizes.values() if "cohens_d" in es])

            if all_effect_sizes:
                mean_effect_size = np.mean([abs(es) for es in all_effect_sizes])
                if mean_effect_size >= 0.8:
                    overall_magnitude = "Large"
                elif mean_effect_size >= 0.5:
                    overall_magnitude = "Medium"
                elif mean_effect_size >= 0.2:
                    overall_magnitude = "Small"
                else:
                    overall_magnitude = "Negligible"
            else:
                overall_magnitude = "Unknown"

            return {
                "primary_metric_effect_size": primary_effect,
                "primary_metric_name": primary_metric_name,
                "model_type": model_type,
                "metric_effect_sizes": effect_sizes,
                "overall_magnitude": overall_magnitude,
                "practical_significance": overall_magnitude in ["Medium", "Large"],
                "effect_size_interpretation": {
                    "small": "0.2 - Small practical difference",
                    "medium": "0.5 - Medium practical difference",
                    "large": "0.8 - Large practical difference",
                },
            }

        except Exception as e:
            return {"error": f"Effect size analysis failed: {str(e)}"}

    def _perform_power_analysis(
        self, y_true: np.ndarray, pred_ref: np.ndarray, pred_curr: np.ndarray
    ) -> Dict[str, Any]:
        """Perform statistical power analysis"""

        try:
            n = len(y_true)

            # Calculate observed effect size
            ref_accuracy = np.mean(pred_ref == y_true)
            curr_accuracy = np.mean(pred_curr == y_true)

            # Pooled standard deviation for effect size
            pooled_std = np.sqrt((np.var(pred_ref == y_true) + np.var(pred_curr == y_true)) / 2)
            observed_effect_size = abs(ref_accuracy - curr_accuracy) / pooled_std if pooled_std > 0 else 0

            # Estimate statistical power using normal approximation
            # This is a simplified power calculation
            from scipy.stats import norm

            z_alpha = norm.ppf(1 - self.alpha / 2)  # Two-tailed test
            z_beta = norm.ppf(0.8)  # 80% power

            # Calculate minimum detectable effect size
            min_detectable_effect = (z_alpha + z_beta) * np.sqrt(2 / n)

            # Estimate power for observed effect
            if observed_effect_size > 0:
                z_power = observed_effect_size * np.sqrt(n / 2) - z_alpha
                estimated_power = norm.cdf(z_power)
            else:
                estimated_power = self.alpha  # No effect, power equals Type I error rate

            # Power interpretation
            if estimated_power >= 0.8:
                power_level = "Adequate"
            elif estimated_power >= 0.6:
                power_level = "Marginal"
            else:
                power_level = "Low"

            return {
                "sample_size": int(n),
                "observed_effect_size": float(observed_effect_size),
                "estimated_power": float(estimated_power),
                "power_level": power_level,
                "minimum_detectable_effect": float(min_detectable_effect),
                "recommendations": {
                    "sample_size_adequate": n >= 100,
                    "power_adequate": estimated_power >= 0.8,
                    "effect_detectable": observed_effect_size >= min_detectable_effect,
                },
            }

        except Exception as e:
            return {"error": f"Power analysis failed: {str(e)}"}

    def _apply_multiple_comparisons_correction(self, hypothesis_results: Dict) -> Dict[str, Any]:
        """Apply multiple comparisons correction (Bonferroni)"""

        try:
            if "individual_tests" not in hypothesis_results:
                return {"error": "No test results to correct"}

            tests = hypothesis_results["individual_tests"]
            n_tests = len(tests)

            if n_tests <= 1:
                return {
                    "correction_applied": False,
                    "reason": "Only one test conducted, no correction needed",
                    "corrected_tests": tests,
                }

            # Apply Bonferroni correction
            bonferroni_alpha = self.alpha / n_tests

            corrected_tests = {}
            significant_after_correction = 0

            for test_name, test_data in tests.items():
                corrected_test = test_data.copy()
                corrected_test["corrected_alpha"] = bonferroni_alpha
                corrected_test["significant_after_correction"] = test_data["p_value"] < bonferroni_alpha
                corrected_test["bonferroni_adjusted_p"] = min(test_data["p_value"] * n_tests, 1.0)

                if corrected_test["significant_after_correction"]:
                    significant_after_correction += 1

                corrected_tests[test_name] = corrected_test

            return {
                "correction_applied": True,
                "correction_method": "Bonferroni",
                "original_alpha": self.alpha,
                "corrected_alpha": bonferroni_alpha,
                "number_of_tests": n_tests,
                "significant_before_correction": hypothesis_results["summary"]["significant_tests"],
                "significant_after_correction": significant_after_correction,
                "corrected_tests": corrected_tests,
            }

        except Exception as e:
            return {"error": f"Multiple comparisons correction failed: {str(e)}"}

    def _generate_statistical_interpretation(
        self, hypothesis_results: Dict, effect_size_results: Dict, multiple_comp_results: Dict
    ) -> Dict[str, Any]:
        """Generate comprehensive statistical interpretation"""

        try:
            interpretation = {
                "statistical_significance": "Not determined",
                "practical_significance": "Not determined",
                "overall_conclusion": "",
                "confidence_level": f"{(1-self.alpha)*100}%",
                "key_findings": [],
            }

            # Statistical significance interpretation
            if "summary" in hypothesis_results:
                significant_tests = hypothesis_results["summary"]["significant_tests"]
                total_tests = hypothesis_results["summary"]["total_tests_conducted"]

                if multiple_comp_results.get("correction_applied", False):
                    corrected_significant = multiple_comp_results["significant_after_correction"]
                    if corrected_significant > 0:
                        interpretation["statistical_significance"] = "Significant (after correction)"
                        interpretation["key_findings"].append(
                            f"{corrected_significant}/{total_tests} tests remain significant after Bonferroni correction"
                        )
                    else:
                        interpretation["statistical_significance"] = "Not significant (after correction)"
                        interpretation["key_findings"].append(
                            "No tests remain significant after multiple comparisons correction"
                        )
                else:
                    if significant_tests > 0:
                        interpretation["statistical_significance"] = "Significant"
                        interpretation["key_findings"].append(
                            f"{significant_tests}/{total_tests} tests show statistical significance"
                        )
                    else:
                        interpretation["statistical_significance"] = "Not significant"
                        interpretation["key_findings"].append("No tests show statistical significance")

            # Practical significance interpretation
            if "overall_magnitude" in effect_size_results:
                magnitude = effect_size_results["overall_magnitude"]
                interpretation["practical_significance"] = magnitude

                if magnitude in ["Medium", "Large"]:
                    interpretation["key_findings"].append(f"{magnitude.lower()} practical effect size detected")
                else:
                    interpretation["key_findings"].append(f"{magnitude.lower()} practical effect size")

            # Overall conclusion
            stat_sig = interpretation["statistical_significance"]
            prac_sig = interpretation["practical_significance"]

            if "Significant" in stat_sig and prac_sig in ["Medium", "Large"]:
                interpretation["overall_conclusion"] = (
                    "Both statistical and practical significance detected - models differ meaningfully"
                )
            elif "Significant" in stat_sig and prac_sig in ["Small", "Negligible"]:
                interpretation["overall_conclusion"] = (
                    "Statistical significance without practical importance - differences may not be meaningful"
                )
            elif "Not significant" in stat_sig and prac_sig in ["Medium", "Large"]:
                interpretation["overall_conclusion"] = (
                    "Practical differences exist but lack statistical evidence - may need larger sample"
                )
            else:
                interpretation["overall_conclusion"] = "No meaningful differences detected between models"

            return interpretation

        except Exception as e:
            return {"error": f"Statistical interpretation failed: {str(e)}"}

    def _generate_statistical_recommendations(
        self, interpretation: Dict, effect_size_results: Dict, power_results: Dict
    ) -> Dict[str, Any]:
        """Generate statistical recommendations based on analysis"""

        try:
            recommendations = {"immediate_actions": [], "methodological_considerations": [], "future_analyses": []}

            # Based on statistical significance
            stat_sig = interpretation.get("statistical_significance", "")
            if "Significant" in stat_sig:
                recommendations["immediate_actions"].append("Investigate root causes of detected differences")
                recommendations["immediate_actions"].append("Consider model retraining or intervention")
            else:
                recommendations["immediate_actions"].append("Continue monitoring - no immediate action required")

            # Based on practical significance
            prac_sig = interpretation.get("practical_significance", "")
            if prac_sig in ["Medium", "Large"]:
                recommendations["immediate_actions"].append("Evaluate business impact of observed differences")

            # Based on power analysis
            if "power_level" in power_results:
                power_level = power_results["power_level"]
                if power_level == "Low":
                    recommendations["methodological_considerations"].append(
                        "Consider increasing sample size for better statistical power"
                    )
                    recommendations["future_analyses"].append("Repeat analysis with larger dataset when available")
                elif power_level == "Marginal":
                    recommendations["methodological_considerations"].append(
                        "Results should be interpreted cautiously due to marginal statistical power"
                    )

            # General methodological recommendations
            recommendations["methodological_considerations"].extend(
                [
                    "Consider effect size alongside statistical significance",
                    "Validate findings with independent dataset when possible",
                    "Monitor for Type I error inflation with multiple testing",
                ]
            )

            recommendations["future_analyses"].extend(
                [
                    "Track statistical metrics over time for trend analysis",
                    "Consider non-parametric alternatives if assumptions are violated",
                    "Implement continuous monitoring for early drift detection",
                ]
            )

            return recommendations

        except Exception as e:
            return {"error": f"Statistical recommendations generation failed: {str(e)}"}

    def _get_test_description(self, test_name: str) -> str:
        """Get description for statistical test"""

        descriptions = {
            "mcnemar": "Tests significance of differences in binary classification accuracy using contingency table analysis",
            "delong": "Compares ROC curves and AUC values for statistical significance in probabilistic classification",
            "bootstrap": "Non-parametric test using bootstrap resampling to estimate confidence intervals",
            "five_two_cv": "Robust cross-validation test that accounts for training set variance in model comparison",
            "paired_t_test": "Classical parametric test for comparing paired observations of model performance",
            "diebold_mariano": "Tests forecast accuracy differences, primarily used for regression models",
        }

        return descriptions.get(test_name, "Statistical significance test for model comparison")


# Service instance
statistical_significance_service = StatisticalSignificanceService()
