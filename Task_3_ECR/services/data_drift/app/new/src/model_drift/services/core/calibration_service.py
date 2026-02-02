"""
Calibration Service - Implements model calibration metrics
Following research specifications for Brier Score, ECE, MCE, and Confidence Entropy
"""

import warnings
from typing import Any, Dict

import numpy as np

warnings.filterwarnings("ignore")


class CalibrationService:
    """Service for calculating model calibration metrics and confidence analysis"""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def brier_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calculate Brier Score - measures accuracy and calibration

        Formula: BS = (1/n) * Σ(p_i - o_i)²
        where p_i is predicted probability and o_i is binary outcome

        Args:
            y_true: True binary labels (0 or 1)
            y_prob: Predicted probabilities

        Returns:
            Brier score (lower is better)
        """
        try:
            return float(np.mean((y_prob - y_true) ** 2))
        except Exception as e:
            raise ValueError(f"Brier score calculation failed: {str(e)}")

    def expected_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Expected Calibration Error (ECE)

        ECE measures the average difference between predicted confidence and actual accuracy
        across equally-sized bins

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities

        Returns:
            Dictionary with ECE value and bin details
        """
        try:
            # Create bins
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece = 0
            bin_details = []

            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find samples in this bin
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()

                if prop_in_bin > 0:
                    # Calculate accuracy and confidence for this bin
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()

                    # Calibration error for this bin
                    bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                    ece += bin_error * prop_in_bin

                    bin_details.append(
                        {
                            "bin_range": f"{bin_lower:.1f}-{bin_upper:.1f}",
                            "count": int(in_bin.sum()),
                            "proportion": float(prop_in_bin),
                            "avg_confidence": float(avg_confidence_in_bin),
                            "accuracy": float(accuracy_in_bin),
                            "calibration_error": float(bin_error),
                        }
                    )
                else:
                    bin_details.append(
                        {
                            "bin_range": f"{bin_lower:.1f}-{bin_upper:.1f}",
                            "count": 0,
                            "proportion": 0.0,
                            "avg_confidence": 0.0,
                            "accuracy": 0.0,
                            "calibration_error": 0.0,
                        }
                    )

            return {"ece": float(ece), "n_bins": self.n_bins, "bin_details": bin_details}

        except Exception as e:
            raise ValueError(f"ECE calculation failed: {str(e)}")

    def maximum_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Maximum Calibration Error (MCE)

        MCE is the worst-case calibration error across all bins

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities

        Returns:
            Dictionary with MCE value and worst bin details
        """
        try:
            ece_result = self.expected_calibration_error(y_true, y_prob)

            # Find maximum calibration error across bins
            max_error = 0
            worst_bin = None

            for bin_detail in ece_result["bin_details"]:
                if bin_detail["calibration_error"] > max_error:
                    max_error = bin_detail["calibration_error"]
                    worst_bin = bin_detail

            return {"mce": float(max_error), "worst_bin": worst_bin, "n_bins": self.n_bins}

        except Exception as e:
            raise ValueError(f"MCE calculation failed: {str(e)}")

    def confidence_entropy(self, y_prob: np.ndarray) -> float:
        """
        Calculate Confidence Entropy - measures uncertainty in predictions

        Lower entropy indicates more certainty in predictions

        Args:
            y_prob: Predicted probabilities

        Returns:
            Confidence entropy value
        """
        try:
            # Calculate entropy: -Σ(p*log(p) + (1-p)*log(1-p))
            # Add small epsilon to avoid log(0)
            epsilon = 1e-15
            p = np.clip(y_prob, epsilon, 1 - epsilon)

            entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
            return float(np.mean(entropy))

        except Exception as e:
            raise ValueError(f"Confidence entropy calculation failed: {str(e)}")

    def calibration_curve_data(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Generate data for calibration curve plotting

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities

        Returns:
            Dictionary with curve data points
        """
        try:
            ece_result = self.expected_calibration_error(y_true, y_prob)

            # Extract data for plotting
            mean_predicted_probs = []
            fraction_positives = []
            bin_counts = []

            for bin_detail in ece_result["bin_details"]:
                if bin_detail["count"] > 0:
                    mean_predicted_probs.append(bin_detail["avg_confidence"])
                    fraction_positives.append(bin_detail["accuracy"])
                    bin_counts.append(bin_detail["count"])

            return {
                "mean_predicted_probability": mean_predicted_probs,
                "fraction_of_positives": fraction_positives,
                "bin_counts": bin_counts,
                "perfect_calibration": mean_predicted_probs,  # For reference line
                "n_bins": self.n_bins,
            }

        except Exception as e:
            raise ValueError(f"Calibration curve data generation failed: {str(e)}")

    def confidence_distribution(self, y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Calculate confidence score distribution for histogram plotting with adaptive binning

        Args:
            y_prob: Predicted probabilities

        Returns:
            Dictionary with distribution data
        """
        try:
            # Flatten array if needed
            if len(y_prob.shape) > 1:
                y_prob = y_prob.flatten()

            # Check for extreme distributions
            unique_values = np.unique(y_prob)
            prob_range = np.max(y_prob) - np.min(y_prob)

            # Use adaptive binning for highly skewed distributions
            use_adaptive = len(unique_values) < self.n_bins or prob_range < 0.2

            if use_adaptive and len(unique_values) > 3:
                # For highly skewed distributions, use percentile-based binning
                try:
                    percentiles = np.linspace(0, 100, self.n_bins + 1)
                    bin_edges = np.percentile(y_prob, percentiles)
                    # Ensure bins are unique
                    bin_edges = np.unique(bin_edges)
                    # Add endpoints if needed
                    if bin_edges[0] > 0:
                        bin_edges = np.insert(bin_edges, 0, 0)
                    if bin_edges[-1] < 1:
                        bin_edges = np.append(bin_edges, 1)

                    binning_method = "adaptive_percentile"
                except:
                    # Fallback to regular binning if percentile fails
                    bin_edges = np.linspace(0, 1, self.n_bins + 1)
                    binning_method = "linear_fallback"
            else:
                # Use regular binning for well-distributed probabilities
                bin_edges = np.linspace(0, 1, self.n_bins + 1)
                binning_method = "linear"

            # Calculate histogram
            bin_counts, _ = np.histogram(y_prob, bins=bin_edges)

            # Create bin labels with better formatting
            bin_labels = []
            for i in range(len(bin_edges) - 1):
                if binning_method == "adaptive_percentile":
                    bin_labels.append(f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}")
                else:
                    bin_labels.append(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}")

            # Calculate statistics
            total_samples = len(y_prob)
            bin_proportions = (bin_counts / total_samples).tolist() if total_samples > 0 else []

            # Quality metrics
            non_empty_bins = np.sum(bin_counts > 0)
            distribution_entropy = -np.sum([p * np.log(p) for p in bin_proportions if p > 0])

            return {
                "bin_labels": bin_labels,
                "bin_counts": bin_counts.tolist(),
                "bin_proportions": bin_proportions,
                "bin_edges": bin_edges.tolist(),
                "total_samples": total_samples,
                "n_bins": len(bin_labels),
                "binning_method": binning_method,
                "distribution_stats": {
                    "non_empty_bins": int(non_empty_bins),
                    "distribution_entropy": float(distribution_entropy),
                    "probability_range": float(prob_range),
                    "unique_values": len(unique_values),
                    "mean_probability": float(np.mean(y_prob)),
                    "std_probability": float(np.std(y_prob)),
                },
            }

        except Exception as e:
            raise ValueError(f"Confidence distribution calculation failed: {str(e)}")

    def comprehensive_calibration_analysis(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive calibration analysis

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities

        Returns:
            Dictionary with all calibration metrics and analysis
        """
        try:
            # Calculate all calibration metrics
            brier = self.brier_score(y_true, y_prob)
            ece_result = self.expected_calibration_error(y_true, y_prob)
            mce_result = self.maximum_calibration_error(y_true, y_prob)
            conf_entropy = self.confidence_entropy(y_prob)

            # Generate curve and distribution data
            curve_data = self.calibration_curve_data(y_true, y_prob)
            dist_data = self.confidence_distribution(y_prob)

            # Assess calibration quality
            ece_value = ece_result["ece"]
            if ece_value <= 0.05:
                calibration_quality = "Excellent"
            elif ece_value <= 0.10:
                calibration_quality = "Good"
            elif ece_value <= 0.15:
                calibration_quality = "Fair"
            else:
                calibration_quality = "Poor"

            return {
                "metrics": {
                    "brier_score": brier,
                    "expected_calibration_error": ece_value,
                    "maximum_calibration_error": mce_result["mce"],
                    "confidence_entropy": conf_entropy,
                },
                "calibration_assessment": {
                    "quality": calibration_quality,
                    "ece_threshold_excellent": 0.05,
                    "ece_threshold_good": 0.10,
                    "ece_threshold_fair": 0.15,
                },
                "detailed_results": {
                    "ece_details": ece_result,
                    "mce_details": mce_result,
                    "calibration_curve": curve_data,
                    "confidence_distribution": dist_data,
                },
            }

        except Exception as e:
            return {"error": f"Comprehensive calibration analysis failed: {str(e)}"}

    def compare_calibrations(
        self, y_true: np.ndarray, y_prob_ref: np.ndarray, y_prob_curr: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare calibration between reference and current models

        Args:
            y_true: True binary labels
            y_prob_ref: Reference model probabilities
            y_prob_curr: Current model probabilities

        Returns:
            Dictionary with calibration comparison
        """
        try:
            # Analyze both models
            ref_analysis = self.comprehensive_calibration_analysis(y_true, y_prob_ref)
            curr_analysis = self.comprehensive_calibration_analysis(y_true, y_prob_curr)

            # Calculate differences
            ref_metrics = ref_analysis["metrics"]
            curr_metrics = curr_analysis["metrics"]

            metric_changes = {}
            for metric in ref_metrics:
                if metric in curr_metrics:
                    change = curr_metrics[metric] - ref_metrics[metric]
                    metric_changes[f"{metric}_change"] = {
                        "reference": ref_metrics[metric],
                        "current": curr_metrics[metric],
                        "absolute_change": change,
                        "relative_change": (change / ref_metrics[metric] * 100) if ref_metrics[metric] != 0 else 0,
                    }

            # Determine overall calibration drift
            ece_change = abs(metric_changes["expected_calibration_error_change"]["absolute_change"])
            if ece_change >= 0.05:
                calibration_drift = "High"
            elif ece_change >= 0.02:
                calibration_drift = "Medium"
            elif ece_change >= 0.01:
                calibration_drift = "Low"
            else:
                calibration_drift = "Minimal"

            return {
                "reference_analysis": ref_analysis,
                "current_analysis": curr_analysis,
                "metric_changes": metric_changes,
                "calibration_drift": {
                    "severity": calibration_drift,
                    "ece_change": ece_change,
                    "interpretation": f"Calibration has {calibration_drift.lower()} drift",
                },
            }

        except Exception as e:
            return {"error": f"Calibration comparison failed: {str(e)}"}


# Service instance
calibration_service = CalibrationService()
