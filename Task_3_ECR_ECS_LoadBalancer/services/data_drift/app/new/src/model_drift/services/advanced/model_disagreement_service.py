"""
Model Disagreement Service - Implements model disagreement analysis
Following research specifications for the Degradation Metrics tab - Model Disagreement sub-tab
"""

import warnings
from typing import Any, Dict, List

import numpy as np
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")


class ModelDisagreementService:
    """Service for analyzing disagreement between reference and current models"""

    def __init__(self):
        pass

    def prediction_scatter_data(
        self, pred_ref: np.ndarray, pred_curr: np.ndarray, sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate scatter plot data for model prediction comparison

        Args:
            pred_ref: Reference model predictions (probabilities)
            pred_curr: Current model predictions (probabilities)
            sample_size: Number of points to sample for visualization

        Returns:
            Dictionary with scatter plot data
        """
        try:
            n_samples = len(pred_ref)

            # Sample data if too large for frontend visualization
            if n_samples > sample_size:
                indices = np.random.choice(n_samples, size=sample_size, replace=False)
                pred_ref_sample = pred_ref[indices]
                pred_curr_sample = pred_curr[indices]
            else:
                pred_ref_sample = pred_ref
                pred_curr_sample = pred_curr

            # Create scatter plot data
            scatter_data = []
            for i in range(len(pred_ref_sample)):
                scatter_data.append(
                    {
                        "reference_prediction": float(pred_ref_sample[i]),
                        "current_prediction": float(pred_curr_sample[i]),
                    }
                )

            return {
                "scatter_data": scatter_data,
                "total_samples": int(n_samples),
                "displayed_samples": len(scatter_data),
                "sampled": n_samples > sample_size,
            }

        except Exception as e:
            return {"error": f"Scatter plot data generation failed: {str(e)}"}

    def disagreement_analysis(self, pred_ref: np.ndarray, pred_curr: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive disagreement statistics between models

        Args:
            pred_ref: Reference model predictions (probabilities)
            pred_curr: Current model predictions (probabilities)

        Returns:
            Dictionary with disagreement statistics
        """
        try:
            # Calculate differences
            differences = pred_ref - pred_curr
            abs_differences = np.abs(differences)

            # Basic statistics
            mean_abs_diff = float(np.mean(abs_differences))
            max_diff = float(np.max(abs_differences))
            std_diff = float(np.std(differences))

            # Pearson correlation
            correlation, p_value = pearsonr(pred_ref, pred_curr)

            # Additional metrics
            median_abs_diff = float(np.median(abs_differences))
            percentile_95_diff = float(np.percentile(abs_differences, 95))

            # Count extreme disagreements (>0.3 difference)
            extreme_disagreements = np.sum(abs_differences > 0.3)
            extreme_disagreement_rate = float(extreme_disagreements / len(pred_ref))

            # Agreement levels
            high_agreement = np.sum(abs_differences <= 0.05)  # Within 5%
            medium_agreement = np.sum((abs_differences > 0.05) & (abs_differences <= 0.15))  # 5-15%
            low_agreement = np.sum(abs_differences > 0.15)  # >15%

            return {
                "mean_absolute_difference": mean_abs_diff,
                "maximum_difference": max_diff,
                "standard_deviation": std_diff,
                "median_absolute_difference": median_abs_diff,
                "percentile_95_difference": percentile_95_diff,
                "pearson_correlation": float(correlation),
                "correlation_p_value": float(p_value),
                "extreme_disagreement_count": int(extreme_disagreements),
                "extreme_disagreement_rate": extreme_disagreement_rate,
                "agreement_levels": {
                    "high_agreement_count": int(high_agreement),
                    "medium_agreement_count": int(medium_agreement),
                    "low_agreement_count": int(low_agreement),
                    "high_agreement_rate": float(high_agreement / len(pred_ref)),
                    "medium_agreement_rate": float(medium_agreement / len(pred_ref)),
                    "low_agreement_rate": float(low_agreement / len(pred_ref)),
                },
            }

        except Exception as e:
            return {"error": f"Disagreement analysis failed: {str(e)}"}

    def decision_threshold_analysis(
        self, pred_ref: np.ndarray, pred_curr: np.ndarray, thresholds: List[float] = [0.3, 0.5, 0.7]
    ) -> Dict[str, Any]:
        """
        Analyze model agreement at different decision thresholds

        Args:
            pred_ref: Reference model predictions (probabilities)
            pred_curr: Current model predictions (probabilities)
            thresholds: List of decision thresholds to analyze

        Returns:
            Dictionary with threshold analysis results
        """
        try:
            threshold_results = []

            for threshold in thresholds:
                # Convert probabilities to binary decisions
                decisions_ref = (pred_ref >= threshold).astype(int)
                decisions_curr = (pred_curr >= threshold).astype(int)

                # Calculate agreement
                agreements = decisions_ref == decisions_curr
                agreement_count = int(np.sum(agreements))
                disagreement_count = len(agreements) - agreement_count

                agreement_rate = float(agreement_count / len(agreements))
                disagreement_rate = 1.0 - agreement_rate

                # Detailed disagreement analysis
                ref_pos_curr_neg = int(np.sum((decisions_ref == 1) & (decisions_curr == 0)))
                ref_neg_curr_pos = int(np.sum((decisions_ref == 0) & (decisions_curr == 1)))

                threshold_results.append(
                    {
                        "threshold": float(threshold),
                        "agreement_count": agreement_count,
                        "disagreement_count": disagreement_count,
                        "agreement_percentage": float(agreement_rate * 100),
                        "disagreement_percentage": float(disagreement_rate * 100),
                        "ref_positive_curr_negative": ref_pos_curr_neg,
                        "ref_negative_curr_positive": ref_neg_curr_pos,
                    }
                )

            # Find optimal threshold (highest agreement)
            optimal_threshold_info = max(threshold_results, key=lambda x: x["agreement_percentage"])

            return {
                "threshold_analysis": threshold_results,
                "optimal_threshold": {
                    "threshold": optimal_threshold_info["threshold"],
                    "agreement_rate": optimal_threshold_info["agreement_percentage"],
                },
                "summary": {
                    "best_agreement_threshold": optimal_threshold_info["threshold"],
                    "best_agreement_rate": optimal_threshold_info["agreement_percentage"],
                    "average_agreement_rate": float(np.mean([t["agreement_percentage"] for t in threshold_results])),
                },
            }

        except Exception as e:
            return {"error": f"Decision threshold analysis failed: {str(e)}"}

    def disagreement_heatmap_data(self, pred_ref: np.ndarray, pred_curr: np.ndarray, bins: int = 10) -> Dict[str, Any]:
        """
        Generate heatmap data for model disagreement visualization

        Args:
            pred_ref: Reference model predictions
            pred_curr: Current model predictions
            bins: Number of bins for heatmap

        Returns:
            Dictionary with heatmap data
        """
        try:
            # Create 2D histogram
            hist, x_edges, y_edges = np.histogram2d(pred_ref, pred_curr, bins=bins, range=[[0, 1], [0, 1]])

            # Convert to percentages
            hist_percent = hist / np.sum(hist) * 100

            # Create bin labels
            x_labels = [f"{x_edges[i]:.1f}-{x_edges[i+1]:.1f}" for i in range(len(x_edges) - 1)]
            y_labels = [f"{y_edges[i]:.1f}-{y_edges[i+1]:.1f}" for i in range(len(y_edges) - 1)]

            # Create heatmap data structure
            heatmap_data = []
            for i in range(len(x_labels)):
                for j in range(len(y_labels)):
                    heatmap_data.append(
                        {
                            "reference_bin": x_labels[i],
                            "current_bin": y_labels[j],
                            "count": int(hist[i, j]),
                            "percentage": float(hist_percent[i, j]),
                        }
                    )

            return {
                "heatmap_data": heatmap_data,
                "x_labels": x_labels,
                "y_labels": y_labels,
                "total_samples": len(pred_ref),
                "bins": bins,
            }

        except Exception as e:
            return {"error": f"Heatmap data generation failed: {str(e)}"}

    def comprehensive_disagreement_analysis(
        self, pred_ref: np.ndarray, pred_curr: np.ndarray, sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model disagreement analysis for the frontend tab

        Args:
            pred_ref: Reference model predictions (probabilities)
            pred_curr: Current model predictions (probabilities)
            sample_size: Sample size for scatter plot

        Returns:
            Dictionary with all disagreement analysis components
        """
        try:
            # Validate inputs
            if len(pred_ref) != len(pred_curr):
                return {"error": "Prediction arrays must have the same length"}

            if len(pred_ref) == 0:
                return {"error": "Prediction arrays cannot be empty"}

            # Generate all analysis components
            scatter_data = self.prediction_scatter_data(pred_ref, pred_curr, sample_size)
            disagreement_stats = self.disagreement_analysis(pred_ref, pred_curr)
            threshold_analysis = self.decision_threshold_analysis(pred_ref, pred_curr)
            heatmap_data = self.disagreement_heatmap_data(pred_ref, pred_curr)

            # Overall disagreement assessment
            if "error" not in disagreement_stats:
                mean_diff = disagreement_stats["mean_absolute_difference"]
                correlation = disagreement_stats["pearson_correlation"]

                if mean_diff >= 0.2:
                    disagreement_level = "High"
                elif mean_diff >= 0.1:
                    disagreement_level = "Medium"
                elif mean_diff >= 0.05:
                    disagreement_level = "Low"
                else:
                    disagreement_level = "Minimal"

                if correlation >= 0.9:
                    correlation_level = "Very Strong"
                elif correlation >= 0.7:
                    correlation_level = "Strong"
                elif correlation >= 0.5:
                    correlation_level = "Moderate"
                elif correlation >= 0.3:
                    correlation_level = "Weak"
                else:
                    correlation_level = "Very Weak"
            else:
                disagreement_level = "Unknown"
                correlation_level = "Unknown"

            return {
                "analysis_type": "model_disagreement",
                "summary": {
                    "disagreement_level": disagreement_level,
                    "correlation_level": correlation_level,
                    "total_predictions": len(pred_ref),
                },
                "scatter_plot": scatter_data,
                "disagreement_statistics": disagreement_stats,
                "threshold_analysis": threshold_analysis,
                "heatmap": heatmap_data,
            }

        except Exception as e:
            return {"error": f"Comprehensive disagreement analysis failed: {str(e)}"}


# Service instance
model_disagreement_service = ModelDisagreementService()
