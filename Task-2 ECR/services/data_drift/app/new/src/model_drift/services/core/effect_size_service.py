"""
Effect Size Service - Implements effect size measures
Following research specifications for Cohen's d and other effect size calculations
"""

import warnings
from typing import Any, Dict

import numpy as np

warnings.filterwarnings("ignore")


class EffectSizeService:
    """Service for calculating effect sizes to measure practical significance"""

    def __init__(self):
        pass

    def cohens_d(self, group1: np.ndarray, group2: np.ndarray, pooled: bool = True) -> Dict[str, Any]:
        """
        Calculate Cohen's d effect size

        Cohen's d measures the standardized difference between two groups

        Interpretation:
        - Small effect: d = 0.2
        - Medium effect: d = 0.5
        - Large effect: d = 0.8

        Args:
            group1: First group of values (e.g., reference model metrics)
            group2: Second group of values (e.g., current model metrics)
            pooled: Whether to use pooled standard deviation (default: True)

        Returns:
            Dictionary with Cohen's d and interpretation
        """
        try:
            n1, n2 = len(group1), len(group2)
            mean1, mean2 = np.mean(group1), np.mean(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

            if pooled:
                # Pooled standard deviation
                pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
                d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            else:
                # Use standard deviation of group1 (control group)
                std1 = np.sqrt(var1)
                d = (mean1 - mean2) / std1 if std1 > 0 else 0

            # Interpret effect size
            abs_d = abs(d)
            if abs_d < 0.2:
                interpretation = "Negligible"
            elif abs_d < 0.5:
                interpretation = "Small"
            elif abs_d < 0.8:
                interpretation = "Medium"
            else:
                interpretation = "Large"

            # Direction
            direction = "Positive" if d > 0 else "Negative" if d < 0 else "No difference"

            return {
                "cohens_d": float(d),
                "absolute_effect_size": float(abs_d),
                "interpretation": interpretation,
                "direction": direction,
                "thresholds": {"small": 0.2, "medium": 0.5, "large": 0.8},
                "group_statistics": {
                    "group1_mean": float(mean1),
                    "group1_std": float(np.sqrt(var1)),
                    "group1_n": int(n1),
                    "group2_mean": float(mean2),
                    "group2_std": float(np.sqrt(var2)),
                    "group2_n": int(n2),
                },
            }

        except Exception as e:
            return {"error": f"Cohen's d calculation failed: {str(e)}"}

    def hedges_g(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Hedges' g effect size

        Hedges' g is a corrected version of Cohen's d for small sample sizes

        Args:
            group1: First group of values
            group2: Second group of values

        Returns:
            Dictionary with Hedges' g and interpretation
        """
        try:
            # First calculate Cohen's d
            cohens_result = self.cohens_d(group1, group2, pooled=True)

            if "error" in cohens_result:
                return cohens_result

            d = cohens_result["cohens_d"]
            n1, n2 = len(group1), len(group2)

            # Correction factor for small samples
            df = n1 + n2 - 2
            correction_factor = 1 - (3 / (4 * df - 1)) if df > 0 else 1

            g = d * correction_factor

            # Use same interpretation as Cohen's d
            abs_g = abs(g)
            if abs_g < 0.2:
                interpretation = "Negligible"
            elif abs_g < 0.5:
                interpretation = "Small"
            elif abs_g < 0.8:
                interpretation = "Medium"
            else:
                interpretation = "Large"

            return {
                "hedges_g": float(g),
                "cohens_d": float(d),
                "correction_factor": float(correction_factor),
                "absolute_effect_size": float(abs_g),
                "interpretation": interpretation,
                "degrees_freedom": int(df),
            }

        except Exception as e:
            return {"error": f"Hedges' g calculation failed: {str(e)}"}

    def glass_delta(self, group1: np.ndarray, group2: np.ndarray, control_group: int = 1) -> Dict[str, Any]:
        """
        Calculate Glass's Δ (Delta) effect size

        Glass's Delta uses only the control group's standard deviation

        Args:
            group1: First group of values
            group2: Second group of values
            control_group: Which group to use as control (1 or 2)

        Returns:
            Dictionary with Glass's Delta and interpretation
        """
        try:
            mean1, mean2 = np.mean(group1), np.mean(group2)

            if control_group == 1:
                control_std = np.std(group1, ddof=1)
                delta = (mean2 - mean1) / control_std if control_std > 0 else 0
            else:
                control_std = np.std(group2, ddof=1)
                delta = (mean1 - mean2) / control_std if control_std > 0 else 0

            # Interpret using Cohen's d thresholds
            abs_delta = abs(delta)
            if abs_delta < 0.2:
                interpretation = "Negligible"
            elif abs_delta < 0.5:
                interpretation = "Small"
            elif abs_delta < 0.8:
                interpretation = "Medium"
            else:
                interpretation = "Large"

            return {
                "glass_delta": float(delta),
                "absolute_effect_size": float(abs_delta),
                "interpretation": interpretation,
                "control_group": control_group,
                "control_std": float(control_std),
            }

        except Exception as e:
            return {"error": f"Glass's Delta calculation failed: {str(e)}"}

    def probability_superiority(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Probability of Superiority (Common Language Effect Size)

        The probability that a randomly selected score from group1
        will be greater than a randomly selected score from group2

        Args:
            group1: First group of values
            group2: Second group of values

        Returns:
            Dictionary with probability of superiority
        """
        try:
            # Count how many pairs show group1 > group2
            superior_count = 0
            total_count = 0

            for val1 in group1:
                for val2 in group2:
                    if val1 > val2:
                        superior_count += 1
                    total_count += 1

            prob_superiority = superior_count / total_count if total_count > 0 else 0.5

            # Interpretation
            if prob_superiority > 0.71:
                interpretation = "Large advantage for group 1"
            elif prob_superiority > 0.64:
                interpretation = "Medium advantage for group 1"
            elif prob_superiority > 0.56:
                interpretation = "Small advantage for group 1"
            elif prob_superiority < 0.29:
                interpretation = "Large advantage for group 2"
            elif prob_superiority < 0.36:
                interpretation = "Medium advantage for group 2"
            elif prob_superiority < 0.44:
                interpretation = "Small advantage for group 2"
            else:
                interpretation = "No meaningful difference"

            return {
                "probability_superiority": float(prob_superiority),
                "interpretation": interpretation,
                "superior_pairs": int(superior_count),
                "total_pairs": int(total_count),
                "thresholds": {"large_effect": 0.71, "medium_effect": 0.64, "small_effect": 0.56},
            }

        except Exception as e:
            return {"error": f"Probability of superiority calculation failed: {str(e)}"}

    def cliff_delta(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Cliff's Delta effect size

        Non-parametric effect size measure based on dominance

        Args:
            group1: First group of values
            group2: Second group of values

        Returns:
            Dictionary with Cliff's Delta and interpretation
        """
        try:
            n1, n2 = len(group1), len(group2)

            # Count dominance relationships
            greater_count = 0
            less_count = 0

            for val1 in group1:
                for val2 in group2:
                    if val1 > val2:
                        greater_count += 1
                    elif val1 < val2:
                        less_count += 1

            # Calculate Cliff's delta
            total_pairs = n1 * n2
            delta = (greater_count - less_count) / total_pairs if total_pairs > 0 else 0

            # Interpretation (Romano et al., 2006)
            abs_delta = abs(delta)
            if abs_delta < 0.147:
                interpretation = "Negligible"
            elif abs_delta < 0.33:
                interpretation = "Small"
            elif abs_delta < 0.474:
                interpretation = "Medium"
            else:
                interpretation = "Large"

            return {
                "cliff_delta": float(delta),
                "absolute_effect_size": float(abs_delta),
                "interpretation": interpretation,
                "greater_pairs": int(greater_count),
                "less_pairs": int(less_count),
                "total_pairs": int(total_pairs),
                "thresholds": {"small": 0.147, "medium": 0.33, "large": 0.474},
            }

        except Exception as e:
            return {"error": f"Cliff's Delta calculation failed: {str(e)}"}

    def comprehensive_effect_analysis(
        self, group1: np.ndarray, group2: np.ndarray, group1_name: str = "Reference", group2_name: str = "Current"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive effect size analysis using multiple measures

        Args:
            group1: First group of values (e.g., reference model metrics)
            group2: Second group of values (e.g., current model metrics)
            group1_name: Name for first group
            group2_name: Name for second group

        Returns:
            Dictionary with all effect size measures and overall assessment
        """
        try:
            # Calculate all effect sizes
            cohens_result = self.cohens_d(group1, group2)
            hedges_result = self.hedges_g(group1, group2)
            glass_result = self.glass_delta(group1, group2)
            prob_sup_result = self.probability_superiority(group1, group2)
            cliff_result = self.cliff_delta(group1, group2)

            # Determine overall effect magnitude
            effect_sizes = []
            if "error" not in cohens_result:
                effect_sizes.append(cohens_result["absolute_effect_size"])
            if "error" not in hedges_result:
                effect_sizes.append(hedges_result["absolute_effect_size"])
            if "error" not in cliff_result:
                effect_sizes.append(cliff_result["absolute_effect_size"])

            if effect_sizes:
                avg_effect = np.mean(effect_sizes)
                if avg_effect >= 0.8:
                    overall_magnitude = "Large"
                elif avg_effect >= 0.5:
                    overall_magnitude = "Medium"
                elif avg_effect >= 0.2:
                    overall_magnitude = "Small"
                else:
                    overall_magnitude = "Negligible"
            else:
                overall_magnitude = "Cannot determine"

            return {
                "group_names": {"group1": group1_name, "group2": group2_name},
                "effect_sizes": {
                    "cohens_d": cohens_result,
                    "hedges_g": hedges_result,
                    "glass_delta": glass_result,
                    "probability_superiority": prob_sup_result,
                    "cliff_delta": cliff_result,
                },
                "overall_assessment": {
                    "magnitude": overall_magnitude,
                    "average_effect_size": float(np.mean(effect_sizes)) if effect_sizes else None,
                    "practical_significance": overall_magnitude in ["Medium", "Large"],
                },
            }

        except Exception as e:
            return {"error": f"Comprehensive effect analysis failed: {str(e)}"}


# Service instance
effect_size_service = EffectSizeService()
