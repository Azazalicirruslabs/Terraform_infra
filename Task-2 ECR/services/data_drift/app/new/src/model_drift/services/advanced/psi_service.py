"""
PSI (Population Stability Index) Service - Implements PSI calculation
Following research specifications for feature and prediction drift detection
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class PSIService:
    """Service for calculating Population Stability Index (PSI) for drift detection"""

    def __init__(self):
        # PSI interpretation thresholds based on research
        self.psi_thresholds = {
            "minimal": 0.1,  # PSI < 0.1: No significant change
            "medium": 0.25,  # PSI 0.1-0.25: Some change, 0.25+: Major change
        }

    def calculate_psi_continuous(
        self, reference: np.ndarray, current: np.ndarray, bins: int = 10, epsilon: float = 1e-8
    ) -> Dict[str, Any]:
        """
        Calculate PSI for continuous variables using equal-width binning

        Args:
            reference: Reference dataset values
            current: Current dataset values
            bins: Number of bins for discretization
            epsilon: Small value to handle zero proportions

        Returns:
            Dictionary with PSI value and detailed bin analysis
        """
        try:
            # Determine bin edges from reference data
            bin_edges = np.linspace(np.min(reference), np.max(reference), bins + 1)
            bin_edges[0] = -np.inf  # Handle edge cases
            bin_edges[-1] = np.inf

            # Calculate proportions for each bin
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            curr_counts, _ = np.histogram(current, bins=bin_edges)

            # Convert to proportions and add epsilon for stability
            ref_props = (ref_counts / len(reference)) + epsilon
            curr_props = (curr_counts / len(current)) + epsilon

            # Normalize to ensure proportions sum to 1
            ref_props = ref_props / np.sum(ref_props)
            curr_props = curr_props / np.sum(curr_props)

            # Calculate PSI for each bin
            psi_bins = (curr_props - ref_props) * np.log(curr_props / ref_props)
            psi_total = np.sum(psi_bins)

            # Create detailed bin analysis
            bin_details = []
            for i in range(bins):
                bin_label = (
                    f"Bin_{i+1}"
                    if bin_edges[i] == -np.inf or bin_edges[i + 1] == np.inf
                    else f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
                )
                bin_details.append(
                    {
                        "bin": bin_label,
                        "bin_range": [
                            float(bin_edges[i]) if bin_edges[i] != -np.inf else "min",
                            float(bin_edges[i + 1]) if bin_edges[i + 1] != np.inf else "max",
                        ],
                        "reference_count": int(ref_counts[i]),
                        "current_count": int(curr_counts[i]),
                        "reference_proportion": float(ref_props[i]),
                        "current_proportion": float(curr_props[i]),
                        "psi_contribution": float(psi_bins[i]),
                    }
                )

            return {
                "psi": float(psi_total),
                "interpretation": self._interpret_psi(psi_total),
                "bins": bins,
                "bin_details": bin_details,
                "reference_samples": len(reference),
                "current_samples": len(current),
            }

        except Exception as e:
            return {"error": f"PSI calculation for continuous variable failed: {str(e)}"}

    def calculate_psi_categorical(
        self, reference: np.ndarray, current: np.ndarray, epsilon: float = 1e-8
    ) -> Dict[str, Any]:
        """
        Calculate PSI for categorical variables

        Args:
            reference: Reference dataset categorical values
            current: Current dataset categorical values
            epsilon: Small value to handle zero proportions

        Returns:
            Dictionary with PSI value and detailed category analysis
        """
        try:
            # Get unique categories from both datasets
            all_categories = sorted(list(set(np.concatenate([reference, current]))))

            # Calculate proportions for each category
            ref_props = {}
            curr_props = {}

            for category in all_categories:
                ref_count = np.sum(reference == category)
                curr_count = np.sum(current == category)

                ref_props[category] = (ref_count / len(reference)) + epsilon
                curr_props[category] = (curr_count / len(current)) + epsilon

            # Normalize proportions
            ref_total = sum(ref_props.values())
            curr_total = sum(curr_props.values())

            ref_props = {k: v / ref_total for k, v in ref_props.items()}
            curr_props = {k: v / curr_total for k, v in curr_props.items()}

            # Calculate PSI
            psi_total = 0
            category_details = []

            for category in all_categories:
                ref_prop = ref_props[category]
                curr_prop = curr_props[category]

                psi_contrib = (curr_prop - ref_prop) * np.log(curr_prop / ref_prop)
                psi_total += psi_contrib

                category_details.append(
                    {
                        "category": str(category),
                        "reference_count": int(np.sum(reference == category)),
                        "current_count": int(np.sum(current == category)),
                        "reference_proportion": float(ref_prop),
                        "current_proportion": float(curr_prop),
                        "psi_contribution": float(psi_contrib),
                    }
                )

            return {
                "psi": float(psi_total),
                "interpretation": self._interpret_psi(psi_total),
                "categories": len(all_categories),
                "category_details": category_details,
                "reference_samples": len(reference),
                "current_samples": len(current),
            }

        except Exception as e:
            return {"error": f"PSI calculation for categorical variable failed: {str(e)}"}

    def calculate_feature_psi(
        self, df_ref: pd.DataFrame, df_curr: pd.DataFrame, feature_name: str, feature_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Calculate PSI for a specific feature

        Args:
            df_ref: Reference dataframe
            df_curr: Current dataframe
            feature_name: Name of the feature to analyze
            feature_type: "continuous", "categorical", or "auto" for automatic detection

        Returns:
            Dictionary with PSI analysis for the feature
        """
        try:
            if feature_name not in df_ref.columns or feature_name not in df_curr.columns:
                return {"error": f"Feature '{feature_name}' not found in one or both datasets"}

            ref_values = df_ref[feature_name].dropna().values
            curr_values = df_curr[feature_name].dropna().values

            if len(ref_values) == 0 or len(curr_values) == 0:
                return {"error": f"No valid values found for feature '{feature_name}'"}

            # Auto-detect feature type if needed
            if feature_type == "auto":
                if pd.api.types.is_numeric_dtype(df_ref[feature_name]):
                    # Check if it looks categorical (low cardinality)
                    unique_values = len(np.unique(ref_values))
                    total_values = len(ref_values)
                    if unique_values / total_values < 0.05 or unique_values < 20:
                        feature_type = "categorical"
                    else:
                        feature_type = "continuous"
                else:
                    feature_type = "categorical"

            # Calculate PSI based on feature type
            if feature_type == "continuous":
                result = self.calculate_psi_continuous(ref_values, curr_values)
            else:  # categorical
                result = self.calculate_psi_categorical(ref_values, curr_values)

            # Add feature metadata
            if "error" not in result:
                result["feature_name"] = feature_name
                result["feature_type"] = feature_type
                result["missing_values"] = {
                    "reference": int(df_ref[feature_name].isna().sum()),
                    "current": int(df_curr[feature_name].isna().sum()),
                }

            return result

        except Exception as e:
            return {"error": f"Feature PSI calculation failed: {str(e)}"}

    def calculate_prediction_psi(self, pred_ref: np.ndarray, pred_curr: np.ndarray, bins: int = 10) -> Dict[str, Any]:
        """
        Calculate PSI for model predictions (typically continuous probabilities)

        Args:
            pred_ref: Reference model predictions
            pred_curr: Current model predictions
            bins: Number of bins for discretization

        Returns:
            Dictionary with prediction PSI analysis
        """
        try:
            result = self.calculate_psi_continuous(pred_ref, pred_curr, bins)

            if "error" not in result:
                result["analysis_type"] = "prediction_psi"
                result["prediction_range"] = {
                    "reference_min": float(np.min(pred_ref)),
                    "reference_max": float(np.max(pred_ref)),
                    "current_min": float(np.min(pred_curr)),
                    "current_max": float(np.max(pred_curr)),
                }

            return result

        except Exception as e:
            return {"error": f"Prediction PSI calculation failed: {str(e)}"}

    def comprehensive_psi_analysis(
        self,
        df_ref: pd.DataFrame,
        df_curr: pd.DataFrame,
        pred_ref: Optional[np.ndarray] = None,
        pred_curr: Optional[np.ndarray] = None,
        exclude_features: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive PSI analysis for all features and predictions

        Args:
            df_ref: Reference dataframe
            df_curr: Current dataframe
            pred_ref: Reference model predictions (optional)
            pred_curr: Current model predictions (optional)
            exclude_features: Features to exclude from analysis

        Returns:
            Dictionary with comprehensive PSI analysis
        """
        try:
            if exclude_features is None:
                exclude_features = []

            results = {"analysis_type": "comprehensive_psi", "feature_psi": {}, "prediction_psi": None, "summary": {}}

            # Analyze all features
            common_features = set(df_ref.columns) & set(df_curr.columns)
            features_to_analyze = [f for f in common_features if f not in exclude_features]

            psi_values = []
            high_drift_features = []
            medium_drift_features = []
            stable_features = []

            for feature in features_to_analyze:
                feature_result = self.calculate_feature_psi(df_ref, df_curr, feature)
                results["feature_psi"][feature] = feature_result

                if "error" not in feature_result:
                    psi_val = feature_result["psi"]
                    psi_values.append(psi_val)

                    if psi_val >= self.psi_thresholds["medium"]:
                        high_drift_features.append({"feature": feature, "psi": psi_val})
                    elif psi_val >= self.psi_thresholds["minimal"]:
                        medium_drift_features.append({"feature": feature, "psi": psi_val})
                    else:
                        stable_features.append({"feature": feature, "psi": psi_val})

            # Analyze predictions if provided
            if pred_ref is not None and pred_curr is not None:
                pred_result = self.calculate_prediction_psi(pred_ref, pred_curr)
                results["prediction_psi"] = pred_result

            # Summary statistics
            if psi_values:
                avg_psi = np.mean(psi_values)
                max_psi = np.max(psi_values)

                # Overall drift assessment
                if max_psi >= self.psi_thresholds["medium"]:
                    overall_drift = "High"
                elif max_psi >= self.psi_thresholds["minimal"]:
                    overall_drift = "Medium"
                else:
                    overall_drift = "Low"

                results["summary"] = {
                    "total_features_analyzed": len(features_to_analyze),
                    "average_psi": float(avg_psi),
                    "maximum_psi": float(max_psi),
                    "overall_drift_level": overall_drift,
                    "high_drift_count": len(high_drift_features),
                    "medium_drift_count": len(medium_drift_features),
                    "stable_count": len(stable_features),
                    "high_drift_features": high_drift_features,
                    "medium_drift_features": medium_drift_features,
                    "psi_thresholds": self.psi_thresholds,
                }

            return results

        except Exception as e:
            return {"error": f"Comprehensive PSI analysis failed: {str(e)}"}

    def _interpret_psi(self, psi_value: float) -> Dict[str, Any]:
        """
        Interpret PSI value according to standard thresholds

        Args:
            psi_value: Calculated PSI value

        Returns:
            Dictionary with interpretation
        """
        if psi_value < self.psi_thresholds["minimal"]:
            level = "Low"
            description = "No significant population shift"
            action = "Continue monitoring"
        elif psi_value < self.psi_thresholds["medium"]:
            level = "Medium"
            description = "Some population shift detected"
            action = "Investigate potential causes"
        else:
            level = "High"
            description = "Major population shift detected"
            action = "Immediate investigation required"

        return {
            "level": level,
            "description": description,
            "recommended_action": action,
            "psi_value": float(psi_value),
            "thresholds": self.psi_thresholds,
        }


# Service instance
psi_service = PSIService()
