from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .base_model_service import BaseModelService

try:
    from lime.lime_tabular import LimeTabularExplainer

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")


class PredictionService:
    """
    Service for individual prediction analysis and what-if scenarios.
    """

    def __init__(self, base_service: BaseModelService):
        self.base = base_service

    def _get_feature_impact_with_fallback(self, instance_df: pd.DataFrame) -> Tuple[Dict[str, float], str]:
        """
        Calculate feature impact for a single instance with fallback methods.

        Args:
            instance_df: DataFrame containing the single instance to analyze

        Returns:
            Tuple of (feature_impact_dict, method_used)
        """
        # Method 1: Try SHAP first (best local explanation)
        if self.base.explainer:
            try:
                shap_values = self.base.explainer.shap_values(instance_df)
                if isinstance(shap_values, list):
                    # For classification, use positive class SHAP values
                    shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                else:
                    if len(shap_values.shape) == 3:
                        # Handle 3D SHAP values (instance, feature, class)
                        shap_vals = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
                    elif len(shap_values.shape) == 2:
                        shap_vals = shap_values[0]
                    else:
                        shap_vals = shap_values

                if len(shap_vals) == len(self.base.feature_names):
                    feature_impact = dict(zip(self.base.feature_names, [float(v) for v in shap_vals]))
                    return feature_impact, "shap"
            except Exception as e:
                print(f"SHAP calculation failed: {e}")

        # Method 2: Try LIME (good local explanation)
        if LIME_AVAILABLE:
            try:
                # Create LIME explainer
                lime_explainer = LimeTabularExplainer(
                    training_data=self.base.X_df.values,
                    feature_names=self.base.feature_names,
                    mode="classification",
                    class_names=["0", "1"],
                    discretize_continuous=False,
                )

                # Get explanation for this instance
                instance_array = instance_df.values[0]
                explanation = lime_explainer.explain_instance(
                    instance_array, self.base.model.predict_proba, num_features=len(self.base.feature_names)
                )

                # Extract feature impacts from LIME explanation
                feature_impact = {}
                for feature_idx, impact_value in explanation.as_list():
                    # LIME returns (feature_name_or_idx, impact_value) pairs
                    if isinstance(feature_idx, str):
                        feature_name = feature_idx
                    else:
                        # If it's an index, get the feature name
                        feature_name = self.base.feature_names[int(feature_idx)]
                    feature_impact[feature_name] = float(impact_value)

                # Ensure all features are included (LIME might not return all)
                for feature_name in self.base.feature_names:
                    if feature_name not in feature_impact:
                        feature_impact[feature_name] = 0.0

                return feature_impact, "lime"

            except Exception as e:
                print(f"LIME calculation failed: {e}")

        # Method 3: Feature Ablation (basic local explanation)
        try:
            base_prediction_proba = self.base.safe_predict_proba(instance_df)[0]
            base_prediction = float(base_prediction_proba[1])  # Positive class probability
            feature_impact = {}

            for feature_name in self.base.feature_names:
                # Create modified instance with feature set to dataset mean
                modified_instance = instance_df.copy()
                modified_instance[feature_name] = self.base.X_df[feature_name].mean()

                # Calculate impact as difference in prediction probability
                modified_prediction_proba = self.base.safe_predict_proba(modified_instance)[0]
                modified_prediction = float(modified_prediction_proba[1])  # Positive class probability
                impact = base_prediction - modified_prediction
                feature_impact[feature_name] = impact

            return feature_impact, "ablation"

        except Exception as e:
            print(f"Feature ablation failed: {e}")

        # Method 4: Return informative error if all methods fail
        raise ValueError(
            "Unable to calculate feature explanations. "
            "This model is not compatible with SHAP, LIME, or ablation methods. "
            "Consider using a different model (RandomForest, XGBoost, LogisticRegression) "
        )

    def individual_prediction(self, payload: Dict[str, Any] = None, instance_idx: int = 0) -> Dict[str, Any]:
        """Get detailed prediction analysis for a single instance."""
        self.base._ensure_loaded_with_payload(payload)

        if not (0 <= instance_idx < len(self.base.X_df)):
            raise ValueError(
                f"Instance index {instance_idx} is out of range. Dataset has {len(self.base.X_df)} instances."
            )

        instance_data = self.base.X_df.iloc[instance_idx]
        instance_df = instance_data.to_frame().T

        # Get prediction probabilities for classification
        prediction_proba = self.base.model.predict_proba(instance_df)[0]
        proba = float(prediction_proba[1])  # Positive class probability
        predicted_class = int(np.argmax(prediction_proba))
        confidence = max(proba, 1.0 - proba)  # Distance from 0.5 decision boundary

        # Get feature impact using fallback methods
        try:
            feature_impact, impact_method = self._get_feature_impact_with_fallback(instance_df)
            shap_vals_for_instance = np.array([feature_impact[name] for name in self.base.feature_names])
        except Exception as e:
            print(f"Feature impact computation failed for instance {instance_idx}: {e}")
            shap_vals_for_instance = np.zeros(len(self.base.feature_names))
            impact_method = "none"

        # Handle case where explainer might not be available
        base_value = 0.0
        if self.base.explainer and hasattr(self.base.explainer, "expected_value"):
            if isinstance(self.base.explainer.expected_value, (list, np.ndarray)):
                base_value = float(
                    self.base.explainer.expected_value[1]
                    if len(self.base.explainer.expected_value) > 1
                    else self.base.explainer.expected_value[0]
                )
            else:
                base_value = float(self.base.explainer.expected_value)

        contributions = [
            {
                "name": name,
                "value": self.base._safe_float(instance_data[name]),
                "shap": float(shap_vals_for_instance[i]),
            }
            for i, name in enumerate(self.base.feature_names)
        ]
        contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)

        return {
            "prediction_percentage": proba * 100.0,
            "predicted_class": predicted_class,
            "actual_outcome": int(self.base.y_s.iloc[instance_idx]),
            "confidence_score": confidence,
            "base_value": base_value,
            "shap_values": [float(v) for v in shap_vals_for_instance],
            "feature_contributions": contributions,
            "impact_method": impact_method,  # New field to indicate which method was used
            "model_type": "classification",
        }

    def explain_instance(self, payload: Dict[str, Any] = None, instance_idx: int = 0) -> Dict[str, Any]:
        """Explain a single instance prediction with detailed SHAP analysis."""
        self.base._ensure_loaded_with_payload(payload)

        if not (0 <= instance_idx < len(self.base.X_df)):
            raise ValueError(
                f"Instance index {instance_idx} is out of range. Dataset has {len(self.base.X_df)} instances."
            )

        instance_data = self.base.X_df.iloc[instance_idx]

        # Create single-row DataFrame for prediction to maintain feature names
        instance_df = pd.DataFrame([instance_data], columns=self.base.feature_names)

        # Get feature impact using fallback methods
        try:
            feature_impact, impact_method = self._get_feature_impact_with_fallback(instance_df)
            shap_vals_for_instance = np.array([feature_impact[name] for name in self.base.feature_names])
        except Exception as e:
            print(f"Feature impact computation failed for instance {instance_idx}: {e}")
            shap_vals_for_instance = np.zeros(len(self.base.feature_names))
            impact_method = "none"

        # Handle case where explainer might not be available
        base_value = 0.0
        if self.base.explainer and hasattr(self.base.explainer, "expected_value"):
            if isinstance(self.base.explainer.expected_value, (list, np.ndarray)):
                base_value = float(
                    self.base.explainer.expected_value[1]
                    if len(self.base.explainer.expected_value) > 1
                    else self.base.explainer.expected_value[0]
                )
            else:
                base_value = float(self.base.explainer.expected_value)

        # Get model prediction for classification
        prediction_proba = self.base.safe_predict_proba(instance_df)[0]
        prediction_prob = float(prediction_proba[1])  # Positive class probability
        predicted_class = int(np.argmax(prediction_proba))
        confidence = max(prediction_prob, 1.0 - prediction_prob)  # Distance from 0.5 decision boundary

        # Prepare both mapping and ordered arrays for convenience on the frontend
        shap_mapping = dict(zip(self.base.feature_names, shap_vals_for_instance))
        ordered = sorted(shap_mapping.items(), key=lambda kv: abs(kv[1]), reverse=True)
        ordered_features = [name for name, _ in ordered]
        ordered_values = [float(val) for _, val in ordered]
        ordered_feature_values = [instance_data[name] for name in ordered_features]

        return {
            "instance_id": instance_idx,
            "features": instance_data.to_dict(),
            "prediction": prediction_prob,
            "predicted_class": predicted_class,
            "actual_value": int(self.base.y_s.iloc[instance_idx]),
            "base_value": float(base_value),
            "confidence": confidence,  # Include confidence value
            "shap_values_map": shap_mapping,
            "ordered_contributions": {
                "feature_names": ordered_features,
                "feature_values": [self.base._safe_float(v) for v in ordered_feature_values],
                "shap_values": ordered_values,
            },
            "impact_method": impact_method,  # New field to indicate which method was used
            "model_type": "classification",
        }

    def perform_what_if(self, payload: Dict[str, Any] = None, features: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform what-if analysis by modifying feature values."""
        self.base._ensure_loaded_with_payload(payload)

        try:
            # Start with the first instance as base
            base_instance = self.base.X_df.iloc[0].copy()

            # Update with provided feature values
            for feature_name, value in features.items():
                if feature_name in base_instance.index:
                    base_instance[feature_name] = value
                else:
                    raise ValueError(f"Feature '{feature_name}' not found in dataset.")

            # Make prediction
            instance_df = pd.DataFrame([base_instance], columns=self.base.feature_names)

            # Align dtypes with training data to avoid dtype issues
            for col in instance_df.columns:
                if col in self.base.X_df.columns:
                    instance_df[col] = instance_df[col].astype(self.base.X_df[col].dtype)

            # Cast any remaining bool columns to int
            for col in instance_df.select_dtypes(include=[bool]).columns:
                instance_df[col] = instance_df[col].astype(int)

            # Classification model predictions
            try:
                prediction_proba = self.base.safe_predict_proba(instance_df)[0]
                prediction_prob = float(prediction_proba[1])  # Positive class probability
                prediction_class = int(np.argmax(prediction_proba))
                confidence = max(prediction_prob, 1.0 - prediction_prob)
            except Exception as pred_e:
                print(f"[DEBUG] Prediction error: {pred_e}")
                import traceback

                traceback.print_exc()
                raise

            # Get feature impact using fallback methods
            try:
                feature_impact, impact_method = self._get_feature_impact_with_fallback(instance_df)
            except Exception as e:
                print(f"Feature impact computation failed: {e}")
                feature_impact = {name: 0.0 for name in self.base.feature_names}
                impact_method = "none"

            # Prepare ordered contributions
            ordered = sorted(feature_impact.items(), key=lambda kv: abs(kv[1]), reverse=True)
            ordered_features = [name for name, _ in ordered]
            ordered_values = [float(val) for _, val in ordered]
            ordered_feature_values = [base_instance[name] for name in ordered_features]

            return {
                "modified_features": features,
                "prediction_probability": prediction_prob,
                "predicted_class": prediction_class,
                "confidence": confidence,  # Add confidence to what-if results
                "feature_values": base_instance.to_dict(),
                "shap_explanations": feature_impact,  # Keep same key name for backward compatibility
                "ordered_contributions": {
                    "feature_names": ordered_features,
                    "feature_values": [self.base._safe_float(v) for v in ordered_feature_values],
                    "shap_values": ordered_values,
                },
                "impact_method": impact_method,  # New field to indicate which method was used
                "model_type": "classification",
                "feature_ranges": self._get_feature_ranges(),
            }

        except Exception as e:
            raise ValueError(f"What-if analysis failed: {str(e)}")

    def _get_feature_ranges(self) -> Dict[str, Any]:
        """Get feature ranges and metadata for what-if analysis."""
        self.base._is_ready()

        feature_ranges = {}
        for feature_name in self.base.feature_names:
            col = self.base.X_df[feature_name]
            is_numeric = pd.api.types.is_numeric_dtype(col.dtype)
            is_bool = pd.api.types.is_bool_dtype(col.dtype)

            if is_numeric and not is_bool:
                feature_ranges[feature_name] = {
                    "type": "numeric",
                    "min": float(col.min()),
                    "max": float(col.max()),
                    "mean": float(col.mean()),
                    "std": float(col.std()),
                    "median": float(col.median()),
                    "step": self._calculate_step(col),
                }
            elif is_bool:
                # For boolean, treat as categorical with fixed categories [0, 1]
                value_counts = col.value_counts()
                feature_ranges[feature_name] = {
                    "type": "boolean",
                    "categories": [0, 1],
                    "frequencies": [int((col == 0).sum()), int((col == 1).sum())],
                    "most_common": int(col.mode().iloc[0]) if not col.mode().empty else None,
                    "step": 1,
                }
            else:
                # For categorical features, provide the most common categories
                value_counts = col.value_counts()
                feature_ranges[feature_name] = {
                    "type": "categorical",
                    "categories": value_counts.index.tolist(),
                    "frequencies": value_counts.values.tolist(),
                    "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                }

        return feature_ranges

    def _calculate_step(self, column: pd.Series) -> float:
        """Calculate appropriate step size for numeric column."""
        col_range = column.max() - column.min()

        # For very small ranges (< 1), use smaller steps
        if col_range < 1:
            return 0.01
        # For medium ranges (1-100), use 0.1 or 1
        elif col_range < 100:
            return 0.1 if col_range < 10 else 1
        # For large ranges, use larger steps
        elif col_range < 1000:
            return 10
        else:
            return 100
