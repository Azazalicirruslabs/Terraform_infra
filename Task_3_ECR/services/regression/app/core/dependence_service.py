from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_model_service import BaseModelService


class DependenceService:
    """
    Service for feature dependence analysis including partial dependence plots,
    SHAP dependence plots, and ICE plots.
    """

    def __init__(self, base_service: BaseModelService):
        self.base = base_service

    def get_feature_dependence(self, feature_name: str) -> Dict[str, Any]:
        """Get basic feature dependence using SHAP values."""
        self.base._is_ready()

        if feature_name not in self.base.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in dataset.")

        if self.base.shap_values is None:
            raise ValueError("SHAP values not available. Cannot compute feature dependence.")

        feature_idx = self.base.feature_names.index(feature_name)
        shap_vals = self.base._get_shap_values_for_analysis()

        if shap_vals is None:
            raise ValueError("Failed to get SHAP values for dependence analysis.")

        return {
            "feature_values": self.base.X_df[feature_name].tolist(),
            "shap_values": shap_vals[:, feature_idx].tolist(),
        }

    def partial_dependence(self, feature_name: str, num_points: int = 20) -> Dict[str, Any]:
        """Compute partial dependence plot data for a feature."""
        self.base._is_ready()

        if feature_name not in self.base.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in dataset.")

        col = self.base.X_df[feature_name]
        is_numeric = pd.api.types.is_numeric_dtype(col.dtype)

        # Create grid of values
        grid: List[Any]
        if is_numeric:
            min_val, max_val = col.min(), col.max()
            grid = np.linspace(min_val, max_val, num_points).tolist()
        else:
            # For categorical, use top categories
            top_categories = col.value_counts().head(num_points)
            grid = top_categories.index.tolist()

        # Compute partial dependence
        preds: List[float] = []

        for v in grid:
            # Create modified dataset with feature set to current value
            X_modified = self.base.X_df.copy()
            X_modified[feature_name] = v

            # Get predictions for all instances and average (regression only)
            y_pred = self.base.safe_predict(X_modified)
            avg_pred = float(np.mean(y_pred))
            preds.append(avg_pred)

        # Impact metrics
        effect_range = float(np.max(preds) - np.min(preds)) if len(preds) > 0 else 0.0
        direction = "increasing"
        if is_numeric and len(preds) > 2:
            # Simple trend analysis
            if preds[-1] < preds[0]:
                direction = "decreasing"
            elif abs(preds[-1] - preds[0]) < 0.01:
                direction = "flat"

        # Approximate importance from SHAP global importance
        try:
            shap_vals = self.base._get_shap_values_for_analysis()
            if shap_vals is not None:
                feature_idx = self.base.feature_names.index(feature_name)
                importance = np.abs(shap_vals[:, feature_idx]).mean()
                total_importance = np.abs(shap_vals).mean(axis=0).sum()
                importance_pct = (importance / total_importance) * 100 if total_importance > 0 else 0
            else:
                importance_pct = 0.0
        except Exception:
            importance_pct = 0.0

        impact = {
            "impact_summary": "High influence on model predictions" if importance_pct > 5 else "Moderate influence",
            "feature_type": "numerical" if is_numeric else "categorical",
            "importance_percentage": round(importance_pct, 2),
            "effect_range": round(effect_range, 6),
            "trend_analysis": {"direction": direction, "variability": "continuous" if is_numeric else "discrete"},
            "confidence_score": int(min(100, max(50, importance_pct))),
        }

        return {
            "feature": feature_name,
            "x": [float(x) if is_numeric else x for x in grid],
            "y": preds,
            "impact": impact,
            "plot_metadata": {
                "title": f"Partial Dependence Plot for {feature_name}",
                "x_label": feature_name,
                "y_label": "Predicted Value",
                "plot_type": "partial_dependence",
            },
        }

    def shap_dependence(self, feature_name: str, color_by: Optional[str] = None) -> Dict[str, Any]:
        """Compute SHAP dependence plot data for a feature."""
        self.base._is_ready()

        if feature_name not in self.base.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in dataset.")

        idx = self.base.feature_names.index(feature_name)
        shap_mat = self.base._get_shap_matrix()
        shap_vec = np.asarray(shap_mat[:, idx]).reshape(-1)
        feature_vals = self.base.X_df[feature_name]

        payload: Dict[str, Any] = {
            "feature": feature_name,
            "feature_values": (
                feature_vals.astype(float).tolist()
                if pd.api.types.is_numeric_dtype(feature_vals.dtype)
                else feature_vals.astype(str).tolist()
            ),
            "shap_values": np.asarray(shap_vec, dtype=float).reshape(-1).tolist(),
        }

        # Build plot metadata
        plot_metadata = {
            "title": f"SHAP Dependence Plot for {feature_name}",
            "x_label": feature_name,
            "y_label": f"SHAP value for {feature_name}",
            "plot_type": "shap_dependence",
        }

        if color_by and color_by in self.base.feature_names:
            color_vals = self.base.X_df[color_by]
            payload["color_by"] = color_by
            payload["color_values"] = (
                color_vals.astype(float).tolist()
                if pd.api.types.is_numeric_dtype(color_vals.dtype)
                else color_vals.astype(str).tolist()
            )
            plot_metadata["color_scale_label"] = color_by

        payload["plot_metadata"] = plot_metadata
        return payload

    def ice_plot(self, feature_name: str, num_points: int = 20, num_instances: int = 20) -> Dict[str, Any]:
        """Compute Individual Conditional Expectation (ICE) plot data."""
        self.base._is_ready()

        if feature_name not in self.base.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in dataset.")

        col = self.base.X_df[feature_name]
        is_numeric = pd.api.types.is_numeric_dtype(col.dtype)

        if not is_numeric:
            return {"error": "ICE plots are only supported for numerical features."}
        else:
            min_val, max_val = col.min(), col.max()
            grid = np.linspace(min_val, max_val, num_points)

        n = min(num_instances, len(self.base.X_df))
        sample_idx = list(range(n))
        curves = []

        for i in sample_idx:
            instance = self.base.X_df.iloc[i].copy()
            curve_preds = []

            for val in grid:
                # Modify the feature value
                instance_modified = instance.copy()
                instance_modified[feature_name] = val

                # Get prediction (regression only)
                instance_df = pd.DataFrame([instance_modified], columns=self.base.feature_names)
                pred = float(self.base.safe_predict(instance_df)[0])
                curve_preds.append(pred)

            curves.append(
                {
                    "instance_id": i,
                    "x": grid.tolist(),
                    "y": curve_preds,
                    "original_value": float(instance[feature_name]),
                    "label": f"Instance {i}",
                }
            )

        return {
            "feature": feature_name,
            "curves": curves,
            "plot_metadata": {
                "title": f"ICE Plot for {feature_name}",
                "x_label": feature_name,
                "y_label": "Predicted Value",
                "plot_type": "ice",
                "legend_entries": [f"Instance {i}" for i in sample_idx],
            },
        }
