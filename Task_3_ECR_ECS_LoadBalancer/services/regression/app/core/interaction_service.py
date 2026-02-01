from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_model_service import BaseModelService


class InteractionService:
    """
    Service for feature interaction analysis including interaction networks
    and pairwise analysis.
    """

    def __init__(self, base_service: BaseModelService):
        self.base = base_service

    def interaction_network(self, top_k: int = 30, sample_rows: int = 200) -> Dict[str, Any]:
        """Compute feature interaction network analysis."""
        self.base._is_ready()

        # Sample to keep interaction computation tractable
        df = self.base.X_df.iloc[: min(sample_rows, len(self.base.X_df))]

        try:
            # For true interaction analysis, we would compute SHAP interaction values
            # This is computationally expensive, so we approximate with correlation-based interactions

            # Encode mixed dataframe for interaction computation
            encoded_cols = {}
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c].dtype):
                    encoded_cols[c] = df[c].fillna(df[c].median())
                else:
                    # Label encode categorical variables
                    encoded_cols[c] = pd.Categorical(df[c].fillna("missing")).codes

            encoded_df = pd.DataFrame(encoded_cols)

            # Compute pairwise interactions (simplified as correlation-based)
            corr_matrix = encoded_df.corr().abs()

            # Convert to interaction matrix (higher correlation = higher interaction)
            mean_abs = corr_matrix.values

        except Exception:
            # Fallback: create identity-like matrix
            F = len(self.base.feature_names)
            mean_abs = np.eye(F) * 0.1 + np.random.random((F, F)) * 0.05

        F = len(self.base.feature_names)
        # Ensure matrix alignment with number of features
        mean_abs = np.asarray(mean_abs)
        if mean_abs.ndim != 2 or mean_abs.shape[0] != F or mean_abs.shape[1] != F:
            mean_abs = np.eye(F) * 0.1

        # If interactions are degenerate (all zeros), fallback to correlation proxy
        max_val = float(np.max(mean_abs)) if mean_abs.size else 0.0
        if max_val <= 0:
            np.fill_diagonal(mean_abs, 0.1)
            max_val = 0.1

        # Normalize to [0, 1] to align with frontend threshold slider
        norm_mat = mean_abs / max_val if max_val > 0 else mean_abs

        # Compute node importance from global SHAP
        node_importance = []
        try:
            if self.base.explainer is not None and self.base.shap_values is not None:
                shap_vals_all = self.base._get_shap_matrix()
                if shap_vals_all.ndim == 2 and shap_vals_all.shape[1] == F:
                    node_importance = np.abs(shap_vals_all).mean(axis=0).tolist()
                else:
                    # Fallback: uniform importance
                    node_importance = [1.0] * F
            else:
                # Fallback: uniform importance when no SHAP available
                node_importance = [1.0] * F
        except Exception:
            # Fallback: uniform importance
            node_importance = [1.0] * F

        nodes = []
        for i, name in enumerate(self.base.feature_names):
            nodes.append(
                {
                    "id": name,
                    "label": name,
                    "importance": float(node_importance[i]),
                    "type": "numerical" if pd.api.types.is_numeric_dtype(self.base.X_df[name].dtype) else "categorical",
                }
            )

        edges = []
        for i in range(F):
            for j in range(i + 1, F):
                strength = float(norm_mat[i, j])
                if strength > 0.01:  # Only include meaningful interactions
                    interaction_type = "synergistic" if strength > 0.5 else "independent"
                    edges.append(
                        {
                            "source": self.base.feature_names[i],
                            "target": self.base.feature_names[j],
                            "strength": strength,
                            "type": interaction_type,
                        }
                    )

        # Top K edges by strength
        edges.sort(key=lambda e: e["strength"], reverse=True)
        total_edges_before_filtering = len(edges)  # Store total before filtering
        edges = edges[:top_k]

        top_pairs = [
            {
                "feature_pair": [e["source"], e["target"]],
                "interaction_score": e["strength"],
                "classification": e["type"],
            }
            for e in edges[:10]
        ]

        # Top features by importance (for heatmap and display)
        imp = np.asarray(node_importance, dtype=float)
        if np.max(imp) > 0:
            imp_norm = imp / np.max(imp)
        else:
            imp_norm = np.ones_like(imp)
        order = np.argsort(imp)[::-1]
        top_n = 20
        top_feature_indices = order[:top_n]
        top_feature_names = [self.base.feature_names[i] for i in top_feature_indices]
        # Reduced matrix for top features only
        matrix = norm_mat[np.ix_(top_feature_indices, top_feature_indices)].astype(float)
        top_features = [
            {"name": self.base.feature_names[i], "importance": float(imp[i]), "normalized": float(imp_norm[i])}
            for i in top_feature_indices
        ]

        independent_count = int(sum(1 for e in edges if e["type"] == "independent"))

        # Calculate additional statistics for frontend compatibility
        edge_strengths = [e["strength"] for e in edges]
        mean_strength = float(np.mean(edge_strengths)) if edge_strengths else 0.0
        median_strength = float(np.median(edge_strengths)) if edge_strengths else 0.0
        independence_ratio = independent_count / len(edges) if edges else 0.0

        return {
            "nodes": nodes,
            "edges": edges,
            # Only show heatmap for top N features
            "interaction_matrix": matrix.tolist(),
            "matrix": matrix.tolist(),  # Frontend expects this name
            "feature_names": top_feature_names,
            "matrix_features": top_feature_names,  # Frontend expects this name
            "top_interactions": top_pairs,
            "top_features": top_features,
            "summary": {
                # Original fields
                "total_interactions": total_edges_before_filtering,  # Total before top_k filtering
                "synergistic_count": len(edges) - independent_count,
                "independent_count": independent_count,
                "avg_interaction_strength": mean_strength,
                # Frontend-expected fields
                "total_edges": len(edges),  # Top K edges actually returned
                "mean_strength": mean_strength,
                "median_strength": median_strength,
                "independence_ratio": independence_ratio,
            },
        }

    def pairwise_analysis(
        self, feature1: str, feature2: str, color_by: Optional[str] = None, sample_size: int = 1000
    ) -> Dict[str, Any]:
        """Perform detailed pairwise analysis between two features."""
        self.base._is_ready()

        if feature1 not in self.base.feature_names:
            raise ValueError(f"Feature '{feature1}' not found in dataset.")
        if feature2 not in self.base.feature_names:
            raise ValueError(f"Feature '{feature2}' not found in dataset.")

        # Sample data for efficiency
        sample_df = self.base.X_df.sample(n=min(sample_size, len(self.base.X_df)), random_state=42)
        sample_y = self.base.y_s.loc[sample_df.index]

        # Get feature values
        f1_values = sample_df[feature1]
        f2_values = sample_df[feature2]
        target_values = sample_y.values.flatten() if hasattr(sample_y, "values") else sample_y

        # Get predictions for the sample (regression predictions)
        predictions = self.base.model.predict(sample_df)

        # Prepare data for scatter plot
        scatter_data = {
            "feature1_values": f1_values.tolist(),
            "feature2_values": f2_values.tolist(),
            "predictions": predictions.tolist(),
            "actual_targets": target_values.tolist(),  # Changed from actual_labels to actual_targets
        }

        # Add color-by feature if specified
        if color_by and color_by in self.base.feature_names:
            color_values = sample_df[color_by]
            scatter_data["color_by"] = color_by
            scatter_data["color_values"] = color_values.tolist()

        # Compute interaction strength (simplified correlation-based)
        try:
            if pd.api.types.is_numeric_dtype(f1_values.dtype) and pd.api.types.is_numeric_dtype(f2_values.dtype):
                # Numeric-numeric interaction
                interaction_strength = float(abs(np.corrcoef(f1_values, f2_values)[0, 1]))
                interaction_type = "correlation"

                # Also compute interaction with target
                f1_target_corr = float(abs(np.corrcoef(f1_values, target_values)[0, 1]))
                f2_target_corr = float(abs(np.corrcoef(f2_values, target_values)[0, 1]))

            else:
                # At least one categorical feature - use contingency-based analysis
                interaction_strength = 0.5  # Placeholder
                interaction_type = "categorical"
                f1_target_corr = 0.0
                f2_target_corr = 0.0

        except Exception:
            interaction_strength = 0.0
            interaction_type = "unknown"
            f1_target_corr = 0.0
            f2_target_corr = 0.0

        # Feature statistics
        f1_stats = {
            "type": "numerical" if pd.api.types.is_numeric_dtype(f1_values.dtype) else "categorical",
            "missing_count": int(f1_values.isnull().sum()),
            "unique_count": int(f1_values.nunique()),
        }

        f2_stats = {
            "type": "numerical" if pd.api.types.is_numeric_dtype(f2_values.dtype) else "categorical",
            "missing_count": int(f2_values.isnull().sum()),
            "unique_count": int(f2_values.nunique()),
        }

        if f1_stats["type"] == "numerical":
            f1_stats.update(
                {
                    "mean": float(f1_values.mean()),
                    "std": float(f1_values.std()),
                    "min": float(f1_values.min()),
                    "max": float(f1_values.max()),
                }
            )

        if f2_stats["type"] == "numerical":
            f2_stats.update(
                {
                    "mean": float(f2_values.mean()),
                    "std": float(f2_values.std()),
                    "min": float(f2_values.min()),
                    "max": float(f2_values.max()),
                }
            )

        return {
            "feature1": feature1,
            "feature2": feature2,
            # Frontend compatibility: provide data in expected format
            "x": f1_values.tolist(),
            "y": f2_values.tolist(),
            "prediction": predictions.tolist(),
            "scatter_data": scatter_data,
            "interaction_analysis": {
                "strength": interaction_strength,
                "type": interaction_type,
                "feature1_target_correlation": f1_target_corr,
                "feature2_target_correlation": f2_target_corr,
                "interpretation": (
                    "Strong interaction"
                    if interaction_strength > 0.7
                    else "Moderate interaction" if interaction_strength > 0.3 else "Weak interaction"
                ),
            },
            "feature_statistics": {feature1: f1_stats, feature2: f2_stats},
            "sample_size": len(sample_df),
        }
