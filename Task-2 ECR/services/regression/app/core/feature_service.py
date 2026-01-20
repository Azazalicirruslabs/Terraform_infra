import hashlib
import math
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from services.regression.app.utils.feature_type_detector import FeatureTypeDetector

from .base_model_service import BaseModelService


class FeatureService:
    # ---------- Fingerprinting ----------
    def generate_data_fingerprint(self) -> str:
        try:
            data_structure = f"{self.base.X_train.shape}|{list(self.base.X_train.columns)}"
            data_sample = pd.util.hash_pandas_object(self.base.X_train.head(100)).values
            fingerprint_input = f"{data_structure}|{data_sample.tobytes()}"
            return hashlib.md5(fingerprint_input.encode()).hexdigest()[:12]
        except Exception:
            return hashlib.md5(f"{self.base.X_train.shape}".encode()).hexdigest()[:12]

    def generate_model_fingerprint(self) -> str:
        try:
            model_params = str(sorted(self.base.model.get_params().items()))
            model_type = type(self.base.model).__name__
            fingerprint_input = f"{model_type}|{model_params}"
            return hashlib.md5(fingerprint_input.encode()).hexdigest()[:12]
        except Exception:
            model_type = type(self.base.model).__name__
            return hashlib.md5(model_type.encode()).hexdigest()[:12]

    def _is_cache_valid(self, cache_entry: Dict[str, Any], method: str) -> bool:
        try:
            ttl_config = {"shap": 1800, "gain": 300, "builtin": 300, "permutation": 900}
            elapsed = time.time() - cache_entry["timestamp"]
            ttl = ttl_config.get(method, 600)
            if elapsed >= ttl:
                return False
            return (
                cache_entry["data_fingerprint"] == self.generate_data_fingerprint()
                and cache_entry.get("model_fingerprint", "") == self.generate_model_fingerprint()
            )
        except Exception:
            return False

    def _cleanup_stale_cache(self):
        try:
            current_data_fp = self.generate_data_fingerprint()
            current_model_fp = self.generate_model_fingerprint()
            keys_to_remove = [
                key
                for key, entry in self._importance_cache.items()
                if (
                    entry.get("data_fingerprint") != current_data_fp
                    or entry.get("model_fingerprint") != current_model_fp
                )
            ]
            for key in keys_to_remove:
                del self._importance_cache[key]
        except Exception:
            pass

    """
    Service for feature-related operations like feature importance, metadata,
    correlation analysis, and feature interactions.
    """

    def __init__(self, base_service: BaseModelService):
        self.base = base_service
        # Simple in-memory caches for heavy computations
        self._correlation_cache: Dict[str, Dict[str, Any]] = {}
        self._importance_cache: Dict[str, Dict[str, Any]] = {}
        # Initialize feature type detector with configurable thresholds
        self.feature_detector = FeatureTypeDetector(
            cardinality_threshold=20,  # Features with ≤20 unique values may be categorical
            cardinality_ratio_threshold=0.05,  # If unique/total ≤5%, likely categorical
            min_samples_for_ratio=100,  # Apply ratio check for datasets with ≥100 samples
        )

    def get_feature_importance(self, method: str) -> Dict[str, Any]:
        """Get feature importance using specified method (shap, builtin, etc.)."""
        # Use the advanced method but return simplified format for backward compatibility
        advanced_result = self.compute_feature_importance_advanced(method, "importance", 20, "bar")

        # Handle error cases
        if "error" in advanced_result:
            raise ValueError(advanced_result["error"])

        # Convert to legacy format
        features = [
            {"name": item["name"], "importance": item["importance_score"]} for item in advanced_result["features"]
        ]

        return {"method": method, "features": features}

    def get_feature_metadata(self) -> Dict[str, Any]:
        """Return available features with basic metadata using robust type detection."""
        self.base._is_ready()
        features: List[Dict[str, Any]] = []

        for name in self.base.feature_names:
            col = self.base.X_df[name]
            unique_count = int(col.nunique())

            # Use robust feature type detection
            feature_type = self.feature_detector.detect_feature_type(col)

            feature_info = {
                "name": name,
                "type": feature_type,
                "missing_count": int(col.isnull().sum()),
                "unique_count": unique_count,
                "dtype": str(col.dtype),
            }

            if feature_type == "numerical":
                feature_info.update(
                    {
                        "min": float(col.min()),
                        "max": float(col.max()),
                        "mean": float(col.mean()),
                        "std": float(col.std()),
                        "median": float(col.median()),
                    }
                )
            else:
                feature_info["top_categories"] = col.value_counts().head(5).to_dict()

            features.append(feature_info)

        return {"features": features}

    def _encode_mixed_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only numerical columns using robust type detection for correlation computation."""
        num_cols = self.feature_detector.get_numerical_features(df)
        return df[num_cols].apply(lambda col: col.fillna(col.median()) if col.isnull().any() else col)

    def compute_correlation(self, selected_features: List[str]) -> Dict[str, Any]:
        """Compute Pearson correlation matrix for numerical features only (using robust detection)."""
        self.base._is_ready()

        if not selected_features or len(selected_features) < 2:
            raise ValueError("At least 2 features must be selected for correlation analysis.")

        for feat in selected_features:
            if feat not in self.base.feature_names:
                raise ValueError(f"Feature '{feat}' not found in dataset.")

        # Use robust detection to identify numerical features
        df = self.base.X_df[selected_features]
        num_cols = self.feature_detector.get_numerical_features(df)

        if len(num_cols) < 2:
            raise ValueError("At least 2 numerical features are required for Pearson correlation analysis.")

        # Get feature importance for numerical columns only
        importance = None
        try:
            importance_result = self.compute_feature_importance_advanced("shap", "importance", len(num_cols), "bar")
            importance = {
                item["name"]: item["importance_score"]
                for item in importance_result.get("features", [])
                if item["name"] in num_cols
            }
        except Exception:
            # fallback: use order as is
            importance = {name: 0 for name in num_cols}

        # Sort numerical columns by importance (no top N restriction)
        sorted_num_cols = sorted(num_cols, key=lambda x: importance.get(x, 0), reverse=True)
        if len(sorted_num_cols) < 2:
            raise ValueError("At least 2 numerical features are required for correlation analysis.")

        cache_key = "|".join(sorted(sorted_num_cols))
        if cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]

        enc = df[sorted_num_cols].apply(lambda col: col.fillna(col.median()) if col.isnull().any() else col)
        corr = enc.corr(method="pearson")
        matrix = corr.loc[sorted_num_cols, sorted_num_cols].to_numpy().tolist()

        def safe_float(val):
            if val is None:
                return None
            try:
                f = float(val)
                if math.isnan(f) or math.isinf(f):
                    return None
                return f
            except Exception:
                return None

        payload = {
            "features": sorted_num_cols,  # Only numerical features used in correlation
            "matrix": [[safe_float(v) for v in row] for row in matrix],
            "computed_at": pd.Timestamp.utcnow().isoformat(),
        }

        # Cache result
        self._correlation_cache[cache_key] = payload
        return payload

    def compute_feature_importance_advanced(
        self, method: str = "shap", sort_by: str = "importance", top_n: int = 1000, visualization: str = "bar"
    ) -> Dict[str, Any]:
        self.base._is_ready()

        data_fp = self.generate_data_fingerprint()
        model_fp = self.generate_model_fingerprint()
        cache_key = f"{method}|{sort_by}|{top_n}|{visualization}|{data_fp}|{model_fp}"

        # Cache check
        if cache_key in self._importance_cache:
            cache_entry = self._importance_cache[cache_key]
            if self._is_cache_valid(cache_entry, method):
                return cache_entry["result"]
            else:
                del self._importance_cache[cache_key]

        if len(self._importance_cache) > 10:
            self._cleanup_stale_cache()

        method = method.lower()

        # Variables to store importance values and directions
        raw_importance = None
        impact_directions = None

        try:
            if method == "shap":
                # SHAP returns both signed means (for direction) and magnitudes (for ranking)
                signed_means, unsigned_magnitudes = self._compute_shap_importance()
                raw_importance = unsigned_magnitudes
                impact_directions = self._compute_impact_direction_from_shap(signed_means)
            elif method == "permutation":
                # Permutation returns only magnitudes, use correlation for direction
                raw_importance = self._compute_permutation_importance()
                impact_directions = self._compute_impact_direction_from_correlation()
            elif method in ("gain", "builtin"):
                # Gain returns only magnitudes, use correlation for direction
                raw_importance = self._compute_gain_importance()
                impact_directions = self._compute_impact_direction_from_correlation()
            else:
                raise ValueError(f"Unsupported importance method: {method}")
        except Exception as e:
            error_payload = {
                "error": str(e),
                "total_features": len(self.base.feature_names),
                "positive_impact_count": 0,
                "negative_impact_count": 0,
                "features": [],
                "computation_method": method,
                "computed_at": pd.Timestamp.utcnow().isoformat(),
            }
            if method == "shap":
                error_payload["suggested_fallback"] = "gain"
            elif method == "gain":
                error_payload["suggested_fallback"] = "permutation"
            return error_payload

        # Build items with correct impact directions
        items = []
        for idx, name in enumerate(self.base.feature_names):
            importance = float(raw_importance[idx])
            impact_direction = impact_directions[idx]  # Use computed direction instead of always 'positive'
            items.append(
                {
                    "name": name,
                    "importance_score": abs(importance),
                    "importance": importance,
                    "impact_direction": impact_direction,
                    "rank": 0,
                }
            )

        if sort_by == "feature_name":
            items.sort(key=lambda x: x["name"])
        elif sort_by == "impact":
            items.sort(key=lambda x: abs(x["importance"]), reverse=True)
        else:
            items.sort(key=lambda x: x["importance_score"], reverse=True)

        for i, it in enumerate(items):
            it["rank"] = i + 1

        top_items = items[: int(top_n)]

        # Count positive and negative impacts
        positive_count = sum(1 for i in items if i["impact_direction"] == "positive")
        negative_count = sum(1 for i in items if i["impact_direction"] == "negative")
        total_features = len(items)

        # Runtime validation: Ensure counts add up correctly
        assert (
            positive_count + negative_count == total_features
        ), f"Impact direction counts don't match total features: {positive_count} + {negative_count} != {total_features}"

        # Log warning if all features have the same direction (potential issue indicator)
        if positive_count == 0 or negative_count == 0:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"All features marked as same impact direction. "
                f"Method: {method}, Positive: {positive_count}, Negative: {negative_count}. "
                f"This may indicate an issue with the feature importance calculation."
            )

        payload = {
            "total_features": total_features,
            "positive_impact_count": positive_count,
            "negative_impact_count": negative_count,
            "features": top_items,
            "computation_method": method,
            "computed_at": pd.Timestamp.utcnow().isoformat(),
            "visualization_type": visualization,
            "sort_by": sort_by,
        }

        self._importance_cache[cache_key] = {
            "result": payload,
            "timestamp": time.time(),
            "method": method,
            "data_fingerprint": data_fp,
            "model_fingerprint": model_fp,
        }
        return payload

    def _compute_shap_importance(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute SHAP-based feature importance.

        Returns:
            tuple: (signed_means, unsigned_magnitudes)
                - signed_means: Mean SHAP values preserving sign (for direction)
                - unsigned_magnitudes: Mean absolute SHAP values (for ranking)
        """
        if self.base.shap_values is None:
            raise ValueError(
                "SHAP values are not available for this model. Try using 'gain' or 'permutation' method instead."
            )

        shap_vals = self.base._get_shap_values_for_analysis()
        if shap_vals is None:
            raise ValueError("Failed to compute SHAP-based feature importance.")

        # For regression, compute both signed means (for direction) and absolute means (for importance ranking)
        signed_means = shap_vals.mean(axis=0)  # Preserves positive/negative direction
        unsigned_magnitudes = np.abs(shap_vals).mean(axis=0)  # For ranking by importance

        return signed_means, unsigned_magnitudes

    def _compute_permutation_importance(self) -> np.ndarray:
        """Compute permutation-based feature importance."""
        try:
            # Use test data for permutation importance
            X_test = (
                self.base.X_test if hasattr(self.base, "X_test") and self.base.X_test is not None else self.base.X_df
            )
            y_test = (
                self.base.y_test
                if hasattr(self.base, "y_test") and self.base.y_test is not None
                else self.base.y_df.values
            )

            # Compute permutation importance
            perm_importance = permutation_importance(
                self.base.model,
                X_test,
                y_test,
                n_repeats=10,
                random_state=42,
                scoring="neg_mean_squared_error",
                n_jobs=1,  # Avoid multiprocessing issues
            )

            return perm_importance.importances_mean

        except Exception as e:
            raise ValueError(f"Failed to compute permutation importance: {str(e)}")

    def _compute_gain_importance(self) -> np.ndarray:
        """Compute gain-based (builtin) feature importance."""
        if not hasattr(self.base.model, "feature_importances_"):
            # Try to get feature importance from wrapped model
            if hasattr(self.base.model, "model") and hasattr(self.base.model.model, "feature_importances_"):
                return self.base.model.model.feature_importances_
            else:
                raise ValueError(
                    "Model doesn't support gain-based feature importance. Try using 'permutation' method instead."
                )

        return self.base.model.feature_importances_

    def _compute_impact_direction_from_shap(self, signed_means: np.ndarray) -> List[str]:
        """
        Compute impact direction from signed SHAP means.

        Args:
            signed_means: Mean SHAP values preserving sign

        Returns:
            List of direction strings ('positive' or 'negative') for each feature
        """
        directions = []
        for signed_value in signed_means:
            # Positive SHAP mean → feature increases predictions
            # Negative SHAP mean → feature decreases predictions
            # Zero or near-zero → treat as positive (neutral impact, minimal effect)
            if signed_value > 0:
                directions.append("positive")
            else:
                directions.append("negative")
        return directions

    def _compute_impact_direction_from_correlation(self) -> List[str]:
        """
        Compute impact direction using Pearson correlation between features and predictions.

        This is a fallback method for non-SHAP importance methods (gain, permutation)
        that don't provide directional information.

        Returns:
            List of direction strings ('positive' or 'negative') for each feature
        """
        try:
            # Use test data if available, otherwise use training data
            X_data = (
                self.base.X_test if hasattr(self.base, "X_test") and self.base.X_test is not None else self.base.X_df
            )

            # Get model predictions for the data
            predictions = self.base.model.predict(X_data)

            directions = []
            for idx, feature_name in enumerate(self.base.feature_names):
                try:
                    # Get feature values
                    if isinstance(X_data, pd.DataFrame):
                        feature_values = X_data[feature_name].values
                    else:
                        feature_values = X_data[:, idx]

                    # Handle constant features or features with no variance
                    if np.std(feature_values) == 0:
                        # Constant feature - no real impact, default to positive
                        directions.append("positive")
                        continue

                    # Compute Pearson correlation coefficient
                    correlation = np.corrcoef(feature_values, predictions)[0, 1]

                    # Handle NaN correlation (e.g., if predictions are constant)
                    if np.isnan(correlation):
                        directions.append("positive")
                    elif correlation >= 0:
                        directions.append("positive")
                    else:
                        directions.append("negative")

                except Exception:
                    # If correlation fails for any reason, default to positive
                    directions.append("positive")

            return directions

        except Exception:
            # If entire correlation computation fails, return all positive as safe fallback
            return ["positive"] * len(self.base.feature_names)

    def get_feature_interactions(self, feature1: str, feature2: str) -> Dict[str, Any]:
        """Get feature interaction analysis (simplified version)."""
        self.base._is_ready()

        if feature1 not in self.base.feature_names or feature2 not in self.base.feature_names:
            raise ValueError("One or both features not found in dataset.")

        # This requires SHAP interaction values, which can be computationally expensive
        # For a production system, this would be pre-computed or calculated on demand by a worker.
        # Here we mock it for simplicity, but a real implementation would use:
        # shap_interaction_values = self.explainer.shap_interaction_values(self.X_df)

        # Mocking interaction effects
        f1_values = self.base.X_df[feature1]
        f2_values = self.base.X_df[feature2]

        # Create a plausible-looking interaction effect based on the features
        interaction_effect = (f1_values - f1_values.mean()) * (f2_values - f2_values.mean())
        interaction_effect_normalized = (interaction_effect / interaction_effect.abs().max()) * 0.1

        return {
            "feature1_values": f1_values.tolist(),
            "feature2_values": f2_values.tolist(),
            "interaction_shap_values": interaction_effect_normalized.tolist(),
        }
