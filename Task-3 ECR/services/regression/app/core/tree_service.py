from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import _tree

from .base_model_service import BaseModelService


class TreeService:
    """
    Service for regression decision tree analysis and visualization.
    """

    def __init__(self, base_service: BaseModelService):
        self.base = base_service

    def get_decision_tree(self) -> Dict[str, Any]:
        """Extract regression decision tree structure for visualization."""
        self.base._is_ready()

        # Verify this is a regression model
        if not self.base._is_regression_model():
            return {"error": "This feature is only available for regression models."}

        # Check if the model has trees
        if hasattr(self.base.model, "estimators_"):
            # Handle different ensemble model structures
            estimators_attr = self.base.model.estimators_

            # Check if it's a GradientBoosting model (numpy array structure)
            if isinstance(estimators_attr, np.ndarray):
                # For regression, GradientBoosting has shape (n_estimators, 1)
                if estimators_attr.ndim == 2:
                    tree_estimators = [estimators_attr[i, 0] for i in range(estimators_attr.shape[0])]
                else:
                    tree_estimators = list(estimators_attr.flatten())
                model_type = "gradient_boosting"
            else:
                # RandomForest, ExtraTrees: estimators_ is a list of tree objects
                tree_estimators = estimators_attr
                model_type = "ensemble"

            is_ensemble = True
        elif hasattr(self.base.model, "tree_"):
            # Single decision tree regressor
            tree_estimators = [self.base.model]
            is_ensemble = False
            model_type = "single_tree"
        else:
            return {
                "error": "The current model does not contain decision trees. Please use a tree-based regression model."
            }

        # Collect data for all trees
        trees_data = []
        for idx, tree_estimator in enumerate(tree_estimators):
            tree_data = self._extract_regression_tree_structure(tree_estimator, idx, model_type)
            trees_data.append(tree_data)

            # For ensemble models, limit to first few trees
            if is_ensemble and idx >= 4:  # Show max 5 trees
                break

        return {
            "trees": trees_data,
            "model_type": model_type,
            "num_trees": len(trees_data),
            "total_trees": len(tree_estimators) if is_ensemble else 1,
            "algorithm": type(self.base.model).__name__,
            "task_type": "regression",
        }

    def _extract_regression_tree_structure(
        self, tree_estimator, tree_idx: int, model_type: str = "ensemble"
    ) -> Dict[str, Any]:
        """Extract the structure of a single regression decision tree."""
        try:
            tree = tree_estimator.tree_
            feature_names = self.base.feature_names

            # Calculate tree performance metrics on test set
            tree_metrics = {"r2": 0.0, "mse": 0.0, "mae": 0.0}

            if self.base.X_test is not None and self.base.y_test is not None:
                try:
                    if model_type == "gradient_boosting":
                        # For GB, individual trees predict residuals, so use full model
                        predictions = self.base.safe_predict(self.base.X_test)
                    else:
                        # For RF/ET, individual trees can make predictions
                        predictions = tree_estimator.predict(self.base.X_test)

                    tree_metrics["r2"] = float(r2_score(self.base.y_test, predictions))
                    tree_metrics["mse"] = float(mean_squared_error(self.base.y_test, predictions))
                    tree_metrics["mae"] = float(mean_absolute_error(self.base.y_test, predictions))
                except Exception:
                    pass  # Keep default metrics if calculation fails

            # Get feature importance for this tree
            tree_importance = None
            if hasattr(tree_estimator, "feature_importances_"):
                tree_importance = tree_estimator.feature_importances_.tolist()

            def recurse(node, depth):
                if tree.feature[node] != _tree.TREE_UNDEFINED:
                    # Internal node (split)
                    feature = feature_names[tree.feature[node]]
                    threshold = float(tree.threshold[node])
                    samples = int(tree.n_node_samples[node])

                    # For regression trees, impurity is MSE (mean squared error)
                    mse = float(tree.impurity[node])

                    # Calculate the mean prediction value at this node
                    node_value = float(tree.value[node][0][0])

                    # Calculate reduction in variance
                    left_child = tree.children_left[node]
                    right_child = tree.children_right[node]
                    left_variance = float(tree.impurity[left_child]) if left_child != _tree.TREE_UNDEFINED else 0.0
                    right_variance = float(tree.impurity[right_child]) if right_child != _tree.TREE_UNDEFINED else 0.0
                    reduction_in_variance = mse - (left_variance + right_variance)

                    return {
                        "type": "split",
                        "feature": feature,
                        "threshold": threshold,
                        "samples": samples,
                        "mse": mse,
                        "std_dev": np.sqrt(mse),  # Standard deviation
                        "mean_value": node_value,
                        "reduction_in_variance": reduction_in_variance,
                        "node_id": f"node_{node}",
                        "depth": depth,
                        "left": recurse(tree.children_left[node], depth + 1),
                        "right": recurse(tree.children_right[node], depth + 1),
                    }
                else:
                    # Leaf node
                    samples = int(tree.n_node_samples[node])

                    # For regression: extract the actual predicted continuous value
                    predicted_value = float(tree.value[node][0][0])

                    # Parent node's MSE (uncertainty measure)
                    parent_node = -1
                    for i in range(tree.node_count):
                        if tree.children_left[i] == node or tree.children_right[i] == node:
                            parent_node = i
                            break

                    parent_mse = float(tree.impurity[parent_node]) if parent_node >= 0 else 0.0

                    return {
                        "type": "leaf",
                        "samples": samples,
                        "prediction": predicted_value,  # The actual continuous value!
                        "parent_mse": parent_mse,
                        "parent_std": np.sqrt(parent_mse),
                        "node_id": f"node_{node}",
                        "depth": depth,
                    }

            # Calculate tree statistics
            total_nodes = tree.node_count
            leaf_nodes = sum(1 for i in range(total_nodes) if tree.feature[i] == _tree.TREE_UNDEFINED)
            max_depth = tree.max_depth

            # Calculate average leaf prediction
            leaf_predictions = []
            for i in range(total_nodes):
                if tree.feature[i] == _tree.TREE_UNDEFINED:  # Is leaf
                    leaf_predictions.append(float(tree.value[i][0][0]))

            avg_leaf_value = np.mean(leaf_predictions) if leaf_predictions else 0.0
            std_leaf_value = np.std(leaf_predictions) if leaf_predictions else 0.0

            return {
                "tree_index": tree_idx,
                "metrics": tree_metrics,
                "importance": tree_importance,
                "total_nodes": total_nodes,
                "leaf_nodes": leaf_nodes,
                "max_depth": max_depth,
                "avg_leaf_prediction": float(avg_leaf_value),
                "std_leaf_prediction": float(std_leaf_value),
                "tree_structure": recurse(0, 0),
            }

        except Exception as e:
            return {"tree_index": tree_idx, "error": f"Failed to extract tree structure: {str(e)}"}

    def get_tree_rules(self, tree_idx: int = 0, max_depth: int = 3) -> Dict[str, Any]:
        """Extract regression rules from a specific tree."""
        self.base._is_ready()

        # Verify this is a regression model
        if not self.base._is_regression_model():
            return {"error": "This feature is only available for regression models."}

        # Get the tree
        if hasattr(self.base.model, "estimators_"):
            estimators_attr = self.base.model.estimators_

            # Handle GradientBoosting numpy array structure
            if isinstance(estimators_attr, np.ndarray):
                if estimators_attr.ndim == 2:
                    tree_estimators = [estimators_attr[i, 0] for i in range(estimators_attr.shape[0])]
                else:
                    tree_estimators = list(estimators_attr.flatten())

                if tree_idx >= len(tree_estimators):
                    raise ValueError(f"Tree index {tree_idx} out of range. Model has {len(tree_estimators)} trees.")
                tree_estimator = tree_estimators[tree_idx]
            else:
                if tree_idx >= len(self.base.model.estimators_):
                    raise ValueError(
                        f"Tree index {tree_idx} out of range. Model has {len(self.base.model.estimators_)} trees."
                    )
                tree_estimator = self.base.model.estimators_[tree_idx]
        elif hasattr(self.base.model, "tree_"):
            if tree_idx != 0:
                raise ValueError("Single tree model only has tree index 0.")
            tree_estimator = self.base.model
        else:
            raise ValueError("Model does not contain decision trees.")

        try:
            tree = tree_estimator.tree_
            feature_names = self.base.feature_names

            def extract_rules(node_id: int, depth: int = 0, conditions: List[str] = None) -> List[Dict[str, Any]]:
                """Recursively extract regression rules from tree."""
                if conditions is None:
                    conditions = []

                rules = []

                # Stop if we've reached max depth
                if depth >= max_depth:
                    return rules

                # Check if node is a leaf
                if tree.children_left[node_id] == tree.children_right[node_id]:
                    # Leaf node - create regression rule
                    rule_text = " AND ".join(conditions) if conditions else "Always"
                    samples = int(tree.n_node_samples[node_id])
                    support = float(samples / tree.n_node_samples[0])

                    # Extract the predicted continuous value
                    predicted_value = float(tree.value[node_id][0][0])

                    # Get parent's MSE for uncertainty measure
                    parent_node = -1
                    for i in range(tree.node_count):
                        if tree.children_left[i] == node_id or tree.children_right[i] == node_id:
                            parent_node = i
                            break

                    uncertainty = float(tree.impurity[parent_node]) if parent_node >= 0 else 0.0

                    rules.append(
                        {
                            "rule": rule_text,
                            "conditions": conditions.copy(),
                            "predicted_value": predicted_value,
                            "samples": samples,
                            "support": support,
                            "uncertainty_mse": uncertainty,
                            "uncertainty_std": np.sqrt(uncertainty),
                        }
                    )
                else:
                    # Internal node
                    feature_idx = tree.feature[node_id]
                    feature_name = (
                        feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}"
                    )
                    threshold = float(tree.threshold[node_id])

                    # Format threshold based on value magnitude
                    if abs(threshold) < 0.01 or abs(threshold) > 1000:
                        threshold_str = f"{threshold:.2e}"
                    else:
                        threshold_str = f"{threshold:.3f}"

                    # Left child (<=)
                    left_conditions = conditions + [f"{feature_name} <= {threshold_str}"]
                    rules.extend(extract_rules(tree.children_left[node_id], depth + 1, left_conditions))

                    # Right child (>)
                    right_conditions = conditions + [f"{feature_name} > {threshold_str}"]
                    rules.extend(extract_rules(tree.children_right[node_id], depth + 1, right_conditions))

                return rules

            rules = extract_rules(0)

            # Sort rules by support (most common paths first)
            rules.sort(key=lambda x: x["support"], reverse=True)

            # Calculate rule statistics
            prediction_values = [r["predicted_value"] for r in rules]

            return {
                "tree_index": tree_idx,
                "max_depth": max_depth,
                "rules": rules,
                "total_rules": len(rules),
                "prediction_range": {
                    "min": float(np.min(prediction_values)) if prediction_values else 0.0,
                    "max": float(np.max(prediction_values)) if prediction_values else 0.0,
                    "mean": float(np.mean(prediction_values)) if prediction_values else 0.0,
                    "std": float(np.std(prediction_values)) if prediction_values else 0.0,
                },
            }

        except Exception as e:
            return {"error": f"Failed to extract tree rules: {str(e)}"}
