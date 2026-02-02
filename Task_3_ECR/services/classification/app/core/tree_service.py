from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree

from .base_model_service import BaseModelService


class TreeService:
    """
    Service for decision tree analysis and visualization.
    """

    def __init__(self, base_service: BaseModelService):
        self.base = base_service

    def get_decision_tree(self, payload: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract decision tree structure for visualization."""
        self.base._ensure_loaded_with_payload(payload)

        # Check if the model has trees (ensemble models or single decision tree)
        if hasattr(self.base.model, "estimators_"):
            # Handle different ensemble model structures
            estimators_attr = self.base.model.estimators_

            # Check if it's a GradientBoosting model (numpy array structure)
            if isinstance(estimators_attr, np.ndarray):
                # GradientBoosting: estimators_ is numpy array with shape (n_estimators, n_classes)
                # For binary classification, we take the first class estimators
                if estimators_attr.ndim == 2:
                    # Extract trees from the first class (class 0)
                    tree_estimators = [estimators_attr[i, 0] for i in range(estimators_attr.shape[0])]
                else:
                    # Fallback: flatten if unexpected structure
                    tree_estimators = estimators_attr.flatten()
                model_type = "gradient_boosting"
            else:
                # RandomForest, ExtraTrees: estimators_ is a list of tree objects
                tree_estimators = estimators_attr
                model_type = "ensemble"

            is_ensemble = True
        elif hasattr(self.base.model, "tree_"):
            # Single decision tree (DecisionTreeClassifier)
            tree_estimators = [self.base.model]
            is_ensemble = False
            model_type = "single_tree"
        else:
            return {
                "error": "The current model does not contain decision trees. Please use a tree-based model like DecisionTree, RandomForest, ExtraTrees, GradientBoosting, etc."
            }

        # Collect data for all trees (ensemble) or single tree
        trees_data = []
        for idx, tree_estimator in enumerate(tree_estimators):
            tree_data = self._extract_tree_structure(tree_estimator, idx, model_type)
            trees_data.append(tree_data)

            # For ensemble models, limit to first few trees to avoid overwhelming the frontend
            if is_ensemble and idx >= 4:  # Show max 5 trees
                break

        return {
            "trees": trees_data,
            "model_type": model_type,
            "num_trees": len(trees_data),
            "total_trees": len(tree_estimators) if is_ensemble else 1,
            "algorithm": type(self.base.model).__name__,
        }

    def _extract_tree_structure(self, tree_estimator, tree_idx: int, model_type: str = "ensemble") -> Dict[str, Any]:
        """Extract the structure of a single decision tree."""
        try:
            tree = tree_estimator.tree_
            feature_names = self.base.feature_names

            # Calculate tree accuracy on test set
            if self.base.X_test is not None and self.base.y_test is not None:
                try:
                    # For gradient boosting, individual trees predict continuous values
                    # We can't compute classification accuracy for individual trees in GB
                    if model_type == "gradient_boosting":
                        # For GB, use the full model accuracy as approximation
                        full_model_predictions = self.base.safe_predict(self.base.X_test)
                        tree_accuracy = float(np.mean(full_model_predictions == self.base.y_test))
                    else:
                        # For other ensemble methods, individual trees can be evaluated

                        tree_predictions = tree_estimator.predict(self.base.X_test)
                        tree_accuracy = float(accuracy_score(self.base.y_test, tree_predictions))
                except Exception as e:
                    # Fallback if accuracy calculation fails
                    tree_accuracy = 0.0
            else:
                tree_accuracy = 0.0

            # Get feature importance for this tree
            # Use the actual feature_importances_ of the individual tree if available
            if hasattr(tree_estimator, "feature_importances_"):
                tree_importance = tree_estimator.feature_importances_.tolist()
            else:
                tree_importance = None

            def recurse(node, depth):
                if tree.feature[node] != _tree.TREE_UNDEFINED:
                    feature = feature_names[tree.feature[node]]
                    threshold = float(tree.threshold[node])
                    samples = int(tree.n_node_samples[node])

                    # Calculate node purity (1 - gini impurity)
                    gini = float(tree.impurity[node])
                    purity = 1 - gini

                    return {
                        "type": "split",
                        "feature": feature,
                        "threshold": threshold,
                        "samples": samples,
                        "purity": purity,
                        "gini": gini,
                        "node_id": f"node_{node}",
                        "left": recurse(tree.children_left[node], depth + 1),
                        "right": recurse(tree.children_right[node], depth + 1),
                    }
                else:
                    # Leaf node
                    values = tree.value[node][0]
                    samples = int(tree.n_node_samples[node])

                    # For classification, calculate prediction and confidence
                    total_samples = sum(values)
                    if total_samples > 0:
                        prediction = np.argmax(values)
                        confidence = values[prediction] / total_samples
                    else:
                        prediction = 0
                        confidence = 0.0

                    # Calculate class distribution
                    class_distribution = {}
                    for i, val in enumerate(values):
                        class_distribution[f"class_{i}"] = val

                    return {
                        "type": "leaf",
                        "samples": samples,
                        "prediction": float(prediction),
                        "confidence": float(confidence),
                        "purity": 1.0,  # Leaf nodes are pure by definition
                        "gini": 0.0,  # Leaf nodes have no impurity
                        "node_id": f"node_{node}",
                        "class_distribution": class_distribution,
                    }

            # Calculate tree statistics
            total_nodes = tree.node_count
            leaf_nodes = sum(1 for i in range(total_nodes) if tree.feature[i] == _tree.TREE_UNDEFINED)
            max_depth = tree.max_depth

            return {
                "tree_index": tree_idx,
                "accuracy": tree_accuracy,
                "importance": tree_importance,
                "total_nodes": total_nodes,
                "leaf_nodes": leaf_nodes,
                "max_depth": max_depth,
                "tree_structure": recurse(0, 0),
            }

        except Exception as e:
            return {"tree_index": tree_idx, "error": f"Failed to extract tree structure: {str(e)}"}

    def get_tree_rules(self, payload: Dict[str, Any] = None, tree_idx: int = 0, max_depth: int = 3) -> Dict[str, Any]:
        """Extract decision rules from a specific tree."""
        self.base._ensure_loaded_with_payload(payload)

        # Get the tree
        if hasattr(self.base.model, "estimators_"):
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
                """Recursively extract rules from tree."""
                if conditions is None:
                    conditions = []

                rules = []

                # Stop if we've reached max depth
                if depth >= max_depth:
                    return rules

                # Check if node is a leaf
                if tree.children_left[node_id] == tree.children_right[node_id]:
                    # Leaf node - create rule
                    value = tree.value[node_id]
                    if value.ndim > 1:
                        class_probs = value[0] / value[0].sum()
                        predicted_class = int(np.argmax(class_probs))
                        confidence = float(np.max(class_probs))
                    else:
                        predicted_class = int(value[0])
                        confidence = 1.0

                    rule_text = " AND ".join(conditions) if conditions else "Always"

                    rules.append(
                        {
                            "rule": rule_text,
                            "conditions": conditions.copy(),
                            "predicted_class": predicted_class,
                            "confidence": confidence,
                            "samples": int(tree.n_node_samples[node_id]),
                            "support": float(tree.n_node_samples[node_id] / tree.n_node_samples[0]),
                        }
                    )
                else:
                    # Internal node
                    feature_idx = tree.feature[node_id]
                    feature_name = (
                        feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}"
                    )
                    threshold = float(tree.threshold[node_id])

                    # Left child (<=)
                    left_conditions = conditions + [f"{feature_name} <= {threshold:.3f}"]
                    rules.extend(extract_rules(tree.children_left[node_id], depth + 1, left_conditions))

                    # Right child (>)
                    right_conditions = conditions + [f"{feature_name} > {threshold:.3f}"]
                    rules.extend(extract_rules(tree.children_right[node_id], depth + 1, right_conditions))

                return rules

            rules = extract_rules(0)

            # Sort rules by confidence and support
            rules.sort(key=lambda x: (x["confidence"], x["support"]), reverse=True)

            return {"tree_index": tree_idx, "max_depth": max_depth, "rules": rules, "total_rules": len(rules)}

        except Exception as e:
            return {"error": f"Failed to extract tree rules: {str(e)}"}
