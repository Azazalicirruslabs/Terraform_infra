from typing import Any, Dict, List, Optional

from .analysis_service import AnalysisService
from .base_model_service import BaseModelService
from .dependence_service import DependenceService
from .feature_service import FeatureService
from .interaction_service import InteractionService
from .prediction_service import PredictionService
from .tree_service import TreeService


class ModelService:
    """
    Unified model service that orchestrates all the specialized services.
    This maintains the same interface as the original ModelService for backward compatibility.
    """

    def __init__(self):
        # Initialize base service
        self.base = BaseModelService()

        # Initialize specialized services
        self.analysis = AnalysisService(self.base)
        self.feature = FeatureService(self.base)
        self.prediction = PredictionService(self.base)
        self.dependence = DependenceService(self.base)
        self.interaction = InteractionService(self.base)
        self.tree = TreeService(self.base)

        print("ModelService initialized with specialized services.")

    # === Model Loading Methods (delegated to base service) ===
    def load_model_and_datasets(
        self,
        model_path: str,
        data_path: str = None,
        train_data_path: str = None,
        test_data_path: str = None,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Unified method to load model and dataset(s) from local files or S3."""
        return self.base.load_model_and_datasets(
            model_path, data_path, train_data_path, test_data_path, target_column, test_size, random_state
        )

    # === Analysis Methods (delegated to analysis service) ===

    def get_model_overview(self) -> Dict[str, Any]:
        """Get comprehensive model overview."""
        return self.analysis.get_model_overview()

    def get_regression_stats(self) -> Dict[str, Any]:
        """Get detailed regression statistics."""
        return self.analysis.get_regression_stats()

    def list_instances(self, sort_by: str = "prediction", limit: int = 100) -> Dict[str, Any]:
        """List instances for UI selection."""
        return self.analysis.list_instances(sort_by, limit)

    def get_dataset_comparison(self) -> Dict[str, Any]:
        """Compare train vs test dataset characteristics."""
        return self.analysis.get_dataset_comparison()

    # === Feature Methods (delegated to feature service) ===

    def get_feature_importance(self, method: str) -> Dict[str, Any]:
        """Get feature importance using specified method."""
        return self.feature.get_feature_importance(method)

    def get_feature_metadata(self) -> Dict[str, Any]:
        """Get feature metadata."""
        return self.feature.get_feature_metadata()

    def compute_correlation(self, selected_features: List[str]) -> Dict[str, Any]:
        """Compute correlation matrix for selected features."""
        return self.feature.compute_correlation(selected_features)

    def compute_feature_importance_advanced(
        self, method: str = "shap", sort_by: str = "importance", top_n: int = 1000, visualization: str = "bar"
    ) -> Dict[str, Any]:
        """Compute advanced feature importance with detailed analysis."""
        return self.feature.compute_feature_importance_advanced(method, sort_by, top_n, visualization)

    def get_feature_interactions(self, feature1: str, feature2: str) -> Dict[str, Any]:
        """Get feature interaction analysis."""
        return self.feature.get_feature_interactions(feature1, feature2)

    # === Prediction Methods (delegated to prediction service) ===

    def individual_prediction(self, instance_idx: int) -> Dict[str, Any]:
        """Get detailed prediction analysis for a single instance."""
        return self.prediction.individual_prediction(instance_idx)

    def explain_instance(self, instance_idx: int) -> Dict[str, Any]:
        """Explain a single instance prediction with SHAP analysis."""
        return self.prediction.explain_instance(instance_idx)

    def perform_what_if(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Perform what-if analysis by modifying feature values."""
        return self.prediction.perform_what_if(features)

    # === Dependence Methods (delegated to dependence service) ===

    def get_feature_dependence(self, feature_name: str) -> Dict[str, Any]:
        """Get basic feature dependence using SHAP values."""
        return self.dependence.get_feature_dependence(feature_name)

    def partial_dependence(self, feature_name: str, num_points: int = 20) -> Dict[str, Any]:
        """Compute partial dependence plot data."""
        return self.dependence.partial_dependence(feature_name, num_points)

    def shap_dependence(self, feature_name: str, color_by: Optional[str] = None) -> Dict[str, Any]:
        """Compute SHAP dependence plot data."""
        return self.dependence.shap_dependence(feature_name, color_by)

    def ice_plot(self, feature_name: str, num_points: int = 20, num_instances: int = 20) -> Dict[str, Any]:
        """Compute Individual Conditional Expectation plot data."""
        return self.dependence.ice_plot(feature_name, num_points, num_instances)

    # === Interaction Methods (delegated to interaction service) ===

    def interaction_network(self, top_k: int = 30, sample_rows: int = 200) -> Dict[str, Any]:
        """Compute feature interaction network analysis."""
        return self.interaction.interaction_network(top_k, sample_rows)

    def pairwise_analysis(
        self, feature1: str, feature2: str, color_by: Optional[str] = None, sample_size: int = 1000
    ) -> Dict[str, Any]:
        """Perform detailed pairwise analysis between two features."""
        return self.interaction.pairwise_analysis(feature1, feature2, color_by, sample_size)

    # === Tree Methods (delegated to tree service) ===

    def get_decision_tree(self) -> Dict[str, Any]:
        """Extract decision tree structure for visualization."""
        return self.tree.get_decision_tree()

    def get_tree_rules(self, tree_idx: int = 0, max_depth: int = 3) -> Dict[str, Any]:
        """Extract decision rules from a specific tree."""
        return self.tree.get_tree_rules(tree_idx, max_depth)

    # === Utility Properties for backward compatibility ===

    @property
    def model(self):
        """Access to the underlying model."""
        return self.base.model

    @property
    def X_train(self):
        """Access to training features."""
        return self.base.X_train

    @property
    def y_train(self):
        """Access to training target."""
        return self.base.y_train

    @property
    def X_test(self):
        """Access to test features."""
        return self.base.X_test

    @property
    def y_test(self):
        """Access to test target."""
        return self.base.y_test

    @property
    def X_df(self):
        """Access to feature dataframe (training data)."""
        return self.base.X_df

    @property
    def y_s(self):
        """Access to target series (training data)."""
        return self.base.y_s

    @property
    def feature_names(self):
        """Access to feature names."""
        return self.base.feature_names

    @property
    def target_name(self):
        """Access to target name."""
        return self.base.target_name

    @property
    def explainer(self):
        """Access to SHAP explainer."""
        return self.base.explainer

    @property
    def shap_values(self):
        """Access to SHAP values."""
        return self.base.shap_values

    @property
    def model_info(self):
        """Access to model metadata."""
        return self.base.model_info
