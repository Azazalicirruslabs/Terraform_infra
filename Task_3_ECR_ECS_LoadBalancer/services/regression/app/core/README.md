# Model Service Architecture

The ModelService has been refactored into a modular architecture with specialized services for better maintainability and organization.

## Architecture Overview

```
ModelService (Unified Interface)
├── BaseModelService (Core model and data management)
├── AnalysisService (Basic analysis operations)
├── FeatureService (Feature-related operations)
├── ClassificationService (Classification-specific analysis)
├── PredictionService (Individual predictions and what-if)
├── DependenceService (Feature dependence analysis)
├── InteractionService (Feature interaction analysis)
└── TreeService (Decision tree analysis)
```

## Service Breakdown

### 1. BaseModelService (`base_model_service.py`)
**Purpose**: Core model and data management
- Model loading (scikit-learn only - preserves tree structure for explainability)
- Data preprocessing and validation
- SHAP explainer initialization
- Common utility methods

**Key Methods**:
- `load_model_and_data()`
- `load_model_and_separate_datasets()`
- `_get_classification_metrics()`
- `_get_shap_values_for_analysis()`

### 2. AnalysisService (`analysis_service.py`)
**Purpose**: Basic model analysis operations
- Model overview and metadata
- Classification statistics
- Dataset comparison
- Instance listing

**API Endpoints Served**:
- `/analysis/overview`
- `/analysis/classification-stats`
- `/analysis/instances`
- `/analysis/dataset-comparison`

**Key Methods**:
- `get_model_overview()`
- `get_classification_stats()`
- `list_instances()`
- `get_dataset_comparison()`

### 3. FeatureService (`feature_service.py`)
**Purpose**: Feature-related operations
- Feature importance computation
- Feature metadata extraction
- Correlation analysis
- Basic feature interactions

**API Endpoints Served**:
- `/api/features`
- `/api/correlation`
- `/api/feature-importance`
- `/analysis/feature-importance`
- `/analysis/feature-interactions`

**Key Methods**:
- `get_feature_importance()`
- `get_feature_metadata()`
- `compute_correlation()`
- `compute_feature_importance_advanced()`
- `get_feature_interactions()`

### 4. ClassificationService (`classification_service.py`)
**Purpose**: Classification-specific analysis
- ROC curve analysis
- Precision-recall curves
- Threshold optimization
- Binary/multiclass metrics

**API Endpoints Served**:
- `/api/roc-analysis`
- `/api/threshold-analysis`

**Key Methods**:
- `roc_analysis()`
- `threshold_analysis()`

### 5. PredictionService (`prediction_service.py`)
**Purpose**: Individual prediction analysis
- Single instance explanations
- What-if scenario analysis
- SHAP-based explanations

**API Endpoints Served**:
- `/api/individual-prediction`
- `/analysis/explain-instance/{instance_idx}`
- `/analysis/what-if`

**Key Methods**:
- `individual_prediction()`
- `explain_instance()`
- `perform_what_if()`

### 6. DependenceService (`dependence_service.py`)
**Purpose**: Feature dependence analysis
- Partial dependence plots
- SHAP dependence plots
- Individual Conditional Expectation (ICE) plots

**API Endpoints Served**:
- `/analysis/feature-dependence/{feature_name}`
- `/api/partial-dependence`
- `/api/shap-dependence`
- `/api/ice-plot`

**Key Methods**:
- `get_feature_dependence()`
- `partial_dependence()`
- `shap_dependence()`
- `ice_plot()`

### 7. InteractionService (`interaction_service.py`)
**Purpose**: Feature interaction analysis
- Feature interaction networks
- Pairwise feature analysis
- Interaction strength computation

**API Endpoints Served**:
- `/api/interaction-network`
- `/api/pairwise-analysis`

**Key Methods**:
- `interaction_network()`
- `pairwise_analysis()`

### 8. TreeService (`tree_service.py`)
**Purpose**: Decision tree analysis
- Tree structure extraction
- Decision rule generation
- Tree visualization data

**API Endpoints Served**:
- `/analysis/decision-tree`

**Key Methods**:
- `get_decision_tree()`
- `get_tree_rules()`

## Benefits of the New Architecture

### 1. **Separation of Concerns**
Each service has a clear, focused responsibility, making the code easier to understand and maintain.

### 2. **Improved Testability**
Individual services can be tested in isolation, making unit testing more straightforward.

### 3. **Better Code Organization**
Related functionality is grouped together, reducing the cognitive load when working on specific features.

### 4. **Easier Maintenance**
Changes to specific functionality (e.g., ROC analysis) only require modifications to the relevant service.

### 5. **Scalability**
New analysis features can be added by creating new services or extending existing ones without affecting the entire codebase.

### 6. **Caching Independence**
Each service can implement its own caching strategy based on its specific needs.

## Backward Compatibility

The `ModelService` class maintains the same public interface as the original implementation, ensuring that existing API endpoints continue to work without modification. It acts as a facade that delegates calls to the appropriate specialized services.

## Usage Examples

### Direct Service Usage
```python
# Initialize services
base_service = BaseModelService()
analysis_service = AnalysisService(base_service)

# Load model and data
base_service.load_model_and_data(model_path, data_path, target_column)

# Get model overview
overview = analysis_service.get_model_overview()
```

### Unified Interface (Recommended)
```python
# Use the unified ModelService (maintains backward compatibility)
model_service = ModelService()
model_service.load_model_and_data(model_path, data_path, target_column)
overview = model_service.get_model_overview()
```

## API Endpoint Mapping

| Endpoint | Service | Method |
|----------|---------|---------|
| `/analysis/overview` | AnalysisService | `get_model_overview()` |
| `/analysis/classification-stats` | AnalysisService | `get_classification_stats()` |
| `/api/features` | FeatureService | `get_feature_metadata()` |
| `/api/roc-analysis` | ClassificationService | `roc_analysis()` |
| `/api/individual-prediction` | PredictionService | `individual_prediction()` |
| `/api/partial-dependence` | DependenceService | `partial_dependence()` |
| `/api/interaction-network` | InteractionService | `interaction_network()` |
| `/analysis/decision-tree` | TreeService | `get_decision_tree()` |

## Future Enhancements

1. **Async Support**: Services can be enhanced to support asynchronous operations for long-running computations.

2. **Caching Strategies**: Implement more sophisticated caching mechanisms per service.

3. **Service Registration**: Implement a service registry pattern for dynamic service discovery.

4. **Plugin Architecture**: Allow third-party services to be plugged into the system.

5. **Performance Monitoring**: Add performance monitoring and metrics collection per service.

## Migration Notes

- The original `model_service.py` has been backed up as `model_service_backup.py`
- All existing API endpoints continue to work without changes
- No changes required to the frontend or API consumers
- The refactor is transparent to external users
