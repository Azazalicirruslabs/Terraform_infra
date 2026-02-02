from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class MetadataResponse(BaseModel):
    session_id: str
    target_column: Optional[str] = None
    feature_names: List[str]
    processed_feature_names: List[str]
    uses_clean_names: bool
    n_features: int
    n_samples: int
    explainer_type: Optional[str] = None
    shap_available: bool
    model_type: Optional[str] = None
    feature_info: Dict[str, Any] = {}
    created_at: Optional[str] = None
