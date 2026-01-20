from fastapi import APIRouter, Depends, HTTPException

from services.what_if.app.schema.llmexplainer import DashboardSummary, LLMExplainerResponse
from services.what_if.app.utils.llm_analyzer import LLMAnalyzer
from services.what_if.app.utils.session_manager import SessionManager
from shared.auth import get_current_user

session_manager = SessionManager()
llm_analyzer = LLMAnalyzer()
import pandas as pd

router = APIRouter(prefix="/what_if", tags=["What If Analysis"])


@router.post("/llm-explainer/{session_id}", response_model=LLMExplainerResponse)
async def get_llm_analysis(
    session_id: str, dashboard_data: DashboardSummary, current_user: dict = Depends(get_current_user)
):
    """Generate LLM-powered analysis of the dashboard metrics"""
    try:
        # Validate session exists
        session_data = session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(404, "Session not found")

        # Convert dashboard data to dictionary for LLM analysis
        dashboard_summary = {
            "model_info": {
                "n_features": dashboard_data.model_info.n_features,
                "n_samples": dashboard_data.model_info.n_samples,
                "target_column": dashboard_data.model_info.target_column,
            },
            "feature_importance": [
                {"feature": fi.feature, "importance": fi.importance, "rank": fi.rank}
                for fi in dashboard_data.feature_importance
            ],
            "shap_analysis": {
                "explanation": dashboard_data.shap_analysis.explanation,
                "confidence_level": dashboard_data.shap_analysis.confidence_level,
                "interactions": dashboard_data.shap_analysis.interactions,
            },
            "current_prediction": {
                "value": dashboard_data.current_prediction.value,
                "interpretation": dashboard_data.current_prediction.interpretation,
                "confidence": dashboard_data.current_prediction.confidence,
            },
            "what_if_insights": {
                "sensitivity": dashboard_data.what_if_insights.sensitivity,
                "key_drivers": dashboard_data.what_if_insights.key_drivers,
                "scenarios": dashboard_data.what_if_insights.scenarios,
            },
        }

        # Generate LLM analysis - both business and technical insights
        business_analysis = llm_analyzer.analyze_dashboard_metrics(dashboard_summary)
        technical_analysis = llm_analyzer.generate_technical_insights(dashboard_summary)
        # Add this right before the return statement
        # Validate responses before returning
        if not business_analysis or business_analysis.startswith("⛔"):
            print(f"❌ Invalid business analysis: {business_analysis[:200] if business_analysis else 'EMPTY'}")
            business_analysis = "Business analysis unavailable due to LLM service issues."

        if not technical_analysis or technical_analysis.startswith("⛔"):
            print(f"❌ Invalid technical analysis: {technical_analysis[:200] if technical_analysis else 'EMPTY'}")
            technical_analysis = "Technical analysis unavailable due to LLM service issues."

        # print(f"✅ Final response validation passed")

        return {
            "session_id": session_id,
            "business_analysis": business_analysis,
            "technical_analysis": technical_analysis,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        import traceback

        # Quick debug info
        error_details = traceback.format_exc()
        print(f"❌ LLM Analysis Failed:")
        print(f"   Error: {str(e)}")
        print(f"   Bedrock Available: {llm_analyzer.bedrock is not None}")
        print(f"   Full Trace:\n{error_details}")

        # Test each LLM function separately
        try:
            business_test = llm_analyzer.analyze_dashboard_metrics(dashboard_summary)
            print(
                f"   Business Analysis: {'SUCCESS' if business_test and not business_test.startswith('⛔') else 'FAILED'}"
            )
        except:
            print(f"   Business Analysis: EXCEPTION")

        try:
            technical_test = llm_analyzer.generate_technical_insights(dashboard_summary)
            print(
                f"   Technical Analysis: {'SUCCESS' if technical_test and not technical_test.startswith('⛔') else 'FAILED'}"
            )
        except:
            print(f"   Technical Analysis: EXCEPTION")

        raise HTTPException(500, f"LLM analysis error: {str(e)}")
