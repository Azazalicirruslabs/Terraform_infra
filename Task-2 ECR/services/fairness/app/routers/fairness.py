"""
main.py
FastAPI backend for Fairness Analysis.
Supports user-uploaded models and datasets, flexible mitigation, and robust error handling.
"""

import base64
import io
import logging
import traceback
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from fairlearn.metrics import (  # Corrected import for fairlearn metrics
    MetricFrame,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, ExponentiatedGradient, GridSearch
from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import JSONResponse
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (  # Keep accuracy_score from sklearn.metrics
    accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from services.fairness.app.database.connections import get_db

# Import necessary functions from the refactored functions.py
from services.fairness.app.routers.functions import FairnessAnalyzer
from services.fairness.app.schemas.fairness_schemas import DatasetPreviewResponse, FullResponse, ListDatasetsResponse
from shared.auth import get_current_user

# Instantiate the FairnessAnalyzer
fairness_analyzer = FairnessAnalyzer()

# get_s3_file_metadata = fairness_analyzer.get_s3_file_metadata
download_and_load_dataframe = fairness_analyzer.download_and_load_dataframe
load_and_preprocess_data = fairness_analyzer.load_and_preprocess_data
prepare_training_and_test_data = fairness_analyzer.prepare_training_and_test_data
generate_plot_descriptions = fairness_analyzer.generate_plot_descriptions
download_and_load_model = fairness_analyzer.download_and_load_model

routers = APIRouter(prefix="/fairness", tags=["Fairness Old"])


def plot_metrics_local(frame, title_suffix=""):
    """
    Generates and encodes two plots (Accuracy by Group and Selection Rate by Group)
    from a MetricFrame object into base64 PNG images.
    """
    plots = {}
    fig_accuracy, ax_accuracy = plt.subplots(figsize=(8, 5))
    frame.by_group["accuracy"].plot.bar(ax=ax_accuracy, rot=45, color="skyblue")
    ax_accuracy.set_title(f"Accuracy by Group {title_suffix}")
    ax_accuracy.set_ylabel("Accuracy")
    ax_accuracy.set_xlabel("Group")
    plt.tight_layout()
    buf_accuracy = io.BytesIO()
    plt.savefig(buf_accuracy, format="png")
    plt.close(fig_accuracy)
    plots["accuracy_plot"] = base64.b64encode(buf_accuracy.getvalue()).decode("utf-8")

    fig_selection, ax_selection = plt.subplots(figsize=(8, 5))
    frame.by_group["selection_rate"].plot.bar(ax=ax_selection, rot=45, color="lightcoral")
    ax_selection.set_title(f"Selection Rate by Group {title_suffix}")
    ax_selection.set_ylabel("Selection Rate")
    ax_selection.set_xlabel("Group")
    plt.tight_layout()

    buf_selection = io.BytesIO()
    plt.savefig(buf_selection, format="png")
    plt.close(fig_selection)
    plots["selection_rate_plot"] = base64.b64encode(buf_selection.getvalue()).decode("utf-8")

    if "true_positive_rate" in frame.by_group.columns:
        fig_tpr, ax_tpr = plt.subplots(figsize=(10, 6))
        frame.by_group["true_positive_rate"].plot.bar(ax=ax_tpr, rot=45, color="lightgreen")
        ax_tpr.set_title(f"True Positive Rate by Group {title_suffix}")
        ax_tpr.set_ylabel("True Positive Rate")
        ax_tpr.set_xlabel("Group")
        ax_tpr.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        buf_tpr = io.BytesIO()
        plt.savefig(buf_tpr, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig_tpr)
        plots["tpr_plot"] = base64.b64encode(buf_tpr.getvalue()).decode("utf-8")

    return plots


def evaluate_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    A_test: pd.Series,
    is_pretrained: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, str], MetricFrame, np.ndarray, Optional[np.ndarray]]:
    """
    Evaluates a model (either pre-trained or fits a new one) and computes fairness metrics.

    Args:
        model: The model to evaluate. If not pre-trained, must be a classifier with fit()
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        A_test: Sensitive attribute for test set
        is_pretrained: Whether the model is already trained

    Returns:
        Tuple of (metrics_dict, plots_dict, metric_frame, y_pred, y_prob)
    """
    # Train model if not pre-trained
    if not is_pretrained and hasattr(model, "fit"):
        model.fit(X_train, y_train)

    # Get predictions
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # preds = wrapper.predict(X_test)
        # probs = wrapper.predict_proba(X_test)  # if implemented
        # print(preds.shape, probs.shape if probs is not None else None)

        # --- SANITY CHECKS ---
        print("\n--- SANITY CHECKS ---")
        unique_preds, counts = np.unique(y_pred, return_counts=True)
        print(f"Unique predictions: {dict(zip(unique_preds, counts))}")
        if len(unique_preds) == 1:
            logging.warning(
                f"Model predicted only one class ({unique_preds[0]}) for all test samples. Fairness metrics may be misleading."
            )

        group_counts = A_test.value_counts()
        print(f"Sensitive feature groups in test set:\n{group_counts.to_string()}")
        if len(group_counts) < 2:
            logging.warning(
                "Test data contains less than two sensitive groups. Group-wise fairness metrics cannot be computed."
            )
        print("---------------------\n")
        # --- END SANITY CHECKS ---
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Model prediction failed. Ensure it's compatible with the data. Error: {str(e)}"
        )
    # --- INSTRUMENTATION ---
    print("\n--- DEBUGGING INFO ---")
    print(f"Sensitive feature groups in test set (A_test):\n{A_test.value_counts().to_string()}")
    print(f"Unique predictions generated by model (y_pred): {np.unique(y_pred, return_counts=True)}")
    if y_prob is not None:
        print(f"Shape of probability predictions (y_prob): {y_prob.shape}")
    else:
        print("Probability predictions (y_prob): Not available.")
    print("----------------------\n")
    # Define metrics to compute
    metrics = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
    }

    # Add AUC-ROC if we have probabilities
    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score

    # Compute metrics
    metric_frame = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=A_test)

    # Format results
    group_metrics = {}
    for group in metric_frame.by_group.index:
        group_metrics[str(group)] = {
            "accuracy": float(metric_frame.by_group.loc[group, "accuracy"]),
            "selection_rate": float(metric_frame.by_group.loc[group, "selection_rate"]),
            "tpr": float(metric_frame.by_group.loc[group, "true_positive_rate"]),
            "fpr": float(metric_frame.by_group.loc[group, "false_positive_rate"]),
        }
        if "auc_roc" in metrics:
            group_metrics[str(group)]["auc_roc"] = float(metric_frame.by_group.loc[group, "auc_roc"])

    # Generate plots
    plots = plot_metrics_local(metric_frame, "(Before Mitigation)")

    return group_metrics, plots, metric_frame, y_pred, y_prob


def apply_mitigation(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    A_train: pd.Series,
    A_test: pd.Series,
    is_pretrained: bool = False,
    strategy: Optional[str] = None,
    y_pred: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]], Dict[str, str], MetricFrame, Dict[str, float]]:
    """
    Applies fairness mitigation appropriate for the model type.
    Returns:
      - y_pred_mitigated: mitigated predictions
      - group_metrics: per-group {accuracy, selection_rate, tpr, fpr}
      - plots: dict of plot references
      - metric_frame: the Fairlearn MetricFrame after mitigation
      - overall_metrics: dict with accuracy, selection_rate, tpr, fpr
    """
    try:
        # ------ Mitigation step ------
        if is_pretrained:
            # Post-processing via ThresholdOptimizer
            if not hasattr(model, "predict_proba") or y_prob is None:
                raise HTTPException(
                    status_code=400, detail="Model must support predict_proba() and y_prob for post-processing"
                )
            postprocess_est = ThresholdOptimizer(
                estimator=model, constraints="demographic_parity", objective="accuracy_score", prefit=True
            )
            postprocess_est.fit(X_test, y_test, sensitive_features=A_test)
            y_pred_mitigated = postprocess_est.predict(X_test, sensitive_features=A_test, random_state=42)

        else:
            strat = (strategy or "exponentiated_gradient").lower()

            if strat == "grid_search":
                mitigator = GridSearch(
                    estimator=model,
                    constraints=DemographicParity(),
                )
                mitigator.fit(X_train, y_train, sensitive_features=A_train)
                y_pred_mitigated = mitigator.predict(X_test)

            elif strat == "reweighing":
                # Require binary sensitive feature
                unique_vals = sorted(A_train.unique())
                if len(unique_vals) != 2:
                    raise HTTPException(
                        status_code=400, detail=f"Reweighing requires binary sensitive feature, got {unique_vals}"
                    )
                sens_name = A_train.name or "sensitive_feature"
                unprivileged_groups = [{sens_name: unique_vals[0]}]
                privileged_groups = [{sens_name: unique_vals[1]}]

                train_bld = BinaryLabelDataset(
                    df=pd.concat([X_train, y_train, A_train], axis=1),
                    label_names=[y_train.name],
                    protected_attribute_names=[sens_name],
                )
                reweigher = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
                train_bld_transf = reweigher.fit_transform(train_bld)
                sample_weights = train_bld_transf.instance_weights

                model.fit(X_train, y_train, sample_weight=sample_weights)
                y_pred_mitigated = model.predict(X_test)

            elif strat == "postprocessing":
                postproc = ThresholdOptimizer(estimator=model, constraints="demographic_parity", prefit=True)
                postproc.fit(X_test, y_test, sensitive_features=A_test)
                y_pred_mitigated = postproc.predict(X_test, sensitive_features=A_test, random_state=42)

            else:  # exponentiated_gradient
                mitigator = ExponentiatedGradient(
                    estimator=model,
                    constraints=DemographicParity(),
                )
                mitigator.fit(X_train, y_train, sensitive_features=A_train)
                y_pred_mitigated = mitigator.predict(X_test)

        # ------ Compute overall metrics (same four as before) ------
        overall_metrics = {
            "accuracy": accuracy_score(y_test, y_pred_mitigated),
            "selection_rate": float(np.mean(y_pred_mitigated)),
            "tpr": float(recall_score(y_test, y_pred_mitigated, pos_label=1, zero_division=0)),
            "fpr": float(1 - recall_score(y_test, y_pred_mitigated, pos_label=0, zero_division=1)),
        }

        # ------ Build MetricFrame for group-wise metrics ------
        metric_fns = {
            "accuracy": accuracy_score,
            "selection_rate": lambda yt, yp: float(np.mean(yp)),
            "tpr": lambda yt, yp: float(recall_score(yt, yp, pos_label=1, zero_division=0)),
            "fpr": lambda yt, yp: float(1 - recall_score(yt, yp, pos_label=0, zero_division=1)),
        }
        # Always use the mitigated predictions for metrics
        metric_frame = MetricFrame(
            metrics=metric_fns, y_true=y_test, y_pred=y_pred_mitigated, sensitive_features=A_test
        )

        # ------ Extract group-wise metrics ------
        group_metrics: Dict[str, Dict[str, float]] = {}
        for group in metric_frame.by_group.index:
            row = metric_frame.by_group.loc[group]
            group_metrics[str(group)] = {
                "accuracy": float(row["accuracy"]),
                "selection_rate": float(row["selection_rate"]),
                "tpr": float(row["tpr"]),
                "fpr": float(row["fpr"]),
            }

        # ------ Generate plots ------
        plots = plot_metrics_local(metric_frame, "(After Mitigation)")

        # NEW LINE at the end of apply_mitigation
        return y_pred_mitigated, group_metrics, plots, metric_frame, overall_metrics

    except Exception as e:
        error_msg = f"Error applying mitigation: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        # Re-raise the exception to be handled by the main endpoint
        raise RuntimeError(error_msg)


@routers.get("/list_datasets_model/{project_id}", response_model=ListDatasetsResponse)
async def list_datasets(project_id: str, current_user=Depends(get_current_user)):
    """
    Endpoint to list available dataset files (metadata) from the S3 bucket.
    """
    token = current_user.get("token")
    try:
        file_metadata = fairness_analyzer.get_s3_file_metadata(token, project_id)
        if file_metadata is None:
            # This case should ideally be caught by the RuntimeError below,
            # but as a fallback for unexpected None returns.
            raise HTTPException(status_code=500, detail="Failed to retrieve dataset list from S3: Unknown error.")
        return {"files": file_metadata}
    except RuntimeError as e:
        # Catch the specific RuntimeError from functions.py and pass its message
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dataset list from S3: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while listing datasets: {e}")


@routers.get("/preview_dataset", response_model=DatasetPreviewResponse)
async def preview_dataset(file_url: str, current_user=Depends(get_current_user)):
    """
    Endpoint to download a dataset from a given URL and return its first few rows as JSON.
    This acts as a proxy to avoid CORS issues when fetching directly from frontend.
    """
    try:
        df = download_and_load_dataframe(file_url)
        # Return first 5 rows and column names
        return JSONResponse(content={"columns": df.columns.tolist(), "data": df.head(5).to_dict(orient="records")})
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to preview dataset: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during preview: {e}")


@routers.post("/perform_analysis", response_model=FullResponse)
async def perform_analysis(
    reference_file_url: str = Form(...),
    current_file_url: str = Form(None),
    target_column: str = Form(...),
    sensitive_feature: str = Form(...),
    mitigation_strategy: str = Form("exponentiated_gradient"),
    model_url: str = Form(None),
    model_format: str = Form("onnx"),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Endpoint to perform fairness analysis on a dataset and model. (V2 - Enhanced Logic)
    Accepts dataset URL, target col, sensitive feature, mitigation, and optional model.
    Returns detailed fairness metrics, scores, plots, and LLM recommendations.
    """
    try:
        # 1. Load and Preprocess Data (This part remains the same)
        df_reference = download_and_load_dataframe(reference_file_url)
        df_reference_processed = fairness_analyzer.load_and_preprocess_data(
            df_reference.copy(), target_column, [sensitive_feature]
        )

        X_all = df_reference_processed.drop(columns=[target_column, sensitive_feature], errors="ignore")
        y_all = df_reference_processed[target_column]
        A_all = df_reference_processed[sensitive_feature]

        X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
            X_all, y_all, A_all, test_size=0.3, random_state=42, stratify=y_all
        )

        # 2. Model Selection (This part remains the same)
        is_pretrained = model_url is not None
        if is_pretrained:
            model = download_and_load_model(model_url, model_format)
        else:
            model = RandomForestClassifier(random_state=42)

        # --- NEW: Logic to validate and adjust mitigation strategy ---
        in_processing_strategies = ["exponentiated_gradient", "grid_search", "reweighing"]
        if is_pretrained and mitigation_strategy in in_processing_strategies:
            logging.warning(
                f"Incompatible mitigation '{mitigation_strategy}' for a pre-trained model. Switching to 'postprocessing'."
            )
            mitigation_strategy = "postprocessing"

        # 3. Evaluate Before Mitigation (Using base model)
        base_model = model if is_pretrained else RandomForestClassifier(random_state=42)
        if not is_pretrained:
            base_model.fit(X_train, y_train)

        y_pred_before = base_model.predict(X_test)
        y_prob_before = None
        if hasattr(base_model, "predict_proba"):
            try:
                y_prob_before = base_model.predict_proba(X_test)[:, 1]
            except (AttributeError, ValueError):
                y_prob_before = None  # Handle cases where it might fail

        # 4. Apply Mitigation
        model_for_mitigation = model if is_pretrained else RandomForestClassifier(random_state=42)

        # --- THIS IS THE CORRECTED CALL ---
        y_pred_after, _, _, _, _ = apply_mitigation(
            model=model_for_mitigation,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            A_train=A_train,
            A_test=A_test,
            strategy=mitigation_strategy,
            y_pred=y_pred_before,
            y_prob=y_prob_before,
        )
        # The variable y_pred_after is now correctly a NumPy array.

        # 5. --- REPLACEMENT START: Use the new fairness calculation logic ---
        fairness_metrics_before = fairness_analyzer.calculate_fairness_metrics(y_test.to_numpy(), y_pred_before, A_test)

        # This call will now succeed because y_pred_after is an array
        fairness_metrics_after = fairness_analyzer.calculate_fairness_metrics(y_test.to_numpy(), y_pred_after, A_test)

        overall_score_before = fairness_analyzer.calculate_overall_fairness_score(fairness_metrics_before)
        overall_score_after = fairness_analyzer.calculate_overall_fairness_score(fairness_metrics_after)

        bias_detected = fairness_analyzer.detect_bias(fairness_metrics_before)
        recommendations = fairness_analyzer.recommend_mitigation_strategies(bias_detected)

        # 6. Generate Plots
        metrics_for_plotting = {
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
        }
        mf_before = MetricFrame(
            metrics=metrics_for_plotting, y_true=y_test, y_pred=y_pred_before, sensitive_features=A_test
        )
        mf_after = MetricFrame(
            metrics=metrics_for_plotting, y_true=y_test, y_pred=y_pred_after, sensitive_features=A_test
        )

        plots_before = plot_metrics_local(mf_before, "(Before Mitigation)")
        plots_after = plot_metrics_local(mf_after, "(After Mitigation)")

        # 7. Generate LLM Report
        llm_summary_data = {
            "Fairness Metrics Before Mitigation": fairness_metrics_before,
            "Fairness Metrics After Mitigation": fairness_metrics_after,
            "Overall Fairness Score Before": asdict(overall_score_before),
            "Overall Fairness Score After": asdict(overall_score_after),
        }
        llm_analysis_report = fairness_analyzer.generate_plot_descriptions(
            title=target_column, data_dict=llm_summary_data
        )

        # 8. --- REPLACEMENT END: Structure the new, improved response ---
        response_data = {
            "overview": {
                "fairness_score_before": asdict(overall_score_before),
                "fairness_score_after": asdict(overall_score_after),
                "mitigation_strategy_used": mitigation_strategy,
            },
            "features": [sensitive_feature],
            "metrics": {
                sensitive_feature: {
                    "fairness_metrics_before": fairness_metrics_before,
                    "fairness_metrics_after": fairness_metrics_after,
                    "group_performance_before": mf_before.by_group.to_dict("index"),
                    "group_performance_after": mf_after.by_group.to_dict("index"),
                    "bias_detected": bias_detected,
                    "recommendations": recommendations,
                }
            },
            "plots": {"before_mitigation": plots_before, "after_mitigation": plots_after},
            "llm_analysis_report": llm_analysis_report,
        }
        # # 9. Store in DB with unique analysis_id and project_id
        # user_id = current_user.get("user_id")
        # # Generate analysis_id
        # analysis_id = generate_analysis_id(user_id=user_id, data_preview=response_data)
        # def get_existing():
        #     return db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()
        # existing = await run_in_threadpool(get_existing)

        # if not existing:
        #     new_entry = AnalysisResult(
        #         analysis_id=analysis_id,
        #         user_id=user_id,
        #         analysis_type="fairness",
        #         analysis_tab="perform_analysis",
        #         project_id="N/A",
        #         json_result=json.dumps(response_data),
        #         )
        #     try:
        #         await run_in_threadpool(db.add, new_entry)
        #         await run_in_threadpool(db.commit)
        #         await run_in_threadpool(db.refresh, new_entry)
        #     except Exception as e:
        #         await run_in_threadpool(db.rollback)
        #         logging.error(f"Database error during AnalysisResult commit: {e}")
        #         raise HTTPException(status_code=500, detail=f"Database error: unable to store analysis result")

        return JSONResponse(content=response_data)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Data processing error: {ve}")
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        logging.error(f"An unexpected error occurred during analysis: {e}\nTraceback:\n{tb}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during analysis: {e}")


@routers.get("/mitigation_strategies")
async def mitigation_strategies(current_user=Depends(get_current_user)):
    """
    Endpoint to return available mitigation strategies for the frontend dropdown.
    """
    return {"strategies": fairness_analyzer.get_mitigation_strategies()}
