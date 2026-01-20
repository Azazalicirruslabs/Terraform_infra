import json
import os
import tempfile

import numpy as np
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, MulticlassClassification, Report
from evidently.metrics import *
from evidently.presets import ClassificationPreset
from onnxruntime import InferenceSession


def drop_high_cardinality(df_train: pd.DataFrame, df_test: pd.DataFrame, thresh: float = 0.9):
    """
    Drop any string/object column where unique_ratio > thresh.
    Returns cleaned copies.
    """
    obj_cols = df_train.select_dtypes(include="object").columns
    to_drop = [c for c in obj_cols if df_train[c].nunique() / len(df_train) > thresh]
    if to_drop:
        print(f"🧹 Dropping high-cardinality cols: {to_drop}")
        df_train = df_train.drop(columns=to_drop, errors="ignore")
        df_test = df_test.drop(columns=to_drop, errors="ignore")
    return df_train.copy(), df_test.copy()


class ONNXProbaWrapper:
    """
    Wrap an ONNX InferenceSession so it behaves like sklearn's predict_proba.
    It picks out the probability output and, if that output is a list of dicts
    (ZipMap format), converts it to a proper 2D numpy float array.
    """

    def __init__(self, session: InferenceSession):
        self.session = session
        self.input_name = session.get_inputs()[0].name

        # Find the "probability" output name
        outputs = session.get_outputs()
        names = [o.name for o in outputs]
        self.proba_name = next((n for n in names if "probab" in n.lower()), None)
        if not self.proba_name:
            raise ValueError(f"No probability output in ONNX model. Available: {names}")
        print("🔍 Using ONNX proba output:", self.proba_name)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # 1) Run the session for only the proba output
        arr = X.values.astype(np.float32)
        outputs = self.session.run([self.proba_name], {self.input_name: arr})
        raw = outputs[0]

        # 2) ZipMap case: list of dicts
        if isinstance(raw, list) and isinstance(raw[0], dict):
            dicts = raw
            # Get a stable ordering of class labels
            classes = list(dicts[0].keys())
            try:
                # Try numeric sort
                classes = sorted(classes, key=float)
            except:
                pass
            # Build a true N×C array
            proba = np.array([[d[c] for c in classes] for d in dicts], dtype=float)

        else:
            # Already a numeric array (e.g. shape (N,) or (N,C))
            proba = np.array(raw, dtype=float)

        # 3) If you got a 1-D array of positives, stack it into [[1-p, p],…]
        if proba.ndim == 1:
            proba = np.vstack([1 - proba, proba]).T

        return proba

    def __getattr__(self, name):
        return getattr(self.session, name)


def generate_predictions(df: pd.DataFrame, features: list, model, task_type: str = "binary") -> pd.DataFrame:
    """
    Creates both:
      - df["prediction_proba"]: float score for the positive class
      - df["prediction"]: hard label (>=0.5)
    Works for sklearn pipelines *and* ONNX sessions.
    """
    df = df.copy()

    # wrap ONNX if needed
    if isinstance(model, InferenceSession):
        model = ONNXProbaWrapper(model)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df[features])
        # assume positive class = index 1
        df["prediction_proba"] = proba[:, 1]
        df["prediction"] = (df["prediction_proba"] >= 0.5).astype(int)

    elif hasattr(model, "predict"):
        # no proba available → use hard labels and mirror them as floats
        df["prediction"] = model.predict(df[features])
        df["prediction_proba"] = df["prediction"].astype(float)

    else:
        raise ValueError("Model has neither predict_proba nor predict().")

    return df


def get_data_definition(target_col: str, label_col: str, proba_col: str, task_type: str) -> DataDefinition:
    if task_type == "binary":
        return DataDefinition(
            classification=[
                BinaryClassification(target=target_col, prediction_labels=label_col, prediction_probas=proba_col)
            ]
        )
    elif task_type == "multiclass":
        return DataDefinition(
            classification=[
                MulticlassClassification(target=target_col, prediction=label_col, prediction_probas=proba_col)
            ]
        )
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")


def prepare_evidently_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model,
    target_col: str,
    task_type: str = "binary",
    high_card_thresh: float = 0.9,
):
    # 1) Drop high-cardinality features
    train_df, test_df = drop_high_cardinality(train_df, test_df, thresh=high_card_thresh)

    # 2) Build fresh feature list (exclude target)
    features = [c for c in train_df.columns if c != target_col]

    # 3) Generate predictions & probas
    train_df = generate_predictions(train_df, features, model, task_type)
    test_df = generate_predictions(test_df, features, model, task_type)

    # 4) Build Evidently DataDefinition
    data_def = get_data_definition(
        target_col=target_col, label_col="prediction", proba_col="prediction_proba", task_type=task_type
    )

    # 5) Convert to Evidently Datasets
    ref_data = Dataset.from_pandas(train_df, data_definition=data_def)
    cur_data = Dataset.from_pandas(test_df, data_definition=data_def)

    # 6) Run report
    # metrics = [ClassificationPreset(), RocAuc()],
    report = Report(metrics=[ClassificationPreset(), RocAuc()], include_tests=True)
    result = report.run(reference_data=ref_data, current_data=cur_data)

    # output_dir = "reports"
    # os.makedirs(output_dir, exist_ok=True)

    # # 2) Build a timestamped filename

    # html_path = os.path.join(output_dir, f"evidently_report_.html")

    # # 3) Save the report HTML to disk
    # result.save_html(html_path)
    # print(f"✅ Report saved to {html_path}")

    # 7) Extract HTML
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    tmp.close()
    result.save_html(tmp.name)
    with open(tmp.name, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp.name)

    # 8) Extract JSON
    # result.save_json("classification_report.json")
    try:
        json_dict = json.loads(result.json())
    except Exception as e:
        json_dict = {"error": f"Failed to parse report JSON: {e}"}

    return html, json_dict
