#!/usr/bin/env python3
"""
Test script for the enhanced class imbalance analysis
Tests both simulated and real prediction scenarios
"""

import io
import json
import pickle

import pandas as pd
import requests
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Configuration
API_BASE = "http://localhost:8001"  # Adjust port as needed


def create_test_data():
    """Create test datasets with known class imbalance"""
    print("Creating test datasets...")

    # Create reference data (balanced)
    X_ref, y_ref = make_classification(
        n_samples=1000, n_features=10, n_classes=2, weights=[0.7, 0.3], random_state=42  # 70-30 split
    )

    # Create current data (more imbalanced)
    X_curr, y_curr = make_classification(
        n_samples=800,
        n_features=10,
        n_classes=2,
        weights=[0.85, 0.15],  # 85-15 split (more imbalanced)
        random_state=123,
    )

    # Create DataFrames
    feature_names = [f"feature_{i}" for i in range(10)]

    reference_df = pd.DataFrame(X_ref, columns=feature_names)
    reference_df["target"] = y_ref

    current_df = pd.DataFrame(X_curr, columns=feature_names)
    current_df["target"] = y_curr

    return reference_df, current_df


def train_model_and_get_predictions(reference_df, current_df):
    """Train a model and get predictions for both datasets"""
    print("Training model and generating predictions...")

    # Prepare training data
    X_ref = reference_df.drop("target", axis=1)
    y_ref = reference_df["target"]

    X_curr = current_df.drop("target", axis=1)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_ref, y_ref)

    # Generate predictions
    ref_predictions = model.predict(X_ref).tolist()
    curr_predictions = model.predict(X_curr).tolist()

    # Save model for upload
    model_bytes = pickle.dumps(model)

    return model_bytes, ref_predictions, curr_predictions


def test_unified_upload_with_class_config():
    """Test the enhanced unified upload with class imbalance configuration"""
    print("\n=== Testing Enhanced Unified Upload ===")

    # Create test data
    reference_df, current_df = create_test_data()
    model_bytes, ref_predictions, curr_predictions = train_model_and_get_predictions(reference_df, current_df)

    # Save to CSV for upload
    reference_csv = reference_df.to_csv(index=False)
    current_csv = current_df.to_csv(index=False)

    # Prepare upload request
    files = {
        "reference_data": ("reference.csv", io.StringIO(reference_csv), "text/csv"),
        "current_data": ("current.csv", io.StringIO(current_csv), "text/csv"),
        "model_file": ("model.pkl", io.BytesIO(model_bytes), "application/octet-stream"),
    }

    data = {
        "target_column": "target",
        "reference_predictions": json.dumps(ref_predictions),
        "current_predictions": json.dumps(curr_predictions),
    }

    try:
        response = requests.post(f"{API_BASE}/api/v1/upload", files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            print("✅ Upload successful!")
            print(f"Session ID: {result['session_id']}")
            print(f"Class imbalance ready: {result['data']['class_imbalance_ready']}")
            print(f"Target column: {result['data']['target_column']}")
            print(f"Predictions provided: {result['data']['predictions_provided']}")
            return result["session_id"]
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None


def test_class_imbalance_analysis(session_id):
    """Test the enhanced class imbalance analysis"""
    print(f"\n=== Testing Class Imbalance Analysis (Session: {session_id}) ===")

    try:
        response = requests.get(f"{API_BASE}/data-drift/class-imbalance/analysis/{session_id}")

        if response.status_code == 200:
            result = response.json()
            data = result["data"]

            print("✅ Analysis successful!")
            print(f"Overall imbalance score: {data['overall_imbalance_score']:.3f}")
            print(f"Severity level: {data['severity_level']}")
            print(f"Minority class: {data['minority_class_label']}")
            print(
                f"Chi-square test: {data['chi_square_test']['interpretation']} (p={data['chi_square_test']['p_value']:.3f})"
            )

            # Check per-class performance metrics
            print(f"\nPer-class performance metrics (count: {len(data['per_class_performance'])}):")
            for metric in data["per_class_performance"][:6]:  # Show first 6 metrics
                print(
                    f"  Class {metric['class_label']} {metric['metric']}: {metric['reference_value']:.3f} -> {metric['current_value']:.3f} (Δ {metric['delta']:+.3f})"
                )

            print(f"\nAnalysis text: {data['analysis_text'][:150]}...")
            print(f"Recommendations: {len(data['recommendations'])} provided")
            print(f"Predictions available: {data['predictions_available']}")

            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(response.text)
            return False

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False


def test_configure_endpoint(session_id):
    """Test the configure endpoint"""
    print(f"\n=== Testing Configure Endpoint (Session: {session_id}) ===")

    try:
        # Create new predictions for testing
        reference_df, current_df = create_test_data()
        _, ref_predictions, curr_predictions = train_model_and_get_predictions(reference_df, current_df)

        config_data = {
            "target_column": "target",
            "reference_predictions": ref_predictions,
            "current_predictions": curr_predictions,
        }

        response = requests.post(f"{API_BASE}/data-drift/class-imbalance/configure/{session_id}", json=config_data)

        if response.status_code == 200:
            result = response.json()
            print("✅ Configuration successful!")
            print(f"Configuration: {result['configuration']}")
            return True
        else:
            print(f"❌ Configuration failed: {response.status_code}")
            print(response.text)
            return False

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Starting Class Imbalance Analysis Tests")
    print("=" * 50)

    # Test 1: Enhanced unified upload
    session_id = test_unified_upload_with_class_config()
    if not session_id:
        print("❌ Cannot proceed without successful upload")
        return

    # Test 2: Class imbalance analysis with real predictions
    analysis_success = test_class_imbalance_analysis(session_id)

    # Test 3: Configure endpoint
    configure_success = test_configure_endpoint(session_id)

    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Summary:")
    print(f"  Upload: {'✅' if session_id else '❌'}")
    print(f"  Analysis: {'✅' if analysis_success else '❌'}")
    print(f"  Configure: {'✅' if configure_success else '❌'}")

    if session_id and analysis_success:
        print("\n🎉 All critical features working! Production ready!")
    else:
        print("\n⚠️  Some tests failed. Check API server and logs.")


if __name__ == "__main__":
    main()
