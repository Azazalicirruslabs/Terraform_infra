def test_classification_stats_success(client):
    payload = {
        "ref_dataset": "s3://bucket/ref.csv",
        "cur_dataset": "s3://bucket/cur.csv",
        "model": "model.joblib",
        "target_column": "label",
    }

    response = client.post("/classification/analysis/classification-stats", json=payload)

    assert response.status_code == 200


def test_missing_target_column(client):
    payload = {"ref_dataset": "s3://bucket/ref.csv", "cur_dataset": "s3://bucket/cur.csv", "model": "model.joblib"}

    response = client.post("/classification/analysis/classification-stats", json=payload)

    assert response.status_code == 400
    assert "target_column" in response.json()["detail"]


def test_missing_model(client):
    payload = {"ref_dataset": "s3://bucket/ref.csv", "cur_dataset": "s3://bucket/cur.csv", "target_column": "label"}

    response = client.post("/classification/analysis/classification-stats", json=payload)

    assert response.status_code == 400
    assert "model" in response.json()["detail"]
