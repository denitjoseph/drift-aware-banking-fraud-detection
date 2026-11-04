"""Tests for dataset upload, validation, preview, and model training endpoints."""

import io
import time
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.model_service import FEATURE_COLUMNS


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan events executed."""
    with TestClient(app) as test_client:
        yield test_client


def _generate_synthetic_csv_bytes(num_rows: int = 30, include_class: bool = True, num_fraud: int = 6) -> bytes:
    """Helper to generate a small synthetic dataset for testing."""
    data = {}
    data["Time"] = [float(i * 10) for i in range(num_rows)]
    for i in range(1, 29):
        data[f"V{i}"] = [0.05 * i + (j * 0.01) for j in range(num_rows)]
    data["Amount"] = [25.0 + float(i * 5) for i in range(num_rows)]

    if include_class:
        # Create a mix of 0 (legitimate) and 1 (fraud)
        classes = [0] * (num_rows - num_fraud) + [1] * num_fraud
        data["Class"] = classes

    df = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def test_upload_non_csv_rejected(client):
    """Uploading a non-csv file should result in HTTP 400."""
    response = client.post(
        "/datasets/upload",
        files={"file": ("test.txt", b"some text content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only CSV files" in response.json()["detail"]


def test_upload_empty_file_rejected(client):
    """Uploading an empty file should result in HTTP 400."""
    response = client.post(
        "/datasets/upload",
        files={"file": ("empty.csv", b"", "text/csv")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_upload_valid_dataset(client):
    """Uploading a valid CSV dataset should return HTTP 201 with validation summary."""
    csv_bytes = _generate_synthetic_csv_bytes(num_rows=25, num_fraud=5)
    response = client.post(
        "/datasets/upload",
        files={"file": ("transactions.csv", csv_bytes, "text/csv")}
    )
    assert response.status_code == 201
    data = response.json()
    assert "dataset_id" in data
    assert data["filename"] == "transactions.csv"
    assert data["row_count"] == 25
    assert data["column_count"] == 31
    assert data["validation"]["is_valid"] is True
    assert data["validation"]["class_distribution"]["fraud"] == 5
    assert data["validation"]["class_distribution"]["legitimate"] == 20


def test_preview_dataset(client):
    """Previewing an uploaded dataset should return rows and columns."""
    csv_bytes = _generate_synthetic_csv_bytes(num_rows=15, num_fraud=3)
    upload_res = client.post(
        "/datasets/upload",
        files={"file": ("preview_test.csv", csv_bytes, "text/csv")}
    )
    dataset_id = upload_res.json()["dataset_id"]

    response = client.get(f"/datasets/{dataset_id}/preview?limit=3")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset_id"] == dataset_id
    assert len(data["sample_rows"]) == 3
    assert len(data["columns"]) == 31
    assert data["validation"]["is_valid"] is True


def test_preview_nonexistent_dataset(client):
    """Previewing an unknown dataset ID should return 404."""
    response = client.get("/datasets/nonexistent-id-12345/preview")
    assert response.status_code == 404


def test_validate_dataset_missing_columns(client):
    """Uploading dataset missing required columns flags validation errors."""
    # Data without V1 and without Class
    df = pd.DataFrame({"Time": [1.0, 2.0], "Amount": [10.0, 20.0]})
    buf = io.StringIO()
    df.to_csv(buf, index=False)

    upload_res = client.post(
        "/datasets/upload",
        files={"file": ("missing_cols.csv", buf.getvalue().encode("utf-8"), "text/csv")}
    )
    assert upload_res.status_code == 201
    data = upload_res.json()
    assert data["validation"]["is_valid"] is False
    assert any("Missing required feature columns" in err for err in data["validation"]["errors"])
    assert any("Missing target column: 'Class'" in err for err in data["validation"]["errors"])


def test_training_workflow_and_metrics(client):
    """Full workflow: upload valid dataset, trigger training, poll until completed, verify metrics."""
    csv_bytes = _generate_synthetic_csv_bytes(num_rows=30, num_fraud=6)
    upload_res = client.post(
        "/datasets/upload",
        files={"file": ("train_dataset.csv", csv_bytes, "text/csv")}
    )
    dataset_id = upload_res.json()["dataset_id"]

    # Trigger training
    train_res = client.post("/training", json={"dataset_id": dataset_id})
    assert train_res.status_code == 202
    train_data = train_res.json()
    training_id = train_data["training_id"]
    assert train_data["status"] in ("queued", "running")

    # Poll status until completed (timeout: 10s)
    status_data = None
    for _ in range(20):
        time.sleep(0.5)
        poll_res = client.get(f"/training/{training_id}")
        assert poll_res.status_code == 200
        status_data = poll_res.json()
        if status_data["status"] in ("completed", "failed"):
            break

    assert status_data is not None
    assert status_data["status"] == "completed", f"Training failed: {status_data.get('error')}"
    assert status_data["candidate_model_saved"] is True
    assert status_data["metrics"] is not None
    metrics = status_data["metrics"]
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "pr_auc" in metrics
    assert "confusion_matrix" in metrics
    cm = metrics["confusion_matrix"]
    assert all(k in cm for k in ("true_negatives", "false_positives", "false_negatives", "true_positives"))


def test_training_rejected_on_invalid_dataset(client):
    """Training on a dataset that failed validation must return HTTP 400."""
    df = pd.DataFrame({"Time": [1.0], "Amount": [10.0]})
    buf = io.StringIO()
    df.to_csv(buf, index=False)

    upload_res = client.post(
        "/datasets/upload",
        files={"file": ("invalid_for_train.csv", buf.getvalue().encode("utf-8"), "text/csv")}
    )
    dataset_id = upload_res.json()["dataset_id"]

    train_res = client.post("/training", json={"dataset_id": dataset_id})
    assert train_res.status_code == 400
    assert "Cannot train on invalid dataset" in train_res.json()["detail"]


def test_get_training_not_found(client):
    """Requesting unknown training ID returns 404."""
    response = client.get("/training/nonexistent-job-999")
    assert response.status_code == 404


def test_list_datasets_and_trainings(client):
    """Listing endpoints should return 200 with list type."""
    datasets_res = client.get("/datasets")
    assert datasets_res.status_code == 200
    assert isinstance(datasets_res.json(), list)

    trainings_res = client.get("/training")
    assert trainings_res.status_code == 200
    assert isinstance(trainings_res.json(), list)
