"""Unit and integration tests for the Banking Fraud Detection FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.model_service import FEATURE_COLUMNS, model_service


# Sample legitimate transaction from dataset (Class = 0)
LEGITIMATE_TRANSACTION = {
    "Time": 0.0,
    "V1": -1.3598071336738,
    "V2": -0.0727811733010012,
    "V3": 2.53634673796914,
    "V4": 1.37815522427443,
    "V5": -0.338320769942518,
    "V6": 0.462387777762292,
    "V7": 0.239598554061194,
    "V8": 0.0986979012610507,
    "V9": 0.363786969611505,
    "V10": 0.0907941719789316,
    "V11": -0.551599533260813,
    "V12": -0.617800855762348,
    "V13": -0.991389847235408,
    "V14": -0.311169353699879,
    "V15": 1.46817697209427,
    "V16": -0.470400525259478,
    "V17": 0.207971241929242,
    "V18": 0.0257905801985591,
    "V19": 0.403992960255733,
    "V20": 0.251412098239705,
    "V21": -0.018306777944153,
    "V22": 0.277837575558899,
    "V23": -0.110473910188767,
    "V24": 0.0669280749146731,
    "V25": 0.128539358273528,
    "V26": -0.189114843888824,
    "V27": 0.133558376740387,
    "V28": -0.0210530534538215,
    "Amount": 149.62,
}

# Sample fraudulent transaction from dataset (Class = 1)
FRAUDULENT_TRANSACTION = {
    "Time": 406.0,
    "V1": -2.3122265423263,
    "V2": 1.95199201064158,
    "V3": -1.60985073229769,
    "V4": 3.9979055875468,
    "V5": -0.522187864667764,
    "V6": -1.42654531920595,
    "V7": -2.53738730624579,
    "V8": 1.39165724829804,
    "V9": -2.77008927719433,
    "V10": -2.77227214465915,
    "V11": 3.20203320709635,
    "V12": -2.89990738849473,
    "V13": -0.595221881324605,
    "V14": -4.2892537824421,
    "V15": 0.389724120274487,
    "V16": -1.14074717981127,
    "V17": -2.83005567450437,
    "V18": -0.0168224681808257,
    "V19": 0.416955705037907,
    "V20": 0.126910559061474,
    "V21": 0.517232370861764,
    "V22": -0.0350493686052974,
    "V23": -0.465211076182388,
    "V24": 0.320198198514526,
    "V25": 0.0445191674731724,
    "V26": 0.177839798284401,
    "V27": 0.261145002567677,
    "V28": -0.143275874698919,
    "Amount": 0.0,
}


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan events executed."""
    with TestClient(app) as test_client:
        yield test_client


def test_feature_columns_definition():
    """Verify that all 30 features are present and Time/Amount are in proper positions."""
    assert len(FEATURE_COLUMNS) == 30
    assert FEATURE_COLUMNS[0] == "Time"
    assert FEATURE_COLUMNS[-1] == "Amount"
    for i in range(1, 29):
        assert f"V{i}" in FEATURE_COLUMNS


def test_root_endpoint(client):
    """Test GET / returns healthy status and friendly message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Banking Fraud Detection API" in data["message"]


def test_health_endpoint(client):
    """Test GET /health returns model loaded status and threshold."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "Random Forest"
    assert data["classification_threshold"] == 0.40


def test_predict_legitimate_transaction(client):
    """Test POST /predict with a known legitimate transaction."""
    response = client.post("/predict", json=LEGITIMATE_TRANSACTION)
    assert response.status_code == 200
    data = response.json()

    assert "fraud_probability" in data
    assert "classification_threshold" in data
    assert "prediction" in data

    assert isinstance(data["fraud_probability"], float)
    assert data["classification_threshold"] == 0.40
    assert data["prediction"] == "LEGITIMATE"
    assert data["fraud_probability"] < 0.40


def test_predict_fraudulent_transaction(client):
    """Test POST /predict with a known fraudulent transaction."""
    response = client.post("/predict", json=FRAUDULENT_TRANSACTION)
    assert response.status_code == 200
    data = response.json()

    assert "fraud_probability" in data
    assert "classification_threshold" in data
    assert "prediction" in data

    assert isinstance(data["fraud_probability"], float)
    assert data["classification_threshold"] == 0.40
    assert data["prediction"] == "FRAUD"
    assert data["fraud_probability"] >= 0.40


def test_predict_missing_feature(client):
    """Test POST /predict with missing required fields returns HTTP 422."""
    incomplete_data = LEGITIMATE_TRANSACTION.copy()
    del incomplete_data["Amount"]

    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422


def test_predict_negative_amount(client):
    """Test POST /predict with negative amount violates validation (HTTP 422)."""
    invalid_data = LEGITIMATE_TRANSACTION.copy()
    invalid_data["Amount"] = -50.0

    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422


def test_predict_invalid_data_type(client):
    """Test POST /predict with string in numeric field returns HTTP 422."""
    invalid_data = LEGITIMATE_TRANSACTION.copy()
    invalid_data["V1"] = "not-a-number"

    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422

