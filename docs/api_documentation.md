# Banking Fraud Detection Inference API

## 1. Overview

The **Banking Fraud Detection API** is a production-grade inference service built with **FastAPI** and **Pydantic**. It serves the trained Scikit-learn Random Forest pipeline (`models/fraud_detection_pipeline.pkl`) configured in Phase 2 to predict whether banking transactions are legitimate or fraudulent in real-time.

Key operational characteristics:
- **Low latency**: Pre-trained model pipeline and configuration are loaded once into memory during application startup (using FastAPI lifespan events).
- **Strict validation**: All 30 incoming transaction features are validated against strict types and bounds via Pydantic before reaching the model.
- **Configurable decision threshold**: Reads the classification threshold (default `0.40`) from `configs/model_config.json`.
- **Cloud and Container Ready**: Compatible with Windows development and containerized Linux deployments (e.g., Docker and Kubernetes).

---

## 2. Architecture & File Structure

```text
├── configs/
│   └── model_config.json            # Model path, name, and classification threshold
├── models/
│   └── fraud_detection_pipeline.pkl # Trained Scikit-learn Pipeline (preprocessing + RF)
├── src/
│   └── api/
│       ├── __init__.py
│       ├── main.py                  # FastAPI application and endpoint definitions
│       ├── model_service.py         # Model loading, path resolution, and inference logic
│       └── schemas.py               # Pydantic request/response validation schemas
├── tests/
│   └── test_api.py                  # Comprehensive unit and integration test suite
├── pytest.ini                       # Test configuration
└── requirements.txt                 # Project dependencies
```

---

## 3. API Endpoints

### 3.1 `GET /` — API Availability
Checks whether the API is online.

- **URL**: `http://127.0.0.1:8000/`
- **Response**: `200 OK`
```json
{
  "message": "Banking Fraud Detection API is running",
  "status": "healthy"
}
```

---

### 3.2 `GET /health` — Readiness & Liveness Probe
Designed for Kubernetes probes and monitoring tools to verify that the ML model pipeline is loaded and operational.

- **URL**: `http://127.0.0.1:8000/health`
- **Response**: `200 OK` (when healthy) or `503 Service Unavailable` (if pipeline failed to load)
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Random Forest",
  "classification_threshold": 0.4
}
```

---

### 3.3 `POST /predict` — Fraud Inference
Evaluates a transaction and returns the estimated fraud probability along with the classification label.

- **URL**: `http://127.0.0.1:8000/predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**: A JSON object containing all 30 features.
- **Response**: `200 OK`
```json
{
  "fraud_probability": 0.6,
  "classification_threshold": 0.4,
  "prediction": "FRAUD"
}
```

---

### 3.4 Interactive Documentation
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 4. Required Transaction Features

The model expects exactly **30 numeric features** per transaction:

| Feature | Type | Constraints | Description |
|---|---|---|---|
| `Time` | `float` | Required | Seconds elapsed since the first transaction in the dataset |
| `V1` – `V28` | `float` | Required | 28 principal component features derived via PCA |
| `Amount` | `float` | `Amount >= 0.0` | Transaction currency amount |

> [!NOTE]
> The Scikit-learn pipeline automatically standardizes `Time` and `Amount` using the fitted `StandardScaler` inside the pipeline before passing all 30 features to the `RandomForestClassifier`. Raw, unscaled `Time` and `Amount` values should be provided to the API.

---

## 5. How to Run the API Locally

From the root of the repository, start the Uvicorn server:

```powershell
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Flags explained:
- `src.api.main:app`: Refers to the `app` instance in `src/api/main.py`.
- `--reload`: Enables hot-reloading on code changes during development.
- `--host 127.0.0.1`: Binds to local loopback (use `0.0.0.0` when running inside Docker).
- `--port 8000`: Port number.

---

## 6. How to Test Predictions

### 6.1 Using Swagger UI (`/docs`)
1. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.
2. Click on `POST /predict`.
3. Click **Try it out** (a complete valid example is pre-populated automatically).
4. Click **Execute** and view the prediction response.

### 6.2 Using PowerShell (`Invoke-RestMethod`)

#### Test a Legitimate Transaction:
```powershell
$body = @{
    Time = 0.0
    V1 = -1.3598; V2 = -0.0728; V3 = 2.5363; V4 = 1.3782; V5 = -0.3383
    V6 = 0.4624; V7 = 0.2396; V8 = 0.0987; V9 = 0.3638; V10 = 0.0908
    V11 = -0.5516; V12 = -0.6178; V13 = -0.9914; V14 = -0.3112; V15 = 1.4682
    V16 = -0.4704; V17 = 0.2080; V18 = 0.0258; V19 = 0.4040; V20 = 0.2514
    V21 = -0.0183; V22 = 0.2778; V23 = -0.1105; V24 = 0.0669; V25 = 0.1285
    V26 = -0.1891; V27 = 0.1336; V28 = -0.0211
    Amount = 149.62
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

#### Test a Fraudulent Transaction:
```powershell
$fraudBody = @{
    Time = 406.0
    V1 = -2.3122; V2 = 1.9520; V3 = -1.6099; V4 = 3.9979; V5 = -0.5222
    V6 = -1.4265; V7 = -2.5374; V8 = 1.3917; V9 = -2.7701; V10 = -2.7723
    V11 = 3.2020; V12 = -2.8999; V13 = -0.5952; V14 = -4.2893; V15 = 0.3897
    V16 = -1.1407; V17 = -2.8301; V18 = -0.0168; V19 = 0.4170; V20 = 0.1269
    V21 = 0.5172; V22 = -0.0350; V23 = -0.4652; V24 = 0.3202; V25 = 0.0445
    V26 = 0.1778; V27 = 0.2611; V28 = -0.1433
    Amount = 0.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body $fraudBody -ContentType "application/json"
```

### 6.3 Using cURL
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Time":0.0,"V1":-1.3598,"V2":-0.0728,"V3":2.5363,"V4":1.3782,"V5":-0.3383,"V6":0.4624,"V7":0.2396,"V8":0.0987,"V9":0.3638,"V10":0.0908,"V11":-0.5516,"V12":-0.6178,"V13":-0.9914,"V14":-0.3112,"V15":1.4682,"V16":-0.4704,"V17":0.2080,"V18":0.0258,"V19":0.4040,"V20":0.2514,"V21":-0.0183,"V22":0.2778,"V23":-0.1105,"V24":0.0669,"V25":0.1285,"V26":-0.1891,"V27":0.1336,"V28":-0.0211,"Amount":149.62}'
```

---

## 7. Automated Testing

Run the test suite using `pytest`:

```powershell
pytest -v
```

This verifies:
- Pipeline feature definition and alignment.
- Root endpoint response.
- Health probe status and model loaded flag.
- Prediction for legitimate transactions (`prediction == "LEGITIMATE"`).
- Prediction for fraudulent transactions (`prediction == "FRAUD"`).
- Input validation: missing fields, negative amounts, and invalid data types returning HTTP 422.

