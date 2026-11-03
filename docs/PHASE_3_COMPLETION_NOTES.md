# Phase 3: Model Serving with FastAPI and Docker

## 1. Objective

Package the trained machine learning pipeline from Phase 2 into a production-grade inference service using FastAPI, validate transaction features with Pydantic, establish health check probes, and containerize the application with Docker for local execution and future Kubernetes deployment.

---

## 2. API Architecture & Modular Design

The application layer was constructed with clear separation of concerns under `src/api/`:

- **Request & Response Schemas (`src/api/schemas.py`)**:
  - `TransactionInput`: Validates all 30 incoming features (`Time`, `V1` to `V28`, and `Amount`). Enforces constraints such as `Amount >= 0.0` and provides complete example payloads for interactive Swagger documentation.
  - `PredictionResponse`: Standardized JSON output format returning `fraud_probability`, `classification_threshold`, and `prediction` (`"FRAUD"` or `"LEGITIMATE"`).
  - `HealthResponse`: Health payload reporting overall service status, model readiness, model name, and the active decision threshold.

- **Model Service Layer (`src/api/model_service.py`)**:
  - Encapsulates model loading, configuration parsing, and inference execution.
  - Loads the trained artifact once into memory at startup, preventing per-request disk I/O.
  - Supports dynamic path resolution (works identically across local Windows directories, Linux containers at `/app`, and custom environment variables `CONFIG_PATH` and `MODEL_PATH`).
  - Enforces strict feature column ordering (`Time`, `V1`–`V28`, `Amount`) before feeding data into the Scikit-learn pipeline.

- **FastAPI Endpoints (`src/api/main.py`)**:
  - Implements an asynchronous `lifespan` handler to load the ML pipeline cleanly during application startup.
  - `GET /`: Root endpoint confirming service status.
  - `GET /health`: Readiness and liveness check for container orchestrators (Kubernetes) and monitoring agents.
  - `POST /predict`: Inference endpoint scoring transactions against the configured decision threshold.
  - Handles validation errors (HTTP 422), uninitialized service states (HTTP 503), and runtime exceptions (HTTP 500).

---

## 3. Machine Learning Model Integration

- **Model Artifact**: `models/fraud_detection_pipeline.pkl` (reusable Scikit-learn Pipeline containing `ColumnTransformer`, `StandardScaler`, and `RandomForestClassifier`).
- **Configuration**: `configs/model_config.json`.
- **Classification Threshold**: `0.40` (selected during Phase 2 validation).
- **Model Preserved**: No model retraining or parameter changes were introduced.

---

## 4. Docker Containerization

- **Base Image**: `python:3.13-slim` (Debian-based, lightweight Python 3.13 runtime matching local development).
- **Security Best Practices**:
  - Creates and switches to a non-root system user (`appuser`, UID 1000).
  - Sets application ownership to `appuser`.
- **Environment Flags**:
  - `PYTHONDONTWRITEBYTECODE=1`: Prevents `.pyc` clutter on disk.
  - `PYTHONUNBUFFERED=1`: Ensures stdout/stderr stream directly to container logs.
- **Layer Caching**: Copies and installs `requirements.txt` before copying source code to avoid unnecessary reinstallations during development.
- **Cross-Platform Compatibility**: Added `; sys_platform == 'win32'` marker to `pywinpty` in `requirements.txt`, allowing seamless Linux container builds from Windows development environments.
- **Build Context Optimization (`.dockerignore`)**: Excludes `.git`, `.venv`, cache files, and the large `data/` directory (~150MB) to keep image builds fast and minimal.

---

## 5. Verification & Testing Results

### 5.1 Automated Pytest Suite (`tests/test_api.py`)
Executed test suite covering unit schemas and HTTP integration:

| Test Case | Method / Endpoint | Expected Outcome | Result |
|---|---|---|---|
| `test_feature_columns_definition` | Internal Service | 30 ordered feature columns | PASSED |
| `test_root_endpoint` | `GET /` | Status 200, healthy status message | PASSED |
| `test_health_endpoint` | `GET /health` | Status 200, `model_loaded: true` | PASSED |
| `test_predict_legitimate_transaction` | `POST /predict` | Status 200, `probability < 0.40`, `LEGITIMATE` | PASSED |
| `test_predict_fraudulent_transaction` | `POST /predict` | Status 200, `probability >= 0.40`, `FRAUD` | PASSED |
| `test_predict_missing_feature` | `POST /predict` | Status 422 Unprocessable Entity | PASSED |
| `test_predict_negative_amount` | `POST /predict` | Status 422 Unprocessable Entity | PASSED |
| `test_predict_invalid_data_type` | `POST /predict` | Status 422 Unprocessable Entity | PASSED |

**Summary**: 8 passed in 2.73s.

### 5.2 Docker Image Build
- Command: `docker build -t fraud-detection-api:v1 .`
- Result: Successfully built and tagged `fraud-detection-api:v1`.

### 5.3 Live Container Execution
- Started container: `docker run -d --name fraud-api -p 8000:8000 fraud-detection-api:v1`
- Container startup logs:
  ```text
  INFO:     Started server process [1]
  INFO:     Waiting for application startup.
  [INFO] fraud_api - Initializing API and loading model artifacts...
  [INFO] fraud_api.model_service - Loading model pipeline from /app/models/fraud_detection_pipeline.pkl
  [INFO] fraud_api.model_service - Model successfully loaded. Name: Random Forest, Threshold: 0.40
  [INFO] fraud_api - Model pipeline ready to serve predictions.
  INFO:     Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8000
  ```
- Endpoint verified: `GET /health` returned `200 OK` with `model_loaded: true`.
- Endpoint verified: `POST /predict` returned `200 OK` with `fraud_probability: 0.60` and prediction `"FRAUD"`.
- Endpoint verified: `GET /docs` returned `200 OK` (interactive Swagger UI available).

---

## 6. Key Project Artifacts Added in Phase 3

- `Dockerfile`: Production container build specification.
- `.dockerignore`: Build context exclusions.
- `src/api/__init__.py`: Package indicator.
- `src/api/main.py`: FastAPI server and routing.
- `src/api/model_service.py`: Pipeline loading and prediction service.
- `src/api/schemas.py`: Pydantic input/output validation models.
- `tests/test_api.py`: Automated integration and unit tests.
- `pytest.ini`: Test discovery configuration.
- `docs/api_documentation.md`: Comprehensive API reference manual.
- `docs/PHASE_3_COMPLETION_NOTES.md`: Phase 3 summary documentation.

---

## 7. Phase 3 Outcome

Phase 3 successfully transitioned the offline machine learning model developed in Phase 2 into a self-contained, containerized microservice ready for automated testing, continuous integration, and Kubernetes orchestration.

