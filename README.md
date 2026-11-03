# Drift-Aware Banking Fraud Detection Platform on Kubernetes

An end-to-end MLOps platform for detecting fraudulent banking transactions, monitoring model performance and data drift, and automating model retraining and deployment on Kubernetes.

> **Project Status:** Phase 3 Completed — Model Serving with FastAPI and Docker

---

## Overview

This project builds an industrial-grade MLOps platform for banking fraud detection.

The platform combines machine learning to identify fraudulent transactions with automated DevOps and MLOps practices across the full model lifecycle:

- Train, evaluate, and tune fraud detection models.
- Package inference services into lightweight, secure Docker containers.
- Deploy and orchestrate model services on Kubernetes.
- Provide health probes, request validation, and interactive API documentation.
- Monitor application performance, metrics, and data drift *(upcoming)*.
- Automate retraining and GitOps deployment *(upcoming)*.

---

## Business Use Case

**Banking Fraud Detection**

Financial institutions must identify fraudulent transactions in real-time while minimizing false alerts for legitimate customers.

This platform uses an imbalanced credit card transaction dataset and trains an optimized Random Forest pipeline with a tuned classification threshold (`0.40`), prioritizing recall and precision-recall AUC (PR-AUC) over naive accuracy.

---

## Planned Architecture

```text
              DATA SOURCE
                   |
                   v
          DATA VALIDATION
                   |
                   v
          DATA VERSIONING
                (DVC)
                   |
                   v
          MODEL TRAINING
                   |
                   v
       EXPERIMENT TRACKING
              (MLflow)
                   |
                   v
          MODEL EVALUATION
                   |
             Quality Gate
                   |
                   v
           DOCKER IMAGE  <--- [PHASE 3 COMPLETED]
                   |
                   v
         CONTAINER REGISTRY
                   |
                   v
          KUBERNETES DEPLOYMENT
                   |
                   v
             MODEL API  <--- [FASTAPI SERVING]
                   |
          -------------------
          |                 |
          v                 v
     PROMETHEUS          EVIDENTLY
       GRAFANA          DRIFT CHECK
          |                 |
          v                 v
    INFRASTRUCTURE     DRIFT DETECTED
      MONITORING             |
                            v
                    AUTOMATED RETRAINING
                            |
                            v
                     MODEL VALIDATION
                            |
                            v
                     GITOPS DEPLOYMENT
```

---

## Technology Stack

| Category | Technologies | Status |
|---|---|---|
| Programming | Python 3.13 | Active |
| Data Processing | Pandas, NumPy | Active |
| Machine Learning | Scikit-learn (Random Forest Pipeline) | Active |
| API Development | FastAPI, Pydantic, Uvicorn | Active |
| Containerization | Docker (python:3.13-slim) | Active |
| Testing | Pytest, FastAPI TestClient | Active |
| Experiment Tracking | MLflow | Planned |
| Data Versioning | DVC | Planned |
| Orchestration | Kubernetes (EKS) | Planned |
| Infrastructure as Code | Terraform | Planned |
| CI/CD | GitHub Actions | Planned |
| GitOps | Argo CD | Planned |
| Monitoring | Prometheus, Grafana | Planned |
| Drift Detection | Evidently | Planned |

---

## API Endpoints

The inference API is built using **FastAPI** and validates incoming transactions against 30 required numerical features using **Pydantic**:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status and root welcome message |
| `GET` | `/health` | Liveness & readiness probe returning model loading status |
| `POST` | `/predict` | Evaluates a transaction and returns fraud prediction |
| `GET` | `/docs` | Interactive Swagger UI documentation |
| `GET` | `/redoc` | ReDoc interactive documentation |

### Example Prediction Request (`POST /predict`)

```json
{
  "Time": 406.0,
  "V1": -2.3122, "V2": 1.9520, "V3": -1.6099, "V4": 3.9979, "V5": -0.5222,
  "V6": -1.4265, "V7": -2.5374, "V8": 1.3917, "V9": -2.7701, "V10": -2.7723,
  "V11": 3.2020, "V12": -2.8999, "V13": -0.5952, "V14": -4.2893, "V15": 0.3897,
  "V16": -1.1407, "V17": -2.8301, "V18": -0.0168, "V19": 0.4170, "V20": 0.1269,
  "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3202, "V25": 0.0445,
  "V26": 0.1778, "V27": 0.2611, "V28": -0.1433,
  "Amount": 0.0
}
```

### Example Prediction Response

```json
{
  "fraud_probability": 0.6,
  "classification_threshold": 0.4,
  "prediction": "FRAUD"
}
```

---

## Project Structure

```text
drift-aware-banking-fraud-detection/
│
├── configs/
│   └── model_config.json                 # Model path, name, and classification threshold
│
├── data/
│   ├── raw/                              # Original dataset (excluded from git)
│   └── processed/                        # Cleaned dataset (excluded from git)
│
├── docker/
├── docs/
│   ├── dataset_analysis.md               # Phase 1 dataset exploration
│   ├── PHASE_2_COMPLETION_NOTES.md       # Phase 2 ML model training documentation
│   ├── PHASE_3_COMPLETION_NOTES.md       # Phase 3 FastAPI & Docker documentation
│   └── api_documentation.md              # Detailed API usage and reference manual
│
├── infrastructure/
│   └── terraform/
│
├── kubernetes/
├── models/
│   └── fraud_detection_pipeline.pkl      # Trained Scikit-learn Pipeline (excluded from git)
│
├── monitoring/
├── notebooks/
├── pipelines/
│   └── retraining/
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                       # FastAPI application and endpoints
│   │   ├── model_service.py              # Model loading and prediction logic
│   │   └── schemas.py                    # Pydantic request/response schemas
│   ├── data/
│   │   ├── inspect_dataset.py
│   │   ├── analyze_duplicates.py
│   │   └── preprocess_data.py
│   ├── features/
│   │   ├── inspect_features.py
│   │   ├── split_data.py
│   │   └── scale_features.py
│   ├── models/
│   │   ├── train_logistic_regression.py
│   │   ├── train_random_forest.py
│   │   ├── compare_models.py
│   │   ├── tune_threshold.py
│   │   ├── train_pipeline.py
│   │   ├── final_evaluation.py
│   │   ├── save_final_model.py
│   │   └── predict.py
│   └── monitoring/
│
├── tests/
│   └── test_api.py                       # Automated API unit & integration tests
│
├── .dockerignore                         # Docker build context exclusion rules
├── .gitignore                            # Git tracking exclusions
├── Dockerfile                            # Production Docker image specification
├── LICENSE
├── pytest.ini                            # Pytest configuration
├── README.md
└── requirements.txt                      # Project dependencies
```

---

## Development Roadmap

- [x] **Phase 1:** Repository, development environment, and dataset exploration
- [x] **Phase 2:** Data preprocessing, model training, threshold tuning, and pipeline export
- [x] **Phase 3:** Model serving with FastAPI, request validation, and Docker containerization
- [ ] **Phase 4:** CI/CD pipeline with GitHub Actions
- [ ] **Phase 5:** Terraform infrastructure provisioning and AWS EKS Kubernetes deployment
- [ ] **Phase 6:** GitOps deployment using Argo CD
- [ ] **Phase 7:** Infrastructure monitoring (Prometheus/Grafana) and drift detection (Evidently)
- [ ] **Phase 8:** Automated model retraining trigger and redeployment
- [ ] **Phase 9:** End-to-end integration testing and platform demonstration

---

## Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/denitjoseph/drift-aware-banking-fraud-detection.git
cd drift-aware-banking-fraud-detection
```

### 2. Set Up Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Run Automated Tests

```bash
pytest -v
```

### 4. Run the FastAPI Application Locally

```powershell
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

- Open Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Open Health Endpoint: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## Docker Build & Container Execution

### 1. Build the Docker Image

```bash
docker build -t fraud-detection-api:v1 .
```

### 2. Run the Docker Container

```bash
docker run -d --name fraud-api -p 8000:8000 fraud-detection-api:v1
```

### 3. Verify Container Health

```bash
# Using curl (or browser)
curl http://127.0.0.1:8000/health

# Using PowerShell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

### 4. View Container Logs

```bash
docker logs -f fraud-api
```

### 5. Stop and Remove the Container

```bash
docker stop fraud-api
docker rm fraud-api
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Denit Joseph**

GitHub: [@denitjoseph](https://github.com/denitjoseph)
