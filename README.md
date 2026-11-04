# Drift-Aware Banking Fraud Detection Platform on Kubernetes

An end-to-end MLOps platform for detecting fraudulent banking transactions, monitoring model performance and data drift, and automating model retraining and deployment on Kubernetes.

> **Project Status:** Phase 4 Completed — Clean Backend/Frontend Architecture with React & Vite Dashboard

---

## Overview

This project builds an industrial-grade MLOps platform for banking fraud detection.

The platform combines machine learning to identify fraudulent transactions with automated DevOps and MLOps practices across the full model lifecycle:

- Train, evaluate, and tune fraud detection models.
- Modularized service layout: decoupled `backend/` and `frontend/` microservices.
- Package inference services into lightweight, secure Docker containers.
- Deploy and orchestrate model services on Kubernetes.
- Provide health probes, request validation, and interactive API documentation.
- Serve a modern, responsive React/Vite dashboard for real-time transaction scoring.
- Monitor application performance, metrics, and data drift *(upcoming)*.
- Automate retraining and GitOps deployment *(upcoming)*.

---

## Business Use Case

**Banking Fraud Detection**

Financial institutions must identify fraudulent transactions in real-time while minimizing false alerts for legitimate customers.

This platform uses an imbalanced credit card transaction dataset and trains an optimized Random Forest pipeline with a tuned classification threshold (`0.40`), prioritizing recall and precision-recall AUC (PR-AUC) over naive accuracy.

---

## Architecture

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
             BACKEND API <--- [FASTAPI SERVING]
                   ^
                   |  (REST API / CORS)
                   |
         REACT DASHBOARD <--- [PHASE 4 COMPLETED]
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
| Backend API | FastAPI, Pydantic, Uvicorn | Active |
| Data Processing | Pandas, NumPy | Active |
| Machine Learning | Scikit-learn (Random Forest Pipeline) | Active |
| Frontend Dashboard | React 18, Vite 6, JavaScript, CSS | Active |
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

## Frontend Dashboard (Phase 4)

The platform provides a modern, responsive web dashboard built with **React** and **Vite** in the `frontend/` directory:

- **Live Microservice Monitoring:** Polls `GET /health` every 15 seconds to display real-time backend connectivity, loaded model name, and active classification threshold.
- **Transaction Prediction Interface:** Accepts all 30 model features (`Time`, `V1`–`V28`, `Amount`) with numerical validation.
- **Instant Test Presets:** One-click sample loaders for verified **Legitimate** and **Fraudulent** transactions from the credit card dataset.
- **Color-Coded Verdicts:** Highlights high-risk `FRAUD` in red and `LEGITIMATE` in green with exact probability scores and decision explanations.
- **Model Architecture Reference:** Displays pipeline specifications, threshold tuning details, and dataset benchmark counts.

---

## API Endpoints

The inference API is located in `backend/` and validates incoming transactions against 30 required numerical features using **Pydantic**:

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
├── backend/                              # Backend inference service and ML code
│   ├── configs/
│   │   └── model_config.json             # Model path, name, and classification threshold
│   │
│   ├── data/
│   │   ├── raw/                          # Original dataset (excluded from git)
│   │   └── processed/                    # Cleaned dataset (excluded from git)
│   │
│   ├── models/
│   │   └── fraud_detection_pipeline.pkl  # Trained Scikit-learn Pipeline (excluded from git)
│   │
│   ├── pipelines/
│   │   └── retraining/
│   │
│   ├── src/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                   # FastAPI application with CORS
│   │   │   ├── model_service.py          # Model loading and prediction logic
│   │   │   └── schemas.py                # Pydantic request/response schemas
│   │   ├── data/
│   │   │   ├── inspect_dataset.py
│   │   │   ├── analyze_duplicates.py
│   │   │   └── preprocess_data.py
│   │   ├── features/
│   │   │   ├── inspect_features.py
│   │   │   ├── split_data.py
│   │   │   └── scale_features.py
│   │   ├── models/
│   │   │   ├── train_logistic_regression.py
│   │   │   ├── train_random_forest.py
│   │   │   ├── compare_models.py
│   │   │   ├── tune_threshold.py
│   │   │   ├── train_pipeline.py
│   │   │   ├── final_evaluation.py
│   │   │   ├── save_final_model.py
│   │   │   └── predict.py
│   │   └── monitoring/
│   │
│   ├── tests/
│   │   └── test_api.py                   # Automated API unit & integration tests
│   │
│   ├── .dockerignore
│   ├── Dockerfile                        # Production Docker container build file
│   ├── pytest.ini                        # Pytest configuration
│   └── requirements.txt                  # Python dependencies
│
├── frontend/                             # Frontend React + Vite Dashboard
│   ├── public/
│   │   └── shield.svg
│   ├── src/
│   │   ├── components/
│   │   │   ├── ApiStatus.jsx             # Live health status indicator
│   │   │   ├── Navbar.jsx                # Header component
│   │   │   ├── PredictionForm.jsx        # 30-feature transaction input form
│   │   │   ├── PredictionResult.jsx      # Verdict display component
│   │   │   ├── Sidebar.jsx               # Navigation sidebar
│   │   │   └── StatCard.jsx              # Summary metric card
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx             # Platform overview page
│   │   │   ├── ModelInfo.jsx             # Architecture reference page
│   │   │   └── Prediction.jsx            # Transaction evaluation page
│   │   ├── services/
│   │   │   └── api.js                    # Fetch API client
│   │   ├── styles/
│   │   │   └── dashboard.css             # SaaS theme stylesheet
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── .env.example
│   ├── .gitignore
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── infrastructure/
│   └── terraform/
│
├── kubernetes/
├── monitoring/
├── notebooks/
├── docs/
│   ├── dataset_analysis.md               # Phase 1 dataset exploration
│   ├── PHASE_2_COMPLETION_NOTES.md       # Phase 2 ML model training documentation
│   ├── PHASE_3_COMPLETION_NOTES.md       # Phase 3 FastAPI & Docker documentation
│   └── api_documentation.md              # Detailed API usage and reference manual
│
├── .dockerignore                         # Root Docker build context exclusion rules
├── .gitignore                            # Root Git tracking exclusions
├── LICENSE
└── README.md
```

---

## Development Roadmap

- [x] **Phase 1:** Repository, development environment, and dataset exploration
- [x] **Phase 2:** Data preprocessing, model training, threshold tuning, and pipeline export
- [x] **Phase 3:** Model serving with FastAPI, request validation, and Docker containerization
- [x] **Phase 4:** Frontend dashboard with React, Vite, and real-time fraud prediction
- [ ] **Phase 5:** CI/CD pipeline with GitHub Actions
- [ ] **Phase 6:** Terraform infrastructure provisioning and AWS EKS Kubernetes deployment
- [ ] **Phase 7:** GitOps deployment using Argo CD
- [ ] **Phase 8:** Infrastructure monitoring (Prometheus/Grafana) and drift detection (Evidently)
- [ ] **Phase 9:** Automated model retraining trigger and redeployment
- [ ] **Phase 10:** End-to-end integration testing and platform demonstration

---

## Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/denitjoseph/drift-aware-banking-fraud-detection.git
cd drift-aware-banking-fraud-detection
```

### 2. Run the Backend API

```powershell
# Navigate to the backend directory
cd backend

# Set up Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run automated tests
python -m pytest -v

# Start FastAPI server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Endpoint: [http://localhost:8000/health](http://localhost:8000/health)

### 3. Run the Frontend Dashboard

Open a second terminal window from the project root:

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies and start Vite dev server
npm install
npm run dev
```

- Open Dashboard: **[http://localhost:5173](http://localhost:5173)**

---

## Docker Build & Container Execution

### 1. Build the Docker Image

You can build the Docker image using either of the following commands:

```bash
# Build pointing to the backend directory
docker build -t fraud-detection-api:v1 backend/

# Or from inside the backend directory:
cd backend
docker build -t fraud-detection-api:v1 .
```

### 2. Run the Docker Container

```bash
docker run -d --name fraud-api -p 8000:8000 fraud-detection-api:v1
```

### 3. Verify Container Health

```bash
curl http://localhost:8000/health
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
