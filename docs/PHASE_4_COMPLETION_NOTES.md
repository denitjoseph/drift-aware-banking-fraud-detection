# Phase 4 Completion Notes — Fraud Detection Frontend Dashboard & Service Decoupling

## Overview

Phase 4 delivers a modern, high-performance web dashboard built with **React** and **Vite** that connects directly to the production-ready **FastAPI** model serving backend. Additionally, the project architecture has been reorganized to clearly separate backend and frontend codebases for clean microservice development and future Kubernetes container orchestration.

---

## 1. Project Architecture Decoupling

To prepare the repository for production deployment and cloud orchestration:
- **`backend/`**: Encapsulates the FastAPI microservice, Scikit-learn model loading, feature scaling pipelines, Pydantic schemas, automated pytest suite, configuration, and Docker container build specifications.
- **`frontend/`**: Encapsulates the React/Vite dashboard, modular UI components, live API health polling, prediction forms, preset test scenarios, and modern dark navy / slate styling.
- **`infrastructure/`**, **`kubernetes/`**, **`monitoring/`**: Scaffolding reserved for upcoming infrastructure-as-code, cluster manifests, and telemetry.

```text
drift-aware-banking-fraud-detection/
├── backend/
│   ├── configs/
│   │   └── model_config.json
│   ├── data/
│   │   ├── raw/ (excluded from git)
│   │   └── processed/ (excluded from git)
│   ├── models/
│   │   └── fraud_detection_pipeline.pkl (excluded from git)
│   ├── pipelines/
│   │   └── retraining/
│   ├── src/
│   │   ├── api/
│   │   │   ├── main.py
│   │   │   ├── model_service.py
│   │   │   └── schemas.py
│   │   ├── data/
│   │   ├── features/
│   │   └── models/
│   ├── tests/
│   │   └── test_api.py
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── pytest.ini
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   │   └── shield.svg
│   ├── src/
│   │   ├── components/
│   │   │   ├── ApiStatus.jsx
│   │   │   ├── Navbar.jsx
│   │   │   ├── PredictionForm.jsx
│   │   │   ├── PredictionResult.jsx
│   │   │   ├── Sidebar.jsx
│   │   │   └── StatCard.jsx
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── ModelInfo.jsx
│   │   │   └── Prediction.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── styles/
│   │   │   └── dashboard.css
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
├── docs/
│   ├── PHASE_2_COMPLETION_NOTES.md
│   ├── PHASE_3_COMPLETION_NOTES.md
│   ├── PHASE_4_COMPLETION_NOTES.md
│   ├── api_documentation.md
│   └── dataset_analysis.md
│
├── infrastructure/
├── kubernetes/
├── monitoring/
├── .dockerignore
├── .gitignore
├── LICENSE
└── README.md
```

---

## 2. Frontend Features & Components

### A. Navigation & Shell
- **`Sidebar.jsx`**: Responsive sidebar navigation for switching between Overview Dashboard, Fraud Prediction, Model Architecture reference, and Swagger API Documentation (`http://localhost:8000/docs`).
- **`Navbar.jsx`**: Displays real-time API connectivity status, active classification threshold (`0.40`), and a manual health refresh trigger.
- **`ApiStatus.jsx`**: Visual indicator badge (`Connected` in green, `Checking` in yellow, `Offline` in red).

### B. Pages
- **`Dashboard.jsx`**: Platform overview showing key business metrics (283,726 total transactions, 473 fraud cases, 283,253 legitimate), microservice component status table, and threshold tuning summary.
- **`Prediction.jsx`**: Interactive transaction inference interface with tabbed input controls and verdict panel.
- **`ModelInfo.jsx`**: Architectural reference detailing Random Forest hyperparameters, standard scaling, and PCA components.

### C. Prediction Workflow
- **`PredictionForm.jsx`**: Accepts all 30 model features (`Time`, `V1` through `V28`, `Amount`). Features instant 1-click presets:
  - *Legitimate Sample Transaction*: High-volume genuine transaction from dataset.
  - *Fraud Sample Transaction*: Confirmed fraud case.
  - *Reset Form*: Clears inputs to default state.
- **`PredictionResult.jsx`**: Evaluates response from `POST /predict`. Displays color-coded verdict (`FRAUD` in red, `LEGITIMATE` in green), predicted probability score, threshold comparison, and security recommendation.

---

## 3. Backend Integration & Verification

The frontend communicates with the FastAPI backend via `fetch` API configured with CORS headers:
- `GET /` — Root service verification.
- `GET /health` — Liveness & readiness probe returning model loading status and tuned threshold.
- `POST /predict` — Evaluates all 30 features and returns classification verdict.

### Verification Results

1. **Backend Test Suite (`pytest`)**:
   ```text
   tests/test_api.py ........ [100%]
   ================== 8 passed in 4.20s ==================
   ```
   - `test_feature_columns_definition`: Verified all 30 features and positions.
   - `test_root_endpoint`: Verified `GET /` returns status healthy.
   - `test_health_endpoint`: Verified model loaded and threshold equals `0.4`.
   - `test_predict_legitimate_transaction`: Correctly predicts `LEGITIMATE` with probability `0.0`.
   - `test_predict_fraudulent_transaction`: Correctly predicts `FRAUD` with probability `0.6`.
   - `test_predict_missing_feature`: Rejects missing features with HTTP 422.
   - `test_predict_negative_amount`: Rejects invalid negative amounts with HTTP 422.
   - `test_predict_invalid_data_type`: Rejects string inputs with HTTP 422.

2. **Frontend Production Build (`vite build`)**:
   ```text
   ✓ 37 modules transformed.
   dist/index.html                   0.73 kB │ gzip:  0.41 kB
   dist/assets/index-Boh-qgFX.css   10.62 kB │ gzip:  2.57 kB
   dist/assets/index-BxJjjRRY.js   171.33 kB │ gzip: 53.40 kB
   ✓ built in 694ms
   ```

---

## 4. Local Execution Guide

### Start Backend Service
```bash
cd backend
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at: `http://localhost:8000/docs`

### Start Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```
Dashboard will open at: `http://localhost:5173`

---

## 5. Next Steps (Phase 5+)

- **Kubernetes Deployment**: Author Kubernetes manifests (`Deployment`, `Service`, `HorizontalPodAutoscaler`, `Ingress`).
- **Helm Packaging**: Parameterize backend and frontend configuration for cluster environments.
- **Monitoring & Observability**: Integrate Prometheus metrics exporter and Grafana dashboards for drift detection.
