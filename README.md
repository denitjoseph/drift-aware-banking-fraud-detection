# Drift-Aware Banking Fraud Detection Platform on Kubernetes

An end-to-end MLOps platform for detecting fraudulent banking transactions, monitoring model performance and data drift, and automating model retraining and deployment on Kubernetes.

> **Project Status:** Phase 1 — Foundation and Dataset Preparation

## Overview

This project aims to build an industrial-style MLOps platform for banking fraud detection.

The platform will use machine learning to identify potentially fraudulent transactions and combine DevOps and MLOps practices to automate the model lifecycle.

The long-term objective is to create a system that can:

- Train and validate fraud detection models.
- Track experiments and manage model versions.
- Package models using Docker.
- Deploy model services on Kubernetes.
- Monitor application health and model performance.
- Detect data drift and prediction drift.
- Trigger automated model retraining when drift is detected.
- Deploy validated models through a GitOps workflow.

**Note:** This project is under active development. Planned capabilities are not yet implemented.

## Business Use Case

**Banking Fraud Detection**

Financial institutions need to identify fraudulent transactions while minimizing false alerts on legitimate transactions.

This project uses an anonymized credit card transaction dataset to develop and demonstrate a fraud detection pipeline.

The dataset is highly imbalanced, making fraud recall, precision and PR-AUC important evaluation metrics.

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
           DOCKER IMAGE
                   |
                   v
         CONTAINER REGISTRY
                   |
                   v
          KUBERNETES DEPLOYMENT
                   |
                   v
             MODEL API
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

*Architecture represents the planned target system and may evolve during implementation.*

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Experiment Tracking | MLflow (planned) |
| Data Versioning | DVC (planned) |
| API Development | FastAPI (planned) |
| Containerization | Docker (planned) |
| Orchestration | Kubernetes (planned) |
| Infrastructure as Code | Terraform (planned) |
| CI/CD | GitHub Actions (planned) |
| GitOps | Argo CD (planned) |
| Monitoring | Prometheus, Grafana (planned) |
| Drift Detection | Evidently (planned) |
| Cloud | AWS EKS (planned) |

## Dataset

**Dataset:** Credit Card Fraud Detection

**Source:** [Kaggle — mlg-ulb Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Dataset characteristics:

| Metric | Value |
|---|---:|
| Total transactions | 284,807 |
| Total columns | 31 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Missing values | 0 |
| Duplicate occurrences | 1,081 |
| Fraud percentage | 0.1727% |

### Dataset Features

- `Time`: Time elapsed since the first transaction.
- `V1` to `V28`: Anonymized PCA-transformed features.
- `Amount`: Transaction amount.
- `Class`: Target variable (`0` = legitimate, `1` = fraudulent).

The original dataset is preserved in the local `data/raw/` directory and excluded from Git tracking.

This is an anonymized benchmark dataset, not real production banking data.

## Project Structure

```text
drift-aware-banking-fraud-detection/
│
├── configs/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docker/
│
├── docs/
│   └── dataset_analysis.md
│
├── infrastructure/
│   └── terraform/
│
├── kubernetes/
│
├── monitoring/
│
├── notebooks/
│
├── pipelines/
│   └── retraining/
│
├── src/
│   ├── api/
│   ├── data/
│   │   └── inspect_dataset.py
│   ├── features/
│   ├── models/
│   └── monitoring/
│
├── tests/
│
├── .gitignore
├── LICENSE
└── README.md
```

## Development Roadmap

- [x] Phase 1: Repository, development environment and dataset preparation foundation
- [ ] Phase 2: Data processing and machine learning model development
- [ ] Phase 3: Experiment tracking and model versioning with MLflow and DVC
- [ ] Phase 4: Dockerization and CI pipeline
- [ ] Phase 5: Terraform infrastructure and Kubernetes deployment
- [ ] Phase 6: GitOps deployment using Argo CD
- [ ] Phase 7: Monitoring and drift detection
- [ ] Phase 8: Automated retraining and model redeployment
- [ ] Phase 9: End-to-end testing and documentation

## Model Evaluation Strategy

Because fraudulent transactions represent only a small fraction of the dataset, the project will evaluate models using:

- Precision
- Recall
- F1-score
- Precision-Recall AUC (PR-AUC)
- Confusion Matrix

Accuracy alone will not be treated as sufficient evidence of model quality.

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/denitjoseph/drift-aware-banking-fraud-detection.git
cd drift-aware-banking-fraud-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 5. Inspect the dataset

After downloading and extracting the Kaggle dataset into `data/raw/`:

```bash
python src/data/inspect_dataset.py
```

**Current functionality:** Dataset inspection and initial analysis only. Model training and deployment instructions will be added as those components are implemented.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Denit Joseph**

GitHub: [@denitjoseph](https://github.com/denitjoseph)

---

*Built as a hands-on DevOps and MLOps learning project, with an emphasis on automation, monitoring, reliability and scalable deployment.*