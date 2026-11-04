"""FastAPI application for the Drift-Aware Banking Fraud Detection Platform."""

import logging
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.dataset_service import dataset_service
from src.api.model_service import model_service
from src.api.schemas import (
    DatasetPreviewResponse,
    DatasetUploadResponse,
    DatasetValidationResult,
    HealthResponse,
    PredictionResponse,
    TrainingRecord,
    TrainingStartRequest,
    TransactionInput,
)
from src.api.training_service import training_service

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("fraud_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load ML model on startup."""
    logger.info("Initializing API and loading model artifacts...")
    try:
        model_service.load_model()
        logger.info("Model pipeline ready to serve predictions.")
    except Exception as exc:
        logger.error("Failed to load model pipeline at startup: %s", exc)
    yield
    logger.info("Shutting down API service.")


app = FastAPI(
    title="Banking Fraud Detection API",
    description=(
        "Production-ready inference API for detecting fraudulent banking transactions. "
        "Powered by a Scikit-learn Random Forest pipeline trained on PCA-transformed credit card features."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def home() -> Dict[str, str]:
    """Root endpoint verifying API availability."""
    return {
        "message": "Banking Fraud Detection API is running",
        "status": "healthy"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Health check endpoint for Kubernetes liveness/readiness probes and monitoring."""
    if not model_service.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded or is unhealthy"
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=model_service.model_name,
        classification_threshold=model_service.threshold
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"]
)
def predict(transaction: TransactionInput) -> PredictionResponse:
    """Predict whether a given credit card transaction is fraudulent.

    Validates all 30 required transaction features, evaluates the transaction using
    the trained Random Forest pipeline, and applies the configured classification threshold.
    """
    if not model_service.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable. Pipeline is not loaded."
        )

    try:
        # transaction.model_dump() provides a dictionary of all 30 features
        prediction_result = model_service.predict(transaction.model_dump())
        return PredictionResponse(**prediction_result)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid transaction features: {str(ve)}"
        )
    except Exception as exc:
        logger.exception("Unexpected error during fraud prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error occurred: {str(exc)}"
        )


# ============================================================================
# Dataset Upload, Validation & Preview Endpoints
# ============================================================================

@app.post(
    "/datasets/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Datasets"]
)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
    """Upload a CSV dataset for fraud detection model training.

    Validates file extension, size limit, CSV structure, presence of all 30
    required features, and binary target column 'Class'.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload."
        )

    try:
        content = await file.read()
        saved = dataset_service.save_upload(content, file.filename)
        return DatasetUploadResponse(**saved)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as exc:
        logger.exception("Failed to process dataset upload")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(exc)}"
        )


@app.get(
    "/datasets/{dataset_id}/preview",
    response_model=DatasetPreviewResponse,
    tags=["Datasets"]
)
def preview_dataset(dataset_id: str, limit: int = 5) -> DatasetPreviewResponse:
    """Return schema info, validation status, and sample rows for an uploaded dataset."""
    try:
        data = dataset_service.get_preview(dataset_id, limit=limit)
        return DatasetPreviewResponse(**data)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found."
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to preview dataset: {str(exc)}"
        )


@app.post(
    "/datasets/{dataset_id}/validate",
    response_model=DatasetValidationResult,
    tags=["Datasets"]
)
def validate_dataset(dataset_id: str) -> DatasetValidationResult:
    """Run full schema and data validation on an uploaded dataset."""
    try:
        val = dataset_service.validate_stored_dataset(dataset_id)
        return DatasetValidationResult(**val)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found."
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation failed: {str(exc)}"
        )


@app.get(
    "/datasets",
    response_model=List[DatasetUploadResponse],
    tags=["Datasets"]
)
def list_datasets() -> List[DatasetUploadResponse]:
    """List all uploaded datasets with metadata and validation status."""
    return [DatasetUploadResponse(**d) for d in dataset_service.list_datasets()]


# ============================================================================
# Model Training Endpoints
# ============================================================================

@app.post(
    "/training",
    response_model=TrainingRecord,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Training"]
)
def start_training(payload: TrainingStartRequest) -> TrainingRecord:
    """Trigger background model training on a validated dataset.

    Trains a Scikit-learn Random Forest pipeline on 80% split, evaluates on
    20% holdout test split at the configured threshold (0.40), computes
    Precision, Recall, F1, PR-AUC, and Confusion Matrix, and saves candidate model.
    """
    try:
        training_id = training_service.start_training(payload.dataset_id)
        record = training_service.get_status(training_id)
        return TrainingRecord(**record)
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf)
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as exc:
        logger.exception("Failed to start model training")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training initiation error: {str(exc)}"
        )


@app.get(
    "/training/{training_id}",
    response_model=TrainingRecord,
    tags=["Training"]
)
def get_training_status(training_id: str) -> TrainingRecord:
    """Fetch status, holdout evaluation metrics, and candidate model info for a training job."""
    try:
        record = training_service.get_status(training_id)
        return TrainingRecord(**record)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job '{training_id}' not found."
        )


@app.get(
    "/training",
    response_model=List[TrainingRecord],
    tags=["Training"]
)
def list_trainings() -> List[TrainingRecord]:
    """List all training runs sorted by creation time descending."""
    return [TrainingRecord(**r) for r in training_service.list_trainings()]
