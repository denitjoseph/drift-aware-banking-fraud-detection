"""FastAPI application for the Drift-Aware Banking Fraud Detection Platform."""

import logging
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.model_service import model_service
from src.api.schemas import HealthResponse, PredictionResponse, TransactionInput

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
