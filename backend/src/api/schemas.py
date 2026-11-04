"""Pydantic schemas for request validation and response formatting."""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class TransactionInput(BaseModel):
    """Transaction features required for fraud prediction.

    Expected features: Time, V1-V28 (PCA transformed components), Amount.
    """

    Time: float = Field(
        ...,
        description="Seconds elapsed between this transaction and the first transaction in the dataset",
        examples=[0.0]
    )
    V1: float = Field(..., description="PCA transformed component V1", examples=[-1.3598])
    V2: float = Field(..., description="PCA transformed component V2", examples=[-0.0728])
    V3: float = Field(..., description="PCA transformed component V3", examples=[2.5363])
    V4: float = Field(..., description="PCA transformed component V4", examples=[1.3782])
    V5: float = Field(..., description="PCA transformed component V5", examples=[-0.3383])
    V6: float = Field(..., description="PCA transformed component V6", examples=[0.4624])
    V7: float = Field(..., description="PCA transformed component V7", examples=[0.2396])
    V8: float = Field(..., description="PCA transformed component V8", examples=[0.0987])
    V9: float = Field(..., description="PCA transformed component V9", examples=[0.3638])
    V10: float = Field(..., description="PCA transformed component V10", examples=[0.0908])
    V11: float = Field(..., description="PCA transformed component V11", examples=[-0.5516])
    V12: float = Field(..., description="PCA transformed component V12", examples=[-0.6178])
    V13: float = Field(..., description="PCA transformed component V13", examples=[-0.9914])
    V14: float = Field(..., description="PCA transformed component V14", examples=[-0.3112])
    V15: float = Field(..., description="PCA transformed component V15", examples=[1.4682])
    V16: float = Field(..., description="PCA transformed component V16", examples=[-0.4704])
    V17: float = Field(..., description="PCA transformed component V17", examples=[0.2080])
    V18: float = Field(..., description="PCA transformed component V18", examples=[0.0258])
    V19: float = Field(..., description="PCA transformed component V19", examples=[0.4040])
    V20: float = Field(..., description="PCA transformed component V20", examples=[0.2514])
    V21: float = Field(..., description="PCA transformed component V21", examples=[-0.0183])
    V22: float = Field(..., description="PCA transformed component V22", examples=[0.2778])
    V23: float = Field(..., description="PCA transformed component V23", examples=[-0.1105])
    V24: float = Field(..., description="PCA transformed component V24", examples=[0.0669])
    V25: float = Field(..., description="PCA transformed component V25", examples=[0.1285])
    V26: float = Field(..., description="PCA transformed component V26", examples=[-0.1891])
    V27: float = Field(..., description="PCA transformed component V27", examples=[0.1336])
    V28: float = Field(..., description="PCA transformed component V28", examples=[-0.0211])
    Amount: float = Field(
        ...,
        ge=0.0,
        description="Transaction amount in currency units (must be non-negative)",
        examples=[149.62]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Time": 0.0,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053,
                "Amount": 149.62
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response returned by the prediction endpoint."""

    fraud_probability: float = Field(
        ...,
        description="Predicted probability that the transaction is fraudulent (0.0 to 1.0)",
        examples=[0.85]
    )
    classification_threshold: float = Field(
        ...,
        description="Configured classification threshold used for decision making",
        examples=[0.40]
    )
    prediction: str = Field(
        ...,
        description="Classification decision: 'FRAUD' or 'LEGITIMATE'",
        examples=["FRAUD"]
    )


class HealthResponse(BaseModel):
    """Response returned by the health check endpoint."""

    status: str = Field("healthy", description="Overall API health status")
    model_loaded: bool = Field(..., description="Whether the ML pipeline is loaded and ready")
    model_name: Optional[str] = Field(None, description="Name of the loaded model")
    classification_threshold: Optional[float] = Field(
        None,
        description="Configured classification threshold"
    )



# ============================================================================
# Dataset Upload & Validation Schemas
# ============================================================================

from typing import Any, Dict, List


class DatasetValidationResult(BaseModel):
    """Result of schema and content validation on an uploaded dataset."""

    is_valid: bool = Field(..., description="Whether the dataset satisfies requirements for ML training")
    errors: List[str] = Field(default_factory=list, description="List of blocking validation errors")
    warnings: List[str] = Field(default_factory=list, description="Non-blocking recommendations")
    missing_values: Dict[str, int] = Field(default_factory=dict, description="Count of nulls per column")
    class_distribution: Optional[Dict[str, int]] = Field(
        None, description="Count of binary targets (Class 0 and Class 1)"
    )
    detected_columns: List[str] = Field(default_factory=list, description="All columns found in CSV")


class DatasetUploadResponse(BaseModel):
    """Metadata returned upon successful CSV upload."""

    dataset_id: str = Field(..., description="Unique identifier for the stored dataset")
    filename: str = Field(..., description="Original filename")
    file_size_bytes: int = Field(..., description="Size in bytes")
    uploaded_at: str = Field(..., description="ISO 8601 upload timestamp")
    row_count: int = Field(..., description="Total rows in dataset")
    column_count: int = Field(..., description="Total columns in dataset")
    validation: DatasetValidationResult


class DatasetPreviewResponse(BaseModel):
    """Detailed preview of an uploaded dataset."""

    dataset_id: str
    filename: str
    row_count: int
    column_count: int
    columns: List[str]
    sample_rows: List[Dict[str, Any]]
    validation: DatasetValidationResult


# ============================================================================
# Model Training Schemas
# ============================================================================

class TrainingStartRequest(BaseModel):
    """Payload to trigger model training on a validated dataset."""

    dataset_id: str = Field(..., description="Identifier of the uploaded dataset to train on")


class ConfusionMatrixInfo(BaseModel):
    """Breakdown of holdout evaluation confusion matrix."""

    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


class TrainingMetrics(BaseModel):
    """Holdout evaluation metrics computed on the test split."""

    precision: float
    recall: float
    f1_score: float
    pr_auc: float
    confusion_matrix: ConfusionMatrixInfo
    threshold: float
    total_samples: int
    train_samples: int
    test_samples: int


class TrainingRecord(BaseModel):
    """Status and metrics of a model training session."""

    training_id: str
    dataset_id: str
    status: str = Field(..., description="'queued' | 'running' | 'completed' | 'failed'")
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    metrics: Optional[TrainingMetrics] = None
    candidate_model_saved: bool = False
