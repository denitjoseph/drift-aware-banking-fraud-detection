"""Service module for loading the trained pipeline and executing fraud predictions."""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

logger = logging.getLogger("fraud_api.model_service")

# Project root directory (3 levels up from src/api/model_service.py)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Feature columns in the exact order expected by the trained pipeline
FEATURE_COLUMNS: List[str] = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]


class ModelService:
    """Manages model pipeline loading, configuration, and inference."""

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.model_name: Optional[str] = None
        self.threshold: float = 0.40
        self.model_path: Optional[Path] = None
        self.config_path: Optional[Path] = None
        self._is_loaded: bool = False

    def _resolve_path(self, given_path: Optional[str], default_rel_path: str) -> Path:
        """Resolve a file path relative to working directory or project root."""
        candidate = Path(given_path) if given_path else Path(default_rel_path)

        # 1. Direct path check (relative to cwd or absolute)
        if candidate.exists():
            return candidate.resolve()

        # 2. Check relative to project root
        candidate_from_base = (BASE_DIR / candidate).resolve()
        if candidate_from_base.exists():
            return candidate_from_base

        return candidate

    def load_model(
        self,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None
    ) -> None:
        """Load the model configuration and the trained Scikit-learn pipeline."""
        # Check environment variables as fallbacks
        cfg_env = os.getenv("CONFIG_PATH", config_path)
        mdl_env = os.getenv("MODEL_PATH", model_path)

        # 1. Resolve configuration path
        resolved_config = self._resolve_path(cfg_env, "configs/model_config.json")
        if not resolved_config.exists():
            raise FileNotFoundError(
                f"Model configuration file not found at: {resolved_config}"
            )

        with open(resolved_config, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.config_path = resolved_config
        self.model_name = config.get("model_name", "Random Forest")
        self.threshold = float(config.get("classification_threshold", 0.40))

        # 2. Resolve model artifact path
        configured_model_file = mdl_env or config.get(
            "model_file", "models/fraud_detection_pipeline.pkl"
        )
        resolved_model = self._resolve_path(
            configured_model_file, "models/fraud_detection_pipeline.pkl"
        )

        if not resolved_model.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at: {resolved_model}"
            )

        logger.info("Loading model pipeline from %s", resolved_model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = joblib.load(resolved_model)

        self.model_path = resolved_model
        self._is_loaded = True
        logger.info(
            "Model successfully loaded. Name: %s, Threshold: %.2f",
            self.model_name,
            self.threshold
        )

    def is_healthy(self) -> bool:
        """Check whether the model service is loaded and ready to serve predictions."""
        return self._is_loaded and self.model is not None

    def predict(self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        """Perform fraud inference on input features.

        Args:
            input_features: Dictionary containing transaction features.

        Returns:
            Dictionary with fraud_probability, classification_threshold, and prediction.
        """
        if not self.is_healthy():
            # Attempt lazy loading if not yet loaded
            self.load_model()

        # Build DataFrame with features in the exact expected order
        df_input = pd.DataFrame([input_features])

        missing_cols = [col for col in FEATURE_COLUMNS if col not in df_input.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        df_ordered = df_input[FEATURE_COLUMNS]

        # Calculate fraud probability (class 1)
        probabilities = self.model.predict_proba(df_ordered)
        fraud_prob = float(probabilities[0][1])

        # Classification decision based on configured threshold
        prediction = "FRAUD" if fraud_prob >= self.threshold else "LEGITIMATE"

        return {
            "fraud_probability": round(fraud_prob, 4),
            "classification_threshold": round(self.threshold, 4),
            "prediction": prediction
        }


# Global model service instance
model_service = ModelService()

