"""Dataset service for managing uploads, storage, schema validation, and row previews."""

import io
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.api.model_service import FEATURE_COLUMNS

logger = logging.getLogger("fraud_api.dataset_service")

# Project base directory (backend/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
METADATA_FILE = UPLOAD_DIR / "datasets_metadata.json"

TARGET_COLUMN = "Class"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# Default max upload size: 100 MB
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", 100 * 1024 * 1024))


class DatasetService:
    """Handles dataset upload processing, storage, validation, and preview."""

    def __init__(self) -> None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self._metadata: Dict[str, Dict[str, Any]] = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        if METADATA_FILE.exists():
            try:
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not read datasets metadata, initializing empty: %s", e)
        return {}

    def _save_metadata(self) -> None:
        try:
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error("Failed to write datasets metadata: %s", e)

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate DataFrame columns, target distribution, missing values, and data types."""
        errors: List[str] = []
        warnings: List[str] = []
        detected_columns = list(df.columns)

        # 1. Row count check
        if len(df) == 0:
            errors.append("Dataset is empty (contains 0 data rows).")
            return {
                "is_valid": False,
                "errors": errors,
                "warnings": warnings,
                "missing_values": {},
                "class_distribution": None,
                "detected_columns": detected_columns
            }

        # 2. Check required feature columns
        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            errors.append(f"Missing required feature columns: {missing_features}")

        # 3. Check target column
        if TARGET_COLUMN not in df.columns:
            errors.append(f"Missing target column: '{TARGET_COLUMN}'. Required for supervised fraud model training.")

        # 4. Check data types and parseability for existing feature columns
        for col in [c for c in FEATURE_COLUMNS if c in df.columns]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' must contain numeric values.")

        # 5. Check missing values
        null_counts = df.isnull().sum().to_dict()
        missing_dict = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}
        total_missing = sum(missing_dict.values())
        if total_missing > 0:
            warnings.append(
                f"Dataset contains {total_missing} missing value(s) across {len(missing_dict)} column(s)."
            )

        # 6. Check target distribution if target column exists
        class_dist: Optional[Dict[str, int]] = None
        if TARGET_COLUMN in df.columns:
            unique_targets = set(df[TARGET_COLUMN].dropna().unique())
            valid_binary = unique_targets.issubset({0, 1, 0.0, 1.0})
            if not valid_binary:
                errors.append(f"Target column '{TARGET_COLUMN}' must be binary (0 for legitimate, 1 for fraud). Detected values: {unique_targets}")
            else:
                counts = df[TARGET_COLUMN].value_counts().to_dict()
                class_dist = {
                    "legitimate": int(counts.get(0, counts.get(0.0, 0))),
                    "fraud": int(counts.get(1, counts.get(1.0, 0)))
                }
                if class_dist["fraud"] == 0:
                    warnings.append("Dataset contains 0 positive fraud samples (Class = 1). Training requires at least 1 fraud record.")
                if class_dist["legitimate"] == 0:
                    errors.append("Dataset contains 0 legitimate samples (Class = 0).")

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "missing_values": missing_dict,
            "class_distribution": class_dist,
            "detected_columns": detected_columns
        }

    def save_upload(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Validate, store CSV, and record dataset metadata."""
        # 1. File extension validation
        if not filename.lower().endswith(".csv"):
            raise ValueError("Only CSV files (.csv) are supported.")

        # 2. File size validation
        size = len(file_bytes)
        if size == 0:
            raise ValueError("Uploaded file is empty (0 bytes).")
        if size > MAX_UPLOAD_SIZE:
            max_mb = MAX_UPLOAD_SIZE // (1024 * 1024)
            raise ValueError(f"File size exceeds maximum allowed limit of {max_mb} MB.")

        # 3. CSV parsing validation
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as exc:
            raise ValueError(f"Malformed CSV file could not be parsed: {str(exc)}")

        dataset_id = uuid.uuid4().hex[:12]
        file_path = UPLOAD_DIR / f"{dataset_id}.csv"

        # Save to disk
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # Run validation
        val_result = self.validate_dataframe(df)

        meta = {
            "dataset_id": dataset_id,
            "filename": filename,
            "file_size_bytes": size,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "validation": val_result
        }

        self._metadata[dataset_id] = meta
        self._save_metadata()

        return meta

    def get_dataset_path(self, dataset_id: str) -> Path:
        """Resolve dataset path on disk."""
        target = UPLOAD_DIR / f"{dataset_id}.csv"
        if not target.exists():
            raise FileNotFoundError(f"Dataset with ID '{dataset_id}' not found.")
        return target

    def validate_stored_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Load and validate an already uploaded dataset from disk."""
        path = self.get_dataset_path(dataset_id)
        df = pd.read_csv(path)
        val_result = self.validate_dataframe(df)
        if dataset_id in self._metadata:
            self._metadata[dataset_id]["validation"] = val_result
            self._save_metadata()
        return val_result

    def get_preview(self, dataset_id: str, limit: int = 5) -> Dict[str, Any]:
        """Generate row preview and schema statistics for an uploaded dataset."""
        path = self.get_dataset_path(dataset_id)
        meta = self._metadata.get(dataset_id, {})

        # Read only up to limit rows for performance
        df_preview = pd.read_csv(path, nrows=limit)

        # Sanitize NaN / Inf to None so JSON serializer outputs standard nulls
        df_clean = df_preview.replace({np.nan: None, np.inf: None, -np.inf: None})
        sample_rows = df_clean.to_dict(orient="records")

        # Full validation if not in metadata
        val_result = meta.get("validation")
        if not val_result:
            df_full = pd.read_csv(path)
            val_result = self.validate_dataframe(df_full)

        return {
            "dataset_id": dataset_id,
            "filename": meta.get("filename", f"{dataset_id}.csv"),
            "row_count": meta.get("row_count", len(df_preview)),
            "column_count": meta.get("column_count", len(df_preview.columns)),
            "columns": list(df_preview.columns),
            "sample_rows": sample_rows,
            "validation": val_result
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Return list of all uploaded datasets."""
        return list(self._metadata.values())


dataset_service = DatasetService()
