"""Model training service executing asynchronous retraining jobs on uploaded datasets."""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.api.dataset_service import TARGET_COLUMN, dataset_service
from src.api.model_service import FEATURE_COLUMNS, model_service

logger = logging.getLogger("fraud_api.training_service")

# Project base directory (backend/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAINED_MODELS_DIR = BASE_DIR / "models" / "trained"
RECORDS_FILE = TRAINED_MODELS_DIR / "training_records.json"


class TrainingService:
    """Manages asynchronous model training, holdout evaluation, and candidate model persistence."""

    def __init__(self) -> None:
        TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, Dict[str, Any]] = self._load_records()
        self._lock = threading.Lock()

    def _load_records(self) -> Dict[str, Dict[str, Any]]:
        if RECORDS_FILE.exists():
            try:
                with open(RECORDS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not read training records, initializing empty: %s", e)
        return {}

    def _save_records(self) -> None:
        try:
            with open(RECORDS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._records, f, indent=2)
        except Exception as e:
            logger.error("Failed to write training records: %s", e)

    def start_training(self, dataset_id: str) -> str:
        """Initiate background model training on a validated dataset.

        Returns:
            training_id: Unique identifier for the training session.
        """
        # 1. Verify dataset exists on disk
        dataset_path = dataset_service.get_dataset_path(dataset_id)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset '{dataset_id}' not found.")

        # 2. Check dataset validation
        preview_info = dataset_service.get_preview(dataset_id, limit=5)
        val = preview_info.get("validation", {})
        if not val.get("is_valid", False):
            errors = val.get("errors", ["Dataset failed validation"])
            raise ValueError(f"Cannot train on invalid dataset: {'; '.join(errors)}")

        training_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()

        record = {
            "training_id": training_id,
            "dataset_id": dataset_id,
            "status": "queued",
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "metrics": None,
            "candidate_model_saved": False
        }

        with self._lock:
            self._records[training_id] = record
            self._save_records()

        # Launch training in background worker thread
        thread = threading.Thread(
            target=self._run_training_job,
            args=(training_id, dataset_path),
            daemon=True
        )
        thread.start()

        return training_id

    def _run_training_job(self, training_id: str, dataset_path: Path) -> None:
        """Worker thread executing training pipeline and metrics evaluation."""
        with self._lock:
            self._records[training_id]["status"] = "running"
            self._records[training_id]["started_at"] = datetime.now(timezone.utc).isoformat()
            self._save_records()

        logger.info("Started training job %s on dataset %s", training_id, dataset_path.name)

        try:
            # 1. Load data
            df = pd.read_csv(dataset_path)

            if len(df) < 10:
                raise ValueError("Dataset has fewer than 10 rows. Insufficient data to perform 80/20 train/test evaluation.")

            # Separate features and target
            X = df[FEATURE_COLUMNS].copy()
            y = df[TARGET_COLUMN].copy().astype(int)

            fraud_count = int((y == 1).sum())
            legit_count = int((y == 0).sum())

            if fraud_count == 0 or legit_count == 0:
                raise ValueError("Dataset must contain both legitimate (0) and fraud (1) samples to evaluate classification metrics.")

            # Use stratified split if both classes have >= 2 instances
            can_stratify = fraud_count >= 2 and legit_count >= 2
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y if can_stratify else None
            )

            # 2. Build Pipeline reusing Phase 2 architecture
            numeric_features = ["Time", "Amount"]
            preprocessor = ColumnTransformer(
                transformers=[
                    ("scaler", StandardScaler(), numeric_features)
                ],
                remainder="passthrough"
            )

            n_trees = 100 if len(X_train) >= 1000 else 50
            classifier = RandomForestClassifier(
                n_estimators=n_trees,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )

            pipeline = Pipeline(
                steps=[
                    ("preprocessing", preprocessor),
                    ("classifier", classifier)
                ]
            )

            # 3. Fit pipeline on training subset
            pipeline.fit(X_train, y_train)

            # 4. Evaluate on holdout test subset
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            threshold = float(model_service.threshold)
            y_pred = (y_prob >= threshold).astype(int)

            # Metrics
            precision = float(precision_score(y_test, y_pred, zero_division=0))
            recall = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))

            # PR-AUC
            try:
                pr_auc = float(average_precision_score(y_test, y_prob))
            except Exception:
                pr_auc = 0.0

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

            metrics = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "pr_auc": round(pr_auc, 4),
                "confusion_matrix": {
                    "true_negatives": tn,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "true_positives": tp
                },
                "threshold": threshold,
                "total_samples": len(df),
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }

            # 5. Save candidate pipeline artifact safely (preserves production pipeline)
            model_file = TRAINED_MODELS_DIR / f"{training_id}_pipeline.pkl"
            joblib.dump(pipeline, model_file)

            with self._lock:
                self._records[training_id]["status"] = "completed"
                self._records[training_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                self._records[training_id]["metrics"] = metrics
                self._records[training_id]["candidate_model_saved"] = True
                self._save_records()

            logger.info("Training job %s successfully completed. F1: %.4f, PR-AUC: %.4f", training_id, f1, pr_auc)

        except Exception as exc:
            logger.exception("Training job %s failed: %s", training_id, exc)
            with self._lock:
                self._records[training_id]["status"] = "failed"
                self._records[training_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                self._records[training_id]["error"] = str(exc)
                self._save_records()

    def get_status(self, training_id: str) -> Dict[str, Any]:
        """Fetch the current status and metrics of a training run."""
        with self._lock:
            record = self._records.get(training_id)
        if not record:
            raise KeyError(f"Training run '{training_id}' not found.")
        return record

    def list_trainings(self) -> List[Dict[str, Any]]:
        """Return list of all training runs sorted by creation time descending."""
        with self._lock:
            records = list(self._records.values())
        return sorted(records, key=lambda r: r.get("created_at", ""), reverse=True)


training_service = TrainingService()
