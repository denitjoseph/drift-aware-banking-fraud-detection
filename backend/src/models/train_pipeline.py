
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    f1_score
)

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/creditcard_cleaned.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Define preprocessing
numeric_features = ["Time", "Amount"]

preprocessor = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), numeric_features)
    ],
    remainder="passthrough"
)

# 4. Create complete ML pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# 5. Train the pipeline
print("Training Random Forest pipeline...")

model_pipeline.fit(X_train, y_train)

# 6. Evaluate the pipeline
y_prob = model_pipeline.predict_proba(X_test)[:, 1]

threshold = 0.35
y_pred = (y_prob >= threshold).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nPR-AUC:")
print(average_precision_score(y_test, y_prob))

print("\nF1 Score at threshold 0.35:")
print(f1_score(y_test, y_pred))

# 7. Save the complete pipeline
output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

joblib.dump(
    model_pipeline,
    output_dir / "fraud_detection_pipeline.pkl"
)

print("\nComplete ML pipeline saved successfully!")
print("Model: models/fraud_detection_pipeline.pkl")
