
import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/creditcard_cleaned.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Recreate the training and test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Combine training and validation data
# These are the 80% of data available before final testing.
X_final_train = X_train.copy()
y_final_train = y_train.copy()

# 4. Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), ["Time", "Amount"])
    ],
    remainder="passthrough"
)

# 5. Create final Random Forest pipeline
final_pipeline = Pipeline(
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

# 6. Train final pipeline
print("Training final Random Forest pipeline...")

final_pipeline.fit(X_final_train, y_final_train)

# 7. Save final pipeline
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

joblib.dump(
    final_pipeline,
    models_dir / "fraud_detection_pipeline.pkl"
)

# 8. Update model configuration
config_path = Path("configs/model_config.json")

with open(config_path, "r") as file:
    config = json.load(file)

config["model_name"] = "Random Forest"
config["model_file"] = "models/fraud_detection_pipeline.pkl"
config["classification_threshold"] = 0.40
config["threshold_selection_metric"] = "F1 Score"
config["threshold_selection_note"] = (
    "Selected using the validation dataset. "
    "Final test evaluation was performed separately."
)

with open(config_path, "w") as file:
    json.dump(config, file, indent=4)

print("\nFinal model saved successfully!")
print("Model: models/fraud_detection_pipeline.pkl")
print("Selected threshold:", config["classification_threshold"])
print("Configuration updated:", config_path)
