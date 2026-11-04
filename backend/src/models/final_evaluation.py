
import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/creditcard_cleaned.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Create training data (80%) and untouched test data (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

# 4. Load model configuration
with open("configs/model_config.json", "r") as file:
    config = json.load(file)

# 5. Load the trained pipeline
model = joblib.load(config["model_file"])

# IMPORTANT:
# This pipeline was trained on the original 80% training split.
# We must retrain it using only the new training subset.
# The existing pipeline cannot be used for an independent
# validation/test evaluation because it has seen the old test data.

from sklearn.base import clone

model = clone(model)

# 6. Train using only the new training subset
print("Training model on training subset...")
model.fit(X_train, y_train)

# 7. Find threshold using validation data
validation_probabilities = model.predict_proba(X_val)[:, 1]

thresholds = [
    0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50
]

best_threshold = 0
best_f1 = 0

for threshold in thresholds:

    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_val,
        validation_predictions,
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print("\nSelected validation threshold:", best_threshold)
print("Validation F1-score:", round(best_f1, 4))

# 8. Evaluate once on untouched test data
test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

# 9. Display final results
print("\nFINAL TEST CLASSIFICATION REPORT")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

print("\nFINAL TEST CONFUSION MATRIX")
print(confusion_matrix(y_test, test_predictions))

print("\nFINAL TEST METRICS")
print("Precision:", precision_score(y_test, test_predictions))
print("Recall:", recall_score(y_test, test_predictions))
print("F1 Score:", f1_score(y_test, test_predictions))
print("PR-AUC:", average_precision_score(y_test, test_probabilities))

# 10. Save final evaluation results
results = {
    "model_name": config["model_name"],
    "selected_threshold": best_threshold,
    "validation_f1": best_f1,
    "test_precision": precision_score(y_test, test_predictions),
    "test_recall": recall_score(y_test, test_predictions),
    "test_f1": f1_score(y_test, test_predictions),
    "test_pr_auc": average_precision_score(y_test, test_probabilities)
}

Path("models").mkdir(exist_ok=True)

with open("models/final_evaluation.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nFinal evaluation saved to models/final_evaluation.json")
