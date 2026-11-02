
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

# 1. Load the cleaned dataset
df = pd.read_csv("data/processed/creditcard_cleaned.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Recreate the same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Load the trained Random Forest and its scaler
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/random_forest_scaler.pkl")

# 4. Scale test features
X_test = X_test.copy()

X_test[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)

# 5. Get fraud probabilities
y_prob = model.predict_proba(X_test)[:, 1]

print("PR-AUC:", average_precision_score(y_test, y_prob))

# 6. Evaluate different thresholds
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

results = []

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp,
        "True Negatives": tn
    })

# 7. Display threshold comparison
comparison = pd.DataFrame(results)

print("\nRANDOM FOREST THRESHOLD COMPARISON")
print("=" * 100)
print(comparison.to_string(index=False))

# 8. Save results
comparison.to_csv(
    "models/threshold_comparison.csv",
    index=False
)

print("\nResults saved to models/threshold_comparison.csv")
