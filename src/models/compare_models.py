
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/creditcard_cleaned.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Recreate the same test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Load both saved models and scalers
models = {
    "Logistic Regression": (
        joblib.load("models/logistic_regression.pkl"),
        joblib.load("models/scaler.pkl")
    ),
    "Random Forest": (
        joblib.load("models/random_forest.pkl"),
        joblib.load("models/random_forest_scaler.pkl")
    )
}

# 4. Prepare test data for each scaler
results = []

for model_name, (model, scaler) in models.items():

    X_test_scaled = X_test.copy()

    X_test_scaled[["Time", "Amount"]] = scaler.transform(
        X_test_scaled[["Time", "Amount"]]
    )

    # 5. Generate predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # 6. Calculate metrics
    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred
    ).ravel()

    results.append({
        "Model": model_name,
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "PR-AUC": average_precision_score(y_test, y_prob),
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp,
        "True Negatives": tn
    })

# 7. Display comparison
comparison = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print("=" * 90)
print(comparison.to_string(index=False))

# 8. Save comparison report
comparison.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nComparison report saved to models/model_comparison.csv")
