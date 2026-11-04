import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/creditcard_cleaned.csv")

# 2. Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Scale Time and Amount
scaler = StandardScaler()

X_train = X_train.copy()
X_test = X_test.copy()

X_train[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

X_test[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)

# 5. Create Logistic Regression model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# 6. Train model
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 8. Evaluate model
print("\n========== LOGISTIC REGRESSION RESULTS ==========")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nPR-AUC (Average Precision):")
print(average_precision_score(y_test, y_prob))

# 9. Save model and scaler
output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

joblib.dump(model, output_dir / "logistic_regression.pkl")
joblib.dump(scaler, output_dir / "scaler.pkl")

print("\nModel saved successfully!")
print("Model: models/logistic_regression.pkl")
print("Scaler: models/scaler.pkl")

