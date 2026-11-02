
import json
import joblib
import pandas as pd

from pathlib import Path

# 1. Load model configuration
config_path = Path("configs/model_config.json")

with open(config_path, "r") as file:
    config = json.load(file)

# 2. Load the complete ML pipeline
model_path = Path(config["model_file"])
model = joblib.load(model_path)

# 3. Load classification threshold
threshold = config["classification_threshold"]

# 4. Prepare a sample transaction
# Use a real row from the cleaned dataset for testing
df = pd.read_csv("data/processed/creditcard_cleaned.csv")

# Select a known fraudulent transaction
fraud_row = df[df["Class"] == 1].iloc[[0]]
sample = fraud_row.drop("Class", axis=1)
actual_class = int(fraud_row["Class"].iloc[0])

# 5. Predict fraud probability
fraud_probability = model.predict_proba(sample)[0][1]

# 6. Classify the transaction
prediction = (
    "FRAUD"
    if fraud_probability >= threshold
    else "LEGITIMATE"
)

# 7. Display prediction
print("TRANSACTION PREDICTION")
print("=" * 40)
print("Actual class:", actual_class)
print("Fraud probability:", round(fraud_probability, 4))
print("Classification threshold:", threshold)
print("Prediction:", prediction)

print("\nPrediction completed successfully!")
