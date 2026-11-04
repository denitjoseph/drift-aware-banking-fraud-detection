import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset
df = pd.read_csv("data/processed/creditcard_cleaned.csv")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create scaler
scaler = StandardScaler()

# Fit only on training data
X_train[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

# Transform test data using training scaler
X_test[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)

print("========== FEATURE SCALING RESULTS ==========")

print("\nTraining feature means:")
print(X_train[["Time", "Amount"]].mean())

print("\nTraining feature standard deviations:")
print(X_train[["Time", "Amount"]].std())

print("\nTesting feature means:")
print(X_test[["Time", "Amount"]].mean())

print("\nFeature scaling completed successfully!")

