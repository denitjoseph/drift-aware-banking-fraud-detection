import pandas as pd

from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("data/processed/creditcard_cleaned.csv")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("========== DATA SPLITTING RESULTS ==========")

print("\nOriginal dataset:", X.shape)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nTraining fraud percentage:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting fraud percentage:")
print(y_test.value_counts(normalize=True) * 100)

print("\nData splitting completed successfully!")

