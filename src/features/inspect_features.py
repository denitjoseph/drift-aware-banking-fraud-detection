import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/creditcard_cleaned.csv")

print("========== FEATURE INSPECTION ==========")

# Inspect Time
print("\nTime Feature:")
print(df["Time"].describe())

# Inspect Amount
print("\nAmount Feature:")
print(df["Amount"].describe())

# Check mean and standard deviation
print("\nFeature Mean:")
print(df[["Time", "Amount"]].mean())

print("\nFeature Standard Deviation:")
print(df[["Time", "Amount"]].std())

print("\nFeature inspection completed!")

