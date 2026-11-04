import pandas as pd
from pathlib import Path

# File paths
input_path = Path("data/raw/creditcard.csv")
output_path = Path("data/processed/creditcard_cleaned.csv")

# Load original dataset
df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# Check missing values
print("\nMissing values:", df.isnull().sum().sum())

# Check duplicate records
print("Duplicate occurrences:", df.duplicated().sum())

# Remove exact duplicate rows
df_cleaned = df.drop_duplicates().copy()

# Display cleaned dataset information
print("\nCleaned dataset shape:", df_cleaned.shape)

print("\nRemaining duplicates:", df_cleaned.duplicated().sum())

# Display fraud distribution
print("\nFraud distribution after cleaning:")
print(df_cleaned["Class"].value_counts())

# Save cleaned dataset
output_path.parent.mkdir(parents=True, exist_ok=True)

df_cleaned.to_csv(output_path, index=False)

print("\nCleaned dataset saved to:", output_path)
print("\nPreprocessing completed successfully!")

