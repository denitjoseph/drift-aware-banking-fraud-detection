import pandas as pd
from pathlib import Path

# Locate the dataset
dataset_path = Path("data/raw/creditcard.csv")

# Load the dataset
df = pd.read_csv(dataset_path)

print("\n========== DATASET OVERVIEW ==========")

# Number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# First 5 records
print("\nFirst 5 Records:")
print(df.head())

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# Duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Fraud distribution
print("\n========== FRAUD DISTRIBUTION ==========")

print(df["Class"].value_counts())

print("\nFraud Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)

# Transaction amount statistics
print("\n========== TRANSACTION AMOUNT ==========")
print(df["Amount"].describe())

print("\n========== INSPECTION COMPLETED ==========")

print("\n========== DUPLICATE ANALYSIS ==========")

duplicate_rows = df[df.duplicated(keep=False)]

print("\nTotal rows involved in duplicates:")
print(len(duplicate_rows))

print("\nClass distribution among duplicate rows:")
print(duplicate_rows["Class"].value_counts())

print("\nFraud distribution in original dataset:")
print(df["Class"].value_counts())