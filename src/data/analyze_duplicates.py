import pandas as pd

# Load original dataset
df = pd.read_csv("data/raw/creditcard.csv")

print("Original dataset shape:", df.shape)

# Count duplicate occurrences
duplicate_count = df.duplicated().sum()

print("\nDuplicate occurrences:", duplicate_count)

# Get every row belonging to a duplicate group
duplicate_rows = df[df.duplicated(keep=False)]

print("\nRows involved in duplicate groups:")
print(len(duplicate_rows))

# Show class distribution, including zero-count classes
print("\nClass distribution among duplicate-group rows:")

print(
    duplicate_rows["Class"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

# Count distinct duplicate groups
unique_duplicate_groups = (
    df.groupby(list(df.columns), dropna=False)
    .size()
)

print("\nNumber of duplicate groups:")
print((unique_duplicate_groups > 1).sum())

print("\nDuplicate analysis completed.")

