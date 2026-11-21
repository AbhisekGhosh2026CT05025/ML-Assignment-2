
import pandas as pd

# Load original dataset
df = pd.read_csv("adult_income.csv")

# Remove unwanted leading/trailing spaces
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Normalize target labels
df["income"] = df["income"].str.rstrip(".")

# 1. Check target distribution
print("\n--- NORMALIZED TARGET DISTRIBUTION ---")
print(df["income"].value_counts())

print("\nTarget percentages:")
print(df["income"].value_counts(normalize=True) * 100)

# 2. Check duplicates
print("\n--- DUPLICATE ANALYSIS ---")
duplicate_count = df.duplicated().sum()
print("Duplicate records:", duplicate_count)

# 3. Check missing values
print("\n--- MISSING VALUE ANALYSIS ---")
print(df.isnull().sum())

# 4. Check numerical statistics
print("\n--- NUMERICAL FEATURE STATISTICS ---")
print(df.describe())

# 5. Check categorical feature distribution
print("\n--- CATEGORICAL FEATURE DETAILS ---")
for col in df.select_dtypes(include="object").columns:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))

print("\n--- DATASET EXPLORATION COMPLETE ---")
