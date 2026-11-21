
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Download Adult Census Income dataset
adult = fetch_ucirepo(id=2)

# Extract features and target
X = adult.data.features
y = adult.data.targets

# Combine into one DataFrame
df = pd.concat([X, y], axis=1)

# Save dataset
df.to_csv("adult_income.csv", index=False)

print("Dataset downloaded successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["income"].value_counts())
