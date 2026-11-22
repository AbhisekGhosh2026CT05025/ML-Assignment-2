
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# --------------------------------------------------
# STEP 1: LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("adult_income.csv")

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# STEP 2: CLEAN CATEGORICAL VALUES
# --------------------------------------------------

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

numerical_features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

# Convert categorical columns to standard object dtype
# and clean leading/trailing spaces.
for col in categorical_features:
    df[col] = df[col].astype(object).map(
        lambda value: (
            np.nan
            if pd.isna(value) or str(value).strip() == "?"
            else str(value).strip()
        )
    )

# Normalize target labels
df["income"] = (
    df["income"]
    .astype(str)
    .str.strip()
    .str.rstrip(".")
)

# --------------------------------------------------
# STEP 3: HANDLE DUPLICATES
# --------------------------------------------------

duplicates = df.duplicated().sum()

print("\nDuplicate records found:", duplicates)

df = df.drop_duplicates().reset_index(drop=True)

print("Shape after removing duplicates:", df.shape)


# --------------------------------------------------
# STEP 4: VERIFY TARGET
# --------------------------------------------------

print("\nTarget distribution:")
print(df["income"].value_counts())

print("\nTarget percentages:")
print(df["income"].value_counts(normalize=True) * 100)

print("\nMissing values after cleaning:")
print(df[categorical_features].isnull().sum())


# --------------------------------------------------
# STEP 5: SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(columns=["income"])
y = df["income"]


# --------------------------------------------------
# STEP 6: TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True) * 100)


# --------------------------------------------------
# STEP 7: NUMERICAL PREPROCESSING
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(
            strategy="median",
            missing_values=np.nan
        )),
        ("scaler", StandardScaler())
    ]
)


# --------------------------------------------------
# STEP 8: CATEGORICAL PREPROCESSING
# --------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(
            strategy="most_frequent",
            missing_values=np.nan
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)


# --------------------------------------------------
# STEP 9: COMBINE PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# --------------------------------------------------
# STEP 10: FIT ONLY ON TRAINING DATA
# --------------------------------------------------

print("\nFitting preprocessing pipeline on training data...")

X_train_processed = preprocessor.fit_transform(X_train)

# Test data is only transformed using the fitted pipeline.
X_test_processed = preprocessor.transform(X_test)


# --------------------------------------------------
# STEP 11: SAVE OUTPUTS
# --------------------------------------------------

os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Save cleaned training and testing datasets
train_data = X_train.copy()
train_data["income"] = y_train

test_data = X_test.copy()
test_data["income"] = y_test

train_data.to_csv(
    "data/train_data.csv",
    index=False
)

test_data.to_csv(
    "test_data.csv",
    index=False
)

# Save fitted preprocessing pipeline
joblib.dump(
    preprocessor,
    "model/preprocessor.pkl"
)

# Save transformed datasets
np.save(
    "data/X_train_processed.npy",
    X_train_processed
)

np.save(
    "data/X_test_processed.npy",
    X_test_processed
)

np.save(
    "data/y_train.npy",
    y_train.to_numpy()
)

np.save(
    "data/y_test.npy",
    y_test.to_numpy()
)

# Save generated feature names
feature_names = preprocessor.get_feature_names_out()

pd.Series(feature_names).to_csv(
    "output/processed_feature_names.csv",
    index=False
)


# --------------------------------------------------
# STEP 12: FINAL SUMMARY
# --------------------------------------------------

print("\n========== PREPROCESSING SUMMARY ==========")

print(
    "Original records:",
    len(pd.read_csv("adult_income.csv"))
)

print("Cleaned records:", len(df))

print("Training records:", len(X_train))

print("Testing records:", len(X_test))

print("Original features:", X.shape[1])

print(
    "Features after encoding:",
    X_train_processed.shape[1]
)

print(
    "Training matrix shape:",
    X_train_processed.shape
)

print(
    "Testing matrix shape:",
    X_test_processed.shape
)

print(
    "\nRemaining missing values in processed training data:",
    np.isnan(X_train_processed).sum()
)

print(
    "Remaining missing values in processed testing data:",
    np.isnan(X_test_processed).sum()
)

print("\n========== VALIDATION ==========")

assert X_train_processed.shape[0] == len(X_train)
assert X_test_processed.shape[0] == len(X_test)

assert X_train_processed.shape[1] == X_test_processed.shape[1]

assert np.isnan(X_train_processed).sum() == 0
assert np.isnan(X_test_processed).sum() == 0

assert set(y.unique()) == {"<=50K", ">50K"}

print("Duplicate removal: Successful")
print("Missing value handling: Successful")
print("Categorical encoding: Successful")
print("Numerical scaling: Successful")
print("Train-test split: Successful")
print("Data leakage prevention: Successful")

print("\nPreprocessing completed successfully!")
