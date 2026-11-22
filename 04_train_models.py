
import os
import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


# --------------------------------------------------
# STEP 1: LOAD TRAINING AND TESTING DATA
# --------------------------------------------------

train_data = pd.read_csv("data/train_data.csv")
test_data = pd.read_csv("data/test_data.csv")

X_train = train_data.drop(columns=["income"])
y_train = train_data["income"]

X_test = test_data.drop(columns=["income"])
y_test = test_data["income"]

# Load preprocessing pipeline
preprocessor = joblib.load("models/preprocessor.pkl")


# --------------------------------------------------
# STEP 2: DEFINE FIVE CLASSIFICATION MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform"
    ),

    "Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# STEP 3: TRAIN AND EVALUATE EACH MODEL
# --------------------------------------------------

results = []

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

for model_name, classifier in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    # Each model receives its own copy of the
    # preprocessing pipeline.
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier)
        ]
    )

    # Train using the same training dataset
    model_pipeline.fit(X_train, y_train)

    # Generate predictions
    y_pred = model_pipeline.predict(X_test)

    # Generate probability for positive class
    classifier_model = model_pipeline.named_steps["classifier"]

    positive_class_index = list(
        classifier_model.classes_
    ).index(">50K")

    y_probability = model_pipeline.predict_proba(
        X_test
    )[:, positive_class_index]


    # --------------------------------------------------
    # STEP 4: CALCULATE SIX EVALUATION METRICS
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        (y_test == ">50K").astype(int),
        y_probability
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=">50K",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=">50K",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=">50K",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)


    # --------------------------------------------------
    # STEP 5: STORE RESULTS
    # --------------------------------------------------

    results.append({
        "ML Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })


    # --------------------------------------------------
    # STEP 6: SAVE COMPLETE MODEL WITH COMPRESSION
    # --------------------------------------------------

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    model_path = os.path.join("models", filename)

    # Save complete pipeline including preprocessing.
    # Compression reduces disk storage requirements.
    joblib.dump(
        model_pipeline,
        model_path,
        compress=3
    )

    # Calculate saved model size
    saved_size_mb = (
        os.path.getsize(model_path) / (1024 * 1024)
    )

    print("Accuracy :", round(accuracy, 4))
    print("AUC      :", round(auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("MCC      :", round(mcc, 4))

    print("Model saved:", filename)
    print("Saved model size:", round(saved_size_mb, 2), "MB")


# --------------------------------------------------
# STEP 7: CREATE COMPARISON TABLE
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        formatters={
            metric: "{:.4f}".format
            for metric in [
                "Accuracy",
                "AUC",
                "Precision",
                "Recall",
                "F1 Score",
                "MCC"
            ]
        }
    )
)

print("\nAll five models trained successfully!")
