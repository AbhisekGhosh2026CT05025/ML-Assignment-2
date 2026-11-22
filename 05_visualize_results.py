
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# Create output directory
os.makedirs("outputs/graphs", exist_ok=True)

# --------------------------------------------------
# GRAPH 1: ALL SIX METRICS
# --------------------------------------------------

results = pd.read_csv("outputs/model_comparison.csv")

metrics = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for ax, metric in zip(axes.flatten(), metrics):

    bars = ax.bar(
        results["ML Model"],
        results[metric]
    )

    ax.set_title(metric, fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=30)

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=9
    )

fig.suptitle(
    "Classification Model Performance Comparison",
    fontsize=18
)

plt.tight_layout()

plt.savefig(
    "outputs/graphs/model_metrics_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# GRAPH 2: CONFUSION MATRICES
# --------------------------------------------------

test_data = pd.read_csv("test_data.csv")

X_test = test_data.drop(columns=["income"])
y_test = test_data["income"]

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for ax, (name, filename) in zip(
    axes.flatten(),
    model_files.items()
):

    model = joblib.load(
        os.path.join("models", filename)
    )

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["<=50K", ">50K"]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["<=50K", ">50K"]
    )

    display.plot(
        ax=ax,
        colorbar=False,
        values_format="d"
    )

    ax.set_title(name)

# Hide unused sixth subplot
axes.flatten()[-1].axis("off")

fig.suptitle(
    "Confusion Matrix Comparison",
    fontsize=18
)

plt.tight_layout()

plt.savefig(
    "outputs/graphs/confusion_matrices.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# GRAPH 3: ROC CURVES
# --------------------------------------------------

plt.figure(figsize=(10, 8))

y_binary = (y_test == ">50K").astype(int)

for name, filename in model_files.items():

    model = joblib.load(
        os.path.join("models", filename)
    )

    probabilities = model.predict_proba(X_test)

    positive_index = list(
        model.classes_
    ).index(">50K")

    positive_probabilities = probabilities[:, positive_index]

    fpr, tpr, thresholds = roc_curve(
        y_binary,
        positive_probabilities
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend(loc="lower right")

plt.grid(alpha=0.3)

plt.savefig(
    "outputs/graphs/roc_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nAll visualizations generated successfully!")

print("\nGenerated files:")

print("1. outputs/graphs/model_metrics_comparison.png")
print("2. outputs/graphs/confusion_matrices.png")
print("3. outputs/graphs/roc_comparison.png")
