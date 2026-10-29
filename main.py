
# CardioSense AI
# Step 1: Load and inspect the UCI Arrhythmia dataset

import os
import pandas as pd

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.metrics import ConfusionMatrixDisplay

# Locate the data folder next to this Python file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

print("Data folder:", DATA_DIR)
print("Files found:", os.listdir(DATA_DIR))

# Find the dataset file
data_file = os.path.join(DATA_DIR, "arrhythmia.data")

# Load the data
df = pd.read_csv(
    data_file,
    header=None,
    na_values=["?"],
    skipinitialspace=True
)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

# Display first five records
print("\nFirst five records:")
print(df.head())

# Check missing values
print("\nTotal missing values:", df.isnull().sum().sum())

# The final column contains the original class label
print("\nOriginal class distribution:")
print(df.iloc[:, -1].value_counts().sort_index())

# Inspect the target column
print("\nLast column name/index:", df.columns[-1])
print("Unique class labels:", sorted(df.iloc[:, -1].dropna().unique()))




# STEP 2: Prepare data for machine learning

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Remove rows where the target class is missing
df = df.dropna(subset=[df.columns[-1]]).copy()

# Convert original class labels to numeric values
df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1])

# Target: class 1 = Normal; classes 2-16 = Arrhythmia
y = (df.iloc[:, -1] != 1).astype(int)

# Features: all columns except the last target column
X = df.iloc[:, :-1].copy()

# Replace infinite values with missing values
X = X.replace([np.inf, -np.inf], np.nan)

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

print("\nFeatures:", X.shape[1])
print("Records available:", len(X))
print("\nBinary target distribution:")
print(y.value_counts().sort_index().rename({
    0: "Normal",
    1: "Arrhythmia"
}))

print("\nNumeric features:", len(numeric_cols))
print("Categorical features:", len(categorical_cols))
print("\nPreprocessing setup complete.")







# STEP 3: Train and compare five ML algorithms

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt

# Five algorithms
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000, class_weight="balanced",
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=42
    ),
    "SVM": SVC(
        probability=True, class_weight="balanced",
        random_state=42
    )
}

# Test fractions for each experiment
splits = {
    "70:30": 0.30,
    "75:25": 0.25,
    "80:20": 0.20
}

results = []
confusion_matrices = {}

for split_name, test_size in splits.items():

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # Fit imputation and scaling using training data only
    preprocessing = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    X_train_ready = preprocessing.fit_transform(X_train)
    X_test_ready = preprocessing.transform(X_test)

    print("\nTraining split:", split_name)

    for model_name, model in models.items():

        # Train
        model.fit(X_train_ready, y_train)

        # Predict
        predictions = model.predict(X_test_ready)
        probabilities = model.predict_proba(X_test_ready)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        auc = roc_auc_score(y_test, probabilities)
        loss = log_loss(y_test, probabilities, labels=[0, 1])

        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        confusion_matrices[(split_name, model_name)] = cm

        results.append({
            "Split": split_name,
            "Algorithm": model_name,
            "Accuracy (%)": round(accuracy * 100, 2),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "Log Loss": round(loss, 4)
        })

        print(
            model_name,
            "| Accuracy:", round(accuracy * 100, 2), "%",
            "| F1:", round(f1, 4)
        )

# Create a results table
results_df = pd.DataFrame(results)

print("\n\nCOMPLETE MODEL COMPARISON")
print(results_df.to_string(index=False))

# Save all results
results_df.to_csv(
    os.path.join(BASE_DIR, "all_model_metrics.csv"),
    index=False
)

# Create the accuracy comparison table
accuracy_table = results_df.pivot(
    index="Algorithm",
    columns="Split",
    values="Accuracy (%)"
)

accuracy_table = accuracy_table[["70:30", "75:25", "80:20"]]

print("\nACCURACY COMPARISON TABLE")
print(accuracy_table)

accuracy_table.to_csv(
    os.path.join(BASE_DIR, "accuracy_comparison.csv")
)

# Plot accuracy comparison
accuracy_table.plot(kind="bar", figsize=(10, 6))
plt.title("CardioSense AI - Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.xlabel("ML Algorithm")
plt.ylim(0, 100)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, "accuracy_comparison.png")
)
plt.show()

# Display confusion matrices for the 70:30 experiment
for model_name in models:
    cm = confusion_matrices[("70:30", model_name)]

    print("\n70:30 Confusion Matrix:", model_name)
    print(cm)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Arrhythmia"]
    ).plot()

    plt.title(model_name + " - 70:30")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            BASE_DIR,
            model_name.lower().replace(" ", "_")
            + "_confusion_matrix.png"
        )
    )
    plt.show()

print("\nCardioSense AI experiments completed.")
print("Results saved in:", BASE_DIR)

