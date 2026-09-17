import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# MLflow
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("telco_churn_experiment")


# ============================================================
# Load Data
# ============================================================

df = pd.read_csv("data/raw/telco.csv")

print("Original dataset shape:", df.shape)


# ============================================================
# Data Cleaning
# ============================================================

# Remove customerID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove missing values
df.dropna(inplace=True)

# Convert target
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# ============================================================
# Features / Target
# ============================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]


print("\nClass distribution:")
print(y.value_counts())
print(y.value_counts(normalize=True))


# ============================================================
# Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# Preprocessing
# ============================================================

categorical_cols = X.select_dtypes(
    include=["object"]
).columns

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns


preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_cols
        ),
        (
            "num",
            StandardScaler(),
            numerical_cols
        )
    ]
)


# ============================================================
# Models
# ============================================================

models = {

    "logistic_regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ============================================================
# Train and Compare Models
# ============================================================

results = []

best_model = None
best_model_name = None
best_f1 = 0


for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    with mlflow.start_run(run_name=model_name):

        # Train
        pipeline.fit(X_train, y_train)

        # Probabilities
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Default threshold
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        # MLflow parameters
        mlflow.log_param(
            "model_type",
            model_name
        )

        # MLflow metrics
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model"
        )

        # Print results
        print("Accuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall, 4))
        print("F1 Score :", round(f1, 4))
        print("ROC-AUC  :", round(roc_auc, 4))

        print("\nConfusion Matrix:")
        print(cm)

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        })

        # Select best model using F1
        if f1 > best_f1:

            best_f1 = f1
            best_model = pipeline
            best_model_name = model_name


# ============================================================
# Model Comparison
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.sort_values(
        "f1",
        ascending=False
    ).to_string(index=False)
)


# ============================================================
# Threshold Optimization
# ============================================================

print("\n")
print("=" * 60)
print("THRESHOLD OPTIMIZATION")
print("=" * 60)


best_probabilities = best_model.predict_proba(
    X_test
)[:, 1]


thresholds = np.arange(
    0.20,
    0.71,
    0.01
)


best_threshold = 0.5
threshold_best_f1 = 0


for threshold in thresholds:

    predictions = (
        best_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    if score > threshold_best_f1:

        threshold_best_f1 = score
        best_threshold = threshold


print(
    "Best Model:",
    best_model_name
)

print(
    "Best Threshold:",
    round(best_threshold, 2)
)

print(
    "Optimized F1:",
    round(threshold_best_f1, 4)
)


# ============================================================
# Final Metrics
# ============================================================

final_predictions = (
    best_probabilities >= best_threshold
).astype(int)


final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

final_precision = precision_score(
    y_test,
    final_predictions,
    zero_division=0
)

final_recall = recall_score(
    y_test,
    final_predictions,
    zero_division=0
)

final_f1 = f1_score(
    y_test,
    final_predictions,
    zero_division=0
)

final_roc_auc = roc_auc_score(
    y_test,
    best_probabilities
)

final_cm = confusion_matrix(
    y_test,
    final_predictions
)


print("\n")
print("=" * 60)
print("FINAL MODEL")
print("=" * 60)

print("Model     :", best_model_name)
print("Threshold :", round(best_threshold, 2))
print("Accuracy  :", round(final_accuracy, 4))
print("Precision :", round(final_precision, 4))
print("Recall    :", round(final_recall, 4))
print("F1 Score  :", round(final_f1, 4))
print("ROC-AUC   :", round(final_roc_auc, 4))

print("\nConfusion Matrix:")
print(final_cm)


# ============================================================
# Save Best Model
# ============================================================

joblib.dump(
    best_model,
    "models/churn_pipeline.pkl"
)

print("\nPipeline saved successfully!")