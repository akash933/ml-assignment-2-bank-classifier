"""
ML Assignment 2 - Model Training Script
Machine Learning (AIML CZG565) - BITS Pilani WILP

Trains 5 classification models on the UCI Bank Marketing dataset
(predicting whether a client subscribes to a term deposit) and saves:
  - one fitted sklearn Pipeline per model  -> model/<model_key>.pkl
  - hold-out test split (raw features + y) -> test_data.csv
  - metrics comparison table               -> model/metrics_comparison.csv

Each saved pipeline bundles the preprocessing (one-hot encoding of
categorical columns + standard scaling of numeric columns) together with
the classifier, so the Streamlit app can score raw CSV rows directly.

Run from the project root:  python model/train_models.py
"""

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
RAW_DATA = os.path.join(MODEL_DIR, "bank_raw.csv")
TEST_DATA = os.path.join(PROJECT_ROOT, "test_data.csv")
METRICS_CSV = os.path.join(MODEL_DIR, "metrics_comparison.csv")

TARGET = "y"
POSITIVE_LABEL = "yes"
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA, sep=";")
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Class balance:\n{df[TARGET].value_counts()}")
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    print(f"Categorical features ({len(categorical)}): {categorical}")
    print(f"Numeric features ({len(numeric)}): {numeric}")
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )


def build_models() -> dict:
    return {
        "logistic_regression": (
            "Logistic Regression",
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "decision_tree": (
            "Decision Tree",
            DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE
            ),
        ),
        "knn": (
            "kNN",
            KNeighborsClassifier(n_neighbors=15, weights="distance"),
        ),
        "naive_bayes": (
            "Naive Bayes (Gaussian)",
            GaussianNB(),
        ),
        "random_forest": (
            "Random Forest (Ensemble)",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    }


def evaluate(name: str, y_true, y_pred, y_proba) -> dict:
    return {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score((y_true == POSITIVE_LABEL).astype(int), y_proba), 4),
        "Precision": round(precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL), 4),
        "Recall": round(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL), 4),
        "F1": round(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def main():
    df = load_dataset()
    X, y = df.drop(columns=[TARGET]), df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # Save the raw hold-out split so it can be uploaded to the Streamlit app
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(TEST_DATA, index=False)
    print(f"Saved hold-out test data -> {TEST_DATA}")

    results = []
    for key, (display_name, clf) in build_models().items():
        pipeline = Pipeline([("preprocessor", build_preprocessor(X_train)), ("classifier", clf)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        pos_idx = list(pipeline.classes_).index(POSITIVE_LABEL)
        y_proba = pipeline.predict_proba(X_test)[:, pos_idx]

        row = evaluate(display_name, y_test, y_pred, y_proba)
        results.append(row)
        print(f"{display_name}: {row}")

        path = os.path.join(MODEL_DIR, f"{key}.pkl")
        joblib.dump(pipeline, path, compress=3)
        print(f"  saved -> {path}")

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(f"\nSaved metrics comparison -> {METRICS_CSV}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
