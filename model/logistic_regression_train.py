"""
ML Assignment 2 - Logistic Regression Training Script
Machine Learning (AIML CZG565) - BITS Pilani WILP

Trains a Logistic Regression classifier on the UCI Bank Marketing dataset
and saves the fitted pipeline -> model/logistic_regression.pkl

Run from the project root:  python model/logistic_regression_train.py
"""

import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import MODEL_DIR, RANDOM_STATE, build_preprocessor, evaluate, get_train_test_split, save_metrics_row

MODEL_KEY = "logistic_regression"
DISPLAY_NAME = "Logistic Regression"


def main():
    X_train, X_test, y_train, y_test = get_train_test_split()

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    pipeline = Pipeline([("preprocessor", build_preprocessor(X_train)), ("classifier", clf)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    pos_idx = list(pipeline.classes_).index("yes")
    y_proba = pipeline.predict_proba(X_test)[:, pos_idx]

    row = evaluate(DISPLAY_NAME, y_test, y_pred, y_proba)
    print(f"{DISPLAY_NAME}: {row}")

    path = os.path.join(MODEL_DIR, f"{MODEL_KEY}.pkl")
    joblib.dump(pipeline, path, compress=3)
    print(f"  saved -> {path}")

    save_metrics_row(row)


if __name__ == "__main__":
    main()
