"""
ML Assignment 2 - Naive Bayes Training Script
Machine Learning (AIML CZG565) - BITS Pilani WILP

Trains a Gaussian Naive Bayes classifier on the UCI Bank Marketing dataset
and saves the fitted pipeline -> model/naive_bayes.pkl

Run from the project root:  python model/naive_bayes_train.py
"""

import os

import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from common import MODEL_DIR, build_preprocessor, evaluate, get_train_test_split, save_metrics_row

MODEL_KEY = "naive_bayes"
DISPLAY_NAME = "Naive Bayes (Gaussian)"


def main():
    X_train, X_test, y_train, y_test = get_train_test_split()

    clf = GaussianNB()
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
