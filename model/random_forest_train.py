"""
ML Assignment 2 - Random Forest (Ensemble) Training Script
Machine Learning (AIML CZG565) - BITS Pilani WILP

Trains a Random Forest classifier on the UCI Bank Marketing dataset
and saves the fitted pipeline -> model/random_forest.pkl

Run from the project root:  python model/random_forest_train.py
"""

import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from common import MODEL_DIR, RANDOM_STATE, build_preprocessor, evaluate, get_train_test_split, save_metrics_row

MODEL_KEY = "random_forest"
DISPLAY_NAME = "Random Forest (Ensemble)"


def main():
    X_train, X_test, y_train, y_test = get_train_test_split()

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
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
