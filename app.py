"""
ML Assignment 2 - Streamlit Web Application
Machine Learning (AIML CZG565) - BITS Pilani WILP

Interactive demo for 5 classification models trained on the UCI Bank
Marketing dataset (predict whether a client subscribes to a term deposit).

Features (as required by the assignment):
  a. Dataset upload option (CSV) - upload the provided test_data.csv
  b. Model selection dropdown
  c. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  d. Confusion matrix + classification report
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")
METRICS_CSV = os.path.join(MODEL_DIR, "metrics_comparison.csv")

TARGET = "y"
POSITIVE_LABEL = "yes"

MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN (K-Nearest Neighbors)": "knn.pkl",
    "Naive Bayes (Gaussian)": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

EXPECTED_FEATURES = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
    "previous", "poutcome",
]

st.set_page_config(
    page_title="Bank Term Deposit Classifier | ML Assignment 2",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource
def load_model(filename: str):
    return joblib.load(os.path.join(MODEL_DIR, filename))


@st.cache_data
def load_comparison_table() -> pd.DataFrame:
    return pd.read_csv(METRICS_CSV)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score((y_true == POSITIVE_LABEL).astype(int), y_proba),
        "Precision": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL),
        "Recall": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL),
        "F1 Score": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve((y_true == POSITIVE_LABEL).astype(int), y_proba)
    auc = roc_auc_score((y_true == POSITIVE_LABEL).astype(int), y_proba)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})", color="#1f77b4")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------- Sidebar ------------------------------------
st.sidebar.title("🏦 Model Playground")
st.sidebar.markdown(
    "Predicting **term deposit subscription** using the "
    "[UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)."
)

selected_model_name = st.sidebar.selectbox(
    "1️⃣ Choose a classification model", list(MODELS.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "2️⃣ Upload test data (CSV)",
    type=["csv"],
    help="Upload the test_data.csv from the repository (raw features, "
    "optionally with the target column 'y' to see evaluation metrics).",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "ML Assignment 2 · AIML CZG565 · BITS Pilani WILP\n\n"
    "Models: Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest"
)

# ------------------------------ Header ------------------------------------
st.title("Bank Term Deposit Subscription Classifier")
st.markdown(
    "This app demonstrates **5 classification models** trained on the same "
    "dataset, evaluated with **6 metrics** (Accuracy, AUC, Precision, Recall, "
    "F1, MCC). Select a model, upload the test CSV, and inspect the results."
)

tab_predict, tab_compare, tab_about = st.tabs(
    ["🔮 Evaluate on Test Data", "📊 Model Comparison", "ℹ️ About Dataset"]
)

# ----------------------- Tab 1: Evaluate / Predict -------------------------
with tab_predict:
    model = load_model(MODELS[selected_model_name])
    st.subheader(f"Selected model: {selected_model_name}")

    if uploaded_file is None:
        st.info(
            "⬅️ Upload **test_data.csv** from the sidebar to evaluate this "
            "model. The file must contain the 16 raw feature columns "
            "(and optionally the target column `y`)."
        )
    else:
        data = pd.read_csv(uploaded_file)
        missing = [c for c in EXPECTED_FEATURES if c not in data.columns]
        if missing:
            st.error(f"Uploaded CSV is missing required feature columns: {missing}")
            st.stop()

        st.markdown(f"**Uploaded data:** {data.shape[0]} rows × {data.shape[1]} columns")
        with st.expander("Preview uploaded data (first 10 rows)"):
            st.dataframe(data.head(10), width="stretch")

        X_new = data[EXPECTED_FEATURES]
        predictions = model.predict(X_new)
        pos_idx = list(model.classes_).index(POSITIVE_LABEL)
        probabilities = model.predict_proba(X_new)[:, pos_idx]

        if TARGET in data.columns:
            y_true = data[TARGET]
            metrics = compute_metrics(y_true, predictions, probabilities)

            st.markdown("#### Evaluation Metrics")
            cols = st.columns(6)
            for col, (name, value) in zip(cols, metrics.items()):
                col.metric(name, f"{value:.4f}")

            left, right = st.columns(2)
            with left:
                st.pyplot(plot_confusion_matrix(y_true, predictions, list(model.classes_)))
            with right:
                st.pyplot(plot_roc_curve(y_true, probabilities, selected_model_name))

            st.markdown("#### Classification Report")
            report = classification_report(y_true, predictions, output_dict=True)
            st.dataframe(
                pd.DataFrame(report).transpose().style.format("{:.4f}"),
                width="stretch",
            )
        else:
            st.warning(
                "No target column `y` found — showing predictions only "
                "(metrics need ground-truth labels)."
            )

        st.markdown("#### Predictions")
        output = data.copy()
        output["prediction"] = predictions
        output["P(subscribe=yes)"] = probabilities.round(4)
        st.dataframe(output.head(50), width="stretch")
        st.download_button(
            "⬇️ Download predictions as CSV",
            output.to_csv(index=False).encode(),
            file_name="predictions.csv",
            mime="text/csv",
        )

# ----------------------- Tab 2: Model Comparison ---------------------------
with tab_compare:
    st.subheader("Evaluation metrics for all 5 models (hold-out test set)")
    comparison = load_comparison_table()
    st.dataframe(
        comparison.style.format(
            {c: "{:.4f}" for c in comparison.columns if c != "ML Model Name"}
        ).highlight_max(
            subset=[c for c in comparison.columns if c != "ML Model Name"],
            color="#d4f7d4",
        ),
        width="stretch",
    )

    metric_to_plot = st.selectbox(
        "Plot a metric across models",
        [c for c in comparison.columns if c != "ML Model Name"],
    )
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.barplot(
        data=comparison, x="ML Model Name", y=metric_to_plot,
        hue="ML Model Name", palette="crest", legend=False, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.set_title(f"{metric_to_plot} by model")
    plt.xticks(rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

# --------------------------- Tab 3: About ----------------------------------
with tab_about:
    st.subheader("Dataset: UCI Bank Marketing")
    st.markdown(
        """
The data relates to **direct marketing campaigns (phone calls) of a Portuguese
banking institution**. The classification goal is to predict whether a client
will **subscribe to a term deposit** (target `y`: yes/no).

- **Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Instances:** 4,521 (assignment minimum: 500 ✅)
- **Features:** 16 (assignment minimum: 12 ✅) — 7 numeric + 9 categorical
- **Problem type:** Binary classification (imbalanced ≈ 88.5% "no" / 11.5% "yes")

| Feature | Type | Description |
|---|---|---|
| age | numeric | Client age |
| job | categorical | Type of job |
| marital | categorical | Marital status |
| education | categorical | Education level |
| default | categorical | Has credit in default? |
| balance | numeric | Average yearly balance (euros) |
| housing | categorical | Has housing loan? |
| loan | categorical | Has personal loan? |
| contact | categorical | Contact communication type |
| day | numeric | Last contact day of month |
| month | categorical | Last contact month |
| duration | numeric | Last contact duration (seconds) |
| campaign | numeric | Contacts performed during this campaign |
| pdays | numeric | Days since client was last contacted (-1 = never) |
| previous | numeric | Contacts performed before this campaign |
| poutcome | categorical | Outcome of previous campaign |

**Preprocessing (bundled inside every saved model pipeline):**
one-hot encoding for categorical features + standard scaling for numeric
features, followed by the classifier. This means raw CSV rows can be scored
directly — no manual preprocessing required.
        """
    )
