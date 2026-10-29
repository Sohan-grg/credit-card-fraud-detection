
from pathlib import Path

import numpy as np
import streamlit as st
import joblib
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"

# Streamlit page settings
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# Load the trained model
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Dashboard", "Fraud Detection", "About"],
)

# Main heading
st.title("💳 Credit Card Fraud Detection")

try:
    model = load_model()
    st.sidebar.success("Trained model loaded successfully")
except Exception as error:
    model = None
    st.sidebar.error(f"Model loading failed: {error}")

# Dashboard page
if page == "Dashboard":
    st.header("Transaction Overview")

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)

        total = len(df)
        fraud = int((df["Class"] == 1).sum())
        legitimate = int((df["Class"] == 0).sum())
        fraud_rate = fraud / total * 100 if total else 0

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Transactions", f"{total:,}")
        col2.metric("Fraud Cases", f"{fraud:,}")
        col3.metric("Legitimate Transactions", f"{legitimate:,}")
        col4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("Dataset not found in the data folder.")

# Fraud detection page


elif page == "Fraud Detection":
    st.header("Fraud Detection")
    st.write("Enter transaction details to estimate fraud risk.")

    with st.form("transaction_form"):
        st.subheader("Transaction Details")

        time_value = st.number_input(
            "Time (seconds since first transaction)",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0,
            step=1.0,
        )

        st.subheader("Anonymized Features (V1–V28)")
        st.caption(
            "Use values from a dataset row for meaningful testing."
        )

        features = {}
        columns = st.columns(2)

        for i in range(1, 29):
            with columns[(i - 1) % 2]:
                features[f"V{i}"] = st.number_input(
                    f"V{i}",
                    value=0.0,
                    format="%.6f",
                    key=f"feature_V{i}",
                )

        submitted = st.form_submit_button("Predict Fraud Risk")

    if submitted:
        feature_order = (
            ["Time"]
            + [f"V{i}" for i in range(1, 29)]
            + ["Amount"]
        )

        values = {
            "Time": time_value,
            **features,
            "Amount": amount,
        }

        transaction = pd.DataFrame(
            [values],
            columns=feature_order,
        )

        if not np.isfinite(transaction.to_numpy()).all():
            st.error("Please enter valid, finite numerical values.")

        elif time_value < 0 or amount < 0:
            st.error("Time and transaction amount cannot be negative.")

        elif model is None:
            st.error(
                "The trained model could not be loaded. "
                "Check the model file and try again."
            )

        else:
            try:
                fraud_probability = float(
                    model.predict_proba(transaction)[0][1]
                )

                prediction = int(
                    fraud_probability >= 0.5
                )

                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Estimated Fraud Probability",
                    f"{fraud_probability * 100:.2f}%",
                )

                if fraud_probability < 0.20:
                    risk_category = "Low Risk"
                elif fraud_probability < 0.50:
                    risk_category = "Medium Risk"
                else:
                    risk_category = "High Risk"

                col2.metric("Fraud Risk Category", risk_category)

                st.progress(fraud_probability)

                if prediction == 1:
                    st.error(
                        "Model prediction: Potentially fraudulent transaction"
                    )
                else:
                    st.success(
                        "Model prediction: Likely legitimate transaction"
                    )

                st.caption(
                    "Risk categories are project-defined thresholds. "
                    "Model probabilities are estimates, not guarantees "
                    "of actual fraud."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# About page
elif page == "About":
    st.header("About This Project")
    st.write(
        "This project explores machine learning methods "
        "for identifying potentially fraudulent credit card transactions."
    )
    st.write("Models evaluated: Logistic Regression and Random Forest.")
    st.warning(
        "Predictions are estimates, not proof of fraudulent activity."
    )
