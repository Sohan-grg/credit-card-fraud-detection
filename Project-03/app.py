
from pathlib import Path

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
    st.info(
        "Transaction input fields and fraud predictions "
        "will be added in the next commits."
    )

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
