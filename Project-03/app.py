
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
    "Navigation",
    [
        "Dashboard",
        "Fraud Detection",
        "Model Performance",
        "About"
    ]
)

# Main heading
st.title("💳 Credit Card Fraud Detection")

try:
    model = load_model()
    st.sidebar.success("Trained model loaded successfully")
except Exception as error:
    model = None
    st.sidebar.error(f"Model loading failed: {error}")

# Store prediction history during the current session
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Dashboard page

if page == "Dashboard":
    st.header("Transaction Overview")

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)

        total_transactions = len(df)
        fraud_count = int((df["Class"] == 1).sum())
        legitimate_count = int((df["Class"] == 0).sum())

        fraud_rate = (
            fraud_count / total_transactions * 100
            if total_transactions > 0
            else 0
        )

        # Dashboard metrics
        st.subheader("Transaction Statistics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )

        col2.metric(
            "Detected Fraud Cases",
            f"{fraud_count:,}"
        )

        col3.metric(
            "Legitimate Transactions",
            f"{legitimate_count:,}"
        )

        col4.metric(
            "Fraud Rate",
            f"{fraud_rate:.3f}%"
        )

        # Fraud distribution
        st.subheader("Fraud Distribution")

        distribution = pd.DataFrame({
            "Transaction Type": [
                "Legitimate",
                "Fraudulent",
            ],
            "Count": [
                legitimate_count,
                fraud_count,
            ],
        })

        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(
                distribution.set_index("Transaction Type")
            )

        with col2:
            st.write("Transaction Class Distribution")
            st.dataframe(
                distribution,
                use_container_width=True,
                hide_index=True,
            )
        # Transaction amount analysis
        st.subheader("Transaction Amount Analysis")

        # Transaction class filter
        selected_classes = st.multiselect(
            "Filter by Transaction Type",
            options=[0, 1],
            default=[0, 1],
            format_func=lambda x: "Legitimate" if x == 0 else "Fraudulent"
        )

        # Amount range filter
        min_amount = float(df["Amount"].min())
        max_amount = float(df["Amount"].max())

        amount_range = st.slider(
            "Select Transaction Amount Range",
            min_value=min_amount,
            max_value=max_amount,
            value=(min_amount, max_amount)
        )

        # Apply filters
        filtered_df = df[
            (df["Class"].isin(selected_classes)) &
            (df["Amount"] >= amount_range[0]) &
            (df["Amount"] <= amount_range[1])
        ]

        st.write(f"Filtered Transactions: {len(filtered_df):,}")

        # Transaction amount chart
        st.subheader("Transaction Amount Distribution")

        st.histogram if False else None

        st.bar_chart(
            filtered_df["Amount"].value_counts().sort_index().head(30)
        )

        # Filtered records
        st.subheader("Filtered Transaction Records")

        st.dataframe(
            filtered_df.head(100),
            use_container_width=True
        )

        # Dataset preview
        st.subheader("Recent Dataset Records")

        st.dataframe(
            df.head(10),
            use_container_width=True,
        )

    else:
        st.warning("Dashboard dataset is not available.")

        st.info(
            "The original credit card dataset is not included "
            "in the GitHub repository. Therefore, transaction "
            "statistics and amount analysis are unavailable "
            "in the deployed version."
        )

        st.success(
            "You can still use Fraud Detection, "
            "Model Performance, and About pages."
        )


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

               # Save this prediction in the current session
                report_row = {
                    **values,
                    "fraud_probability": fraud_probability,
                    "prediction": prediction,
                    "risk_category": risk_category,
                    "prediction_time": str(pd.Timestamp.now()),
                }
                st.session_state.prediction_history.append(report_row)

                # Allow the user to download all predictions from this session
                report_df = pd.DataFrame(
                    st.session_state.prediction_history
                )

                st.download_button(
                    label="Download Prediction Report (CSV)",
                    data=report_df.to_csv(index=False),
                    file_name="fraud_prediction_report.csv",
                    mime="text/csv",
                    key="download_prediction_report",
                ) 

                # Show important model features
                st.subheader("Important Model Features")

                classifier = model

                if hasattr(model, "named_steps"):
                    classifier = model.named_steps.get(
                        "classifier", model
                    )

                if hasattr(classifier, "feature_importances_"):
                    importance = pd.DataFrame({
                        "Feature": feature_order,
                        "Importance": classifier.feature_importances_,
                    })

                    importance = importance.sort_values(
                        "Importance",
                        ascending=False,
                    ).head(10)

                    st.write(
                        "These are the 10 most important features "
                        "across the trained Random Forest model."
                    )

                    st.bar_chart(
                        importance.set_index("Feature")["Importance"]
                    )

                    st.dataframe(
                        importance.reset_index(drop=True),
                        use_container_width=True,
                    )
                else:
                    st.info(
                        "Feature importance is unavailable "
                        "for the loaded model."
                    )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

elif page == "Model Performance":
    st.header("Model Performance Dashboard")

    report_path = BASE_DIR / "reports" / "model_comparison.csv"

    if report_path.exists():
        results = pd.read_csv(report_path)

        st.subheader("Model Comparison")
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Performance Metrics")

        metric = st.selectbox(
            "Select Performance Metric",
            [
                "average_precision",
                "roc_auc",
                "precision",
                "recall",
                "f1_score"
            ]
        )

        st.bar_chart(
            results.set_index("model")[metric]
        )

        st.subheader("Best Performing Model")

        best_model = results.loc[
            results["f1_score"].idxmax()
        ]

        st.success(
            f"Best Model: {best_model['model']}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Precision",
            f"{best_model['precision']:.2%}"
        )

        col2.metric(
            "Recall",
            f"{best_model['recall']:.2%}"
        )

        col3.metric(
            "F1 Score",
            f"{best_model['f1_score']:.2%}"
        )

    else:
        st.error("Model comparison report not found.")

# About page
elif page == "About":
    st.header("About This Project")

    st.write(
        "This project uses machine learning to identify potentially "
        "fraudulent credit card transactions."
    )

    st.subheader("Models Used")
    st.write(
        "The project evaluates Logistic Regression and Random Forest. "
        "Random Forest is used for transaction predictions."
    )

    st.subheader("Model Limitations")

    st.markdown("""
    1. **Class Imbalance:** The dataset contains significantly fewer
       fraudulent transactions than legitimate transactions.

    2. **Anonymized Features:** Features V1 to V28 are PCA-transformed
       and do not have directly understandable meanings.

    3. **False Positives:** Legitimate transactions may sometimes
       be incorrectly classified as fraudulent.

    4. **False Negatives:** Some fraudulent transactions may be
       incorrectly classified as legitimate.

    5. **Prediction Uncertainty:** Fraud probabilities are estimates
       and do not guarantee that a transaction is fraudulent.

    6. **Dataset Limitations:** The model was trained using a specific
       dataset and may not perform equally well on real-world
       transactions.
    """)

    st.warning(
        "This application is an educational machine learning project. "
        "Its predictions should not be treated as a final financial "
        "decision."
    )
