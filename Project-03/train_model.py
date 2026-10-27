
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results = {
        "model": name,
        "average_precision": average_precision_score(
            y_test, probabilities
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
    }

    for metric, value in results.items():
        print(f"{metric}: {value}")

    return results, model


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place creditcard.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    logistic_results, logistic_model = evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    forest_results, random_forest_model = evaluate_model(
        "Random Forest",
        random_forest_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    comparison = pd.DataFrame([
        logistic_results,
        forest_results,
    ])

    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    joblib.dump(
        random_forest_model,
        MODEL_DIR / "random_forest.joblib",
    )

    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print("\nComparison saved to reports/model_comparison.csv")
    print("Random Forest model saved successfully.")


if __name__ == "__main__":
    main()
