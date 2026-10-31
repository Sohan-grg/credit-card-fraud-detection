from pathlib import Path

import joblib
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"

FEATURE_ORDER = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("Trained model not found. Train the model first.")

    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def sample_transaction():
    if not DATA_PATH.exists():
        pytest.skip("Dataset not found.")

    df = pd.read_csv(DATA_PATH)

    return df[FEATURE_ORDER].iloc[[0]]


def test_model_loads_successfully(model):
    assert model is not None


def test_dataset_contains_required_features():
    if not DATA_PATH.exists():
        pytest.skip("Dataset not found.")

    df = pd.read_csv(DATA_PATH)

    assert all(feature in df.columns for feature in FEATURE_ORDER)
    assert "Class" in df.columns


def test_model_predicts_transaction(model, sample_transaction):
    prediction = model.predict(sample_transaction)

    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]


def test_fraud_probability_is_valid(model, sample_transaction):
    probabilities = model.predict_proba(sample_transaction)

    fraud_probability = probabilities[0][1]

    assert 0 <= fraud_probability <= 1