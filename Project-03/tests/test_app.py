
import numpy as np
import pandas as pd
import pytest


FEATURE_ORDER = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


def make_transaction(time_value=406.0, amount=100.0):
    values = {
        "Time": time_value,
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "Amount": amount,
    }

    return pd.DataFrame([values], columns=FEATURE_ORDER)


def validate_transaction(transaction):
    if not np.isfinite(transaction.to_numpy()).all():
        raise ValueError("All values must be finite numbers.")

    if transaction["Time"].iloc[0] < 0:
        raise ValueError("Time cannot be negative.")

    if transaction["Amount"].iloc[0] < 0:
        raise ValueError("Amount cannot be negative.")


def test_transaction_has_30_features():
    transaction = make_transaction()

    assert transaction.shape == (1, 30)
    assert list(transaction.columns) == FEATURE_ORDER


def test_valid_transaction_passes():
    transaction = make_transaction()
    validate_transaction(transaction)


def test_negative_time_is_rejected():
    transaction = make_transaction(time_value=-1)

    with pytest.raises(ValueError, match="Time cannot be negative"):
        validate_transaction(transaction)


def test_negative_amount_is_rejected():
    transaction = make_transaction(amount=-1)

    with pytest.raises(ValueError, match="Amount cannot be negative"):
        validate_transaction(transaction)


def test_infinite_value_is_rejected():
    transaction = make_transaction()
    transaction.loc[0, "V1"] = np.inf

    with pytest.raises(ValueError, match="finite numbers"):
        validate_transaction(transaction)
