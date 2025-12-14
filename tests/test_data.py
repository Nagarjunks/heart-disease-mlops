"""
test_data.py

Unit tests for data preprocessing.
"""

import pandas as pd
from src.preprocess import load_and_preprocess_data


def test_preprocessing():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        "data/heart.csv"
    )

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) > 0
    assert scaler is not None
