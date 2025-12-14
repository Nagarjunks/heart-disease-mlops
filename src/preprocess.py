"""
preprocess.py

This file contains all data preprocessing logic.
Keeping preprocessing separate ensures reproducibility
and clean MLOps practices.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(csv_path):
    """
    Loads the heart disease dataset and performs preprocessing.

    Steps:
    1. Load CSV
    2. Separate features and target
    3. Scale numerical features
    4. Split into train and test sets

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """

    # Load dataset
    df = pd.read_csv(csv_path)

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler
