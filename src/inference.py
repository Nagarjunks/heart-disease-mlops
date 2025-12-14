"""
inference.py

Handles model inference logic.
"""

import numpy as np


def predict(model, scaler, input_data):
    """
    Generates prediction and confidence score.

    Args:
        model: Trained ML model
        scaler: Fitted scaler
        input_data (list): Input feature values

    Returns:
        prediction, confidence
    """

    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled).max()

    return int(prediction), float(confidence)
