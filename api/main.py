"""
main.py

FastAPI application for serving the ML model.
"""

from fastapi import FastAPI
import joblib
import logging
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")

logging.basicConfig(level=logging.INFO)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


@app.post("/predict")
def predict_heart_disease(features: list):
    """
    Predict heart disease from input features.
    """

    logging.info(f"Received input: {features}")

    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)

    prediction = model.predict(features_scaled)[0]
    confidence = model.predict_proba(features_scaled).max()

    return {
        "prediction": int(prediction),
        "confidence": float(confidence)
    }
