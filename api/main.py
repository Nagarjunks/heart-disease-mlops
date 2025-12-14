"""
main.py

FastAPI application for serving the heart disease ML model.
Uses Pydantic for request validation (production best practice).
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")

logging.basicConfig(level=logging.INFO)

# Load model and scaler
model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")


# Request schema
class PredictionRequest(BaseModel):
    features: list


@app.post("/predict")
def predict_heart_disease(request: PredictionRequest):
    """
    Predict heart disease from input features.
    """

    logging.info(f"Received input: {request.features}")

    features_array = np.array(request.features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)

    prediction = model.predict(features_scaled)[0]
    confidence = model.predict_proba(features_scaled).max()

    return {
        "prediction": int(prediction),
        "confidence": float(confidence)
    }
