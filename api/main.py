from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import mlflow
import os

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the input data model using Pydantic
class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Initialize the FastAPI app
app = FastAPI(title="Heart Disease Prediction API")

# Load the preprocessor and model
try:
    preprocessor_path = "models/preprocessor.joblib"
    preprocessor = joblib.load(preprocessor_path)
    
    # Path to the logged model
    model_path = "mlruns/2/models/m-d588745fcc0b4960897a74ec63ecd840/artifacts/model.pkl"
    model = joblib.load(model_path)

except FileNotFoundError as e:
    print(f"Error loading model or preprocessor: {e}")
    # In a real application, you might want to handle this more gracefully
    preprocessor = None
    model = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease Prediction API. Use the /predict endpoint to make predictions."}



@app.post("/predict")

def predict(data: HeartDiseaseInput):

    """

    Predict heart disease based on input data.

    """

    logger.info(f"Received prediction request with data: {data.dict()}")



    if not model or not preprocessor:

        logger.error("Model or preprocessor not loaded. Returning error.")

        return {"error": "Model or preprocessor not loaded. Please check the server logs."}



    # Convert input data to a DataFrame

    input_df = pd.DataFrame([data.dict()])

    

    # Preprocess the input data

    try:

        processed_input = preprocessor.transform(input_df)

    except Exception as e:

        logger.error(f"Error during data preprocessing: {e}")

        return {"error": f"Error during data preprocessing: {e}"}

    

    # Make prediction

    try:

        prediction = model.predict(processed_input)

        prediction_proba = model.predict_proba(processed_input)

        

        # The output is a numpy array, get the first element

        prediction_result = int(prediction[0])

        confidence = float(prediction_proba[0][prediction_result])

        

        response = {

            "prediction": "Heart Disease" if prediction_result == 1 else "No Heart Disease",

            "prediction_label": prediction_result,

            "confidence": confidence

        }

        logger.info(f"Prediction successful: {response}")

        return response

    except Exception as e:

        logger.error(f"Error during prediction: {e}")

        return {"error": f"Error during prediction: {e}"}



if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
