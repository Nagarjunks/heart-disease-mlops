# Heart Disease Prediction – MLOps Assignment

## Overview
End-to-end MLOps pipeline for predicting heart disease using machine learning.

## Features
- Data preprocessing & feature engineering
- MLflow experiment tracking
- FastAPI-based model serving
- Dockerized deployment
- CI/CD with GitHub Actions

## How to Run
```bash
git clone https://github.com/Nagarjunks/heart-disease-mlops.git
cd heart-disease-mlops
pip install -r requirements.txt
python src/train.py
mlflow ui
