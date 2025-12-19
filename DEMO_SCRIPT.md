# MLOps Project Demo Guide: Heart Disease Prediction

This guide outlines the steps to demonstrate the end-to-end MLOps pipeline for the Heart Disease Prediction project. Use this as a script or checklist when recording your explanation video.

## 1. Introduction
*   **Goal**: Predict heart disease risk using a Machine Learning classifier deployed as a scalable API.
*   **Tech Stack**: 
    *   **Model**: Scikit-Learn (Logistic Regression, Random Forest)
    *   **Tracking**: MLflow
    *   **API**: FastAPI
    *   **Containerization**: Docker
    *   **Orchestration**: Kubernetes
    *   **CI/CD**: GitHub Actions

## 2. Setup & Installation
*   **Objective**: Show a clean environment setup.
*   **Steps**:
    1.  Open your terminal in the project root (`D:\BITS\sem3\mlops\heart-disease-mlops`).
    2.  Explain the folder structure briefly (`src/` for code, `api/` for the app, `tests/` for validation, `k8s/` for deployment).
    3.  (Optional) Create a fresh virtual environment:
        ```bash
        python -m venv .venv
        .\.venv\Scripts\activate
        ```
    4.  Install dependencies:
        ```bash
        pip install -r requirements.txt
        pip install -e .
        ```

## 3. Data Acquisition & EDA
*   **Objective**: Show understanding of the data.
*   **Steps**:
    1.  Open the data file `data/heart.csv` to show the raw data.
    2.  Launch Jupyter Notebook:
        ```bash
        jupyter notebook notebooks/eda.ipynb
        ```
    3.  Walk through the EDA notebook:
        *   **Missing Values**: None found.
        *   **Class Balance**: Show the target distribution plot.
        *   **Correlations**: Show the heatmap.
    4.  Show saved figures in `reports/figures/`.

## 4. Feature Engineering & Model Training
*   **Objective**: Demonstrate the training pipeline and experiment tracking.
*   **Steps**:
    1.  Show `src/preprocess.py`: Explain `ColumnTransformer`, `StandardScaler`, and `OneHotEncoder`.
    2.  Show `src/train.py`: Explain how it trains multiple models and logs to MLflow.
    3.  **Run Training**:
        ```bash
        python -m src.train
        ```
    4.  **View Results in MLflow**:
        ```bash
        mlflow ui
        ```
        *   Open `http://localhost:5000` in your browser.
        *   Show the "Heart Disease Prediction" experiment.
        *   Compare runs (Logistic Regression vs Random Forest) based on Accuracy/ROC-AUC.
        *   Show logged artifacts (model, preprocessor).

    > **CRITICAL STEP**: 
    > After training, MLflow creates a new run ID. 
    > 1. Copy the `run_id` or the full artifact path of the best model from the MLflow UI (e.g., `mlruns/1/<RUN_ID>/artifacts/LogisticRegression`).
    > 2. Open `api/main.py` and update the `model_path` variable with this new path to ensure the API uses the latest trained model.

## 5. Automated Testing
*   **Objective**: Verify code quality.
*   **Steps**:
    1.  Run the test suite:
        ```bash
        pytest
        ```
    2.  Show that tests for data processing and training pass.

## 6. Model Containerization (Docker)
*   **Objective**: Show how the model is packaged for production.
*   **Steps**:
    1.  Show `Dockerfile`: Explain the base image (`python:3.13-slim`), dependency installation, and startup command.
    2.  **Build Image**:
        ```bash
        docker build -t heart-disease-api .
        ```
    3.  **Run Container**:
        ```bash
        docker run -p 8000:8000 heart-disease-api
        ```
    4.  **Test API**:
        *   Open `http://localhost:8000/docs` (Swagger UI).
        *   Use the "Try it out" feature on the `/predict` endpoint with sample data:
            ```json
            {
              "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
              "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
              "ca": 0, "thal": 1
            }
            ```
        *   Show the response (Prediction, Confidence).

## 7. CI/CD Pipeline
*   **Objective**: Explain automation.
*   **Steps**:
    1.  Open `.github/workflows/github-actions.yml`.
    2.  Explain the stages:
        *   **Linting**: `ruff check .`
        *   **Testing**: `pytest`
        *   **Training**: `python -m src.train`
    3.  (If applicable) Show the "Actions" tab in your GitHub repository to show a successful run.

## 8. Production Deployment (Kubernetes)
*   **Objective**: Deploy to a scalable cluster.
*   **Steps**:
    1.  Show `k8s/deployment.yaml`: Replicas, container image, ports.
    2.  Show `k8s/service.yaml`: LoadBalancer type.
    3.  **Deploy**:
        ```bash
        kubectl apply -f k8s/
        ```
    4.  **Verify**:
        ```bash
        kubectl get deployments
        kubectl get services
        ```
    5.  Explain that the `EXTERNAL-IP` (or `localhost` if using Docker Desktop) is the entry point for users.

## 9. Monitoring & Logging
*   **Objective**: Show observability.
*   **Steps**:
    1.  Show the terminal where the Docker container or Uvicorn is running.
    2.  Point out the logs generated for each request (Info/Error levels).
    3.  Mention that in a real production environment, these would be shipped to Prometheus/Grafana.

## 10. Conclusion
*   Summarize that the project fulfills all assignment requirements:
    *   Reproducible pipeline.
    *   Experiment tracking.
    *   Containerized API.
    *   Automated CI/CD.
    *   Scalable Kubernetes deployment.
