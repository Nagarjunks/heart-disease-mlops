# Heart Disease Prediction – MLOps Assignment

## 1. Project Overview

This project implements an end-to-end MLOps pipeline for predicting heart disease based on patient health data. The solution adheres to modern MLOps best practices, including automation, experiment tracking, CI/CD, containerization, and API deployment.

**Objective:** To design, develop, and deploy a scalable and reproducible machine learning solution utilizing modern MLOps best practices.

**Dataset:** Heart Disease UCI Dataset, sourced from the UCI Machine Learning Repository. It contains 14+ features (age, sex, blood pressure, cholesterol, etc.) and a binary target (presence/absence of heart disease).

## 2. Features

-   **Data Acquisition & EDA:** Automated data loading and comprehensive Exploratory Data Analysis.
-   **Feature Engineering:** Robust preprocessing pipeline for numerical scaling and categorical encoding.
-   **Model Development:** Training and evaluation of multiple classification models (Logistic Regression, Random Forest).
-   **Experiment Tracking:** Integration with MLflow for logging parameters, metrics, artifacts, and models.
-   **Model Packaging:** Models and preprocessing pipelines are saved for reproducibility.
-   **Automated Testing:** Unit tests for data preprocessing and model training.
-   **CI/CD Pipeline:** GitHub Actions workflow for linting, testing, and model training.
-   **Model Containerization:** FastAPI-based prediction API packaged as a Docker image.
-   **Production Deployment:** Kubernetes manifests for deploying the Dockerized API.
-   **Monitoring & Logging:** Basic request logging for the prediction API.

## 3. Setup & Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPO_LINK_HERE>
    cd heart-disease-mlops
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate # On Windows
    source .venv/bin/activate # On Linux/macOS
    ```

3.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e . # Install project in editable mode
    ```

## 4. Data Acquisition & EDA Summary

The `data/heart.csv` dataset was used directly.

**Key EDA Insights:**
-   The dataset is clean with no missing values, as confirmed by `df.isnull().sum().sum() == 0`.
-   The target variable exhibits a moderate class balance, which was visualized using a histogram.
-   Feature distributions were analyzed using histograms (`df.hist()`), revealing varying spreads and potential outliers.
-   A correlation heatmap (`sns.heatmap(df.corr())`) identified noticeable correlations between certain features (e.g., age, cholesterol, chest pain type) and the presence of heart disease.
-   Feature scaling is required due to varying ranges among numerical features.

EDA plots are saved in `reports/figures/`.

## 5. Model Development & Evaluation

-   **Preprocessing:** The `src/preprocess.py` script implements a `ColumnTransformer` to apply `StandardScaler` to numerical features and `OneHotEncoder` to categorical features.
-   **Models Trained:** Logistic Regression and Random Forest Classifiers were trained.
-   **Evaluation Metrics:** Models were evaluated using accuracy, precision, recall, and ROC-AUC.
-   **Model Selection:** Logistic Regression showed slightly better performance and was chosen for deployment.

## 6. Experiment Tracking (MLflow)

MLflow was integrated to track experiments. The `src/train.py` script:
-   Initializes an MLflow experiment named "Heart Disease Prediction".
-   Logs parameters (e.g., `model_type`).
-   Logs metrics (accuracy, precision, recall, ROC-AUC) for each model.
-   Logs the trained models as artifacts using `mlflow.sklearn.log_model`.
-   The fitted `preprocessor` pipeline is saved as `models/preprocessor.joblib` for reproducibility.

To view MLflow UI:
```bash
mlflow ui
```

## 7. API & Containerization

-   **API Framework:** FastAPI (`api/main.py`) provides a `/predict` endpoint.
-   **Input:** Accepts a JSON payload conforming to the `HeartDiseaseInput` Pydantic model.
-   **Output:** Returns a JSON response with the prediction ("Heart Disease" or "No Heart Disease"), prediction label (0 or 1), and confidence score.
-   **Model Loading:** The API loads the preprocessor (`models/preprocessor.joblib`) and the Logistic Regression model (via `joblib.load` from MLflow artifact path).
-   **Dockerfile:** A `Dockerfile` is provided to containerize the FastAPI application. It creates a `python:3.13-slim` image, installs dependencies, copies necessary code, and exposes port `8000`.

**Building the Docker Image:**
```bash
docker build -t heart-disease-api .
```

**Running the Docker Container Locally:**
```bash
docker run -p 8000:8000 heart-disease-api
```
(After running, access API docs at `http://localhost:8000/docs`)

## 8. CI/CD Pipeline (GitHub Actions)

A GitHub Actions workflow (`.github/workflows/github-actions.yml`) is configured for continuous integration on `push` and `pull_request` events to the `main` branch. The pipeline includes:
-   **Checkout code:** Fetches the repository content.
-   **Setup Python:** Configures Python 3.13.
-   **Cache pip dependencies:** Speeds up installation.
-   **Install dependencies:** Installs `requirements.txt` and the project in editable mode.
-   **Linting:** Runs `ruff check .` for code quality.
-   **Run unit tests:** Executes `pytest tests/`.
-   **Train model:** Runs `python -m src.train` to ensure the training process is functional and logs new experiments to MLflow.

**CI/CD Workflow Screenshot:**
![CI/CD Workflow Screenshot Placeholder](path/to/ci-cd-workflow-screenshot.png)

## 9. Deployment (Kubernetes)

Kubernetes manifests are provided in the `k8s/` directory for deploying the FastAPI application:
-   `deployment.yaml`: Defines a Kubernetes Deployment for the `heart-disease-api` Docker image with 2 replicas.
-   `service.yaml`: Defines a Kubernetes Service of type `LoadBalancer` to expose the API externally on port 80, routing traffic to container port 8000.

**Deploying to Kubernetes:**
(Assuming `kubectl` is configured for your cluster, e.g., Minikube/Docker Desktop Kubernetes)
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

**Verifying Deployment:**
```bash
kubectl get deployments
kubectl get services
# To get the external IP for the LoadBalancer service
kubectl get services heart-disease-api-service
```
Once the external IP is available, the API can be accessed at `http://<EXTERNAL-IP>/docs`.

**Deployment Screenshots:**
![Kubernetes Deployment Screenshot Placeholder](path/to/kubernetes-deployment-screenshot.png)

## 10. Monitoring & Logging

-   **API Logging:** The FastAPI application (`api/main.py`) integrates Python's `logging` module to record incoming requests, prediction results, and any errors. Logs are output to `stdout`, which can be collected by container orchestration systems.
-   **Monitoring:** For a full monitoring solution (e.g., Prometheus and Grafana), additional configuration would be required within the Kubernetes cluster to scrape metrics from the API and visualize them. This would involve:
    -   Instrumenting the FastAPI app with a Prometheus client library to expose custom metrics.
    -   Deploying Prometheus to collect these metrics.
    -   Deploying Grafana to create dashboards for visualization.

## 11. Deliverables

-   **GitHub Repository:** All code, Dockerfile, `requirements.txt`, Jupyter notebooks (`notebooks/eda.ipynb`), `tests/` folder with unit tests, `.github/workflows/github-actions.yml`, and `k8s` manifests.
    -   Link to Code Repository: <YOUR_REPO_LINK_HERE>
-   **Cleaned dataset and download script/instructions:** The dataset `data/heart.csv` is included.
-   **Jupyter notebooks/scripts (EDA, training, inference):** `notebooks/eda.ipynb`, `src/preprocess.py`, `src/train.py`, `src/inference.py` (placeholder, not implemented as a separate script, logic is in API).
-   **Screenshot folder for reporting:** `reports/figures/` contains EDA plots. Placeholders for CI/CD and deployment screenshots are in this README.
-   **Final written report:** This README.md serves as the report.
-   **Deployed API URL:** Instructions provided for local deployment and verification.
-   **Short video containing an end-to-end pipeline:** (Cannot be provided by the agent.)

---
**Note:** For a complete end-to-end deployment with screenshots and a video, some manual steps and external tools are required beyond the agent's capabilities.