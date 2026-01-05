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
-   **Monitoring & Logging:**
    -   Request/response logging for debugging
    -   Prometheus metrics for ML model monitoring
    -   Pre-configured Grafana dashboards
    -   Docker Compose stack for local monitoring
    -   Kubernetes monitoring manifests for production

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

## 10. Monitoring & Logging

### API Logging
The FastAPI application ([api/main.py](api/main.py)) integrates Python's `logging` module to record incoming requests, prediction results, and any errors. Logs are output to `stdout`, which can be collected by container orchestration systems.

### Prometheus Monitoring
The application is instrumented with Prometheus metrics to track:

**Custom ML Metrics:**
- `heart_disease_predictions_total`: Counter tracking total predictions by label (0/1)
- `heart_disease_prediction_confidence`: Histogram of prediction confidence scores
- `heart_disease_prediction_errors_total`: Counter tracking errors by type (model_not_loaded, preprocessing_error, prediction_error)
- `heart_disease_active_requests`: Gauge tracking concurrent requests
- `heart_disease_model_loaded`: Gauge indicating model load status (1=loaded, 0=failed)

**Standard HTTP Metrics** (via prometheus-fastapi-instrumentator):
- Request duration histograms
- Request count by status code
- Request size/response size metrics

**Metrics Endpoint:** Available at `http://localhost:8000/metrics`

### Running with Docker Compose (Recommended)

The complete monitoring stack (API + Prometheus + Grafana) can be started with:

```bash
docker-compose up -d
```

This will start:
- **FastAPI Application** on [http://localhost:8000](http://localhost:8000)
- **Prometheus** on [http://localhost:9090](http://localhost:9090)
- **Grafana** on [http://localhost:3000](http://localhost:3000) (username: `admin`, password: `admin`)

**View the monitoring dashboard:**
1. Access Grafana at [http://localhost:3000](http://localhost:3000)
2. Login with admin/admin
3. Navigate to Dashboards → "Heart Disease ML Model Monitoring"

**Stop the monitoring stack:**
```bash
docker-compose down
```

### Kubernetes Monitoring Deployment

For Kubernetes deployments, additional manifests are provided:

```bash
# Deploy the API
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Deploy Prometheus
kubectl apply -f k8s/prometheus-configmap.yaml
kubectl apply -f k8s/prometheus-deployment.yaml

# Deploy Grafana
kubectl apply -f k8s/grafana-deployment.yaml
```

**Access monitoring services:**
```bash
# Get Prometheus external IP
kubectl get service prometheus

# Get Grafana external IP
kubectl get service grafana
```

### Grafana Dashboard Features

The pre-configured "Heart Disease ML Model Monitoring" dashboard includes:

1. **Model Status** - Real-time model health indicator
2. **Prediction Rate** - Predictions per second by label
3. **Total Predictions** - Cumulative prediction count
4. **Active Requests** - Current concurrent requests
5. **Prediction Distribution** - Pie chart of prediction labels
6. **Prediction Confidence Percentiles** - p50, p90, p95, p99 confidence scores
7. **Error Rate by Type** - Tracking different error categories
8. **Request Latency Percentiles** - API response time distribution
9. **HTTP Request Rate** - Requests per second by status code
10. **API Availability** - Uptime based on 5xx errors

### Monitoring Architecture

```
┌─────────────┐     metrics       ┌────────────┐     queries     ┌─────────┐
│  FastAPI    │ ─────────────────>│ Prometheus │ ───────────────>│ Grafana │
│ (Port 8000) │     metrics       │ (Port 9090)│                 │ (Port   │
└─────────────┘                   └────────────┘                 │  3000)  │
                                                                 └─────────┘
