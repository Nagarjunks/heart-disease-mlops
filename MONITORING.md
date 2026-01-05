# Heart Disease ML Monitoring Guide

This guide explains how to set up and use the Prometheus + Grafana monitoring stack for the Heart Disease Prediction API.

## Quick Start

### Option 1: Docker Compose (Recommended for Local Development)

1. **Start the monitoring stack:**
   ```bash
   docker-compose up -d
   ```

2. **Access the services:**
   - FastAPI: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Prometheus Metrics: http://localhost:8000/metrics
   - Prometheus UI: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

3. **View the dashboard:**
   - Login to Grafana at http://localhost:3000
   - Navigate to Dashboards → "Heart Disease ML Model Monitoring"

4. **Test the API and watch metrics:**
   ```bash
   # Make a prediction request
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "age": 63,
       "sex": 1,
       "cp": 3,
       "trestbps": 145,
       "chol": 233,
       "fbs": 1,
       "restecg": 0,
       "thalach": 150,
       "exang": 0,
       "oldpeak": 2.3,
       "slope": 0,
       "ca": 0,
       "thal": 1
     }'
   ```

5. **Stop the stack:**
   ```bash
   docker-compose down
   ```

### Option 2: Kubernetes Deployment

1. **Deploy the application:**
   ```bash
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   ```

2. **Deploy Prometheus:**
   ```bash
   kubectl apply -f k8s/prometheus-configmap.yaml
   kubectl apply -f k8s/prometheus-deployment.yaml
   ```

3. **Deploy Grafana:**
   ```bash
   kubectl apply -f k8s/grafana-deployment.yaml
   ```

4. **Get service endpoints:**
   ```bash
   kubectl get services
   ```

5. **Access Grafana and configure:**
   - Navigate to the Grafana external IP
   - Login with admin/admin
   - Add Prometheus datasource (http://prometheus:9090)
   - Import the dashboard from `grafana/dashboards/heart-disease-ml-monitoring.json`

## Metrics Explained

### Custom ML Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `heart_disease_predictions_total` | Counter | Total predictions made, labeled by prediction result (0 or 1) |
| `heart_disease_prediction_confidence` | Histogram | Distribution of prediction confidence scores |
| `heart_disease_prediction_errors_total` | Counter | Total errors by type (model_not_loaded, preprocessing_error, prediction_error) |
| `heart_disease_active_requests` | Gauge | Number of currently active prediction requests |
| `heart_disease_model_loaded` | Gauge | Model load status (1=success, 0=failure) |

### Standard HTTP Metrics

These are automatically collected by `prometheus-fastapi-instrumentator`:

| Metric Name | Type | Description |
|-------------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests by method, status, and handler |
| `http_request_duration_seconds` | Histogram | HTTP request latency distribution |
| `http_request_size_bytes` | Histogram | HTTP request body size |
| `http_response_size_bytes` | Histogram | HTTP response body size |

## Dashboard Panels

The pre-configured Grafana dashboard includes:

1. **Model Status Gauge** - Shows if the model is loaded (1) or not (0)
2. **Prediction Rate** - Real-time predictions per second, split by label
3. **Total Predictions Counter** - Cumulative count of all predictions
4. **Active Requests Gauge** - Current concurrent requests being processed
5. **Prediction Distribution Pie Chart** - Proportion of positive/negative predictions
6. **Confidence Percentiles Graph** - p50, p90, p95, p99 confidence scores over time
7. **Error Rate by Type** - Tracks different error categories
8. **Request Latency Percentiles** - API response time distribution
9. **HTTP Request Rate** - Requests per second by status code
10. **API Availability** - Percentage uptime based on 5xx errors

## Prometheus Queries

Here are some useful PromQL queries you can use:

### Prediction Metrics

```promql
# Total predictions in last 5 minutes
sum(increase(heart_disease_predictions_total[5m]))

# Prediction rate per second
rate(heart_disease_predictions_total[1m])

# Percentage of positive predictions
sum(heart_disease_predictions_total{prediction_label="1"}) / sum(heart_disease_predictions_total) * 100

# Average confidence score (p50)
histogram_quantile(0.50, rate(heart_disease_prediction_confidence_bucket[5m]))
```

### Performance Metrics

```promql
# p95 latency for /predict endpoint
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{handler="/predict"}[5m]))

# Requests per second
rate(http_requests_total{handler="/predict"}[1m])

# Error rate (4xx + 5xx)
sum(rate(http_requests_total{handler="/predict",status=~"[45].."}[1m]))

# API availability (% of non-5xx responses)
(1 - sum(rate(http_requests_total{handler="/predict",status=~"5.."}[5m])) / sum(rate(http_requests_total{handler="/predict"}[5m]))) * 100
```

### System Health

```promql
# Model load status
heart_disease_model_loaded

# Current active requests
heart_disease_active_requests

# Total errors by type
sum by (error_type) (heart_disease_prediction_errors_total)
```

## Alerting (Optional)

You can configure Prometheus alerts for critical conditions:

```yaml
# Example alert rules (add to prometheus.yml)
groups:
  - name: heart_disease_api_alerts
    interval: 30s
    rules:
      - alert: ModelNotLoaded
        expr: heart_disease_model_loaded == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "ML model failed to load"
          description: "The heart disease prediction model is not loaded"

      - alert: HighErrorRate
        expr: rate(heart_disease_prediction_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction error rate"
          description: "Error rate is {{ $value }} errors/sec"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{handler="/predict"}[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "API latency is high"
          description: "P95 latency is {{ $value }} seconds"
```

## Troubleshooting

### Prometheus can't scrape metrics

1. Check if the API is running:
   ```bash
   curl http://localhost:8000/metrics
   ```

2. Verify Prometheus configuration:
   ```bash
   # Check targets in Prometheus UI
   # Visit http://localhost:9090/targets
   ```

3. Check network connectivity (Docker Compose):
   ```bash
   docker-compose ps
   docker network inspect heart-disease-mlops_monitoring
   ```

### Grafana dashboard shows "No Data"

1. Verify Prometheus datasource:
   - Go to Configuration → Data Sources
   - Test the Prometheus connection

2. Check if metrics are being collected:
   - Visit http://localhost:9090
   - Run query: `heart_disease_predictions_total`

3. Generate some traffic:
   ```bash
   # Make several prediction requests
   for i in {1..10}; do
     curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d '{"age": 60, "sex": 1, "cp": 3, "trestbps": 140, "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 2.0, "slope": 0, "ca": 0, "thal": 2}'
   done
   ```

### Docker Compose fails to start

1. Check for port conflicts:
   ```bash
   # Ports 8000, 9090, 3000 must be available
   netstat -an | findstr "8000 9090 3000"
   ```

2. Check Docker logs:
   ```bash
   docker-compose logs api
   docker-compose logs prometheus
   docker-compose logs grafana
   ```

## Advanced Configuration

### Customize Scrape Interval

Edit `prometheus.yml`:
```yaml
global:
  scrape_interval: 5s  # Change this value
```

### Add Custom Metrics

Edit [api/main.py](api/main.py) and add new metrics:

```python
from prometheus_client import Counter

# Define custom metric
custom_metric = Counter(
    'my_custom_metric_total',
    'Description of my metric',
    ['label_name']
)

# Use in endpoint
@app.post("/predict")
def predict(data: HeartDiseaseInput):
    custom_metric.labels(label_name="value").inc()
    # ... rest of code
```

### Persist Prometheus Data

Update `docker-compose.yml` to use a named volume:
```yaml
volumes:
  - prometheus_data:/prometheus  # Already configured
```

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [prometheus-fastapi-instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator)
- [PromQL Guide](https://prometheus.io/docs/prometheus/latest/querying/basics/)

## Support

For issues or questions:
1. Check application logs: `docker-compose logs -f api`
2. Check Prometheus logs: `docker-compose logs -f prometheus`
3. Check Grafana logs: `docker-compose logs -f grafana`
