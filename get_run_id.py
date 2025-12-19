
import mlflow

# Set the tracking URI to the local mlruns directory
mlflow.set_tracking_uri("file:///D:/JEET/mlops/heart-disease-mlops/mlruns")

# Get the experiment by name
experiment_name = "Heart Disease Prediction"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    print(f"Experiment '{experiment_name}' not found.")
    exit()

# Search for the latest run of the LogisticRegression model
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="params.model_type = 'LogisticRegression'",
    order_by=["start_time desc"],
    max_results=1
)

if not runs.empty:
    latest_run = runs.iloc[0]
    run_id = latest_run.run_id
    artifact_uri = latest_run.artifact_uri
    print(f"Latest LogisticRegression Run ID: {run_id}")
    print(f"Artifact URI: {artifact_uri}")
else:
    print("No runs found for LogisticRegression model.")
