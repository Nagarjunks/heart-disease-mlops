import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from preprocess import preprocess_data

# Define the models to train
models = {
    "LogisticRegression": LogisticRegression(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
}


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """Trains the model and evaluates it."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }


def main():

    # Load data

    df = pd.read_csv("data/heart.csv")

    # Preprocess data

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    # Set up MLflow experiment

    experiment_name = "Heart Disease V2"

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:

        mlflow.create_experiment(experiment_name)

        mlflow.set_experiment(experiment_name=experiment_name)

    else:

        mlflow.set_experiment(experiment_name=experiment_name)

    # Process the training and test data using the preprocessor

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    # Save the fitted preprocessor

    os.makedirs("models", exist_ok=True)

    preprocessor_path = "models/preprocessor.joblib"

    joblib.dump(preprocessor, preprocessor_path)

    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):

            print(f"Training {model_name}...")

            # Log model type

            mlflow.log_param("model_type", model_name)

            # Create a full pipeline with preprocessor and model

            # Note: We are training on already processed data here

            # but logging the model as a simple estimator.

            # The API will need to chain the saved preprocessor and this model.

            # Train and evaluate

            metrics = train_and_evaluate(
                model, X_train_processed, y_train, X_test_processed, y_test
            )

            # Log metrics

            mlflow.log_metrics(metrics)

            print(f"Metrics for {model_name}: {metrics}")

            # Log the trained model

            mlflow.sklearn.log_model(model, name=model_name)

            print(f"Model {model_name} logged to MLflow.")

    print("Training and evaluation complete.")


if __name__ == "__main__":

    main()
