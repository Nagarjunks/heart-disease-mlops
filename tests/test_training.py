
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from train import train_and_evaluate, main
from preprocess import preprocess_data
from sklearn.linear_model import LogisticRegression

@pytest.fixture
def preprocessed_data():
    """Fixture for preprocessed data."""
    df = pd.read_csv('data/heart.csv')
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    return X_train_processed, y_train, X_test_processed, y_test

def test_train_and_evaluate(preprocessed_data):
    """Test the train_and_evaluate function."""
    X_train, y_train, X_test, y_test = preprocessed_data
    model = LogisticRegression(random_state=42)
    
    metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test)
    
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "roc_auc" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


@patch('train.mlflow')
@patch('train.joblib')
@patch('train.pd.read_csv')
@patch('train.preprocess_data')
def test_training_script_main(mock_preprocess, mock_read_csv, mock_joblib, mock_mlflow):
    """
    Test the main execution block of the training script to ensure it runs
    without error and calls MLflow functions correctly.
    """
    # Mock the dataframe
    mock_df = pd.DataFrame({
        'age': [63, 37], 'sex': [1, 1], 'cp': [3, 2], 'trestbps': [145, 130],
        'chol': [233, 250], 'fbs': [1, 0], 'restecg': [0, 1], 'thalach': [150, 187],
        'exang': [0, 0], 'oldpeak': [2.3, 3.5], 'slope': [0, 0], 'ca': [0, 0],
        'thal': [1, 2], 'target': [1, 1]
    })
    mock_read_csv.return_value = mock_df
    
    
    

    
    # Mock the return value of preprocess_data
    
    X_train = pd.DataFrame({'age': [50, 55], 'sex': [1, 0]})
    X_test = pd.DataFrame({'age': [60, 65], 'sex': [0, 1]})
    y_train = pd.Series([0, 1])
    y_test = pd.Series([1, 0])
    mock_preprocessor = MagicMock()
    mock_preprocessor.fit_transform.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_preprocessor.transform.return_value = [[0.5, 0.6], [0.7, 0.8]]
    
    mock_preprocess.return_value = (
        X_train, X_test, y_train, y_test, mock_preprocessor
    )
    
    # Import main function and run it
    main()

    # Assert that MLflow experiment was set
    mock_mlflow.set_experiment.assert_called_with(experiment_name="Heart Disease V2")
    
    # Assert that MLflow runs were started
    assert mock_mlflow.start_run.call_count == 2 # For two models
    
    # Assert that models and metrics were logged
    assert mock_mlflow.log_param.call_count == 2
    assert mock_mlflow.log_metrics.call_count == 2
    assert mock_mlflow.sklearn.log_model.call_count == 2
    
    # Assert that the preprocessor was saved
    mock_joblib.dump.assert_called_once()
