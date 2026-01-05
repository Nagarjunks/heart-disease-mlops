
import pytest
import pandas as pd
from preprocess import preprocess_data


@pytest.fixture
def heart_data():
    """Fixture to load the heart disease dataset."""
    try:
        df = pd.read_csv('data/heart.csv')
        return df
    except FileNotFoundError:
        pytest.fail("Dataset file not found: data/heart.csv")

def test_data_loading(heart_data):
    """Test that the data loads into a non-empty DataFrame."""
    assert not heart_data.empty
    assert "target" in heart_data.columns

def test_data_shape(heart_data):
    """Test the shape of the loaded data."""
    assert heart_data.shape[0] > 0
    assert heart_data.shape[1] == 14

def test_preprocess_data_output(heart_data):
    """Test the output of the preprocess_data function."""
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(heart_data)

    # Check that the outputs are not empty
    assert not X_train.empty
    assert not X_test.empty
    assert not y_train.empty
    assert not y_test.empty

    # Check shapes
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    # Check preprocessor
    assert preprocessor is not None
    
    # Check that the preprocessor can transform data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    assert X_train_processed.shape[0] == X_train.shape[0]
    assert X_test_processed.shape[0] == X_test.shape[0]

def test_for_missing_values(heart_data):
    """Test for any missing values in the dataset."""
    assert heart_data.isnull().sum().sum() == 0

def test_target_variable_distribution(heart_data):
    """Test the distribution of the target variable."""
    assert heart_data['target'].nunique() == 2