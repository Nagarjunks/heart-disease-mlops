import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df: pd.DataFrame):
    """
    Cleans and preprocesses the heart disease data.

    Args:
        df: Raw data as a pandas DataFrame.

    Returns:
        A tuple containing:
        - X_train, X_test, y_train, y_test
        - The preprocessing pipeline object.
    """
    # Separate target variable
    X = df.drop("target", axis=1)
    y = df["target"]

    # Identify categorical and numerical features
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    # Create preprocessing pipelines for both feature types
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == '__main__':
    # Example of how to use the preprocessor
    df = pd.read_csv('data/heart.csv')
    X_train, X_test, y_train, y_test, preprocessor_pipeline = preprocess_data(df)

    # Fit and transform the training data
    X_train_processed = preprocessor_pipeline.fit_transform(X_train)
    
    # Transform the test data
    X_test_processed = preprocessor_pipeline.transform(X_test)

    print("Data preprocessed successfully.")
    print("X_train shape:", X_train_processed.shape)
    print("X_test shape:", X_test_processed.shape)