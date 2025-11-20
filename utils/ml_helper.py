"""
Machine Learning Helper Functions
Utilities for training and evaluating ML models
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def encode_data(df: pd.DataFrame, feature_cols: list, target_col: str) -> tuple:
    """
    Encode categorical variables in features and target
    
    Args:
        df: Input dataframe
        feature_cols: List of feature column names
        target_col: Target column name
    
    Returns:
        Tuple of (encoded_dataframe, dict_of_encoders)
    """
    df_encoded = df.copy()
    encoders = {}
    
    # Encode features
    for col in feature_cols:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    
    # Encode target
    if df_encoded[target_col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[target_col] = le.fit_transform(df_encoded[target_col].astype(str))
        encoders[target_col] = le
    
    return df_encoded, encoders


def train_model(df: pd.DataFrame, feature_cols: list, target_col: str, 
                model_type: str, test_size: float = 0.2) -> dict:
    """
    Train a machine learning model
    
    Args:
        df: Input dataframe (should be encoded)
        feature_cols: List of feature column names
        target_col: Target column name
        model_type: Type of model ('Logistic Regression', 'Decision Tree', 'Random Forest')
        test_size: Proportion of data to use for testing
    
    Returns:
        Dictionary with model results
    """
    # Prepare features and target
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Select and train model
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'classification_report': report
    }


def get_model_params(model_type: str) -> dict:
    """
    Get default parameters for different model types
    
    Args:
        model_type: Type of model
    
    Returns:
        Dictionary of default parameters
    """
    params = {
        "Logistic Regression": {
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs'
        },
        "Decision Tree": {
            'random_state': 42,
            'max_depth': None,
            'min_samples_split': 2
        },
        "Random Forest": {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': None,
            'min_samples_split': 2
        }
    }
    
    return params.get(model_type, {})


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a trained model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def cross_validate_model(df: pd.DataFrame, feature_cols: list, target_col: str,
                        model_type: str, cv: int = 5) -> dict:
    """
    Perform cross-validation on a model
    
    Args:
        df: Input dataframe
        feature_cols: List of feature column names
        target_col: Target column name
        model_type: Type of model
        cv: Number of cross-validation folds
    
    Returns:
        Dictionary with cross-validation results
    """
    from sklearn.model_selection import cross_val_score
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Select model
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    return {
        'scores': scores,
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'cv_folds': cv
    }