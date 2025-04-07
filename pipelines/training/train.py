import os
import sys
import logging
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classifier")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")

# Configure MLflow
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
except Exception as e:
    logger.error(f"Error setting MLflow tracking URI: {str(e)}")
    logger.warning("Will attempt to continue without MLflow tracking")

def load_data(dataset="iris"):
    """Load dataset based on the specified name."""
    logger.info(f"Loading {dataset} dataset")
    
    if dataset == "iris":
        # Load Iris dataset for classification
        iris = load_iris()
        X = iris.data
        y = iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        # Create a DataFrame for easier handling
        data = pd.DataFrame(X, columns=feature_names)
        data['target'] = y
        data['target_name'] = [target_names[t] for t in y]
        
        dataset_type = "classification"
        return data, feature_names, target_names, dataset_type
    
    elif dataset == "california_housing":
        # Load California Housing dataset for regression
        housing = fetch_california_housing()
        X = housing.data
        y = housing.target
        feature_names = housing.feature_names
        
        # Create a DataFrame for easier handling
        data = pd.DataFrame(X, columns=feature_names)
        data['target'] = y
        
        dataset_type = "regression"
        return data, feature_names, None, dataset_type
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def prepare_data(data, test_size=0.2, random_state=42):
    """Split data into training and test sets."""
    logger.info(f"Splitting data into train and test sets (test_size={test_size})")
    
    if 'target_name' in data.columns:
        X = data.drop(['target', 'target_name'], axis=1)
    else:
        X = data.drop(['target'], axis=1)
    
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) < 10 else None
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, dataset_type, params=None):
    """Train a model based on the dataset type."""
    if dataset_type == "classification":
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        
        logger.info(f"Training RandomForest classifier with params: {params}")
        model = RandomForestClassifier(**params)
    
    elif dataset_type == "regression":
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        
        logger.info(f"Training RandomForest regressor with params: {params}")
        model = RandomForestRegressor(**params)
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, dataset_type):
    """Evaluate model performance based on dataset type."""
    logger.info("Evaluating model performance")
    
    y_pred = model.predict(X_test)
    
    if dataset_type == "classification":
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
    
    elif dataset_type == "regression":
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    return metrics, y_pred

def log_to_mlflow(model, params, metrics, X_train, y_train, X_test, y_test, feature_names, target_names, dataset_type, dataset_name):
    """Log model, parameters, and metrics to MLflow."""
    logger.info("Logging to MLflow")
    
    try:
        # Set experiment
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        # Start run and log everything
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("dataset_type", dataset_type)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            signature = infer_signature(X_train, model.predict(X_train))
            
            # Choose model name based on dataset
            registered_model_name = f"{dataset_name}-model"
            
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=registered_model_name
            )
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            mlflow.log_dict(
                dictionary=feature_importance.to_dict(),
                artifact_file="feature_importance.json"
            )
            
            run_id = run.info.run_id
            logger.info(f"MLflow run completed (run_id: {run_id})")
            
            return run_id
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {str(e)}")
        logger.info("Saving model locally instead")
        
        # Save model locally as fallback
        os.makedirs("models", exist_ok=True)
        local_model_path = os.path.join("models", f"{dataset_name}_model.joblib")
        joblib.dump(model, local_model_path)
        logger.info(f"Model saved locally to: {local_model_path}")
        
        return None

def main(dataset_name="iris", model_params=None):
    """Main training pipeline function."""
    logger.info(f"Starting training pipeline for {dataset_name} dataset")
    
    try:
        # Load and prepare data
        data, feature_names, target_names, dataset_type = load_data(dataset_name)
        X_train, X_test, y_train, y_test = prepare_data(data)
        
        # Define model parameters
        if model_params is None:
            if dataset_type == "classification":
                model_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
            else:  # regression
                model_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
        
        # Train model
        model = train_model(X_train, y_train, dataset_type, params=model_params)
        
        # Evaluate model
        metrics, y_pred = evaluate_model(model, X_test, y_test, dataset_type)
        
        # Log to MLflow
        run_id = log_to_mlflow(
            model=model,
            params=model_params,
            metrics=metrics,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test, 
            y_test=y_test,
            feature_names=feature_names,
            target_names=target_names,
            dataset_type=dataset_type,
            dataset_name=dataset_name
        )
        
        # Always save model locally as a backup
        os.makedirs("models", exist_ok=True)
        local_model_path = os.path.join("models", f"{dataset_name}_model.joblib")
        joblib.dump(model, local_model_path)
        logger.info(f"Model backup saved locally to: {local_model_path}")
        
        logger.info("Training pipeline completed successfully")
        return run_id
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model")
    parser.add_argument("--dataset", type=str, default="iris", choices=["iris", "california_housing"],
                        help="Dataset to use for training (default: iris)")
    args = parser.parse_args()
    
    main(dataset_name=args.dataset) 