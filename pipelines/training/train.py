import os
import sys
import logging
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

def load_data():
    """Load sample Iris dataset."""
    logger.info("Loading Iris dataset")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Create a DataFrame for easier handling
    data = pd.DataFrame(X, columns=feature_names)
    data['target'] = y
    data['target_name'] = [target_names[t] for t in y]
    
    return data, feature_names, target_names

def prepare_data(data, test_size=0.2, random_state=42):
    """Split data into training and test sets."""
    logger.info(f"Splitting data into train and test sets (test_size={test_size})")
    
    X = data.drop(['target', 'target_name'], axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, params=None):
    """Train a RandomForest classifier."""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
    
    logger.info(f"Training RandomForest model with params: {params}")
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    logger.info("Evaluating model performance")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    return metrics, y_pred

def log_to_mlflow(model, params, metrics, X_train, y_train, X_test, y_test, feature_names, target_names):
    """Log model, parameters, and metrics to MLflow."""
    logger.info("Logging to MLflow")
    
    try:
        # Set experiment
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        # Start run and log everything
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            signature = infer_signature(X_train, model.predict(X_train))
            
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name="iris-classifier"
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
        local_model_path = os.path.join("models", "iris_classifier.joblib")
        joblib.dump(model, local_model_path)
        logger.info(f"Model saved locally to: {local_model_path}")
        
        return None

def main(model_params=None):
    """Main training pipeline function."""
    logger.info("Starting training pipeline")
    
    try:
        # Load and prepare data
        data, feature_names, target_names = load_data()
        X_train, X_test, y_train, y_test = prepare_data(data)
        
        # Define model parameters
        if model_params is None:
            model_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        
        # Train model
        model = train_model(X_train, y_train, params=model_params)
        
        # Evaluate model
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        
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
            target_names=target_names
        )
        
        # Always save model locally as a backup
        os.makedirs("models", exist_ok=True)
        local_model_path = os.path.join("models", "iris_classifier.joblib")
        joblib.dump(model, local_model_path)
        logger.info(f"Model backup saved locally to: {local_model_path}")
        
        logger.info("Training pipeline completed successfully")
        return run_id
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an Iris classifier")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees in the forest")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum depth of the trees")
    parser.add_argument("--min-samples-split", type=int, default=2, help="Minimum samples required to split")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="Minimum samples required at a leaf node")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    
    args = parser.parse_args()
    
    model_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'random_state': args.random_state
    }
    
    main(model_params=model_params) 