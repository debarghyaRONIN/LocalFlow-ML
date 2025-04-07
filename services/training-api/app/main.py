import os
import io
import base64
import tempfile
import logging
import uuid
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_absolute_error, mean_squared_error, r2_score
)
import mlflow
import mlflow.sklearn

# Import model implementations
from app.models import get_model_by_name, get_available_models, get_model_task
from app.data_utils import process_dataset, validate_dataset, get_dataset_info, detect_dataset_task
from app.visualization import generate_visualization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "custom-training")

# Configure MLflow
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
except Exception as e:
    logger.error(f"Error setting MLflow tracking URI: {str(e)}")

# Create temp directory for storing uploaded files
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "ml_training_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="ML Training API",
    description="API for uploading datasets, training models, and visualizing results",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Pydantic Models ----

class DatasetInfo(BaseModel):
    id: str
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    column_dtypes: Dict[str, str]
    has_missing_values: bool
    sample_data: List[Dict[str, Any]]
    numeric_columns: List[str]
    categorical_columns: List[str]

class ModelInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    task: str

class TrainingRequest(BaseModel):
    dataset_id: str
    model_name: str
    target_column: str
    features: List[str] = Field(default=[])
    test_size: float = Field(default=0.2, ge=0.1, le=0.9)
    random_state: int = Field(default=42)
    model_parameters: Dict[str, Any] = Field(default={})

class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str

class TrainingResult(BaseModel):
    job_id: str
    model_name: str
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    run_id: Optional[str] = None

class VisualizationRequest(BaseModel):
    dataset_id: str
    viz_type: str
    columns: List[str] = Field(default=[])
    parameters: Dict[str, Any] = Field(default={})

class VisualizationResponse(BaseModel):
    image_data: str  # Base64 encoded image

class PredictionRequest(BaseModel):
    job_id: str
    data: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions: List[Any]
    probability: Optional[List[Dict[str, float]]] = None

# ---- In-memory storage ----
datasets = {}
training_jobs = {}
training_results = {}

# ---- API Endpoints ----

@app.get("/")
def read_root():
    return {"message": "ML Training API is running"}

@app.get("/models", response_model=List[ModelInfo])
def list_models():
    """List all available models that can be trained"""
    return get_available_models()

@app.post("/datasets", response_model=DatasetInfo)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file (CSV, Excel, etc.)"""
    try:
        # Generate unique ID for the dataset
        dataset_id = str(uuid.uuid4())
        
        # Create a file path
        file_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_{file.filename}")
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process and validate the dataset
        df = process_dataset(file_path)
        validation_result = validate_dataset(df)
        
        if not validation_result["valid"]:
            os.remove(file_path)  # Clean up
            raise HTTPException(status_code=400, detail=validation_result["message"])
            
        # Get dataset info
        dataset_info = get_dataset_info(df)
        dataset_info["id"] = dataset_id
        dataset_info["filename"] = file.filename
        
        # Store dataset path and format
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        datasets[dataset_id] = {
            "path": file_path,
            "info": dataset_info,
            "format": file_ext
        }
        
        return dataset_info
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")

@app.get("/datasets/{dataset_id}", response_model=DatasetInfo)
def get_dataset(dataset_id: str):
    """Get dataset information"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return datasets[dataset_id]["info"]

@app.post("/train", response_model=TrainingResponse)
def train_model(training_request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a model training job"""
    dataset_id = training_request.dataset_id
    model_name = training_request.model_name
    
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Check if model is valid
    try:
        model_class = get_model_by_name(model_name)
        model_task = get_model_task(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Load dataset to validate columns
    try:
        df = _load_dataset(dataset_id)
        
        # Validate target column exists
        if training_request.target_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{training_request.target_column}' not found in dataset"
            )
        
        # Validate feature columns exist
        if training_request.features:
            missing_features = [col for col in training_request.features if col not in df.columns]
            if missing_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature column(s) not found in dataset: {', '.join(missing_features)}"
                )
        
        # Validate target column is appropriate for model task
        dataset_task = detect_dataset_task(df, training_request.target_column)
        if dataset_task != model_task:
            raise HTTPException(
                status_code=400,
                detail=f"Model task '{model_task}' does not match dataset task '{dataset_task}' based on target column"
            )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error validating dataset for training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error validating dataset: {str(e)}")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Add to training jobs
    training_jobs[job_id] = {
        "status": "queued",
        "request": training_request.dict(),
        "warnings": []
    }
    
    # Schedule training task
    background_tasks.add_task(
        _train_model_task,
        job_id=job_id, 
        dataset_id=dataset_id,
        training_request=training_request
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Training job queued for model: {model_name}"
    }

@app.get("/train/{job_id}", response_model=dict)
def get_training_status(job_id: str):
    """Get the status of a training job"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job_info = training_jobs[job_id].copy()
    
    # Add results if available
    if job_id in training_results:
        job_info["results"] = training_results[job_id]
    
    return job_info

@app.post("/predict", response_model=PredictionResponse)
def predict(prediction_request: PredictionRequest):
    """Make predictions using a trained model"""
    job_id = prediction_request.job_id
    
    # Check if job exists and is completed
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if training_jobs[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Model training is not completed yet")
    
    if job_id not in training_results:
        raise HTTPException(status_code=404, detail="Training results not found")
    
    try:
        # Get training information
        training_result = training_results[job_id]
        model_path = training_result.get("model_path")
        training_request = training_jobs[job_id]["request"]
        model_task = training_result["task"]
        
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # Load the model
        model = joblib.load(model_path)
        
        # Convert input data to DataFrame
        input_data = pd.DataFrame(prediction_request.data)
        
        # Get features used in training
        features = training_request.get("features", [])
        if not features:
            # Get all features from the original dataset excluding target
            dataset_id = training_request.get("dataset_id")
            if dataset_id in datasets:
                df = _load_dataset(dataset_id)
                target_col = training_request.get("target_column")
                features = [col for col in df.columns if col != target_col]
        
        # Validate input data has required features
        missing_features = [col for col in features if col not in input_data.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Input data is missing required features: {', '.join(missing_features)}"
            )
        
        # Prepare input data (apply same preprocessing as during training)
        X = input_data[features]
        
        # Handle missing values
        for col in X.columns:
            if X[col].isna().any():
                # Use simple imputation as in training
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].mean())
                else:
                    X[col] = X[col].fillna("missing")
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Make predictions
        predictions = model.predict(X).tolist()
        
        # For classification, add probabilities if available
        probabilities = None
        if model_task == "classification" and hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X)
                
                # Get class labels
                if hasattr(model, 'classes_'):
                    class_labels = model.classes_.tolist()
                    
                    # Convert probabilities to dictionaries
                    probabilities = []
                    for p in proba:
                        probabilities.append({str(label): float(prob) for label, prob in zip(class_labels, p)})
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {str(e)}")
        
        return {
            "predictions": predictions,
            "probability": probabilities
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")

@app.post("/visualize", response_model=VisualizationResponse)
def create_visualization(viz_request: VisualizationRequest):
    """Generate a visualization for the dataset"""
    dataset_id = viz_request.dataset_id
    
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Load the dataset
        df = _load_dataset(dataset_id)
        
        # Validate columns exist if specified
        if viz_request.columns:
            missing_columns = [col for col in viz_request.columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Column(s) not found in dataset: {', '.join(missing_columns)}"
                )
        
        # Generate the visualization
        img_data = generate_visualization(
            df=df, 
            viz_type=viz_request.viz_type,
            columns=viz_request.columns,
            parameters=viz_request.parameters
        )
        
        return {"image_data": img_data}
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error generating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

# ---- Helper Functions ----

def _load_dataset(dataset_id: str) -> pd.DataFrame:
    """Load a dataset from storage"""
    if dataset_id not in datasets:
        raise ValueError(f"Dataset ID '{dataset_id}' not found")
    
    dataset_info = datasets[dataset_id]
    file_path = dataset_info["path"]
    file_format = dataset_info.get("format", ".csv")  # Default to CSV
    
    # Load dataset based on file format
    try:
        df = process_dataset(file_path)
        return df
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
        raise ValueError(f"Error loading dataset: {str(e)}")

def _calculate_metrics(y_true, y_pred, task: str) -> Dict[str, float]:
    """Calculate metrics based on task type"""
    metrics = {}
    
    if task == "classification":
        # Classification metrics
        try:
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            
            # For multi-class, use weighted average
            metrics["precision"] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics["f1"] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        except Exception as e:
            logger.warning(f"Error calculating some classification metrics: {str(e)}")
            # Ensure at least accuracy is calculated
            if "accuracy" not in metrics:
                metrics["accuracy"] = float(np.mean(y_true == y_pred))
    
    elif task == "regression":
        # Regression metrics
        try:
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["mse"] = float(mean_squared_error(y_true, y_pred))
            metrics["rmse"] = float(np.sqrt(metrics["mse"]))
            metrics["r2"] = float(r2_score(y_true, y_pred))
            
            # Add explained variance if not 1.0 (perfect prediction)
            if metrics["r2"] < 0.999:
                metrics["explained_variance"] = float(np.var(y_pred) / np.var(y_true))
        except Exception as e:
            logger.warning(f"Error calculating some regression metrics: {str(e)}")
    
    return metrics

# ---- Background Tasks ----

def _train_model_task(job_id: str, dataset_id: str, training_request: TrainingRequest):
    """Background task for model training"""
    try:
        # Update job status
        training_jobs[job_id]["status"] = "running"
        
        # Get model task
        model_name = training_request.model_name
        model_task = get_model_task(model_name)
        
        # Load dataset
        df = _load_dataset(dataset_id)
        
        # Prepare data
        target = training_request.target_column
        
        # Use specified features or all columns except target
        if training_request.features:
            features = training_request.features
        else:
            features = [col for col in df.columns if col != target]
        
        X = df[features]
        y = df[target]
        
        # Handle missing values
        if X.isna().any().any():
            training_jobs[job_id]["warnings"].append(
                "Dataset contains missing values. Simple imputation will be applied."
            )
            # Simple imputation: fill numeric with mean, categorical with mode
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].mean())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "missing")
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            # For simplicity, we'll do a basic one-hot encoding
            training_jobs[job_id]["warnings"].append(
                "Categorical features detected. Basic one-hot encoding will be applied."
            )
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=training_request.test_size, 
            random_state=training_request.random_state,
            stratify=y if model_task == "classification" else None
        )
        
        # Get model
        model_class = get_model_by_name(training_request.model_name)
        
        # Validate model parameters
        valid_params = {}
        if training_request.model_parameters:
            # Only use parameters that are valid for this model
            from inspect import signature
            model_signature = signature(model_class.__init__)
            valid_param_names = [param.name for param in model_signature.parameters.values() 
                                if param.name not in ('self', 'args', 'kwargs')]
            
            for param, value in training_request.model_parameters.items():
                if param in valid_param_names:
                    valid_params[param] = value
                else:
                    training_jobs[job_id]["warnings"].append(
                        f"Parameter '{param}' is not valid for model {training_request.model_name} and will be ignored."
                    )
        
        # Initialize and train model
        model = model_class(**valid_params)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        # Calculate metrics based on task
        metrics = _calculate_metrics(y_test, y_pred, model_task)
        
        # Try to get feature importance
        feature_importance = None
        try:
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                # For linear models
                if len(model.coef_.shape) == 1:
                    feature_importance = dict(zip(X.columns, abs(model.coef_)))
                else:
                    # For multi-class, use mean absolute coefficient
                    feature_importance = dict(zip(X.columns, abs(model.coef_).mean(axis=0)))
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
        
        # Log to MLflow
        run_id = None
        try:
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            
            with mlflow.start_run() as run:
                # Log parameters
                mlflow.log_params(valid_params)
                mlflow.log_param("model_name", training_request.model_name)
                mlflow.log_param("target_column", target)
                mlflow.log_param("features", features)
                mlflow.log_param("test_size", training_request.test_size)
                mlflow.log_param("task", model_task)
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model"
                )
                
                run_id = run.info.run_id
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {str(e)}")
        
        # Save model locally
        model_dir = os.path.join(UPLOAD_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{job_id}.joblib")
        joblib.dump(model, model_path)
        
        # Save results
        training_results[job_id] = {
            "job_id": job_id,
            "model_name": training_request.model_name,
            "task": model_task,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "run_id": run_id,
            "model_path": model_path,
            "features": features,  # Save features for prediction
            "categorical_cols": categorical_cols.tolist() if not categorical_cols.empty else []
        }
        
        # Update job status
        training_jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        logger.error(f"Error in training job {job_id}: {str(e)}")
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e) 