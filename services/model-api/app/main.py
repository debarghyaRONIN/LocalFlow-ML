import os
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException, Request, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Import custom modules
from . import ab_testing
from . import explainer
from data.feature_store import FeatureStore
from data.validation import validate_schema

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="MLOps Model API", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
PREDICTION_COUNT = Counter("model_prediction_count", "Count of predictions made", ["model_name", "version"])
PREDICTION_LATENCY = Histogram("model_prediction_latency_seconds", "Time for prediction", ["model_name", "version"])
VALIDATION_FAILURES = Counter("data_validation_failures", "Count of data validation failures", ["model_name", "version"])

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global model cache
model_cache = {}

# Initialize feature store
feature_store = FeatureStore()

# Data models
class PredictionRequest(BaseModel):
    features: List[List[float]] = Field(..., example=[[5.1, 3.5, 1.4, 0.2]])
    feature_names: Optional[List[str]] = Field(None, example=["sepal_length", "sepal_width", "petal_length", "petal_width"])

class PredictionResponse(BaseModel):
    predictions: List[Union[int, float, str]] = Field(..., example=[0])
    probabilities: Optional[List[List[float]]] = Field(None, example=[[0.9, 0.05, 0.05]])
    model_name: str
    model_version: str
    prediction_time: float
    explanation: Optional[Dict[str, Any]] = None

class ExplanationRequest(BaseModel):
    features: List[List[float]] = Field(..., example=[[5.1, 3.5, 1.4, 0.2]])
    feature_names: Optional[List[str]] = Field(None, example=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    class_names: Optional[List[str]] = Field(None, example=["setosa", "versicolor", "virginica"])
    output_format: Optional[str] = Field("json", example="json")
    max_display: Optional[int] = Field(10, example=10)

class ABTestRequest(BaseModel):
    name: str = Field(..., example="iris-model-comparison")
    variants: List[Dict[str, Any]] = Field(..., example=[
        {"name": "production", "model_name": "iris-classifier", "model_version": "Production"},
        {"name": "candidate", "model_name": "iris-classifier", "model_version": "Staging"}
    ])
    traffic_split: Optional[List[float]] = Field(None, example=[0.9, 0.1])
    custom_metrics: Optional[List[str]] = Field(None, example=["confidence"])

@app.get("/")
async def root():
    return {"message": "Welcome to MLOps Model API", "docs_url": "/docs"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

def get_user_id(user_id: Optional[str] = Header(None, alias="X-User-ID")):
    """Get user ID from header or generate a random one."""
    if user_id:
        return user_id
    # Generate a random user ID if none provided
    return f"anonymous-{time.time()}"

def load_model(model_name, model_version="latest"):
    """Load a model from MLflow."""
    cache_key = f"{model_name}:{model_version}"
    
    if cache_key in model_cache:
        logger.info(f"Using cached model: {cache_key}")
        return model_cache[cache_key]
    
    try:
        logger.info(f"Loading model {model_name} (version: {model_version}) from MLflow")
        
        if model_version == "latest":
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{model_version}"
        
        model = mlflow.pyfunc.load_model(model_uri)
        model_cache[cache_key] = model
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not load model {model_name} (version: {model_version}): {str(e)}"
        )

def preprocess_features(features, feature_names, model_name):
    """
    Preprocess features using feature store.
    If feature store doesn't have transformations for this model, return original features.
    """
    try:
        # Check if feature store has transformations for this model
        feature_sets = feature_store.list_feature_sets()
        model_feature_set = next((fs for fs in feature_sets if fs['feature_set_name'] == model_name), None)
        
        if model_feature_set:
            # Convert features to DataFrame
            df = pd.DataFrame(features, columns=feature_names or [f"feature_{i}" for i in range(len(features[0]))])
            
            # Transform features
            transformed_df = feature_store.transform_features(df, model_name)
            return transformed_df.values
        else:
            # No transformations for this model
            return np.array(features)
    except Exception as e:
        logger.warning(f"Error preprocessing features: {str(e)}")
        # Return original features if any error
        return np.array(features)

def validate_input_data(features, feature_names, model_name):
    """Validate input data against schema."""
    try:
        # Map model name to schema type (simplified)
        schema_mapping = {
            "iris-classifier": "iris",
            "california-housing-model": "california_housing"
        }
        
        schema_type = schema_mapping.get(model_name)
        if not schema_type:
            logger.warning(f"No schema defined for model {model_name}")
            return True, []
        
        # Convert features to DataFrame with feature names
        df = pd.DataFrame(features, columns=feature_names or [f"feature_{i}" for i in range(len(features[0]))])
        
        # For Iris, rename columns to match schema
        if schema_type == "iris" and feature_names:
            column_mapping = {
                "sepal length (cm)": "sepal_length",
                "sepal width (cm)": "sepal_width",
                "petal length (cm)": "petal_length",
                "petal width (cm)": "petal_width"
            }
            df = df.rename(columns={col: column_mapping.get(col, col) for col in df.columns})
        
        # Validate against schema
        is_valid, errors = validate_schema(df, schema_type)
        return is_valid, errors
    except Exception as e:
        logger.error(f"Error validating input data: {str(e)}")
        return False, [str(e)]

@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict(
    model_name: str, 
    request: PredictionRequest, 
    model_version: Optional[str] = "latest",
    explain: Optional[bool] = Query(False, description="Whether to include explanation with prediction"),
    user_id: str = Depends(get_user_id)
):
    """Make predictions using a trained model."""
    start_time = time.time()
    
    try:
        # Apply A/B testing if available
        test_name = f"{model_name}-comparison"
        ab_test = ab_testing.ab_test_manager.get_test(test_name)
        
        if ab_test:
            # Get assigned variant
            variant = ab_test.get_variant(user_id)
            logger.info(f"A/B testing: User {user_id} assigned to variant {variant['name']}")
            
            # Override model name and version from variant
            model_name = variant['model_name']
            model_version = variant['model_version']
        
        # Validate input data
        is_valid, errors = validate_input_data(request.features, request.feature_names, model_name)
        if not is_valid:
            VALIDATION_FAILURES.labels(model_name=model_name, version=model_version).inc()
            raise HTTPException(
                status_code=400,
                detail=f"Input validation failed: {', '.join(errors)}"
            )
        
        # Load model
        model = load_model(model_name, model_version)
        
        # Preprocess features
        processed_features = preprocess_features(request.features, request.feature_names, model_name)
        
        # Make predictions
        preds = model.predict(processed_features)
        
        # Try to get probabilities if available
        probabilities = None
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_features).tolist()
        except Exception as e:
            logger.warning(f"Could not get probabilities: {str(e)}")
        
        # Convert predictions to list if needed
        if isinstance(preds, np.ndarray):
            predictions = preds.tolist()
        elif isinstance(preds, list):
            predictions = preds
        else:
            predictions = [preds]
        
        # Generate explanation if requested
        explanation = None
        if explain:
            try:
                model_explainer = explainer.get_explainer(model_name, model_version)
                explanation = model_explainer.explain(
                    processed_features, 
                    feature_names=request.feature_names,
                    output_format="json"
                )
            except Exception as e:
                logger.warning(f"Could not generate explanation: {str(e)}")
        
        # Calculate confidence (example: max probability for classification)
        confidence = None
        if probabilities and len(probabilities) > 0 and isinstance(probabilities[0], list):
            confidence = max(probabilities[0])
        
        # Record metrics
        prediction_time = time.time() - start_time
        PREDICTION_COUNT.labels(model_name=model_name, version=model_version).inc()
        PREDICTION_LATENCY.labels(model_name=model_name, version=model_version).observe(prediction_time)
        
        # Record A/B test metrics if available
        if ab_test:
            custom_metrics = {}
            if confidence is not None:
                custom_metrics["confidence"] = confidence
            
            ab_test.record_prediction(
                variant_name=variant['name'],
                prediction=predictions[0] if len(predictions) == 1 else predictions,
                latency=prediction_time,
                custom_metrics=custom_metrics
            )
        
        # Log prediction
        logger.info(f"Prediction made with model {model_name} (version: {model_version})")
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_name=model_name,
            model_version=model_version,
            prediction_time=prediction_time,
            explanation=explanation
        )
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/explain/{model_name}")
async def get_explanation(
    model_name: str,
    request: ExplanationRequest,
    model_version: Optional[str] = "latest"
):
    """Generate model explanations."""
    try:
        # Validate input data
        is_valid, errors = validate_input_data(request.features, request.feature_names, model_name)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Input validation failed: {', '.join(errors)}"
            )
        
        # Load model explainer
        model_explainer = explainer.get_explainer(model_name, model_version)
        
        # Preprocess features
        processed_features = preprocess_features(request.features, request.feature_names, model_name)
        
        # Generate explanation
        explanation = model_explainer.explain(
            processed_features,
            feature_names=request.feature_names,
            class_names=request.class_names,
            output_format=request.output_format,
            max_display=request.max_display
        )
        
        return explanation
    
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Explanation error: {str(e)}"
        )

@app.get("/feature-importance/{model_name}")
async def get_feature_importance(
    model_name: str,
    model_version: Optional[str] = "latest",
    feature_names: Optional[List[str]] = Query(None)
):
    """Get global feature importance for a model."""
    try:
        # Load model explainer
        model_explainer = explainer.get_explainer(model_name, model_version)
        
        # Get feature importance
        importance = model_explainer.get_feature_importance(feature_names)
        
        return importance
    
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Feature importance error: {str(e)}"
        )

@app.get("/models")
async def list_models():
    """List available models from MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        registered_models = client.search_registered_models()
        
        models = []
        for rm in registered_models:
            model_versions = client.get_latest_versions(rm.name)
            versions = [{"version": mv.version, "stage": mv.current_stage} for mv in model_versions]
            models.append({
                "name": rm.name,
                "versions": versions
            })
        
        return {"models": models}
    
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not list models: {str(e)}"
        )

@app.post("/ab-test")
async def create_ab_test(request: ABTestRequest):
    """Create an A/B test for model comparison."""
    try:
        test = ab_testing.ab_test_manager.create_test(
            name=request.name,
            variants=request.variants,
            traffic_split=request.traffic_split,
            custom_metrics=request.custom_metrics
        )
        
        return {"message": f"A/B test '{request.name}' created successfully"}
    
    except Exception as e:
        logger.error(f"Error creating A/B test: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating A/B test: {str(e)}"
        )

@app.get("/ab-test/{name}/results")
async def get_ab_test_results(name: str):
    """Get results from an A/B test."""
    try:
        test = ab_testing.ab_test_manager.get_test(name)
        if not test:
            raise HTTPException(
                status_code=404,
                detail=f"A/B test '{name}' not found"
            )
        
        return test.get_results()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting A/B test results: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting A/B test results: {str(e)}"
        )

@app.get("/ab-tests")
async def list_ab_tests():
    """List all A/B tests."""
    try:
        tests = ab_testing.ab_test_manager.list_tests()
        return {"tests": tests}
    
    except Exception as e:
        logger.error(f"Error listing A/B tests: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing A/B tests: {str(e)}"
        )

@app.delete("/ab-test/{name}")
async def delete_ab_test(name: str):
    """Delete an A/B test."""
    try:
        success = ab_testing.ab_test_manager.delete_test(name)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"A/B test '{name}' not found"
            )
        
        return {"message": f"A/B test '{name}' deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting A/B test: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting A/B test: {str(e)}"
        )

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 