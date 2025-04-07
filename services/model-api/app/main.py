import os
import time
import json
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

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

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global model cache
model_cache = {}


# Data models
class PredictionRequest(BaseModel):
    features: List[List[float]] = Field(..., example=[[5.1, 3.5, 1.4, 0.2]])


class PredictionResponse(BaseModel):
    predictions: List[Union[int, float, str]] = Field(..., example=[0])
    model_name: str
    model_version: str
    prediction_time: float


@app.get("/")
async def root():
    return {"message": "Welcome to MLOps Model API", "docs_url": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


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


@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict(model_name: str, request: PredictionRequest, model_version: Optional[str] = "latest"):
    """Make predictions using a trained model."""
    start_time = time.time()
    
    try:
        model = load_model(model_name, model_version)
        
        # Convert features to numpy array or pandas DataFrame as needed
        features = np.array(request.features)
        
        # Make predictions
        predictions = model.predict(features).tolist()
        
        # Record metrics
        prediction_time = time.time() - start_time
        PREDICTION_COUNT.labels(model_name=model_name, version=model_version).inc()
        PREDICTION_LATENCY.labels(model_name=model_name, version=model_version).observe(prediction_time)
        
        # Log prediction
        logger.info(f"Prediction made with model {model_name} (version: {model_version})")
        
        return PredictionResponse(
            predictions=predictions,
            model_name=model_name,
            model_version=model_version,
            prediction_time=prediction_time
        )
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
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