# MLOps Services

This directory contains the microservices for the MLOps platform.

## Model API Service

The `model-api/` directory contains a FastAPI service that serves trained models via a RESTful API.

### Directory Structure

- `app/` - Python application code
  - `main.py` - Main FastAPI application
  - `__init__.py` - Python package initialization
- `Dockerfile` - Docker container definition
- `requirements.txt` - Python dependencies

### API Endpoints

The Model API provides the following endpoints:

- `GET /` - API welcome message and links
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics endpoint
- `GET /models` - List all available models from MLflow
- `POST /predict/{model_name}` - Make predictions using the specified model

### Example Usage

```bash
# Get list of available models
curl -X GET http://localhost:8000/models

# Make predictions
curl -X POST http://localhost:8000/predict/iris-classifier \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'
```

### Building and Running

To build and run the Model API:

```bash
# Build the Docker image
docker build -t model-api:latest ./services/model-api

# Run locally
docker run -p 8000:8000 model-api:latest

# Deploy to Kubernetes
kubectl apply -f infrastructure/kubernetes/model-api.yaml
```

### Environment Variables

The Model API uses the following environment variables:

- `MLFLOW_TRACKING_URI` - URI of the MLflow tracking server
- `MLFLOW_S3_ENDPOINT_URL` - URL of the MinIO S3-compatible storage
- `AWS_ACCESS_KEY_ID` - MinIO access key
- `AWS_SECRET_ACCESS_KEY` - MinIO secret key 