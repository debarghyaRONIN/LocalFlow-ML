# Running LocalFlow-ML Without Kubernetes

This guide provides instructions for running components of the LocalFlow-ML project directly on your local machine without Kubernetes, which can be helpful for development and testing.

## Author
Debarghya Saha

## Prerequisites

- Python 3.8+
- Git
- Docker (optional, but recommended for some components)

## Setup

1. **Clone the repository**:

```bash
git clone https://github.com/debarghyaRONIN/LocalFlow-ML.git
cd LocalFlow-ML
```

2. **Create a Python virtual environment**:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n localflow python=3.10
conda activate localflow
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

## Running Components Locally

### MLflow Server

Run MLflow locally for experiment tracking:

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts
```

### Training a Model

Train a model with MLflow tracking:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python pipelines/training/train.py
```

### Model API Service

Run the model API service locally:

```bash
cd services/model-api
pip install -r requirements.txt
export MLFLOW_TRACKING_URI=http://localhost:5000
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Data Drift Monitoring

Run data drift detection:

```bash
python monitoring/evidently/drift_detection.py --drift-percent 0.2 --output-dir ./drift_reports
```

## Using Docker (Optional)

### MLflow Server

```bash
docker run -p 5000:5000 -v $(pwd)/mlflow-data:/mlflow-data ghcr.io/mlflow/mlflow:v2.7.1 mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow-data/mlflow.db --default-artifact-root ./mlflow-data/artifacts
```

### Model API Service

```bash
cd services/model-api
docker build -t model-api:local .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 model-api:local
```

## Accessing Services

- **MLflow UI**: http://localhost:5000
- **Model API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Tips for Local Development

1. **Environment Variables**: Set these for proper configuration:
   ```bash
   export MLFLOW_TRACKING_URI=http://localhost:5000
   export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 # If using MinIO locally
   export AWS_ACCESS_KEY_ID=minio
   export AWS_SECRET_ACCESS_KEY=minio123
   ```

2. **Use SQLite for Development**: For simplicity, use SQLite as the database backend for MLflow during development.

3. **Mock Services**: When developing specific components, consider mocking external services for faster testing.

## Troubleshooting

- **Connection Issues**: Ensure all services are running and ports are not blocked by firewalls or other applications.
- **Missing Dependencies**: If you encounter module not found errors, install the missing dependencies with pip.
- **Permissions Issues**: Check file permissions for data directories and log files. 