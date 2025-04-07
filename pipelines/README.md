# MLOps Pipelines

This directory contains the ML model training and deployment pipelines.

## Directory Structure

- `training/` - Model training pipelines
- `deployment/` - Model deployment pipelines

## Training Pipeline

The `training/` directory contains scripts for:

- Data preparation
- Model training
- Model evaluation
- MLflow logging
- Model registration

To run the training pipeline:

```bash
python pipelines/training/train.py
```

### Command-line Arguments

The training pipeline accepts the following arguments:

- `--n-estimators` - Number of trees in the random forest (default: 100)
- `--max-depth` - Maximum depth of trees (default: 10)
- `--min-samples-split` - Minimum samples required to split (default: 2)
- `--min-samples-leaf` - Minimum samples required at leaf node (default: 1)
- `--random-state` - Random seed for reproducibility (default: 42)

## Deployment Pipeline

The `deployment/` directory contains scripts for packaging and deploying trained models.

To use the deployment pipeline:

```bash
# From the repository root directory
make deploy-model
```

This will:
1. Fetch the latest model from MLflow
2. Package it into a Docker container
3. Deploy it to Kubernetes

## Environment Variables

Both pipelines use the following environment variables:

- `MLFLOW_TRACKING_URI` - URI of the MLflow tracking server
- `MLFLOW_S3_ENDPOINT_URL` - URL of the MinIO S3-compatible storage
- `AWS_ACCESS_KEY_ID` - MinIO access key
- `AWS_SECRET_ACCESS_KEY` - MinIO secret key 