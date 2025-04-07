# Data Directory

This directory contains the datasets used for training and evaluating models in the MLOps pipeline.

## Structure

- `raw/` - Raw data files before preprocessing
- `processed/` - Cleaned and preprocessed data ready for model training

## Data Sources

For the sample Iris classifier, we use the built-in Iris dataset from scikit-learn, so no external data files are required.

In a real project, you might store:

- CSV, JSON, or Parquet files
- Database dumps
- Data extraction scripts
- Data documentation

## Data Versioning

Data changes should be tracked alongside code. Consider using:

1. DVC (Data Version Control) for larger datasets
2. MinIO with versioning enabled for object storage
3. Git LFS for smaller datasets

## Data Flow

1. Raw data is stored in the `raw/` directory
2. Data preprocessing transforms raw data into processed data
3. Processed data is used for model training
4. Data drift is monitored using the Evidently AI service

## Manual Data Handling

If you need to manually upload data files to the MinIO storage:

```bash
# Start port-forwarding to access MinIO
kubectl port-forward svc/minio -n mlops 9000:9000 9001:9001

# Then open in browser:
# http://localhost:9001

# Login with:
# Username: minio
# Password: minio123
``` 