# ML Training API

This service provides a REST API for uploading datasets, training machine learning models, and generating visualizations. It's designed to be used as a backend for desktop applications that need ML capabilities.

## Features

- Upload datasets in various formats (CSV, Excel, JSON, etc.)
- Train various machine learning models (classification and regression)
- Make predictions with trained models
- Generate visualizations for data exploration
- Track experiments with MLflow
- Background training with status updates

## API Endpoints

### Dataset Management

- `POST /datasets` - Upload a new dataset
- `GET /datasets/{dataset_id}` - Get dataset information

### Model Management

- `GET /models` - List all available models
- `POST /train` - Train a model on a dataset
- `GET /train/{job_id}` - Get training job status
- `POST /predict` - Make predictions using a trained model

### Visualization

- `POST /visualize` - Generate a visualization for a dataset

## Running the Service

### Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)
- MLflow tracking server (optional)

### Starting the Service

```bash
# From the repository root
cd services/training-api
python run.py
```

The service will be available at http://localhost:8000

### Environment Variables

- `PORT` - Port to run the service on (default: 8000)
- `MLFLOW_TRACKING_URI` - URI of MLflow tracking server
- `MLFLOW_EXPERIMENT_NAME` - Name of MLflow experiment
- `MLFLOW_S3_ENDPOINT_URL` - S3-compatible storage for MLflow
- `AWS_ACCESS_KEY_ID` - S3 access key
- `AWS_SECRET_ACCESS_KEY` - S3 secret key

## Example Usage

### Upload a Dataset

```bash
curl -X POST -F "file=@data.csv" http://localhost:8000/datasets
```

### Train a Model

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "dataset_id": "your-dataset-id",
  "model_name": "random_forest",
  "target_column": "target",
  "features": ["feature1", "feature2"],
  "model_parameters": {
    "n_estimators": 100,
    "max_depth": 10
  }
}' http://localhost:8000/train
```

### Get Training Status

```bash
curl -X GET http://localhost:8000/train/your-job-id
```

### Make Predictions

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "job_id": "your-job-id",
  "data": [
    {"feature1": 5.1, "feature2": 3.5},
    {"feature1": 4.9, "feature2": 3.0}
  ]
}' http://localhost:8000/predict
```

### Generate a Visualization

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "dataset_id": "your-dataset-id",
  "viz_type": "histogram",
  "columns": ["feature1", "feature2"]
}' http://localhost:8000/visualize
```

## Extending

### Adding New Models

To add a new model:

1. Add the model class to the `MODEL_REGISTRY` in `app/models.py`
2. Define its parameters and description

### Adding New Visualizations

To add a new visualization type:

1. Add a new method to generate the visualization in `app/visualization.py`
2. Add the visualization type to the conditional in `generate_visualization()`

## Desktop Application Integration

When building a desktop application that uses this API:

1. Start by uploading a dataset and storing the dataset_id
2. Allow users to select from available models
3. Start training and monitor the job status
4. Once training is complete, display metrics and visualizations
5. Enable predictions on new data using the trained model 