# Model Exploration Notebooks

This directory contains Jupyter notebooks for exploring the datasets and models used in the MLOps pipeline.

## Using the Notebooks

1. Start your Minikube cluster:
   ```
   minikube start
   ```

2. Install Jupyter and required dependencies:
   ```
   pip install jupyter notebook
   pip install -r ../../requirements.txt
   ```

3. Start Jupyter notebook server:
   ```
   jupyter notebook
   ```

4. Create a new notebook or use the provided notebook templates.

## Available Notebooks

- `iris_model_exploration.ipynb` - Exploration of the Iris dataset and RandomForest model
- `model_drift_analysis.ipynb` - Analysis of data drift using Evidently AI

## Connecting to MLflow

When working with MLflow in a notebook, you'll need to set the tracking URI to point to your MLflow service running in Minikube:

```python
import mlflow

# Get Minikube IP
import subprocess
minikube_ip = subprocess.check_output(['minikube', 'ip']).decode('utf-8').strip()

# Set MLflow tracking URI
mlflow_port = '30500'  # This might be different - check with `kubectl get svc mlflow -n mlops`
mlflow.set_tracking_uri(f"http://{minikube_ip}:{mlflow_port}")
```

This will allow your notebook to log experiments to the MLflow server running in your Kubernetes cluster. 