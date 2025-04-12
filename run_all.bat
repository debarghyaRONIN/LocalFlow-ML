@echo off
echo ===================================
echo LocalFlow-ML Pipeline Runner
echo Author: Debarghya Saha
echo ===================================
echo.

:: Set environment variables
set MLFLOW_TRACKING_URI=http://localhost:5000
set MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
set AWS_ACCESS_KEY_ID=minio
set AWS_SECRET_ACCESS_KEY=minio123

echo Setting up environment...
:: Check if virtual environment exists, create if it doesn't
if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Start a local MLflow server (in background)
echo Starting local MLflow server...
start "MLflow Server" cmd /c "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts"
:: Wait for MLflow server to start
echo Waiting for MLflow server to start...
timeout /t 5 /nobreak > nul

:: Start Minikube if not already running
echo Checking Minikube status...
minikube status || (
    echo Starting Minikube...
    minikube start --memory=4096 --cpus=2 --disk-size=20g
)

:: Deploy infrastructure
echo Deploying MLOps infrastructure...
kubectl apply -f infrastructure/kubernetes/namespace.yaml
kubectl apply -f infrastructure/kubernetes/minio.yaml
kubectl apply -f infrastructure/kubernetes/mlflow.yaml
kubectl apply -f infrastructure/kubernetes/prometheus.yaml
kubectl apply -f infrastructure/kubernetes/grafana.yaml
kubectl apply -f infrastructure/kubernetes/loki.yaml
kubectl apply -f infrastructure/kubernetes/datahub.yaml

echo Waiting for infrastructure to be ready...
kubectl wait --for=condition=available deployment --all -n mlops --timeout=300s

:: Create directories for outputs
echo Creating directories for outputs...
if not exist models mkdir models
if not exist drift_reports mkdir drift_reports
if not exist mlflow-artifacts mkdir mlflow-artifacts

:: Train model
echo Training ML model...
python pipelines/training/train.py

:: Deploy model
echo Deploying model...
python pipelines/deployment/deploy.py

:: Run data drift monitoring
echo Running data drift detection...
python monitoring/evidently/drift_detection.py --drift-percent 0.2 --output-dir ./drift_reports

:: Display service access information
echo.
echo ===================================
echo Services Access Information:
echo ===================================
echo Local MLflow UI: http://localhost:5000
echo.
echo To access Kubernetes MLflow UI:
minikube service mlflow -n mlops --url
echo.
echo To access Model API:
minikube service model-api -n mlops --url
echo.
echo To access Grafana (credentials: admin/admin):
minikube service grafana -n mlops --url
echo.
echo To access MinIO (credentials: minio/minio123):
minikube service minio -n mlops --url
echo.
echo To access Prometheus:
minikube service prometheus -n mlops --url
echo.
echo To access DataHub (credentials: datahub@example.com/datahub):
minikube service datahub-frontend -n mlops --url
echo.
echo Run completed successfully!
echo.
echo NOTE: A local MLflow server is running in the background.
echo To stop it, close the MLflow server window or use task manager.
echo.

:: Deactivate virtual environment
call venv\Scripts\deactivate