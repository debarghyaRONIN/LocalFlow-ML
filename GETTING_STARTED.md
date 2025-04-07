# Getting Started with LocalFlow-ML

This guide will help you get started with our local MLOps pipeline using Minikube.

## Author
Debarghya Saha

## Prerequisites

Ensure you have the following installed:

1. **Docker** - [Install Docker](https://docs.docker.com/get-docker/)
2. **Minikube** - [Install Minikube](https://minikube.sigs.k8s.io/docs/start/)
3. **kubectl** - [Install kubectl](https://kubernetes.io/docs/tasks/tools/)
4. **Python 3.8+** - [Install Python](https://www.python.org/downloads/)
5. **Git** - [Install Git](https://git-scm.com/downloads)
6. **Make** - [Install Make](https://www.gnu.org/software/make/) (optional, but recommended)

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
conda create -n mlops python=3.10
conda activate mlops
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

## Starting the MLOps infrastructure

1. **Start Minikube**:

```bash
make minikube-start

# Or without make:
minikube start --memory=4096 --cpus=2 --disk-size=20g
```

2. **Deploy the MLOps infrastructure**:

```bash
make deploy-infra

# Or without make:
kubectl apply -f infrastructure/kubernetes/namespace.yaml
kubectl apply -f infrastructure/kubernetes/minio.yaml
kubectl apply -f infrastructure/kubernetes/mlflow.yaml
kubectl apply -f infrastructure/kubernetes/prometheus.yaml
kubectl apply -f infrastructure/kubernetes/grafana.yaml
kubectl apply -f infrastructure/kubernetes/loki.yaml
```

3. **Verify services are running**:

```bash
make services

# Or without make:
kubectl get pods -n mlops
kubectl get services -n mlops
```

## Training and deploying a model

1. **Train a model**:

```bash
make train-model

# Or without make:
python pipelines/training/train.py
```

2. **Deploy the model**:

```bash
make deploy-model

# Or without make:
python pipelines/deployment/deploy.py
```

3. **Test the model API**:

```bash
# Get the model API URL
minikube service model-api -n mlops --url

# Make a prediction
curl -X POST http://<model-api-url>/predict/iris-classifier \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'
```

## Accessing MLOps services

- **MLflow**: `minikube service mlflow -n mlops`
- **Grafana**: `minikube service grafana -n mlops` (credentials: admin/admin)
- **MinIO**: `minikube service minio -n mlops` (credentials: minio/minio123)
- **Prometheus**: `minikube service prometheus -n mlops`

## Monitoring data drift

To run data drift detection:

```bash
python monitoring/evidently/drift_detection.py --drift-percent 0.2 --output-dir ./drift_reports
```

## Running tests

```bash
python -m pytest
```

## Customization

- Edit `pipelines/training/train.py` to use your own dataset and model
- Modify `services/model-api/app/main.py` to customize the API
- Update Kubernetes manifests in `infrastructure/kubernetes/` for your needs

## Stopping and cleaning up

```bash
make clean

# Or without make:
kubectl delete namespace mlops
minikube stop
```

## Next Steps

- Integrate additional ML models
- Set up CI/CD with GitHub Actions
- Add authentication to services
- Implement A/B testing
- Explore advanced monitoring with EvidentlyAI 