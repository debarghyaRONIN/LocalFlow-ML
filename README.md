# LocalFlow-ML

A production-grade, local MLOps system that simulates real-world ML workflows - built, deployed, monitored, and tracked inside Minikube (a local Kubernetes cluster).

## Author
Debarghya Saha

## Core Features

| Function | Description |
|----------|-------------|
| Model Training & Tracking | MLflow to train ML models, log metrics, and manage model versions |
| CI/CD Pipeline | GitHub Actions to retrain and redeploy models automatically on each push |
| Model Deployment | FastAPI (Dockerized) to serve models as REST APIs inside Minikube |
| Data Drift Monitoring | EvidentlyAI to detect and report changes in data distribution |
| Performance Monitoring | Prometheus + Grafana to monitor API metrics (latency, accuracy, usage) |
| Logging | Loki + Grafana to view logs from deployed containers |
| Lineage Tracking | OpenLineage + DataHub to visualize data + model flow and dependencies |
| Artifact Storage | MinIO (S3-compatible) for storing datasets, trained models, etc. |
| Deployment Orchestration | Docker and Minikube (Kubernetes) to manage everything locally |

## Prerequisites

- Docker
- Minikube
- kubectl
- Python 3.8+
- Git

## Getting Started

1. Clone this repository
   ```bash
   git clone https://github.com/debarghyaRONIN/LocalFlow-ML.git
   cd LocalFlow-ML
   ```
2. Start Minikube cluster: `minikube start`
3. Deploy the infrastructure: `make deploy-infra`
4. Train a model: `make train-model`
5. Deploy the model: `make deploy-model`
6. Access the services:
   - MLflow UI: `minikube service mlflow`
   - FastAPI model endpoint: `minikube service model-api`
   - Grafana dashboard: `minikube service grafana`

## Project Structure

```
├── .github/            # GitHub Actions workflows for CI/CD
├── data/               # Sample datasets and data processing scripts
├── infrastructure/     # Kubernetes manifests for the infrastructure
├── models/             # Model training code and evaluation scripts
├── monitoring/         # Monitoring configuration (Prometheus, Grafana, etc.)
├── notebooks/          # Jupyter notebooks for exploration
├── pipelines/          # Training and deployment pipeline code
└── services/           # Microservices (FastAPI model serving, etc.)
```

## License

MIT 