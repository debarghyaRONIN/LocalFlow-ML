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
| Model Versioning & Promotion | Automated promotion across stages (Dev, Staging, Production) with rollback capability |
| Feature Store | Consistent feature engineering between training and inference |
| A/B Testing | Framework for comparing multiple model variants in production |
| Model Explainability | SHAP-based explanations for model predictions |
| Data Validation | Schema enforcement and statistical validation for training and inference data |
| Advanced Testing | Comprehensive unit, integration, and end-to-end tests |
| Complex Dataset Support | Support for multiple dataset types|

## Prerequisites

- Docker
- Minikube
- kubectl
- Python 3.8 - 3.11
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
├── data/
│   ├── feature_store/  # Feature storage and transformation
│   ├── validation/     # Data validation expectations and schemas
│   ├── raw/            # Raw input datasets
│   └── processed/      # Processed datasets ready for training
├── infrastructure/     # Kubernetes manifests for the infrastructure
├── models/             # Model training code and evaluation scripts
├── monitoring/         # Monitoring configuration (Prometheus, Grafana, etc.)
├── pipelines/
│   ├── training/       # Training pipeline code
│   ├── deployment/     # Deployment pipeline code
│   └── management/     # Model lifecycle management
├── services/
│   ├── model-api/      # Model serving API with explainability and A/B testing
│   └── training-api/   # API for triggering model retraining
└── tests/              # Comprehensive test suite
    ├── unit/           # Unit tests for components
    ├── integration/    # Integration tests for services
    └── e2e/            # End-to-end tests for the full pipeline
```

## Advanced Features

### Model Versioning and Promotion
The platform supports automatic versioning of models and a sophisticated promotion workflow:
- Development → Staging → Production
- Promotion based on performance metrics
- One-click rollback capability

### Feature Store
Ensures consistent feature transformations between training and inference:
- Version-controlled feature transformations
- Feature caching for performance
- Schema validation for features

### A/B Testing Framework
Compare multiple model variants in production:
- Traffic splitting with configurable weights
- Sticky sessions for consistent user experience
- Automatic performance tracking and reporting

### Model Explainability
Transparent ML with SHAP-based explanations:
- Feature importance visualization
- Individual prediction explanations
- Global model interpretation

### Data Validation
Ensures data quality throughout the ML lifecycle:
- Schema validation with Pydantic
- Statistical checks with Great Expectations
- Automatic data quality reporting

