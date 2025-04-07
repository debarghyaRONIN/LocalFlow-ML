# Infrastructure Directory

This directory contains the infrastructure-as-code components for the MLOps platform.

## Directory Structure

- `kubernetes/` - Kubernetes manifests for deploying all services

## Kubernetes Components

The `kubernetes/` directory contains manifests for:

- `namespace.yaml` - MLOps namespace definition
- `minio.yaml` - S3-compatible object storage for artifacts
- `mlflow.yaml` - Model tracking and registry service
- `prometheus.yaml` - Metrics collection and storage
- `grafana.yaml` - Dashboards and visualization
- `loki.yaml` - Log aggregation system
- `model-api.yaml` - Model serving API deployment
- `datahub.yaml` - Data lineage tracking

## Deploying Infrastructure

The infrastructure can be deployed using the Makefile target:

```bash
make deploy-infra
```

This will:
1. Start a Minikube cluster if not already running
2. Create the MLOps namespace
3. Deploy all infrastructure components
4. Wait for deployments to be ready

## Accessing Services

Services are exposed as NodePort services in Minikube. You can access them using:

```bash
# Show all services
make services

# Or access individually
minikube service mlflow -n mlops
minikube service grafana -n mlops
minikube service minio -n mlops
```

## Service Ports & Credentials

| Service    | Default Credentials |
|------------|---------------------|
| MinIO      | minio / minio123    |
| Grafana    | admin / admin       |
| MLflow     | N/A (no auth)       |
| Prometheus | N/A (no auth)       |

## Customizing Deployments

To customize the infrastructure:

1. Edit the YAML files in the `kubernetes/` directory
2. Apply changes using `kubectl apply -f <file>`
3. Or re-run `make deploy-infra` to redeploy everything 