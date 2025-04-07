.PHONY: setup minikube-start minikube-stop deploy-infra train-model deploy-model clean all

# Default target
all: setup deploy-infra train-model deploy-model

# Setup environment
setup:
	@echo "Setting up environment..."
	pip install -r requirements.txt

# Minikube commands
minikube-start:
	@echo "Starting Minikube cluster..."
	minikube start --memory=4096 --cpus=2 --disk-size=20g

minikube-stop:
	@echo "Stopping Minikube cluster..."
	minikube stop

# Infrastructure deployment
deploy-infra: minikube-start
	@echo "Deploying MLOps infrastructure..."
	kubectl apply -f infrastructure/kubernetes/namespace.yaml
	kubectl apply -f infrastructure/kubernetes/minio.yaml
	kubectl apply -f infrastructure/kubernetes/mlflow.yaml
	kubectl apply -f infrastructure/kubernetes/prometheus.yaml
	kubectl apply -f infrastructure/kubernetes/grafana.yaml
	kubectl apply -f infrastructure/kubernetes/loki.yaml
	kubectl apply -f infrastructure/kubernetes/datahub.yaml
	@echo "Waiting for infrastructure to be ready..."
	kubectl wait --for=condition=available deployment --all -n mlops --timeout=300s
	@echo "Infrastructure deployed successfully!"

# Model training
train-model:
	@echo "Training ML model..."
	python pipelines/training/train.py

# Model deployment
deploy-model:
	@echo "Deploying model API..."
	python pipelines/deployment/deploy.py
	@echo "Model API deployed!"

# Clean up
clean:
	@echo "Cleaning up..."
	kubectl delete namespace mlops
	minikube stop

# Show available services
services:
	@echo "MLflow UI: $(shell minikube service mlflow -n mlops --url)"
	@echo "Model API: $(shell minikube service model-api -n mlops --url)"
	@echo "Grafana: $(shell minikube service grafana -n mlops --url)"
	@echo "MinIO: $(shell minikube service minio -n mlops --url)"
	@echo "DataHub: $(shell minikube service datahub-frontend -n mlops --url)" 