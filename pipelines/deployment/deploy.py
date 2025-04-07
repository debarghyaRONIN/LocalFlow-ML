import os
import sys
import argparse
import logging
import subprocess
import tempfile
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def get_latest_model(model_name="iris-classifier"):
    """Get the latest version of a registered model from MLflow."""
    logger.info(f"Getting latest version of model: {model_name}")
    
    client = MlflowClient()
    latest_versions = client.get_latest_versions(model_name)
    
    if not latest_versions:
        raise ValueError(f"No versions found for model: {model_name}")
    
    # Get the latest version in Production or Staging, or just the latest version
    production_versions = [v for v in latest_versions if v.current_stage == "Production"]
    staging_versions = [v for v in latest_versions if v.current_stage == "Staging"]
    
    if production_versions:
        latest_version = production_versions[0]
    elif staging_versions:
        latest_version = staging_versions[0]
    else:
        # Sort by version number and get the latest
        latest_version = sorted(latest_versions, key=lambda x: int(x.version), reverse=True)[0]
    
    logger.info(f"Latest model version: {latest_version.version} (stage: {latest_version.current_stage})")
    return latest_version

def build_model_api_image(model_name, model_version, image_tag="latest"):
    """Build Docker image for model API."""
    logger.info(f"Building Docker image for model: {model_name} (version: {model_version})")
    
    docker_command = [
        "docker", "build", 
        "-t", f"model-api:{image_tag}",
        "--build-arg", f"MODEL_NAME={model_name}",
        "--build-arg", f"MODEL_VERSION={model_version}",
        "./services/model-api"
    ]
    
    try:
        subprocess.run(docker_command, check=True)
        logger.info(f"Successfully built Docker image: model-api:{image_tag}")
        return f"model-api:{image_tag}"
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build Docker image: {str(e)}")
        raise

def deploy_to_kubernetes(namespace="mlops"):
    """Deploy the model API to Kubernetes."""
    logger.info(f"Deploying model API to Kubernetes namespace: {namespace}")
    
    try:
        # Apply Kubernetes deployment manifest
        subprocess.run(
            ["kubectl", "apply", "-f", "infrastructure/kubernetes/model-api.yaml"],
            check=True
        )
        
        # Wait for deployment to be ready
        subprocess.run(
            ["kubectl", "rollout", "status", "deployment/model-api", "-n", namespace],
            check=True
        )
        
        logger.info("Model API deployed successfully")
        
        # Get service URL
        try:
            service_url = subprocess.check_output(
                ["minikube", "service", "model-api", "-n", namespace, "--url"],
                text=True
            ).strip()
            logger.info(f"Model API available at: {service_url}")
        except subprocess.CalledProcessError:
            logger.warning("Could not get service URL")
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise

def main():
    """Main function for the deployment pipeline."""
    parser = argparse.ArgumentParser(description="Deploy ML model to Kubernetes")
    parser.add_argument("--model-name", type=str, default="iris-classifier",
                       help="Name of the registered model")
    parser.add_argument("--image-tag", type=str, default="latest",
                       help="Tag for the Docker image")
    parser.add_argument("--namespace", type=str, default="mlops",
                       help="Kubernetes namespace for deployment")
    
    args = parser.parse_args()
    
    try:
        # Get latest model version
        model_version = get_latest_model(args.model_name)
        
        # Build Docker image
        image_name = build_model_api_image(
            args.model_name, 
            model_version.version,
            args.image_tag
        )
        
        # Deploy to Kubernetes
        deploy_to_kubernetes(args.namespace)
        
        logger.info("Deployment pipeline completed successfully")
    
    except Exception as e:
        logger.error(f"Deployment pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 