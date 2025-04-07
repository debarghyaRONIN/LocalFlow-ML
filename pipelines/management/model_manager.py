#!/usr/bin/env python
import os
import sys
import argparse
import logging
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Valid model stages
VALID_STAGES = ["None", "Development", "Staging", "Production", "Archived"]

def setup_mlflow():
    """Set up MLflow client."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        logger.info(f"MLflow client initialized with tracking URI: {MLFLOW_TRACKING_URI}")
        return client
    except Exception as e:
        logger.error(f"Error initializing MLflow client: {str(e)}")
        sys.exit(1)

def list_models(client):
    """List all registered models."""
    try:
        models = client.search_registered_models()
        if not models:
            logger.info("No registered models found.")
            return []
        
        logger.info(f"Found {len(models)} registered models:")
        for model in models:
            logger.info(f"  - {model.name}")
            latest_versions = client.get_latest_versions(model.name)
            for version in latest_versions:
                logger.info(f"      Version: {version.version}, Stage: {version.current_stage}")
        
        return models
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return []

def get_latest_version(client, model_name, stage=None):
    """Get the latest version of a model, optionally filtered by stage."""
    try:
        if stage and stage not in VALID_STAGES:
            logger.error(f"Invalid stage: {stage}. Must be one of {VALID_STAGES}")
            return None
        
        filters = {"name": model_name}
        if stage:
            latest_versions = [v for v in client.get_latest_versions(model_name) if v.current_stage == stage]
        else:
            latest_versions = client.get_latest_versions(model_name)
        
        if not latest_versions:
            logger.warning(f"No versions found for model {model_name}" + 
                         (f" in stage {stage}" if stage else ""))
            return None
        
        # Sort by version number to get the latest
        latest_version = sorted(latest_versions, key=lambda x: int(x.version), reverse=True)[0]
        logger.info(f"Latest version for {model_name}" +
                  (f" in stage {stage}" if stage else "") + 
                  f": Version {latest_version.version} (Stage: {latest_version.current_stage})")
        
        return latest_version
    except Exception as e:
        logger.error(f"Error getting latest version: {str(e)}")
        return None

def promote_model(client, model_name, version, to_stage, from_stage=None):
    """Promote a model version to a new stage."""
    try:
        if to_stage not in VALID_STAGES:
            logger.error(f"Invalid target stage: {to_stage}. Must be one of {VALID_STAGES}")
            return False
        
        # Check if version exists
        try:
            model_version = client.get_model_version(model_name, version)
        except MlflowException:
            logger.error(f"Model version {version} not found for model {model_name}")
            return False
        
        # Check if it's in the correct source stage
        if from_stage and model_version.current_stage != from_stage:
            logger.error(f"Model {model_name} version {version} is in stage {model_version.current_stage}, not {from_stage}")
            return False
        
        # Promote the model
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=to_stage
        )
        
        logger.info(f"Promoted {model_name} version {version} from {model_version.current_stage} to {to_stage}")
        
        # If promoting to Production, archive any other Production models
        if to_stage == "Production":
            production_versions = [v for v in client.get_latest_versions(model_name) 
                                if v.current_stage == "Production" and v.version != version]
            
            for prod_version in production_versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=prod_version.version,
                    stage="Archived"
                )
                logger.info(f"Archived previous production model: {model_name} version {prod_version.version}")
        
        return True
    except Exception as e:
        logger.error(f"Error promoting model: {str(e)}")
        return False

def rollback_model(client, model_name, to_version=None):
    """Rollback the Production model to a previous version or to the current Staging model."""
    try:
        # If version not specified, use the current Staging model
        if not to_version:
            staging_version = get_latest_version(client, model_name, stage="Staging")
            if not staging_version:
                logger.error(f"No Staging version found for model {model_name}")
                return False
            to_version = staging_version.version
        
        # Get current Production model to archive it
        production_version = get_latest_version(client, model_name, stage="Production")
        
        # Promote the target version to Production
        if promote_model(client, model_name, to_version, "Production"):
            # Archive the old production version if it exists
            if production_version and production_version.version != to_version:
                promote_model(client, model_name, production_version.version, "Archived")
                logger.info(f"Rollback successful: {model_name} version {to_version} is now in Production, old version {production_version.version} archived")
            else:
                logger.info(f"Rollback successful: {model_name} version {to_version} is now in Production")
            return True
        
        return False
    except Exception as e:
        logger.error(f"Error rolling back model: {str(e)}")
        return False

def compare_models(client, model_name, version1, version2):
    """Compare metrics between two model versions."""
    try:
        # Get model versions
        try:
            model_version1 = client.get_model_version(model_name, version1)
            model_version2 = client.get_model_version(model_name, version2)
        except MlflowException as e:
            logger.error(f"Error retrieving model versions: {str(e)}")
            return False
        
        # Get run data
        run1 = client.get_run(model_version1.run_id)
        run2 = client.get_run(model_version2.run_id)
        
        # Compare metrics
        metrics1 = run1.data.metrics
        metrics2 = run2.data.metrics
        
        logger.info(f"Comparing {model_name} versions {version1} vs {version2}:")
        logger.info(f"  Version {version1} (Stage: {model_version1.current_stage}):")
        for key, value in metrics1.items():
            logger.info(f"    {key}: {value}")
        
        logger.info(f"  Version {version2} (Stage: {model_version2.current_stage}):")
        for key, value in metrics2.items():
            logger.info(f"    {key}: {value}")
        
        # Determine if version1 is better based on key metrics
        better_metrics = 0
        total_compared = 0
        
        for key in metrics1:
            if key in metrics2:
                total_compared += 1
                # For error metrics (lower is better)
                if "error" in key.lower() or "loss" in key.lower() or key.startswith("mse") or key.startswith("rmse"):
                    if metrics1[key] < metrics2[key]:
                        better_metrics += 1
                # For other metrics (higher is better)
                else:
                    if metrics1[key] > metrics2[key]:
                        better_metrics += 1
        
        if total_compared > 0:
            logger.info(f"Version {version1} is better in {better_metrics}/{total_compared} metrics")
            
        return True
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        return False

def auto_promote_best_model(client, model_name, to_stage, metric_name, higher_is_better=True):
    """Automatically promote the best model based on a specific metric."""
    try:
        # Get all versions in Development stage
        versions = [v for v in client.get_latest_versions(model_name) if v.current_stage == "Development"]
        
        if not versions:
            logger.warning(f"No Development versions found for model {model_name}")
            return False
        
        # Get metrics for each version
        version_metrics = []
        for version in versions:
            run = client.get_run(version.run_id)
            metrics = run.data.metrics
            
            if metric_name in metrics:
                version_metrics.append((version.version, metrics[metric_name]))
        
        if not version_metrics:
            logger.warning(f"No versions found with metric {metric_name}")
            return False
        
        # Find the best version
        best_version, best_value = sorted(
            version_metrics, 
            key=lambda x: x[1], 
            reverse=higher_is_better
        )[0]
        
        logger.info(f"Best version based on {metric_name}: Version {best_version} with value {best_value}")
        
        # Promote the best version
        return promote_model(client, model_name, best_version, to_stage)
    except Exception as e:
        logger.error(f"Error auto-promoting best model: {str(e)}")
        return False

def main():
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(description="MLflow Model Management")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List registered models")
    
    # Promote model command
    promote_parser = subparsers.add_parser("promote", help="Promote a model version to a new stage")
    promote_parser.add_argument("--model-name", type=str, required=True, help="Name of the registered model")
    promote_parser.add_argument("--version", type=str, required=True, help="Version of the model to promote")
    promote_parser.add_argument("--to-stage", type=str, required=True, choices=VALID_STAGES, 
                               help="Target stage for promotion")
    promote_parser.add_argument("--from-stage", type=str, choices=VALID_STAGES,
                               help="Source stage (if specified, will only promote if model is in this stage)")
    
    # Rollback model command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to a previous model version")
    rollback_parser.add_argument("--model-name", type=str, required=True, help="Name of the registered model")
    rollback_parser.add_argument("--to-version", type=str, help="Version to rollback to (if not specified, uses Staging)")
    
    # Compare models command
    compare_parser = subparsers.add_parser("compare", help="Compare metrics between two model versions")
    compare_parser.add_argument("--model-name", type=str, required=True, help="Name of the registered model")
    compare_parser.add_argument("--version1", type=str, required=True, help="First version to compare")
    compare_parser.add_argument("--version2", type=str, required=True, help="Second version to compare")
    
    # Auto-promote command
    auto_parser = subparsers.add_parser("auto-promote", help="Automatically promote the best model based on a metric")
    auto_parser.add_argument("--model-name", type=str, required=True, help="Name of the registered model")
    auto_parser.add_argument("--to-stage", type=str, required=True, choices=["Staging", "Production"],
                            help="Target stage for promotion")
    auto_parser.add_argument("--metric", type=str, required=True, 
                            help="Metric to use for comparison (e.g., 'accuracy' or 'mse')")
    auto_parser.add_argument("--higher-is-better", action="store_true", default=True,
                            help="Whether higher metric values are better (default: True)")
    auto_parser.add_argument("--lower-is-better", action="store_true", default=False,
                            help="Whether lower metric values are better")
    
    args = parser.parse_args()
    
    # Initialize MLflow client
    client = setup_mlflow()
    
    # Execute the appropriate command
    if args.command == "list":
        list_models(client)
    
    elif args.command == "promote":
        success = promote_model(client, args.model_name, args.version, args.to_stage, args.from_stage)
        sys.exit(0 if success else 1)
    
    elif args.command == "rollback":
        success = rollback_model(client, args.model_name, args.to_version)
        sys.exit(0 if success else 1)
    
    elif args.command == "compare":
        success = compare_models(client, args.model_name, args.version1, args.version2)
        sys.exit(0 if success else 1)
    
    elif args.command == "auto-promote":
        higher_is_better = not args.lower_is_better if args.lower_is_better else args.higher_is_better
        success = auto_promote_best_model(client, args.model_name, args.to_stage, args.metric, higher_is_better)
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 