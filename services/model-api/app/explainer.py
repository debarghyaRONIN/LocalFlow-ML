import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from typing import Dict, List, Union, Optional, Any, Tuple
import mlflow
from mlflow.pyfunc import PyFuncModel
import shap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class ModelExplainer:
    """
    Model explainability service using SHAP.
    """
    
    def __init__(self, model_name: str, model_version: str = "latest", 
                background_data_size: int = 100,
                cache_results: bool = True):
        """
        Initialize a model explainer.
        
        Args:
            model_name: Name of the model to explain
            model_version: Version of the model
            background_data_size: Number of background samples to use for SHAP
            cache_results: Whether to cache explainers
        """
        self.model_name = model_name
        self.model_version = model_version
        self.background_data_size = background_data_size
        self.cache_results = cache_results
        
        # Cache for explainers
        self.explainers = {}
        
        # Load the model
        self.model = self._load_model()
        
        # Initialize the explainer if we have a model
        if self.model:
            self._init_explainer()
    
    def _load_model(self) -> Optional[PyFuncModel]:
        """Load the model from MLflow."""
        try:
            logger.info(f"Loading model {self.model_name} (version: {self.model_version}) from MLflow")
            
            if self.model_version == "latest":
                model_uri = f"models:/{self.model_name}/latest"
            else:
                model_uri = f"models:/{self.model_name}/{self.model_version}"
            
            model = mlflow.pyfunc.load_model(model_uri)
            return model
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def _init_explainer(self) -> None:
        """Initialize the SHAP explainer."""
        try:
            # Get the underlying model
            if hasattr(self.model, '_model_impl'):
                underlying_model = self.model._model_impl
            else:
                logger.warning("Could not access underlying model implementation, using wrapper model")
                underlying_model = self.model
            
            # Load some background data from MLflow
            client = mlflow.tracking.MlflowClient()
            
            # Find the run that produced the model
            model_details = client.get_registered_model(self.model_name)
            versions = [v for v in model_details.latest_versions 
                      if v.version == self.model_version or 
                      (self.model_version == "latest" and v.current_stage == "Production")]
            
            if not versions:
                logger.warning(f"Could not find model version {self.model_version}, using default background data")
                # Create some random background data
                # This is not ideal but allows us to proceed
                background_data = np.random.rand(self.background_data_size, 4)  # Assuming 4 features
            else:
                run_id = versions[0].run_id
                
                # Get run info to find the dataset used for training
                run = client.get_run(run_id)
                dataset_path = run.data.params.get("dataset", None)
                
                if dataset_path:
                    # Try to load the dataset
                    try:
                        # This is a simplified approach, in practice you'd need to know the dataset format
                        df = pd.read_csv(dataset_path)
                        # Keep only feature columns
                        background_data = df.drop(['target', 'target_name'], axis=1, errors='ignore').values
                        # Limit to the specified number of background samples
                        if len(background_data) > self.background_data_size:
                            background_data = background_data[:self.background_data_size]
                    except Exception as e:
                        logger.warning(f"Could not load background data: {str(e)}")
                        background_data = np.random.rand(self.background_data_size, 4)
                else:
                    logger.warning("No dataset path found in run parameters")
                    background_data = np.random.rand(self.background_data_size, 4)
            
            # Determine the model type to choose the right explainer
            # This is a simplified approach, in practice, you would need more sophisticated detection
            if hasattr(underlying_model, 'predict_proba'):
                # For sklearn models with predict_proba (classifiers)
                logger.info("Using TreeExplainer for model with predict_proba")
                # Try TreeExplainer first
                try:
                    explainer = shap.TreeExplainer(underlying_model, background_data)
                    # Test the explainer
                    _ = explainer.shap_values(background_data[:1])
                    logger.info("TreeExplainer initialized successfully")
                except Exception as e:
                    logger.warning(f"TreeExplainer failed: {str(e)}, trying KernelExplainer")
                    explainer = shap.KernelExplainer(underlying_model.predict_proba, background_data)
            elif hasattr(underlying_model, 'predict'):
                # For sklearn models with predict (regressors)
                logger.info("Using TreeExplainer for model with predict")
                try:
                    explainer = shap.TreeExplainer(underlying_model, background_data)
                    # Test the explainer
                    _ = explainer.shap_values(background_data[:1])
                    logger.info("TreeExplainer initialized successfully")
                except Exception as e:
                    logger.warning(f"TreeExplainer failed: {str(e)}, trying KernelExplainer")
                    explainer = shap.KernelExplainer(underlying_model.predict, background_data)
            else:
                # For generic models, use KernelExplainer
                logger.info("Using KernelExplainer for generic model")
                explainer = shap.KernelExplainer(self.model.predict, background_data)
            
            # Store the explainer and background data
            self.explainers['default'] = {
                'explainer': explainer,
                'background_data': background_data
            }
            
            logger.info(f"Initialized SHAP explainer for model {self.model_name} (version: {self.model_version})")
        
        except Exception as e:
            logger.error(f"Error initializing explainer: {str(e)}")
    
    def explain(self, features: Union[List[List[float]], np.ndarray], 
                feature_names: Optional[List[str]] = None,
                class_names: Optional[List[str]] = None,
                output_format: str = 'json',
                max_display: int = 10) -> Dict[str, Any]:
        """
        Generate explanations for model predictions.
        
        Args:
            features: Features to explain (2D array)
            feature_names: Names of the features
            class_names: Names of the classes for classification
            output_format: Format of the output ('json' or 'html')
            max_display: Maximum number of features to display
            
        Returns:
            Dictionary with explanations
        """
        try:
            if 'default' not in self.explainers:
                return {'error': 'Explainer not initialized'}
            
            explainer = self.explainers['default']['explainer']
            
            # Convert features to numpy array
            if isinstance(features, list):
                features = np.array(features)
            
            # Generate SHAP values
            shap_values = explainer.shap_values(features)
            
            # Get base (expected) value
            if hasattr(explainer, 'expected_value'):
                expected_value = explainer.expected_value
                if isinstance(expected_value, np.ndarray) and len(expected_value) == 1:
                    expected_value = float(expected_value[0])
                elif isinstance(expected_value, list) and len(expected_value) == 1:
                    expected_value = float(expected_value[0])
            else:
                expected_value = 0.0
            
            # Handle multi-class output
            is_multiclass = isinstance(shap_values, list) and len(shap_values) > 1
            
            # Generate plots
            plots = {}
            if output_format == 'html' or output_format == 'both':
                plots = self._generate_plots(features, shap_values, feature_names, class_names, max_display)
            
            # Prepare feature names if not provided
            if not feature_names:
                feature_names = [f'feature_{i}' for i in range(features.shape[1])]
            
            # Prepare output based on model type
            if is_multiclass:
                # Multi-class classification
                class_names = class_names or [f'class_{i}' for i in range(len(shap_values))]
                
                # Format SHAP values for each class
                class_explanations = []
                for i, class_shap_values in enumerate(shap_values):
                    # For each class, get the explanation for each sample
                    sample_explanations = []
                    for j in range(features.shape[0]):
                        # Get feature impacts for this sample and class
                        feature_impacts = []
                        for k, feature_name in enumerate(feature_names):
                            feature_impacts.append({
                                'feature': feature_name,
                                'impact': float(class_shap_values[j][k]),
                                'value': float(features[j][k])
                            })
                        
                        # Sort by absolute impact
                        feature_impacts.sort(key=lambda x: abs(x['impact']), reverse=True)
                        
                        # Take only top features
                        if max_display > 0 and len(feature_impacts) > max_display:
                            feature_impacts = feature_impacts[:max_display]
                        
                        sample_explanations.append({
                            'sample_index': j,
                            'feature_impacts': feature_impacts,
                            'base_value': float(expected_value[i]) if isinstance(expected_value, (list, np.ndarray)) else float(expected_value)
                        })
                    
                    class_explanations.append({
                        'class': class_names[i],
                        'explanations': sample_explanations
                    })
                
                result = {
                    'model_name': self.model_name,
                    'model_version': self.model_version,
                    'is_multiclass': True,
                    'class_explanations': class_explanations
                }
            
            else:
                # Binary classification or regression
                sample_explanations = []
                for i in range(features.shape[0]):
                    # Get feature impacts for this sample
                    feature_impacts = []
                    for j, feature_name in enumerate(feature_names):
                        feature_impacts.append({
                            'feature': feature_name,
                            'impact': float(shap_values[i][j]),
                            'value': float(features[i][j])
                        })
                    
                    # Sort by absolute impact
                    feature_impacts.sort(key=lambda x: abs(x['impact']), reverse=True)
                    
                    # Take only top features
                    if max_display > 0 and len(feature_impacts) > max_display:
                        feature_impacts = feature_impacts[:max_display]
                    
                    sample_explanations.append({
                        'sample_index': i,
                        'feature_impacts': feature_impacts,
                        'base_value': float(expected_value) if not isinstance(expected_value, (list, np.ndarray)) else float(expected_value[0])
                    })
                
                result = {
                    'model_name': self.model_name,
                    'model_version': self.model_version,
                    'is_multiclass': False,
                    'explanations': sample_explanations
                }
            
            # Add plots if generated
            if plots:
                result['plots'] = plots
            
            return result
        
        except Exception as e:
            logger.error(f"Error generating explanations: {str(e)}")
            return {'error': str(e)}
    
    def _generate_plots(self, features: np.ndarray, shap_values: Union[np.ndarray, List[np.ndarray]],
                       feature_names: Optional[List[str]] = None,
                       class_names: Optional[List[str]] = None,
                       max_display: int = 10) -> Dict[str, str]:
        """
        Generate SHAP plots and return them as HTML.
        
        Args:
            features: Features to explain
            shap_values: SHAP values for the features
            feature_names: Names of the features
            class_names: Names of the classes
            max_display: Maximum number of features to display
            
        Returns:
            Dictionary mapping plot types to HTML representations
        """
        plots = {}
        try:
            # Set up feature names if provided
            if feature_names:
                plt.rc('xtick', labelsize=10)
                plt.rc('ytick', labelsize=10)
            
            # Summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values, 
                features,
                feature_names=feature_names,
                class_names=class_names,
                max_display=max_display,
                show=False
            )
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            plt.close()
            buffer.seek(0)
            plots['summary'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Force plot for the first sample
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # Multi-class, use the first class
                force_plot = shap.force_plot(
                    explainer.expected_value[0] if hasattr(explainer, 'expected_value') else 0,
                    shap_values[0][0],
                    features[0],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
            else:
                # Binary/regression
                force_plot = shap.force_plot(
                    explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                    shap_values[0],
                    features[0],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            plt.close()
            buffer.seek(0)
            plots['force'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return plots
        
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            return {'error': str(e)}
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get global feature importance for the model.
        
        Args:
            feature_names: Names of the features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            if 'default' not in self.explainers:
                return {'error': 'Explainer not initialized'}
            
            explainer = self.explainers['default']['explainer']
            background_data = self.explainers['default']['background_data']
            
            # Generate SHAP values for background data
            shap_values = explainer.shap_values(background_data)
            
            # Handle multi-class output
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # For multi-class, average across classes
                avg_shap_values = np.abs(np.array(shap_values)).mean(axis=0)
                # Get mean absolute SHAP value for each feature
                importance_scores = np.abs(avg_shap_values).mean(axis=0)
            else:
                # Get mean absolute SHAP value for each feature
                importance_scores = np.abs(shap_values).mean(axis=0)
            
            # Prepare feature names if not provided
            if not feature_names:
                feature_names = [f'feature_{i}' for i in range(len(importance_scores))]
            
            # Create a dictionary of feature importance
            importance_dict = {
                feature_names[i]: float(importance_scores[i])
                for i in range(len(importance_scores))
            }
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
            
            return importance_dict
        
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {'error': str(e)}

# Create a singleton instance for each model
explainers = {}

def get_explainer(model_name: str, model_version: str = "latest") -> ModelExplainer:
    """
    Get or create a model explainer.
    
    Args:
        model_name: Name of the model
        model_version: Version of the model
        
    Returns:
        ModelExplainer instance
    """
    key = f"{model_name}_{model_version}"
    
    if key not in explainers:
        explainers[key] = ModelExplainer(model_name, model_version)
    
    return explainers[key]

if __name__ == "__main__":
    # Example usage
    explainer = get_explainer("iris-classifier")
    
    # Example features (Iris setosa)
    features = [
        [5.1, 3.5, 1.4, 0.2]  # Iris-setosa
    ]
    
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    class_names = ['setosa', 'versicolor', 'virginica']
    
    # Generate explanation
    explanation = explainer.explain(features, feature_names, class_names)
    print(json.dumps(explanation, indent=2))
    
    # Get feature importance
    importance = explainer.get_feature_importance(feature_names)
    print(json.dumps(importance, indent=2)) 