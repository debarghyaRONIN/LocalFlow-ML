import os
import time
import json
import logging
import random
from typing import Dict, List, Any, Tuple, Optional
import threading
import numpy as np
import pandas as pd
import mlflow
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prometheus metrics
AB_TEST_ASSIGNMENTS = Counter('ab_test_assignments', 'Count of A/B test assignments', 
                               ['test_name', 'model_name', 'variant'])
AB_TEST_PREDICTIONS = Counter('ab_test_predictions', 'Count of predictions made in A/B tests', 
                               ['test_name', 'model_name', 'variant'])
AB_TEST_PREDICTION_LATENCY = Histogram('ab_test_prediction_latency_seconds', 
                                        'Time for prediction in A/B tests', 
                                        ['test_name', 'model_name', 'variant'])
AB_TEST_PREDICTION_VALUE = Histogram('ab_test_prediction_value', 
                                      'Distribution of prediction values', 
                                      ['test_name', 'model_name', 'variant'])
AB_TEST_METRICS = Gauge('ab_test_metrics', 'Custom metrics for A/B tests',
                         ['test_name', 'model_name', 'variant', 'metric'])

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class ABTest:
    """
    A/B test for comparing multiple model variants.
    """
    
    def __init__(self, name: str, variants: List[Dict[str, Any]], 
                 traffic_split: Optional[List[float]] = None,
                 sticky_sessions: bool = True,
                 sticky_session_ttl: int = 3600,  # 1 hour
                 custom_metrics: List[str] = None):
        """
        Initialize an A/B test.
        
        Args:
            name: Name of the A/B test
            variants: List of variant configurations, each containing:
                      - 'name': Name of the variant
                      - 'model_name': Name of the model to use
                      - 'model_version': Version of the model to use (optional)
            traffic_split: List of traffic proportions for each variant (must sum to 1.0)
            sticky_sessions: Whether to use sticky sessions (same user gets same variant)
            sticky_session_ttl: Time-to-live for sticky sessions in seconds
            custom_metrics: List of custom metrics to track for this A/B test
        """
        self.name = name
        self.variants = variants
        
        # Validate variants
        if not variants or len(variants) < 2:
            raise ValueError("At least two variants are required for an A/B test")
        
        for variant in variants:
            if 'name' not in variant or 'model_name' not in variant:
                raise ValueError("Each variant must have a 'name' and 'model_name'")
            
            # Add default model version if not specified
            if 'model_version' not in variant:
                variant['model_version'] = 'latest'
        
        # Set traffic split
        if traffic_split is None:
            # Equal split by default
            self.traffic_split = [1.0 / len(variants) for _ in variants]
        else:
            if len(traffic_split) != len(variants):
                raise ValueError("Traffic split must have the same length as variants")
            
            if abs(sum(traffic_split) - 1.0) > 0.0001:
                raise ValueError("Traffic split must sum to 1.0")
            
            self.traffic_split = traffic_split
        
        # Set up cumulative split for fast assignment
        self.cumulative_split = np.cumsum(self.traffic_split)
        
        # Set up sticky sessions
        self.sticky_sessions = sticky_sessions
        self.sticky_session_ttl = sticky_session_ttl
        self.session_assignments = {}  # user_id -> (variant_index, timestamp)
        
        # Set up metrics tracking
        self.custom_metrics = custom_metrics or []
        
        # Set up results storage
        self.results = {
            variant['name']: {
                'predictions': [],
                'latencies': [],
                'custom_metrics': {metric: [] for metric in self.custom_metrics}
            } for variant in variants
        }
        
        # Set up a lock for thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Initialized A/B test '{name}' with {len(variants)} variants")
        for i, (variant, split) in enumerate(zip(variants, self.traffic_split)):
            logger.info(f"  Variant {i+1}: {variant['name']} ({variant['model_name']} v{variant['model_version']}) - {split*100:.1f}% traffic")
    
    def get_variant(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the variant for a user.
        
        Args:
            user_id: User ID for sticky sessions
            
        Returns:
            Variant configuration
        """
        with self.lock:
            # If sticky sessions are enabled and user_id is provided
            if self.sticky_sessions and user_id:
                # Check if user already has an assignment
                if user_id in self.session_assignments:
                    variant_index, timestamp = self.session_assignments[user_id]
                    
                    # Check if the assignment is still valid
                    if time.time() - timestamp <= self.sticky_session_ttl:
                        # Update timestamp
                        self.session_assignments[user_id] = (variant_index, time.time())
                        
                        # Return the assigned variant
                        variant = self.variants[variant_index]
                        AB_TEST_ASSIGNMENTS.labels(
                            test_name=self.name,
                            model_name=variant['model_name'],
                            variant=variant['name']
                        ).inc()
                        return variant
            
            # Assign a new variant based on traffic split
            random_value = random.random()
            variant_index = 0
            
            for i, threshold in enumerate(self.cumulative_split):
                if random_value <= threshold:
                    variant_index = i
                    break
            
            # Save assignment if sticky sessions are enabled
            if self.sticky_sessions and user_id:
                self.session_assignments[user_id] = (variant_index, time.time())
            
            # Return the selected variant
            variant = self.variants[variant_index]
            AB_TEST_ASSIGNMENTS.labels(
                test_name=self.name,
                model_name=variant['model_name'],
                variant=variant['name']
            ).inc()
            return variant
    
    def record_prediction(self, variant_name: str, prediction: Any, latency: float,
                         custom_metrics: Dict[str, float] = None) -> None:
        """
        Record prediction and metrics for a variant.
        
        Args:
            variant_name: Name of the variant
            prediction: Prediction result
            latency: Prediction latency in seconds
            custom_metrics: Dictionary of custom metrics
        """
        with self.lock:
            if variant_name not in self.results:
                logger.warning(f"Unknown variant '{variant_name}' in A/B test '{self.name}'")
                return
            
            # Get variant info for Prometheus labels
            variant_info = next((v for v in self.variants if v['name'] == variant_name), None)
            if not variant_info:
                logger.warning(f"Could not find variant info for '{variant_name}'")
                return
            
            model_name = variant_info['model_name']
            
            # Record prediction
            self.results[variant_name]['predictions'].append(prediction)
            self.results[variant_name]['latencies'].append(latency)
            
            # Update Prometheus metrics
            AB_TEST_PREDICTIONS.labels(
                test_name=self.name,
                model_name=model_name,
                variant=variant_name
            ).inc()
            
            AB_TEST_PREDICTION_LATENCY.labels(
                test_name=self.name,
                model_name=model_name,
                variant=variant_name
            ).observe(latency)
            
            # Record prediction value if it's a number
            if isinstance(prediction, (int, float)) or (
                    isinstance(prediction, (list, np.ndarray)) and 
                    len(prediction) == 1 and 
                    isinstance(prediction[0], (int, float))):
                value = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
                AB_TEST_PREDICTION_VALUE.labels(
                    test_name=self.name,
                    model_name=model_name,
                    variant=variant_name
                ).observe(value)
            
            # Record custom metrics
            if custom_metrics:
                for metric_name, value in custom_metrics.items():
                    if metric_name in self.custom_metrics:
                        self.results[variant_name]['custom_metrics'][metric_name].append(value)
                        
                        AB_TEST_METRICS.labels(
                            test_name=self.name,
                            model_name=model_name,
                            variant=variant_name,
                            metric=metric_name
                        ).set(value)
    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current results of the A/B test.
        
        Returns:
            Dictionary of results for each variant
        """
        with self.lock:
            results = {}
            
            for variant_name, data in self.results.items():
                predictions = data['predictions']
                latencies = data['latencies']
                
                variant_results = {
                    'count': len(predictions),
                    'latency': {
                        'mean': np.mean(latencies) if latencies else None,
                        'median': np.median(latencies) if latencies else None,
                        'p95': np.percentile(latencies, 95) if latencies else None,
                        'min': min(latencies) if latencies else None,
                        'max': max(latencies) if latencies else None
                    }
                }
                
                # Add custom metrics
                for metric_name, values in data['custom_metrics'].items():
                    if values:
                        variant_results[metric_name] = {
                            'mean': np.mean(values),
                            'median': np.median(values),
                            'min': min(values),
                            'max': max(values)
                        }
                
                results[variant_name] = variant_results
            
            return results
    
    def cleanup_expired_sessions(self) -> None:
        """Remove expired sticky sessions."""
        with self.lock:
            current_time = time.time()
            expired_users = [
                user_id for user_id, (_, timestamp) in self.session_assignments.items()
                if current_time - timestamp > self.sticky_session_ttl
            ]
            
            for user_id in expired_users:
                del self.session_assignments[user_id]
            
            if expired_users:
                logger.info(f"Cleaned up {len(expired_users)} expired sessions from A/B test '{self.name}'")

class ABTestManager:
    """
    Manager for multiple A/B tests.
    """
    
    def __init__(self):
        """Initialize the A/B test manager."""
        self.tests = {}
        self.lock = threading.Lock()
        
        # Set up cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def create_test(self, name: str, variants: List[Dict[str, Any]], 
                   traffic_split: Optional[List[float]] = None,
                   sticky_sessions: bool = True,
                   sticky_session_ttl: int = 3600,
                   custom_metrics: List[str] = None) -> ABTest:
        """
        Create a new A/B test.
        
        Args:
            name: Name of the A/B test
            variants: List of variant configurations
            traffic_split: List of traffic proportions
            sticky_sessions: Whether to use sticky sessions
            sticky_session_ttl: Time-to-live for sticky sessions
            custom_metrics: List of custom metrics to track
            
        Returns:
            Created ABTest instance
        """
        with self.lock:
            if name in self.tests:
                logger.warning(f"A/B test '{name}' already exists, returning existing test")
                return self.tests[name]
            
            test = ABTest(
                name=name,
                variants=variants,
                traffic_split=traffic_split,
                sticky_sessions=sticky_sessions,
                sticky_session_ttl=sticky_session_ttl,
                custom_metrics=custom_metrics
            )
            
            self.tests[name] = test
            logger.info(f"Created A/B test '{name}'")
            return test
    
    def get_test(self, name: str) -> Optional[ABTest]:
        """
        Get an A/B test by name.
        
        Args:
            name: Name of the A/B test
            
        Returns:
            ABTest instance or None if not found
        """
        with self.lock:
            return self.tests.get(name)
    
    def delete_test(self, name: str) -> bool:
        """
        Delete an A/B test.
        
        Args:
            name: Name of the A/B test
            
        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            if name in self.tests:
                del self.tests[name]
                logger.info(f"Deleted A/B test '{name}'")
                return True
            return False
    
    def list_tests(self) -> List[str]:
        """
        List all A/B tests.
        
        Returns:
            List of A/B test names
        """
        with self.lock:
            return list(self.tests.keys())
    
    def _cleanup_loop(self) -> None:
        """Background thread to clean up expired sessions."""
        while True:
            with self.lock:
                for test in self.tests.values():
                    test.cleanup_expired_sessions()
            
            # Sleep for 5 minutes
            time.sleep(300)

# Create a global instance
ab_test_manager = ABTestManager()

def initialize_default_test():
    """Initialize a default A/B test for the Iris classifier."""
    try:
        # Create a default A/B test for the Iris classifier
        ab_test_manager.create_test(
            name="iris-model-comparison",
            variants=[
                {
                    "name": "production",
                    "model_name": "iris-classifier",
                    "model_version": "Production"
                },
                {
                    "name": "candidate",
                    "model_name": "iris-classifier",
                    "model_version": "Staging"
                }
            ],
            traffic_split=[0.9, 0.1],  # 90% to production, 10% to candidate
            custom_metrics=["confidence"]
        )
        
        logger.info("Initialized default A/B test for Iris classifier")
    except Exception as e:
        logger.error(f"Failed to initialize default A/B test: {str(e)}")

# Initialize default test
initialize_default_test()

if __name__ == "__main__":
    # Example usage
    test = ab_test_manager.create_test(
        name="example-test",
        variants=[
            {"name": "control", "model_name": "model-a", "model_version": "1"},
            {"name": "treatment", "model_name": "model-b", "model_version": "1"}
        ]
    )
    
    # Simulate some predictions
    for _ in range(100):
        user_id = f"user_{random.randint(1, 20)}"
        variant = test.get_variant(user_id)
        
        # Simulate prediction and latency
        prediction = random.random()
        latency = random.uniform(0.01, 0.2)
        
        # Record results
        test.record_prediction(
            variant_name=variant['name'],
            prediction=prediction,
            latency=latency,
            custom_metrics={"confidence": random.random()}
        )
    
    # Print results
    print(json.dumps(test.get_results(), indent=2)) 