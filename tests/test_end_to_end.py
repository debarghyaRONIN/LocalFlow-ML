import os
import sys
import unittest
import tempfile
import shutil
import time
import requests
import subprocess
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.training.train import main as train_main

class TestEndToEnd(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Create temporary directory for artifacts
        cls.temp_dir = tempfile.mkdtemp()
        cls.mlflow_artifacts = os.path.join(cls.temp_dir, "mlflow_artifacts")
        os.makedirs(cls.mlflow_artifacts, exist_ok=True)
        
        # Set environment variables
        os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///" + os.path.join(cls.temp_dir, "mlflow.db")
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "test-iris-classifier"
        
        # Start MLflow server for testing (detached process)
        cls.mlflow_process = subprocess.Popen([
            "mlflow", "server",
            "--host", "127.0.0.1",
            "--port", "5001",
            "--backend-store-uri", os.environ["MLFLOW_TRACKING_URI"],
            "--default-artifact-root", cls.mlflow_artifacts
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give MLflow time to start
        time.sleep(5)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        # Stop MLflow server
        cls.mlflow_process.terminate()
        cls.mlflow_process.wait()
        
        # Clean up temp directory
        shutil.rmtree(cls.temp_dir)
    
    def test_full_pipeline(self):
        """Test the full pipeline from training to deployment."""
        try:
            # 1. Train a model
            run_id = train_main()
            self.assertIsNotNone(run_id, "Model training should return a run ID")
            
            # 2. Verify model was registered in MLflow
            mlflow_api_url = "http://127.0.0.1:5001/api/2.0/mlflow"
            registered_models_response = requests.get(f"{mlflow_api_url}/registered-models/get", 
                                                    params={"name": "iris-classifier"})
            self.assertEqual(registered_models_response.status_code, 200, 
                            "Should be able to get registered model from MLflow")
            
            # 3. Start model API server for testing
            model_api_process = subprocess.Popen([
                "python", "-m", "uvicorn", 
                "services.model_api.app.main:app", 
                "--host", "127.0.0.1", 
                "--port", "8001"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Give API server time to start
            time.sleep(5)
            
            try:
                # 4. Test prediction endpoint
                prediction_response = requests.post(
                    "http://127.0.0.1:8001/predict/iris-classifier",
                    json={"features": [[5.1, 3.5, 1.4, 0.2]]}  # Example Iris-setosa features
                )
                
                self.assertEqual(prediction_response.status_code, 200,
                                "Prediction endpoint should return 200 OK")
                                
                prediction_data = prediction_response.json()
                self.assertIn("predictions", prediction_data,
                            "Response should contain predictions")
                self.assertIn("model_name", prediction_data,
                            "Response should contain model name")
                            
            finally:
                # Clean up API server
                model_api_process.terminate()
                model_api_process.wait()
        
        except Exception as e:
            self.fail(f"End-to-end test failed with exception: {str(e)}")

if __name__ == '__main__':
    unittest.main() 