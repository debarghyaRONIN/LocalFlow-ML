import os
import sys
import unittest
import json
from fastapi.testclient import TestClient

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.model_api.app.main import app

class TestModelAPI(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        # Mock MLflow model loading
        self.mock_model_name = "iris-classifier"
        self.mock_features = [[5.1, 3.5, 1.4, 0.2]]  # Example Iris features
    
    def test_health_endpoint(self):
        """Test the health endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        self.assertIn("docs_url", response.json())
    
    def test_metrics_endpoint(self):
        """Test the metrics endpoint."""
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        # Should contain Prometheus metrics
        self.assertIn("model_prediction_count", response.text)
    
    @unittest.mock.patch("mlflow.pyfunc.load_model")
    def test_prediction_endpoint(self, mock_load_model):
        """Test the prediction endpoint with mocked MLflow."""
        # Mock the model's predict method
        mock_model = unittest.mock.MagicMock()
        mock_model.predict.return_value = [0]  # Return class 0 for Iris-setosa
        mock_load_model.return_value = mock_model
        
        # Test prediction
        response = self.client.post(
            f"/predict/{self.mock_model_name}",
            json={"features": self.mock_features}
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("predictions", response_data)
        self.assertIn("model_name", response_data)
        self.assertIn("model_version", response_data)
        self.assertIn("prediction_time", response_data)
        self.assertEqual(response_data["model_name"], self.mock_model_name)
    
    @unittest.mock.patch("mlflow.tracking.MlflowClient")
    def test_list_models_endpoint(self, mock_mlflow_client):
        """Test the list models endpoint with mocked MLflow client."""
        # Set up mock for registered_models
        mock_registered_model = unittest.mock.MagicMock()
        mock_registered_model.name = "iris-classifier"
        
        # Set up mock for model versions
        mock_model_version = unittest.mock.MagicMock()
        mock_model_version.version = "1"
        mock_model_version.current_stage = "Production"
        
        # Configure mocks
        mock_client_instance = mock_mlflow_client.return_value
        mock_client_instance.search_registered_models.return_value = [mock_registered_model]
        mock_client_instance.get_latest_versions.return_value = [mock_model_version]
        
        # Test endpoint
        response = self.client.get("/models")
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("models", response_data)
        self.assertEqual(len(response_data["models"]), 1)
        self.assertEqual(response_data["models"][0]["name"], "iris-classifier")

if __name__ == '__main__':
    unittest.main() 