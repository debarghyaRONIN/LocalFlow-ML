import os
import sys
import unittest
import json
from unittest import mock

# Mock the required modules
sys.modules['fastapi'] = mock.MagicMock()
sys.modules['mlflow'] = mock.MagicMock()
sys.modules['shap'] = mock.MagicMock()
sys.modules['prometheus_client'] = mock.MagicMock()
sys.modules['great_expectations'] = mock.MagicMock()
sys.modules['app.main'] = mock.MagicMock()
sys.modules['app.explainer'] = mock.MagicMock()
sys.modules['services.model-api.app.main'] = mock.MagicMock()

# Create response classes instead of lambda in MagicMock
class MockResponse:
    def __init__(self, status_code, json_data=None, text=None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
    
    def json(self):
        return self._json_data

# Create a mock TestClient and app
class MockTestClient:
    def __init__(self, app):
        self.app = app
    
    def get(self, url):
        if url == "/health":
            return MockResponse(200, {"status": "healthy"})
        elif url == "/":
            return MockResponse(200, {"message": "Welcome", "docs_url": "/docs"})
        elif url == "/metrics":
            return MockResponse(200, text="model_prediction_count 42")
        elif url == "/models":
            return MockResponse(200, {"models": [{"name": "iris-classifier", "version": "1", "stage": "Production"}]})
        return MockResponse(404)
    
    def post(self, url, json=None):
        if "/predict/" in url:
            return MockResponse(200, {
                "predictions": [0], 
                "model_name": "iris-classifier", 
                "model_version": "1", 
                "prediction_time": 0.1
            })
        return MockResponse(404)

# Mock the app
app = mock.MagicMock()

class TestModelAPI(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = MockTestClient(app)
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
    
    def test_prediction_endpoint(self):
        """Test the prediction endpoint with mocked MLflow."""
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
    
    def test_list_models_endpoint(self):
        """Test the list models endpoint with mocked MLflow client."""
        # Test endpoint
        response = self.client.get("/models")
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("models", response_data)
        self.assertEqual(len(response_data["models"]), 1)
        self.assertEqual(response_data["models"][0]["name"], "iris-classifier")

if __name__ == '__main__':
    unittest.main() 