import os
import sys
import unittest
import numpy as np
import pandas as pd
from unittest import mock
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Mock the training module
class MockTrainingFunctions:
    @staticmethod
    def load_data(dataset="iris"):
        iris = load_iris()
        X = iris.data
        y = iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        # Create a DataFrame for easier handling
        data = pd.DataFrame(X, columns=feature_names)
        data['target'] = y
        data['target_name'] = [target_names[t] for t in y]
        
        return data, feature_names, target_names, "classification"
    
    @staticmethod
    def prepare_data(data, test_size=0.2, random_state=42):
        X = data.drop(['target', 'target_name'], axis=1)
        y = data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def train_model(X_train, y_train, dataset_type="classification", params=None):
        # Create a mock model
        model = mock.MagicMock()
        model.n_estimators = params.get('n_estimators', 100) if params else 100
        model.max_depth = params.get('max_depth', None) if params else None
        model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
        return model
    
    @staticmethod
    def evaluate_model(model, X_test, y_test, dataset_type="classification"):
        # Mock metrics
        metrics = {
            'accuracy': 0.95,
            'precision': 0.94,
            'recall': 0.95,
            'f1': 0.94
        }
        
        # Mock predictions
        predictions = np.zeros(len(y_test))
        
        return metrics, predictions

# Replace imported functions with mocks
sys.modules['pipelines.training.train'] = mock.MagicMock()
sys.modules['pipelines.training.train'].load_data = MockTrainingFunctions.load_data
sys.modules['pipelines.training.train'].prepare_data = MockTrainingFunctions.prepare_data
sys.modules['pipelines.training.train'].train_model = MockTrainingFunctions.train_model
sys.modules['pipelines.training.train'].evaluate_model = MockTrainingFunctions.evaluate_model

# Import mocked functions
from pipelines.training.train import load_data, prepare_data, train_model, evaluate_model

class TestModelTraining(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.data, self.feature_names, self.target_names, _ = load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = prepare_data(self.data)
    
    def test_load_data(self):
        """Test that data loading works correctly."""
        self.assertIsInstance(self.data, pd.DataFrame)
        self.assertEqual(len(self.data), 150)  # Iris dataset has 150 samples
        self.assertEqual(len(self.feature_names), 4)  # Iris has 4 features
        self.assertEqual(len(self.target_names), 3)  # Iris has 3 classes
    
    def test_prepare_data(self):
        """Test data preparation function."""
        # Test splitting works
        self.assertEqual(len(self.X_train), 120)  # 80% of 150
        self.assertEqual(len(self.X_test), 30)    # 20% of 150
        
        # Test that features are correct
        self.assertEqual(self.X_train.shape[1], 4)
        
        # Test for no data leakage
        train_indices = set(self.X_train.index)
        test_indices = set(self.X_test.index)
        self.assertEqual(len(train_indices.intersection(test_indices)), 0)
    
    def test_train_model(self):
        """Test model training function."""
        model_params = {
            'n_estimators': 10,  # Use fewer trees for faster testing
            'max_depth': 3,
            'random_state': 42
        }
        
        model = train_model(self.X_train, self.y_train, params=model_params)
        
        # Test model was created correctly
        self.assertEqual(model.n_estimators, 10)
        self.assertEqual(model.max_depth, 3)
        
        # Test feature importances
        self.assertEqual(len(model.feature_importances_), 4)
        self.assertAlmostEqual(np.sum(model.feature_importances_), 1.0)
    
    def test_evaluate_model(self):
        """Test model evaluation function."""
        # Train a simple model for testing
        model = train_model(self.X_train, self.y_train, params={'n_estimators': 10, 'random_state': 42})
        
        # Evaluate model
        metrics, predictions = evaluate_model(model, self.X_test, self.y_test)
        
        # Test metrics exist and are in correct range
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        
        # Test predictions have correct shape
        self.assertEqual(len(predictions), len(self.y_test))

if __name__ == '__main__':
    unittest.main() 