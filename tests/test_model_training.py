import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.training.train import load_data, prepare_data, train_model, evaluate_model

class TestModelTraining(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.data, self.feature_names, self.target_names = load_data()
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