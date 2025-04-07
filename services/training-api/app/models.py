from typing import List, Dict, Any, Type, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB

# Dictionary of available models
MODEL_REGISTRY = {
    # Classification
    "random_forest": {
        "class": RandomForestClassifier,
        "description": "Random Forest Classifier",
        "parameters": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        },
        "task": "classification"
    },
    "logistic_regression": {
        "class": LogisticRegression,
        "description": "Logistic Regression",
        "parameters": {
            "C": 1.0,
            "max_iter": 100,
            "random_state": 42
        },
        "task": "classification"
    },
    "decision_tree": {
        "class": DecisionTreeClassifier,
        "description": "Decision Tree Classifier",
        "parameters": {
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        },
        "task": "classification"
    },
    "gradient_boosting": {
        "class": GradientBoostingClassifier,
        "description": "Gradient Boosting Classifier",
        "parameters": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42
        },
        "task": "classification"
    },
    "svm": {
        "class": SVC,
        "description": "Support Vector Machine Classifier",
        "parameters": {
            "C": 1.0,
            "kernel": "rbf",
            "probability": True,
            "random_state": 42
        },
        "task": "classification"
    },
    "knn_classifier": {
        "class": KNeighborsClassifier,
        "description": "K-Nearest Neighbors Classifier",
        "parameters": {
            "n_neighbors": 5,
            "weights": "uniform"
        },
        "task": "classification"
    },
    "mlp_classifier": {
        "class": MLPClassifier,
        "description": "Multi-layer Perceptron Classifier",
        "parameters": {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "max_iter": 200,
            "random_state": 42
        },
        "task": "classification"
    },
    "naive_bayes": {
        "class": GaussianNB,
        "description": "Gaussian Naive Bayes",
        "parameters": {},
        "task": "classification"
    },
    "adaboost": {
        "class": AdaBoostClassifier,
        "description": "AdaBoost Classifier",
        "parameters": {
            "n_estimators": 50,
            "learning_rate": 1.0,
            "random_state": 42
        },
        "task": "classification"
    },
    
    # Regression
    "linear_regression": {
        "class": LinearRegression,
        "description": "Linear Regression",
        "parameters": {
            "fit_intercept": True,
            "n_jobs": -1
        },
        "task": "regression"
    },
    "ridge_regression": {
        "class": Ridge,
        "description": "Ridge Regression",
        "parameters": {
            "alpha": 1.0,
            "fit_intercept": True,
            "random_state": 42
        },
        "task": "regression"
    },
    "lasso_regression": {
        "class": Lasso,
        "description": "Lasso Regression",
        "parameters": {
            "alpha": 1.0,
            "fit_intercept": True,
            "random_state": 42
        },
        "task": "regression"
    },
    "decision_tree_regressor": {
        "class": DecisionTreeRegressor,
        "description": "Decision Tree Regressor",
        "parameters": {
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        },
        "task": "regression"
    },
    "random_forest_regressor": {
        "class": RandomForestRegressor,
        "description": "Random Forest Regressor",
        "parameters": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        },
        "task": "regression"
    },
    "svr": {
        "class": SVR,
        "description": "Support Vector Regression",
        "parameters": {
            "C": 1.0,
            "kernel": "rbf",
            "epsilon": 0.1
        },
        "task": "regression"
    },
    "knn_regressor": {
        "class": KNeighborsRegressor,
        "description": "K-Nearest Neighbors Regressor",
        "parameters": {
            "n_neighbors": 5,
            "weights": "uniform"
        },
        "task": "regression"
    },
    "mlp_regressor": {
        "class": MLPRegressor,
        "description": "Multi-layer Perceptron Regressor",
        "parameters": {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "max_iter": 200,
            "random_state": 42
        },
        "task": "regression"
    }
}

def get_model_by_name(model_name: str) -> Type:
    """
    Get a model class by name
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model class
        
    Raises:
        ValueError: If model not found
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry")
    
    return MODEL_REGISTRY[model_name]["class"]

def get_available_models() -> List[Dict[str, Any]]:
    """
    Get a list of all available models
    
    Returns:
        List of model info objects
    """
    return [
        {
            "name": name,
            "description": config["description"],
            "parameters": config["parameters"],
            "task": config["task"]
        }
        for name, config in MODEL_REGISTRY.items()
    ]

def get_model_params(model_name: str) -> Dict[str, Any]:
    """
    Get the default parameters for a model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of default parameters
        
    Raises:
        ValueError: If model not found
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry")
    
    return MODEL_REGISTRY[model_name]["parameters"].copy()

def get_model_by_task(task: str) -> List[Dict[str, Any]]:
    """
    Get all models suitable for a specific task
    
    Args:
        task: Task type ('classification' or 'regression')
        
    Returns:
        List of model info objects for the specified task
    """
    return [
        {
            "name": name,
            "description": config["description"],
            "parameters": config["parameters"]
        }
        for name, config in MODEL_REGISTRY.items()
        if config["task"] == task
    ]

def get_model_task(model_name: str) -> str:
    """
    Get the task type for a model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Task type ('classification' or 'regression')
        
    Raises:
        ValueError: If model not found
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry")
    
    return MODEL_REGISTRY[model_name]["task"] 