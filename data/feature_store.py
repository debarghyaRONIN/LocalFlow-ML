import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Union, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureStore:
    """
    A simple feature store implementation that handles:
    - Feature transformation
    - Feature versioning
    - Feature storage and retrieval
    - Feature validation
    """
    
    def __init__(self, store_dir: str = "data/feature_store"):
        """
        Initialize the feature store.
        
        Args:
            store_dir: Directory where feature transformers and metadata are stored
        """
        self.store_dir = store_dir
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(store_dir, "transformers"), exist_ok=True)
        os.makedirs(os.path.join(store_dir, "schemas"), exist_ok=True)
        os.makedirs(os.path.join(store_dir, "features"), exist_ok=True)
        
        # Transformer dictionary
        self.transformers = {}
        
        # Feature schemas dictionary
        self.schemas = {}
    
    def _get_feature_set_path(self, feature_set_name: str, version: Optional[str] = None) -> str:
        """Get the path for a feature set."""
        if version is None:
            # Find the latest version
            version_dirs = [d for d in os.listdir(os.path.join(self.store_dir, "transformers")) 
                           if d.startswith(f"{feature_set_name}_v")]
            if not version_dirs:
                version = "v1"
            else:
                # Extract version numbers and find the max
                versions = [int(d.split('_v')[1]) for d in version_dirs]
                version = f"v{max(versions)}"
        
        return os.path.join(self.store_dir, "transformers", f"{feature_set_name}_{version}")
    
    def _get_schema_path(self, feature_set_name: str, version: Optional[str] = None) -> str:
        """Get the path for a feature schema."""
        if version is None:
            # Find the latest version
            schema_files = [f for f in os.listdir(os.path.join(self.store_dir, "schemas")) 
                           if f.startswith(f"{feature_set_name}_v") and f.endswith(".json")]
            if not schema_files:
                version = "v1"
            else:
                # Extract version numbers and find the max
                versions = [int(f.split('_v')[1].split('.')[0]) for f in schema_files]
                version = f"v{max(versions)}"
        
        return os.path.join(self.store_dir, "schemas", f"{feature_set_name}_{version}.json")
    
    def _get_features_path(self, feature_set_name: str, version: Optional[str] = None) -> str:
        """Get the path for a feature cache file."""
        if version is None:
            # Find the latest version
            feature_files = [f for f in os.listdir(os.path.join(self.store_dir, "features")) 
                            if f.startswith(f"{feature_set_name}_v") and f.endswith(".parquet")]
            if not feature_files:
                version = "v1"
            else:
                # Extract version numbers and find the max
                versions = [int(f.split('_v')[1].split('.')[0]) for f in feature_files]
                version = f"v{max(versions)}"
        
        return os.path.join(self.store_dir, "features", f"{feature_set_name}_{version}.parquet")
    
    def create_feature_set(self, 
                           feature_set_name: str,
                           df: pd.DataFrame, 
                           numeric_features: List[str],
                           categorical_features: List[str] = None,
                           target_column: str = None,
                           version: str = None) -> Dict:
        """
        Create a new feature set with transformers for numeric and categorical features.
        
        Args:
            feature_set_name: Name of the feature set
            df: DataFrame containing the features
            numeric_features: List of numeric feature column names
            categorical_features: List of categorical feature column names
            target_column: Target column name (if any)
            version: Version string (e.g., 'v1'). If None, increments from the latest version.
            
        Returns:
            Dict with information about the created feature set
        """
        try:
            # Determine version
            if version is None:
                existing_versions = [d for d in os.listdir(os.path.join(self.store_dir, "transformers")) 
                                    if d.startswith(f"{feature_set_name}_v")]
                if not existing_versions:
                    version = "v1"
                else:
                    # Extract version numbers and find the max
                    versions = [int(d.split('_v')[1]) for d in existing_versions]
                    version = f"v{max(versions) + 1}"
            
            logger.info(f"Creating feature set {feature_set_name} version {version}")
            
            # Initialize transformers
            transformers = {}
            
            # Process numeric features
            if numeric_features:
                # Create and fit numeric transformer
                numeric_transformer = {
                    'imputer': SimpleImputer(strategy='mean'),
                    'scaler': StandardScaler()
                }
                
                # Fit imputer
                numeric_transformer['imputer'].fit(df[numeric_features])
                
                # Apply imputer and fit scaler
                imputed_data = numeric_transformer['imputer'].transform(df[numeric_features])
                numeric_transformer['scaler'].fit(imputed_data)
                
                transformers['numeric'] = {
                    'columns': numeric_features,
                    'transformer': numeric_transformer
                }
            
            # Process categorical features
            if categorical_features:
                # Create and fit categorical transformer
                categorical_transformer = {
                    'imputer': SimpleImputer(strategy='most_frequent'),
                    'encoder': OneHotEncoder(sparse=False, handle_unknown='ignore')
                }
                
                # Fit imputer
                categorical_transformer['imputer'].fit(df[categorical_features])
                
                # Apply imputer and fit encoder
                imputed_data = categorical_transformer['imputer'].transform(df[categorical_features])
                categorical_transformer['encoder'].fit(pd.DataFrame(imputed_data, columns=categorical_features))
                
                transformers['categorical'] = {
                    'columns': categorical_features,
                    'transformer': categorical_transformer,
                    'categories': categorical_transformer['encoder'].categories_
                }
            
            # Create feature set directory
            transformer_path = self._get_feature_set_path(feature_set_name, version)
            os.makedirs(transformer_path, exist_ok=True)
            
            # Save transformers
            if 'numeric' in transformers:
                np.save(os.path.join(transformer_path, "numeric_imputer.npy"), 
                        transformers['numeric']['transformer']['imputer'].__dict__)
                np.save(os.path.join(transformer_path, "numeric_scaler.npy"), 
                        transformers['numeric']['transformer']['scaler'].__dict__)
            
            if 'categorical' in transformers:
                np.save(os.path.join(transformer_path, "categorical_imputer.npy"), 
                        transformers['categorical']['transformer']['imputer'].__dict__)
                np.save(os.path.join(transformer_path, "categorical_encoder.npy"), 
                        transformers['categorical']['transformer']['encoder'].__dict__)
                np.save(os.path.join(transformer_path, "categorical_categories.npy"), 
                        transformers['categorical']['categories'])
            
            # Save schema information
            schema = {
                'feature_set_name': feature_set_name,
                'version': version,
                'created_at': datetime.now().isoformat(),
                'numeric_features': numeric_features,
                'categorical_features': categorical_features,
                'target_column': target_column,
                'column_dtypes': {col: str(df[col].dtype) for col in df.columns},
                'feature_statistics': {
                    'numeric': {
                        col: {
                            'mean': float(df[col].mean()),
                            'min': float(df[col].min()),
                            'max': float(df[col].max()),
                            'std': float(df[col].std())
                        } for col in numeric_features
                    } if numeric_features else {},
                    'categorical': {
                        col: {
                            'unique_values': df[col].nunique(),
                            'most_common': df[col].value_counts().index[0]
                        } for col in categorical_features
                    } if categorical_features else {}
                }
            }
            
            # Save schema
            schema_path = self._get_schema_path(feature_set_name, version)
            with open(schema_path, 'w') as f:
                json.dump(schema, f, indent=2)
            
            # Store in memory
            self.transformers[f"{feature_set_name}_{version}"] = transformers
            self.schemas[f"{feature_set_name}_{version}"] = schema
            
            logger.info(f"Successfully created feature set {feature_set_name} version {version}")
            
            return {
                'feature_set_name': feature_set_name,
                'version': version,
                'transformer_path': transformer_path,
                'schema_path': schema_path
            }
        
        except Exception as e:
            logger.error(f"Error creating feature set: {str(e)}")
            raise
    
    def load_feature_set(self, feature_set_name: str, version: Optional[str] = None) -> Dict:
        """
        Load a feature set's transformers and schema.
        
        Args:
            feature_set_name: Name of the feature set
            version: Version to load. If None, loads the latest version.
            
        Returns:
            Dict with transformers and schema
        """
        try:
            # Determine paths
            transformer_path = self._get_feature_set_path(feature_set_name, version)
            schema_path = self._get_schema_path(feature_set_name, version)
            
            if not os.path.exists(transformer_path) or not os.path.exists(schema_path):
                raise FileNotFoundError(f"Feature set {feature_set_name} {version} not found")
            
            # Load schema
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            
            version = schema['version']
            
            # Initialize transformers
            transformers = {}
            
            # Load numeric transformers if they exist
            if os.path.exists(os.path.join(transformer_path, "numeric_imputer.npy")):
                numeric_imputer = SimpleImputer()
                numeric_imputer.__dict__.update(np.load(
                    os.path.join(transformer_path, "numeric_imputer.npy"), 
                    allow_pickle=True).item())
                
                numeric_scaler = StandardScaler()
                numeric_scaler.__dict__.update(np.load(
                    os.path.join(transformer_path, "numeric_scaler.npy"),
                    allow_pickle=True).item())
                
                transformers['numeric'] = {
                    'columns': schema['numeric_features'],
                    'transformer': {
                        'imputer': numeric_imputer,
                        'scaler': numeric_scaler
                    }
                }
            
            # Load categorical transformers if they exist
            if os.path.exists(os.path.join(transformer_path, "categorical_imputer.npy")):
                categorical_imputer = SimpleImputer()
                categorical_imputer.__dict__.update(np.load(
                    os.path.join(transformer_path, "categorical_imputer.npy"),
                    allow_pickle=True).item())
                
                categorical_encoder = OneHotEncoder()
                categorical_encoder.__dict__.update(np.load(
                    os.path.join(transformer_path, "categorical_encoder.npy"),
                    allow_pickle=True).item())
                
                categories = np.load(
                    os.path.join(transformer_path, "categorical_categories.npy"),
                    allow_pickle=True)
                
                transformers['categorical'] = {
                    'columns': schema['categorical_features'],
                    'transformer': {
                        'imputer': categorical_imputer,
                        'encoder': categorical_encoder
                    },
                    'categories': categories
                }
            
            # Store in memory
            self.transformers[f"{feature_set_name}_{version}"] = transformers
            self.schemas[f"{feature_set_name}_{version}"] = schema
            
            logger.info(f"Successfully loaded feature set {feature_set_name} version {version}")
            
            return {
                'transformers': transformers,
                'schema': schema
            }
        
        except Exception as e:
            logger.error(f"Error loading feature set: {str(e)}")
            raise
    
    def transform_features(self, df: pd.DataFrame, feature_set_name: str, version: Optional[str] = None) -> pd.DataFrame:
        """
        Transform features using a feature set's transformers.
        
        Args:
            df: DataFrame to transform
            feature_set_name: Name of the feature set
            version: Version to use. If None, uses the latest version.
            
        Returns:
            Transformed DataFrame
        """
        try:
            # Load feature set if not already loaded
            feature_set_key = f"{feature_set_name}_{version}" if version else None
            
            if feature_set_key not in self.transformers:
                feature_set = self.load_feature_set(feature_set_name, version)
                transformers = feature_set['transformers']
                schema = feature_set['schema']
            else:
                transformers = self.transformers[feature_set_key]
                schema = self.schemas[feature_set_key]
            
            # Validate the dataframe against the schema
            self._validate_dataframe(df, schema)
            
            # Initialize output dataframe
            transformed_dfs = []
            
            # Transform numeric features
            if 'numeric' in transformers:
                numeric_columns = transformers['numeric']['columns']
                numeric_transformer = transformers['numeric']['transformer']
                
                # Handle missing numeric columns by adding them with NaN values
                for col in numeric_columns:
                    if col not in df.columns:
                        df[col] = np.nan
                
                # Select only the columns we need
                numeric_df = df[numeric_columns].copy()
                
                # Apply imputation
                numeric_imputed = numeric_transformer['imputer'].transform(numeric_df)
                
                # Apply scaling
                numeric_scaled = numeric_transformer['scaler'].transform(numeric_imputed)
                
                # Convert to DataFrame
                numeric_result = pd.DataFrame(
                    numeric_scaled, 
                    columns=[f"{col}_scaled" for col in numeric_columns],
                    index=df.index
                )
                
                transformed_dfs.append(numeric_result)
            
            # Transform categorical features
            if 'categorical' in transformers:
                categorical_columns = transformers['categorical']['columns']
                categorical_transformer = transformers['categorical']['transformer']
                
                # Handle missing categorical columns by adding them with NaN values
                for col in categorical_columns:
                    if col not in df.columns:
                        df[col] = np.nan
                
                # Select only the columns we need
                categorical_df = df[categorical_columns].copy()
                
                # Apply imputation
                categorical_imputed = categorical_transformer['imputer'].transform(categorical_df)
                
                # Convert to DataFrame to preserve column names
                categorical_imputed_df = pd.DataFrame(
                    categorical_imputed,
                    columns=categorical_columns,
                    index=df.index
                )
                
                # Apply one-hot encoding
                categorical_encoded = categorical_transformer['encoder'].transform(categorical_imputed_df)
                
                # Get feature names from encoder
                encoded_features = []
                for i, feature in enumerate(categorical_columns):
                    for category in categorical_transformer['encoder'].categories_[i]:
                        encoded_features.append(f"{feature}_{category}")
                
                # Convert to DataFrame
                categorical_result = pd.DataFrame(
                    categorical_encoded,
                    columns=encoded_features,
                    index=df.index
                )
                
                transformed_dfs.append(categorical_result)
            
            # Combine all transformed features
            if transformed_dfs:
                result = pd.concat(transformed_dfs, axis=1)
                
                # Add target column if it exists in the original dataframe
                if schema['target_column'] and schema['target_column'] in df.columns:
                    result[schema['target_column']] = df[schema['target_column']]
                
                return result
            else:
                logger.warning("No features were transformed")
                return df.copy()
        
        except Exception as e:
            logger.error(f"Error transforming features: {str(e)}")
            raise
    
    def _validate_dataframe(self, df: pd.DataFrame, schema: Dict) -> None:
        """
        Validate a dataframe against a schema.
        
        Args:
            df: DataFrame to validate
            schema: Schema dict
            
        Raises:
            ValueError: If validation fails
        """
        missing_required_columns = []
        
        # Check numeric features
        if 'numeric_features' in schema and schema['numeric_features']:
            for col in schema['numeric_features']:
                if col not in df.columns:
                    missing_required_columns.append(col)
        
        # Check categorical features
        if 'categorical_features' in schema and schema['categorical_features']:
            for col in schema['categorical_features']:
                if col not in df.columns:
                    missing_required_columns.append(col)
        
        # Report missing columns
        if missing_required_columns:
            logger.warning(f"Missing columns in dataframe: {missing_required_columns}")
    
    def cache_features(self, df: pd.DataFrame, feature_set_name: str, version: Optional[str] = None) -> str:
        """
        Cache transformed features to disk.
        
        Args:
            df: DataFrame to cache
            feature_set_name: Name of the feature set
            version: Version to use. If None, uses the latest version.
            
        Returns:
            Path to cached features
        """
        try:
            # Transform features
            transformed_df = self.transform_features(df, feature_set_name, version)
            
            # Determine version
            if version is None:
                version = self.schemas[f"{feature_set_name}_{version}" if version else list(filter(
                    lambda k: k.startswith(f"{feature_set_name}_"), self.schemas.keys()
                ))[-1]]['version']
            
            # Save to parquet
            features_path = self._get_features_path(feature_set_name, version)
            transformed_df.to_parquet(features_path)
            
            logger.info(f"Cached features saved to {features_path}")
            
            return features_path
        
        except Exception as e:
            logger.error(f"Error caching features: {str(e)}")
            raise
    
    def get_cached_features(self, feature_set_name: str, version: Optional[str] = None) -> pd.DataFrame:
        """
        Load cached features from disk.
        
        Args:
            feature_set_name: Name of the feature set
            version: Version to load. If None, loads the latest version.
            
        Returns:
            DataFrame with cached features
        """
        try:
            features_path = self._get_features_path(feature_set_name, version)
            
            if not os.path.exists(features_path):
                raise FileNotFoundError(f"Cached features not found at {features_path}")
            
            df = pd.read_parquet(features_path)
            
            logger.info(f"Loaded cached features from {features_path}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading cached features: {str(e)}")
            raise
    
    def list_feature_sets(self) -> List[Dict]:
        """
        List all available feature sets.
        
        Returns:
            List of feature set information
        """
        try:
            feature_sets = []
            
            # Get all schema files
            schema_files = [f for f in os.listdir(os.path.join(self.store_dir, "schemas")) if f.endswith(".json")]
            
            for schema_file in schema_files:
                schema_path = os.path.join(self.store_dir, "schemas", schema_file)
                
                with open(schema_path, 'r') as f:
                    schema = json.load(f)
                
                feature_sets.append({
                    'feature_set_name': schema['feature_set_name'],
                    'version': schema['version'],
                    'created_at': schema['created_at'],
                    'numeric_features': schema['numeric_features'],
                    'categorical_features': schema['categorical_features'],
                    'target_column': schema['target_column']
                })
            
            return feature_sets
        
        except Exception as e:
            logger.error(f"Error listing feature sets: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_iris
    
    # Create feature store
    feature_store = FeatureStore()
    
    # Load sample data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Create feature set
    feature_store.create_feature_set(
        feature_set_name="iris",
        df=df,
        numeric_features=iris.feature_names,
        target_column="target"
    )
    
    # Transform new data
    new_data = pd.DataFrame({
        'sepal length (cm)': [5.1, 6.2],
        'sepal width (cm)': [3.5, 2.9],
        'petal length (cm)': [1.4, 4.3],
        'petal width (cm)': [0.2, 1.3]
    })
    
    transformed = feature_store.transform_features(new_data, "iris")
    print(transformed.head()) 