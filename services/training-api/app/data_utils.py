import os
import io
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple

logger = logging.getLogger(__name__)

def process_dataset(file_path: str) -> pd.DataFrame:
    """
    Process a dataset file and return a DataFrame
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Processed DataFrame
        
    Raises:
        ValueError: If file format is not supported
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif file_ext == '.json':
            df = pd.read_json(file_path)
        elif file_ext == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_ext == '.tsv':
            df = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Basic preprocessing
        df = df.copy()
        
        # Drop rows with all NaN values
        df.dropna(how='all', inplace=True)
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise ValueError(f"Error processing dataset: {str(e)}")

def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate a dataset for training
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        "valid": True,
        "message": "Dataset validation successful",
        "warnings": []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_result["valid"] = False
        validation_result["message"] = "Dataset is empty"
        return validation_result
    
    # Check minimum size
    if len(df) < 10:
        validation_result["valid"] = False
        validation_result["message"] = "Dataset has fewer than 10 samples"
        return validation_result
    
    # Check if there are any columns
    if len(df.columns) < 2:
        validation_result["valid"] = False
        validation_result["message"] = "Dataset needs at least 2 columns (features and target)"
        return validation_result
    
    # Check for too many missing values
    missing_pct = df.isna().mean()
    high_missing_cols = missing_pct[missing_pct > 0.5].index.tolist()
    
    if high_missing_cols:
        validation_result["warnings"].append(
            f"Columns with >50% missing values: {', '.join(high_missing_cols)}"
        )
    
    # Check data types
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    if len(non_numeric_cols) == len(df.columns):
        validation_result["warnings"].append(
            "Dataset has no numeric columns, which might limit model choices"
        )
    
    return validation_result

def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get information about a dataset
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with dataset information
    """
    # Basic info
    info = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "column_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "has_missing_values": df.isna().any().any(),
    }
    
    # Column types
    info["numeric_columns"] = df.select_dtypes(include=['number']).columns.tolist()
    info["categorical_columns"] = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Sample data (first 5 rows)
    sample_data = df.head(5).to_dict(orient='records')
    for record in sample_data:
        for key, val in record.items():
            if isinstance(val, (np.integer, np.floating)):
                record[key] = float(val)
            elif pd.isna(val):
                record[key] = None
    
    info["sample_data"] = sample_data
    
    return info

def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to infer which column might be the target variable
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Name of the inferred target column, or None if can't infer
    """
    # Check for common target column names
    common_target_names = ['target', 'label', 'class', 'y', 'output', 'result', 'outcome']
    for name in common_target_names:
        if name in df.columns:
            return name
    
    # Check if there's a column with fewer unique values (potential classification target)
    if len(df.columns) > 1:
        unique_counts = df.nunique()
        min_unique = unique_counts.min()
        
        # If there's a column with few distinct values, it might be a target
        if min_unique < 10 and min_unique / len(df) < 0.05:
            return unique_counts.idxmin()
    
    return None

def suggest_feature_cols(df: pd.DataFrame, target_col: str) -> List[str]:
    """
    Suggest which columns might be good features
    
    Args:
        df: DataFrame to analyze
        target_col: Name of the target column
        
    Returns:
        List of suggested feature columns
    """
    # Exclude the target column
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Exclude columns with too many missing values
    missing_pct = df[feature_cols].isna().mean()
    feature_cols = [col for col in feature_cols if missing_pct[col] < 0.5]
    
    # Exclude columns with too many unique values (like IDs)
    unique_pct = df[feature_cols].nunique() / len(df)
    feature_cols = [col for col in feature_cols if unique_pct[col] < 0.95]
    
    return feature_cols

def detect_dataset_task(df: pd.DataFrame, target_col: str) -> str:
    """
    Detect if a dataset is for classification or regression based on target variable
    
    Args:
        df: DataFrame to analyze
        target_col: Name of the target column
        
    Returns:
        Task type: 'classification' or 'regression'
    """
    # Get target column data
    y = df[target_col]
    
    # Check data type
    if pd.api.types.is_numeric_dtype(y):
        # If numeric, check if it's likely categorical or continuous
        n_unique = y.nunique()
        
        # Check if values are all integers or close to integers
        is_integer_like = False
        if pd.api.types.is_integer_dtype(y):
            is_integer_like = True
        elif pd.api.types.is_float_dtype(y):
            # Check if float values are close to integers
            is_integer_like = np.allclose(y.dropna(), np.round(y.dropna()), atol=1e-5, rtol=0)
        
        # Criteria for classification:
        # 1. Few unique values (< 10)
        # 2. Small ratio of unique values to total rows (< 5%)
        # 3. Integer-like values and small number of unique values (< 20)
        if (n_unique < 10 or 
            (n_unique / len(y) < 0.05) or 
            (is_integer_like and n_unique < 20)):
            return "classification"
        else:
            return "regression"
    else:
        # Non-numeric features are always for classification
        return "classification" 