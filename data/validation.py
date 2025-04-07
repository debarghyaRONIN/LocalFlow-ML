import os
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime
import great_expectations as ge
from pydantic import BaseModel, validator, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory to store validation reports and expectations
VALIDATION_DIR = os.path.join("data", "validation")
os.makedirs(VALIDATION_DIR, exist_ok=True)

class DataValidator:
    """
    Data validator for ensuring data quality.
    Uses Great Expectations and Pydantic for schema validation.
    """
    
    def __init__(self, dataset_name: str, version: str = "latest"):
        """
        Initialize a data validator.
        
        Args:
            dataset_name: Name of the dataset to validate
            version: Version of the expectation suite
        """
        self.dataset_name = dataset_name
        self.version = version
        
        # Initialize Great Expectations context
        self.expectations_path = os.path.join(VALIDATION_DIR, "expectations")
        self.validations_path = os.path.join(VALIDATION_DIR, "validations")
        
        # Create directories if they don't exist
        os.makedirs(self.expectations_path, exist_ok=True)
        os.makedirs(self.validations_path, exist_ok=True)
        
        # Load expectation suite if it exists
        self.expectation_suite = self._load_expectation_suite()
    
    def _get_expectation_suite_path(self) -> str:
        """Get the path to the expectation suite file."""
        if self.version == "latest":
            # Find the latest version
            suite_files = [f for f in os.listdir(self.expectations_path) 
                          if f.startswith(f"{self.dataset_name}_v") and f.endswith(".json")]
            
            if not suite_files:
                # No existing suite, create a new one with version 1
                return os.path.join(self.expectations_path, f"{self.dataset_name}_v1.json")
            
            # Extract version numbers and find the max
            versions = [int(f.split('_v')[1].split('.')[0]) for f in suite_files]
            latest_version = f"v{max(versions)}"
            
            return os.path.join(self.expectations_path, f"{self.dataset_name}_{latest_version}.json")
        else:
            return os.path.join(self.expectations_path, f"{self.dataset_name}_{self.version}.json")
    
    def _load_expectation_suite(self) -> Optional[Dict]:
        """Load the expectation suite from file if it exists."""
        expectation_path = self._get_expectation_suite_path()
        
        if os.path.exists(expectation_path):
            try:
                with open(expectation_path, 'r') as f:
                    suite = json.load(f)
                logger.info(f"Loaded expectation suite from {expectation_path}")
                return suite
            except Exception as e:
                logger.error(f"Error loading expectation suite: {str(e)}")
                return None
        else:
            logger.info(f"Expectation suite does not exist at {expectation_path}")
            return None
    
    def _save_expectation_suite(self, suite: Dict) -> None:
        """Save the expectation suite to file."""
        expectation_path = self._get_expectation_suite_path()
        
        try:
            with open(expectation_path, 'w') as f:
                json.dump(suite, f, indent=2)
            logger.info(f"Saved expectation suite to {expectation_path}")
        except Exception as e:
            logger.error(f"Error saving expectation suite: {str(e)}")
    
    def generate_expectations(self, df: pd.DataFrame, overwrite: bool = False) -> Dict:
        """
        Generate expectations from a DataFrame.
        
        Args:
            df: DataFrame to generate expectations from
            overwrite: Whether to overwrite existing expectations
            
        Returns:
            Dictionary of expectations
        """
        if self.expectation_suite is not None and not overwrite:
            logger.info("Expectation suite already exists and overwrite=False")
            return self.expectation_suite
        
        logger.info(f"Generating expectations for dataset {self.dataset_name}")
        
        # Convert DataFrame to Great Expectations DataAsset
        ge_df = ge.from_pandas(df)
        
        # Create a new expectation suite
        suite = {
            "dataset_name": self.dataset_name,
            "version": self.version if self.version != "latest" else f"v{self._get_next_version()}",
            "created_at": datetime.now().isoformat(),
            "expectations": []
        }
        
        # Add general expectations
        suite["expectations"].extend([
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {
                    "min_value": 1,
                    "max_value": df.shape[0] * 2  # Allow up to 2x the current size
                }
            },
            {
                "expectation_type": "expect_table_column_count_to_equal",
                "kwargs": {
                    "value": df.shape[1]
                }
            }
        ])
        
        # Add column names expectation
        suite["expectations"].append({
            "expectation_type": "expect_table_columns_to_match_ordered_list",
            "kwargs": {
                "column_list": list(df.columns)
            }
        })
        
        # Add column-specific expectations
        for column in df.columns:
            column_type = df[column].dtype
            
            # Skip columns with object type (strings, etc.)
            if column_type == 'object':
                # For categorical columns, check for unique values
                if df[column].nunique() / len(df) < 0.1:  # Less than 10% unique values
                    unique_values = list(df[column].unique())
                    suite["expectations"].append({
                        "expectation_type": "expect_column_values_to_be_in_set",
                        "kwargs": {
                            "column": column,
                            "value_set": unique_values,
                            "mostly": 0.95  # Allow 5% new values
                        }
                    })
                
                # Check for null values
                null_rate = df[column].isnull().mean()
                suite["expectations"].append({
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {
                        "column": column,
                        "mostly": 1.0 - (null_rate + 0.05)  # Allow 5% more nulls
                    }
                })
            
            # For numeric columns, check ranges
            elif np.issubdtype(column_type, np.number):
                min_value = df[column].min()
                max_value = df[column].max()
                mean_value = df[column].mean()
                std_value = df[column].std()
                
                # Add range expectation
                suite["expectations"].append({
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {
                        "column": column,
                        "min_value": float(min_value - 3 * std_value),
                        "max_value": float(max_value + 3 * std_value),
                        "mostly": 0.99  # Allow 1% outliers
                    }
                })
                
                # Add distribution expectation
                suite["expectations"].append({
                    "expectation_type": "expect_column_mean_to_be_between",
                    "kwargs": {
                        "column": column,
                        "min_value": float(mean_value - std_value),
                        "max_value": float(mean_value + std_value)
                    }
                })
                
                # Check for null values
                null_rate = df[column].isnull().mean()
                suite["expectations"].append({
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {
                        "column": column,
                        "mostly": 1.0 - (null_rate + 0.05)  # Allow 5% more nulls
                    }
                })
        
        # Save the expectation suite
        self.expectation_suite = suite
        self._save_expectation_suite(suite)
        
        return suite
    
    def _get_next_version(self) -> int:
        """Get the next version number for the expectation suite."""
        # Find all suite files for this dataset
        suite_files = [f for f in os.listdir(self.expectations_path) 
                      if f.startswith(f"{self.dataset_name}_v") and f.endswith(".json")]
        
        if not suite_files:
            return 1
        
        # Extract version numbers and find the max
        versions = [int(f.split('_v')[1].split('.')[0]) for f in suite_files]
        return max(versions) + 1
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate data against expectations.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation result dictionary
        """
        # Ensure we have expectations
        if self.expectation_suite is None:
            logger.warning("No expectation suite found, generating from the provided data")
            self.generate_expectations(df)
        
        logger.info(f"Validating data against expectations for {self.dataset_name}")
        
        # Create a results dictionary
        validation_results = {
            "dataset_name": self.dataset_name,
            "validation_time": datetime.now().isoformat(),
            "passed": True,
            "expectation_results": [],
            "summary": {
                "total_expectations": 0,
                "passing_expectations": 0,
                "failing_expectations": 0,
                "success_percent": 0.0
            }
        }
        
        # Convert DataFrame to Great Expectations DataAsset
        ge_df = ge.from_pandas(df)
        
        # Validate each expectation
        for expectation in self.expectation_suite["expectations"]:
            expectation_type = expectation["expectation_type"]
            kwargs = expectation["kwargs"]
            
            # Get the validation method
            validation_method = getattr(ge_df, expectation_type, None)
            
            if validation_method is None:
                logger.warning(f"Unknown expectation type: {expectation_type}")
                continue
            
            # Validate the expectation
            try:
                result = validation_method(**kwargs)
                validation_results["expectation_results"].append({
                    "expectation_type": expectation_type,
                    "kwargs": kwargs,
                    "success": result.success,
                    "result": {
                        "observed_value": result.result.get("observed_value"),
                        "details": result.result
                    }
                })
                
                # Update summary stats
                validation_results["summary"]["total_expectations"] += 1
                if result.success:
                    validation_results["summary"]["passing_expectations"] += 1
                else:
                    validation_results["summary"]["failing_expectations"] += 1
                    validation_results["passed"] = False
            
            except Exception as e:
                logger.error(f"Error validating {expectation_type}: {str(e)}")
                validation_results["expectation_results"].append({
                    "expectation_type": expectation_type,
                    "kwargs": kwargs,
                    "success": False,
                    "error": str(e)
                })
                validation_results["summary"]["total_expectations"] += 1
                validation_results["summary"]["failing_expectations"] += 1
                validation_results["passed"] = False
        
        # Calculate success percentage
        total = validation_results["summary"]["total_expectations"]
        if total > 0:
            success_percent = (validation_results["summary"]["passing_expectations"] / total) * 100
            validation_results["summary"]["success_percent"] = success_percent
        
        # Save validation results
        self._save_validation_results(validation_results)
        
        return validation_results
    
    def _save_validation_results(self, results: Dict) -> None:
        """Save validation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(
            self.validations_path, 
            f"{self.dataset_name}_{timestamp}.json"
        )
        
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved validation results to {results_path}")
        except Exception as e:
            logger.error(f"Error saving validation results: {str(e)}")

# Pydantic models for schema validation

class IrisFeatures(BaseModel):
    """Schema validation model for Iris dataset features."""
    sepal_length: float = Field(..., ge=4.0, le=8.0)
    sepal_width: float = Field(..., ge=2.0, le=4.5)
    petal_length: float = Field(..., ge=1.0, le=7.0)
    petal_width: float = Field(..., ge=0.1, le=2.5)
    
    @validator('*')
    def check_nans(cls, v):
        if pd.isna(v):
            raise ValueError('NaN values are not allowed')
        return v

class CaliforniaHousingFeatures(BaseModel):
    """Schema validation model for California Housing dataset features."""
    median_income: float = Field(..., ge=0.0, le=15.0)
    housing_median_age: float = Field(..., ge=1.0, le=52.0)
    total_rooms: float = Field(..., ge=6.0, le=37937.0)
    total_bedrooms: float = Field(..., ge=1.0, le=6445.0)
    population: float = Field(..., ge=3.0, le=35682.0)
    households: float = Field(..., ge=1.0, le=6082.0)
    latitude: float = Field(..., ge=32.5, le=42.0)
    longitude: float = Field(..., ge=-124.3, le=-114.0)
    
    @validator('*')
    def check_nans(cls, v):
        if pd.isna(v):
            raise ValueError('NaN values are not allowed')
        return v

def validate_schema(data: Union[Dict, pd.DataFrame], schema_type: str) -> Tuple[bool, List[str]]:
    """
    Validate data against a predefined schema.
    
    Args:
        data: Data to validate (dict or DataFrame)
        schema_type: Type of schema to validate against
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    if schema_type == "iris":
        model = IrisFeatures
    elif schema_type == "california_housing":
        model = CaliforniaHousingFeatures
    else:
        return False, [f"Unknown schema type: {schema_type}"]
    
    # If data is a DataFrame, convert it to a dict
    if isinstance(data, pd.DataFrame):
        # Check if this is a batch
        if data.shape[0] > 1:
            # Validate each row
            all_valid = True
            all_errors = []
            
            for i, row in data.iterrows():
                row_dict = row.to_dict()
                is_valid, errors = validate_schema(row_dict, schema_type)
                
                if not is_valid:
                    all_valid = False
                    all_errors.append(f"Row {i}: {', '.join(errors)}")
            
            return all_valid, all_errors
        else:
            # Single row DataFrame
            data = data.iloc[0].to_dict()
    
    # Validate with Pydantic
    try:
        model(**data)
        return True, []
    except Exception as e:
        return False, [str(e)]

if __name__ == "__main__":
    # Example usage with Iris dataset
    from sklearn.datasets import load_iris
    
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Rename columns to match the Pydantic model
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
    
    # Create a data validator
    validator = DataValidator("iris")
    
    # Generate expectations
    expectations = validator.generate_expectations(df)
    
    # Validate data
    results = validator.validate_data(df)
    
    # Print validation summary
    print(f"Validation passed: {results['passed']}")
    print(f"Total expectations: {results['summary']['total_expectations']}")
    print(f"Passing expectations: {results['summary']['passing_expectations']}")
    print(f"Success rate: {results['summary']['success_percent']:.2f}%")
    
    # Schema validation example
    is_valid, errors = validate_schema(df.iloc[0].to_dict(), "iris")
    print(f"Schema validation passed: {is_valid}")
    if not is_valid:
        print(f"Errors: {errors}") 