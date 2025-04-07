import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.datasets import load_iris
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.metrics import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_reference_data():
    """Load reference data (training data)."""
    logger.info("Loading reference (training) data")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    
    # Create a DataFrame
    reference_data = pd.DataFrame(X, columns=feature_names)
    reference_data['target'] = y
    
    return reference_data

def generate_production_data(reference_data, drift_percent=0.1, random_seed=42):
    """
    Generate production-like data with optional drift.
    
    Parameters:
    - reference_data: Reference DataFrame
    - drift_percent: What percentage of the data should have drift
    - random_seed: Random seed for reproducibility
    
    Returns:
    - DataFrame with simulated production data
    """
    logger.info(f"Generating production data with {drift_percent*100}% drift")
    
    np.random.seed(random_seed)
    
    # Clone the reference data
    production_data = reference_data.copy()
    
    # Number of rows to modify
    n_rows = int(len(production_data) * drift_percent)
    
    if n_rows > 0:
        # Select random rows to modify
        rows_to_modify = np.random.choice(len(production_data), size=n_rows, replace=False)
        
        for idx in rows_to_modify:
            # Choose a random feature to modify
            feature_cols = [col for col in production_data.columns if col != 'target']
            feature_to_modify = np.random.choice(feature_cols)
            
            # Modify the feature value (add significant noise)
            # For Iris dataset, add noise that's ~20% of the feature's range
            feature_std = production_data[feature_to_modify].std()
            noise = np.random.normal(0, feature_std * 2)
            production_data.loc[idx, feature_to_modify] += noise
    
    return production_data

def detect_data_drift(reference_data, current_data, output_path=None):
    """
    Detect data drift between reference and current data.
    
    Parameters:
    - reference_data: Reference DataFrame
    - current_data: Current DataFrame to compare against reference
    - output_path: Where to save the drift report
    
    Returns:
    - Drift report
    """
    logger.info("Detecting data drift")
    
    # Use all columns except target for drift detection
    data_columns = [col for col in reference_data.columns if col != 'target']
    
    # Create data drift report
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    
    data_drift_report.run(reference_data=reference_data[data_columns], 
                         current_data=current_data[data_columns])
    
    # If target column exists, also create target drift report
    if 'target' in reference_data.columns and 'target' in current_data.columns:
        target_drift_report = Report(metrics=[
            TargetDriftPreset()
        ])
        
        target_drift_report.run(reference_data=reference_data, 
                              current_data=current_data,
                              column_mapping={'target': 'target'})
    else:
        target_drift_report = None
    
    # Save reports if output path is provided
    if output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_path, exist_ok=True)
        
        # Save data drift report
        data_drift_report.save_html(f"{output_path}/data_drift_report_{timestamp}.html")
        
        # Extract and save drift metrics as JSON
        drift_share = data_drift_report.as_dict()['metrics'][0]['result']['drift_share']
        drift_by_feature = data_drift_report.as_dict()['metrics'][0]['result']['drift_by_columns']
        
        drift_summary = {
            'timestamp': timestamp,
            'drift_share': drift_share,
            'drift_by_feature': drift_by_feature,
            'number_of_columns': len(drift_by_feature),
            'number_of_drifted_columns': sum(1 for f in drift_by_feature.values() if f['drift_detected']),
        }
        
        with open(f"{output_path}/drift_metrics_{timestamp}.json", 'w') as f:
            json.dump(drift_summary, f, indent=2)
        
        # Save target drift report if available
        if target_drift_report:
            target_drift_report.save_html(f"{output_path}/target_drift_report_{timestamp}.html")
    
    return data_drift_report, target_drift_report

def main():
    """Main function for drift detection."""
    parser = argparse.ArgumentParser(description="Detect data drift using Evidently AI")
    parser.add_argument("--drift-percent", type=float, default=0.2, 
                        help="Percentage of data to inject drift into (0.0-1.0)")
    parser.add_argument("--output-dir", type=str, default="./drift_reports", 
                        help="Directory to save drift reports")
    
    args = parser.parse_args()
    
    try:
        # Load reference data (training data)
        reference_data = load_reference_data()
        
        # Generate production-like data with drift
        current_data = generate_production_data(
            reference_data, 
            drift_percent=args.drift_percent
        )
        
        # Detect and report drift
        data_drift_report, target_drift_report = detect_data_drift(
            reference_data, 
            current_data, 
            output_path=args.output_dir
        )
        
        # Get drift summary for output
        drift_share = data_drift_report.as_dict()['metrics'][0]['result']['drift_share']
        drifted_features = sum(1 for f in data_drift_report.as_dict()['metrics'][0]['result']['drift_by_columns'].values() 
                             if f['drift_detected'])
        total_features = len(reference_data.columns) - 1  # Exclude target
        
        logger.info(f"Drift detection complete:")
        logger.info(f"Drift share: {drift_share:.2f}")
        logger.info(f"Drifted features: {drifted_features}/{total_features}")
        logger.info(f"Reports saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 