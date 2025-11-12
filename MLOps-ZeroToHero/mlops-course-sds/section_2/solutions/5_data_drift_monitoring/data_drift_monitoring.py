import os 
import pandas as pd 
import mlflow 
import wandb 
from sklearn.preprocessing import StandardScaler 
from scipy.stats import ks_2samp 
 
# Initialize WandB for tracking the data drift monitoring 
wandb.init(project="credit-card-fraud-detection-test-2", job_type="data-drift-monitoring") 
 
def load_data(): 
    """ 
    Load the current and reference datasets from WandB Artifacts. 
     
    :return: pandas DataFrame, reference and current datasets 
    """ 
    # Load the reference data (previously processed data) 
    ref_artifact = wandb.use_artifact('anicin/credit-card-fraud-detection-test-2/reference_data.csv:latest', type='dataset') 
    ref_artifact_dir = ref_artifact.download() 
    reference_data = pd.read_csv(os.path.join(ref_artifact_dir, 'balanced_dataset.csv')) 
 
    # Load the current data (new data to be monitored) 
    curr_artifact = wandb.use_artifact('anicin/credit-card-fraud-detection-test-2/original_data.csv:latest', type='dataset') 
    curr_artifact_dir = curr_artifact.download() 
    current_data = pd.read_csv(os.path.join(curr_artifact_dir, 'balanced_dataset.csv')) 
 
    return reference_data, current_data 
 
def detect_drift(reference_data, current_data, threshold=0.05): 
    """ 
    Detect data drift between reference and current datasets. 
     
    :param reference_data: pandas DataFrame, the reference dataset 
    :param current_data: pandas DataFrame, the current dataset 
    :param threshold: float, significance level to detect drift 
    :return: dict, drift detection results for each feature 
    """ 
    drift_results = {} 
    scaler = StandardScaler() 
     
    # Scaling both datasets 
    # Remove trans_date_trans_time column 
    reference_data = reference_data.drop(columns=["trans_date_trans_time"]) 
    current_data = current_data.drop(columns=["trans_date_trans_time"]) 
                                          
    reference_data_scaled = pd.DataFrame(scaler.fit_transform(reference_data), columns=reference_data.columns) 
    current_data_scaled = pd.DataFrame(scaler.transform(current_data), columns=current_data.columns) 
     
    # Perform Kolmogorov-Smirnov test for each feature 
    for column in reference_data_scaled.columns: 
        stat, p_value = ks_2samp(reference_data_scaled[column], current_data_scaled[column]) 
        drift_results[column] = {"p_value": p_value, "drift_detected": p_value < threshold} 
         
        # Log drift detection results 
        wandb.log({f"{column}_drift_p_value": p_value, f"{column}_drift_detected": p_value < threshold}) 
 
    return drift_results 
 
def setup_alerts(drift_results): 
    """ 
    Set up alerts for data drift using WandB. 
     
    :param drift_results: dict, results of the drift detection 
    """ 
    drift_detected = any(result["drift_detected"] for result in drift_results.values()) 
    wandb.log({"data_drift_detected": drift_detected}) 
 
    if drift_detected: 
        wandb.alert( 
            title="Data Drift Alert", 
            text="Data drift detected in the monitored dataset. Immediate action may be required.", 
            level=wandb.AlertLevel.WARN 
        ) 
 
def main(): 
    mlflow.start_run() 
 
    # Load reference and current data from WandB 
    reference_data, current_data = load_data() 
 
    # Detect data drift 
    drift_results = detect_drift(reference_data, current_data) 
 
    # Set up alerts if drift is detected 
    setup_alerts(drift_results) 
     
    # Log drift detection summary to MLFlow 
    for feature, result in drift_results.items(): 
        mlflow.log_param(f"{feature}_drift_detected", result["drift_detected"]) 
        mlflow.log_metric(f"{feature}_drift_p_value", result["p_value"]) 
 
    mlflow.end_run() 
 
if __name__ == "__main__": 
    main()