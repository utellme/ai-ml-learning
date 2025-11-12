import subprocess
import mlflow
import wandb

# Initialize WandB for tracking the entire pipeline process
wandb.init(project="credit-card-fraud-detection", job_type="data-pipeline")

def run_component(script_path):
    """
    Run a data processing component via subprocess.
    
    :param script_path: str, path to the component script
    """
    result = subprocess.run(["python", script_path], capture_output=True, text=True)

    # Log the output of the component
    wandb.log({f"{script_path}_stdout": result.stdout})
    wandb.log({f"{script_path}_stderr": result.stderr})

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError(f"Component {script_path} failed with return code {result.returncode}")

def main():
    mlflow.start_run()

    try:
        # Run the data cleaning component
        run_component("./data_cleaning/data_cleaning.py")
        
        # Run the data validation component
        run_component("./data_validation/data_validation.py")
        
        # Run the feature engineering component
        run_component("./feature_engineering/feature_engineering.py")
        
        wandb.log({"pipeline_status": "Success"})
    
    except Exception as e:
        wandb.log({"pipeline_status": "Failed", "error": str(e)})
        raise e
    
    mlflow.end_run()

if __name__ == "__main__":
    main()
