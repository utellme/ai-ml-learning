import subprocess

def run_pipeline():
    # Step 1: Download Data
    subprocess.run(["mlflow", "run", "download_data"])
    
    # Step 2: Clean Data
    subprocess.run(["mlflow", "run", "clean_data"])
    
    # Step 3: Feature Engineering
    subprocess.run(["mlflow", "run", "feature_engineering"])
    
    # Step 4: Train Model
    subprocess.run(["mlflow", "run", "train_model"])

if __name__ == "__main__":
    run_pipeline()
