import mlflow
import wandb
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def prepare_data():
    # Load dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Log dataset info
    mlflow.log_param("dataset", "iris")
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Start wandb run
    wandb.init(project="mlops-component-lesson-12", name="lesson-12-data-preparation", job_type="data-preparation")
    
    with mlflow.start_run(run_name="data_preparation"):
        X_train, X_test, y_train, y_test = prepare_data()
        
        # Save the datasets as artifacts
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_data.to_csv("train_data.csv", index=False)
        test_data.to_csv("test_data.csv", index=False)
        
        mlflow.log_artifact("train_data.csv")
        mlflow.log_artifact("test_data.csv")
        
        # Log datasets to wandb
        artifact = wandb.Artifact("train_data", type="dataset")
        artifact.add_file("train_data.csv")
        wandb.log_artifact(artifact)

        artifact = wandb.Artifact("test_data", type="dataset")
        artifact.add_file("test_data.csv")
        wandb.log_artifact(artifact)

    wandb.finish()