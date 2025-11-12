import os
import mlflow
import wandb
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average='weighted'),
        "recall": recall_score(y_test, predictions, average='weighted'),
        "f1": f1_score(y_test, predictions, average='weighted')
    }
    return metrics, predictions

if __name__ == "__main__":
    # Start wandb run
    wandb.init(project="mlops-component-lesson-12", name="lesson-12-evaluation")
    
    with mlflow.start_run(run_name="model_evaluation"):
        # Download the test data from wandb
        test_artifact = wandb.use_artifact('test_data:latest')
        test_path = test_artifact.download()
        if os.path.islink(test_path):
            print("File is a symlink")
            test_path = os.path.realpath(test_path)
            print(f"Real path: {test_path}")
        else:
            print("File is not a symlink")
        test_data = pd.read_csv(test_path + "/test_data.csv")
        
        X_test = test_data.drop("target", axis=1)
        y_test = test_data["target"]
        
        # Download the trained model from wandb
        model_artifact = wandb.use_artifact('random_forest_model:latest')
        model_path = model_artifact.download()
        model = joblib.load(model_path + "/model.pkl")
        
        # Evaluate the model
        metrics, predictions = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        wandb.log(metrics)
        
        # Save and log predictions
        predictions_df = pd.DataFrame({"predictions": predictions})
        predictions_df.to_csv("predictions.csv", index=False)
        mlflow.log_artifact("predictions.csv")
        wandb.log_artifact("predictions.csv", name="predictions", type="predictions")
    
    wandb.finish()