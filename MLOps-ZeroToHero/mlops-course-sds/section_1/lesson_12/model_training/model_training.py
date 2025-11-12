import os
import mlflow
import mlflow.sklearn
import wandb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env


def train_model(X_train, y_train, hyperparameters):
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Start wandb run
    wandb.init(project="mlops-component-lesson-12", name="lesson-12-training", job_type="training")
    
    with mlflow.start_run(run_name="model_training") as run:
        # Download the training data from wandb
        train_artifact = wandb.use_artifact('train_data:latest')
        train_path = train_artifact.download()
        train_data = pd.read_csv(os.path.join(train_path, "train_data.csv"))
        
        X_train = train_data.drop("target", axis=1)
        y_train = train_data["target"]
        
        # Define hyperparameters
        hyperparameters = {
            "n_estimators": 100,
            "max_depth": 3,
            "random_state": 42
        }
        
        # Log hyperparameters
        mlflow.log_params(hyperparameters)
        wandb.config.update(hyperparameters)
        
        # Train the model
        model = train_model(X_train, y_train, hyperparameters)
        
        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Create an example of model input
        input_example = X_train.iloc[:5].to_dict(orient="split")
        
        # Define conda environment
        conda_env = _mlflow_conda_env(
            additional_pip_deps=["scikit-learn", "pandas"]
        )
        
        # Save the model
        model_path = "random_forest_model"
        mlflow.sklearn.save_model(
            sk_model=model,
            path=model_path,
            conda_env=conda_env,
            signature=signature,
            input_example=input_example
        )
        
        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_path,
            conda_env=conda_env,
            signature=signature,
            input_example=input_example
        )
        
        # Log feature importances
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mlflow.log_table(data=feature_importance, artifact_file="feature_importance.csv")
        
        print(f"Model saved in run {run.info.run_id}")
        print(f"Model path: {os.path.abspath(model_path)}")
    
    wandb.finish()

print("Model training completed. The model is saved and can be served using:")
print(f"mlflow models serve -m {os.path.abspath(model_path)} -p 5000")