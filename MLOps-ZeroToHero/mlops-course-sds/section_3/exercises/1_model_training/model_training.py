import os
import pandas as pd
import numpy as np
import mlflow
import wandb
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib

# Initialize WandB for tracking the model training process
wandb.init(project="credit-card-fraud-detection-1", name="model-training")

def load_data():
    """
    Load the engineered dataset from WandB Artifacts.
    
    :return: pandas DataFrame, feature matrix and target vector
    """
    artifact = wandb.use_artifact('dvalia-self-employed/credit-card-fraud-detection-1/engineered_data.csv:latest', type='dataset')
    artifact_dir = artifact.download()

    # Load the dataset
    data = pd.read_csv(os.path.join(artifact_dir, 'engineered_data.csv'))
    
    # Assume the target variable is 'is_fraud' and features are the rest
    X = data.drop(columns=['is_fraud'])
    y = data['is_fraud']
    
    return X, y

def train_model(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost model and evaluate its performance.
    
    :param X_train: Training feature matrix
    :param y_train: Training target vector
    :param X_test: Test feature matrix
    :param y_test: Test target vector
    :return: trained model, evaluation metrics
    """
    # Initialize XGBoost model with eval_metric='logloss' and use_label_encoder=False
    model = xgb.XGBClassifier(
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics (accuracy, precision, recall, f1_score) and store them in the metrics dictionary
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

    # Log metrics to WandB
    wandb.log(metrics)

    # Infer and return signature for the model deployment
    signature = infer_signature(X_train, model.predict(X_train))

    return model, metrics, signature

def main():
    mlflow.start_run()

    # Load the data from WandB
    X, y = load_data()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model and evaluate its performance
    model, metrics, signature = train_model(X_train, y_train, X_test, y_test)

    # Log parameters and metrics to MLFlow
    # Log model parameters
    mlflow.log_params({
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42,
        'test_size': 0.2
    })
    
    # Log params and metrics
   # mlflow.log_params("model_type", "XGBoost")
   # mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)

    # Save the model to a file
    model_path = "fraud-detection_xgboost_model.json"
    model.save_model(model_path)
   
    # Log the model to WandB and mlflow
    model_artifact = wandb.Artifact(
        name="fraud-detection-model",
        type="model",
        description="XGBoost model for credit card fraud detection"
    )
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)
   # mlflow.log_artifact(model_path)
    
    # Log model to MLFlow and prepare it for deployment
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5),
        registered_model_name="fraud-detection-xgboost"
    )
    

    mlflow.end_run()

if __name__ == "__main__":
    main()
