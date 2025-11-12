import os
import pandas as pd
import numpy as np
import mlflow
import wandb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

def load_data():
    """
    Load the engineered dataset from WandB Artifacts.
    
    :return: pandas DataFrame, feature matrix and target vector
    """
    wandb.init(project="credit-card-fraud-detection-test-515", name="hyperparameter-tuning")

    artifact = wandb.use_artifact('anicin/credit-card-fraud-detection-test-515/engineered_data.csv:latest', type='dataset')
    artifact_dir = artifact.download()

    # Load the dataset
    data = pd.read_csv(os.path.join(artifact_dir, 'engineered_data.csv'))
    
    # Assume the target variable is 'is_fraud' and features are the rest
    X = data.drop(columns=['is_fraud'])
    y = data['is_fraud']
    
    return X, y

def train_model(config=None):
    """
    Train an XGBoost model using the specified hyperparameters and evaluate its performance.
    
    :param config: dict, hyperparameters to use for training the model
    :return: evaluation metrics
    """
    # Initialize a new wandb run
    run = wandb.init(config=config, project="credit-card-fraud-detection-test-515", job_type="train")
    
    # Access all hyperparameter values through wandb.config
    config = wandb.config

    # Load the data
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost model with the provided hyperparameters
    model = xgb.XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score_value = f1_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score_value
    }

    # Log metrics to WandB
    wandb.log(metrics)

    return metrics

def main(run_type="FAST_RUN"):
    mlflow.start_run()

    # Choose the configuration based on the FAST_RUN environment variable
    if run_type == 'FAST_RUN':
        sweep_config = fast_sweep_config
        max_runs = 2
        print("Running fast hyperparameter tuning...")
    else:
        sweep_config = full_sweep_config
        max_runs = None  # Allow full sweep to run to completion
        print("Running full hyperparameter tuning...")

    # Run the WandB Sweep for hyperparameter tuning
    sweep_id = wandb.sweep(sweep_config, project="credit-card-fraud-detection-test-515")
    
    # Custom run function to limit the number of runs
    def run_sweep():
        run_count = 0
        while run_count < max_runs if max_runs is not None else True:
            wandb.agent(sweep_id, function=train_model, count=1)
            run_count += 1
            if max_runs is not None:
                print(f"Completed {run_count}/{max_runs} runs")

    run_sweep()

    # Get the best run from the sweep
    api = wandb.Api()
    sweep = api.sweep(f"credit-card-fraud-detection-test-515/{sweep_id}")
    best_run = sweep.best_run()
    
    print(f"Best run: {best_run}")

    # Train the model with the best hyperparameters
    best_config = best_run.config
    best_model = xgb.XGBClassifier(
        n_estimators=best_config['n_estimators'],
        max_depth=best_config['max_depth'],
        learning_rate=best_config['learning_rate'],
        subsample=best_config['subsample'],
        use_label_encoder=False,
        eval_metric='logloss'
    )

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    best_f1_score = f1_score(y_test, y_pred)

    # Try to load the baseline model from MLflow Models
    client = mlflow.tracking.MlflowClient()
    baseline_model_name = "XGBoost Baseline"
    baseline_model = None

    for mv in client.search_model_versions(f"name='{baseline_model_name}'"):
        baseline_model = mlflow.xgboost.load_model(mv.source)
        break  # Load the first (latest) version

    if baseline_model is not None:
        # Evaluate the baseline model
        baseline_pred = baseline_model.predict(X_test)
        baseline_f1_score = f1_score(y_test, baseline_pred)

        # Compare and choose the best model
        if best_f1_score > baseline_f1_score:
            print(f"NEW MODEL IS BETTER. Using tuned model for production. F1 Score: {best_f1_score}")
            production_model = best_model
            production_f1_score = best_f1_score
        else:
            print(f"BASELINE MODEL IS BETTER. Using baseline model for production. F1 Score: {baseline_f1_score}")
            production_model = baseline_model
            production_f1_score = baseline_f1_score
    else:
        print(f"NO BASELINE MODEL FOUND. Using tuned model for production. F1 Score: {best_f1_score}")
        production_model = best_model
        production_f1_score = best_f1_score

    # Create model signature
    signature = mlflow.models.infer_signature(X_train, production_model.predict(X_train))

    # Register the production model in MLflow
    model_name = "prod_model"
    mlflow.xgboost.log_model(
        production_model,
        "production_model",
        registered_model_name=model_name,
        signature=signature,
        input_example=X_train.iloc[:5]
    )

    # Log the F1 score as a metric in MLflow
    mlflow.log_metric("production_f1_score", production_f1_score)

    # Log the production model to wandb
    wandb.init(project="credit-card-fraud-detection-test-515", name="production-model")
    joblib.dump(production_model, "production_model.pkl")
    artifact = wandb.Artifact("production_model", type="model")
    artifact.add_file("production_model.pkl")
    wandb.log_artifact(artifact)
    wandb.finish()

    mlflow.end_run()

if __name__ == "__main__":
    # Define the full hyperparameter sweep configuration
    full_sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'f1_score',
            'goal': 'maximize'
        },
        'parameters': {
            'n_estimators': {'values': [100, 200, 300]},
            'max_depth': {'values': [3, 4, 5, 6, 7, 8, 9, 10]},
            'learning_rate': {'values': [0.01, 0.05, 0.1, 0.2]},
            'subsample': {'values': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
        }
    }

    # Define a smaller configuration for faster runs
    fast_sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'f1_score',
            'goal': 'maximize'
        },
        'parameters': {
            'n_estimators': {'values': [100, 200]},
            'max_depth': {'values': [3, 5, 7]},
            'learning_rate': {'values': [0.01, 0.1]},
            'subsample': {'values': [0.7, 0.9]}
        }
    }

    main(run_type="FAST_RUN")