import mlflow
import wandb
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def prepare_data():
    # Load dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, hyperparameters):
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average='weighted'),
        "recall": recall_score(y_test, predictions, average='weighted'),
        "f1": f1_score(y_test, predictions, average='weighted')
    }
    return metrics, predictions

def main():
    # Initialize WandB
    wandb.init(project="mlops-component", name="single_component_run")
    
    # Start MLflow run
    with mlflow.start_run(run_name="single_component"):
        # Data preparation
        X_train, X_test, y_train, y_test = prepare_data()
        
        # Log dataset info
        mlflow.log_param("dataset", "iris")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Save and log datasets
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_data.to_csv("train_data.csv", index=False)
        test_data.to_csv("test_data.csv", index=False)
        
        mlflow.log_artifact("train_data.csv")
        mlflow.log_artifact("test_data.csv")
        
        wandb.log_artifact("train_data.csv", name="train_data", type="dataset")
        wandb.log_artifact("test_data.csv", name="test_data", type="dataset")
        
        # Define hyperparameters
        hyperparameters = {
            "n_estimators": 100,
            "max_depth": 3,
            "random_state": 42
        }
        
        # Log hyperparameters
        mlflow.log_params(hyperparameters)
        wandb.config.update(hyperparameters)
        
        # Model training
        model = train_model(X_train, y_train, hyperparameters)
        
        # Log the model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Save and log the model to wandb
        model_path = "random_forest_model.pkl"
        joblib.dump(model, model_path)
        wandb.log_artifact(model_path, name="random_forest_model", type="model")
        
        # Log feature importances
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mlflow.log_table(data=feature_importance, artifact_file="feature_importance.csv")
        wandb.log({"feature_importance": wandb.Table(dataframe=feature_importance)})
        
        # Model evaluation
        metrics, predictions = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        wandb.log(metrics)
        
        # Save and log predictions
        predictions_df = pd.DataFrame({"predictions": predictions})
        predictions_df.to_csv("predictions.csv", index=False)
        mlflow.log_artifact("predictions.csv")
        wandb.log_artifact("predictions.csv", name="predictions", type="predictions")

if __name__ == "__main__":
    main()