import mlflow
# Import necessary libraries
import mlflow  # MLFlow for experiment tracking and model management
import mlflow.sklearn  # MLFlow's Scikit-learn integration
import wandb  # Weights and Biases for experiment tracking
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def main():
    # Initialize WandB
    wandb.init(project="mlops-integration", name="lesson-11-mlflow-wandb")

    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 5: Define model hyperparameters and log to both MLFlow and WandB
    hyperparameters = {
        "n_estimators": 100,
        "max_depth": 3,
        "random_state": 42
    }
    mlflow.log_params(hyperparameters)
    wandb.config.update(hyperparameters)

    # Train a model
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log the accuracy to MLFlow
    mlflow.log_metric("accuracy", accuracy)

    # TODO: Log the accuracy to WandB
    wandb.log({"accuracy": accuracy})

    # Log the model to MLFlow
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Save and log the model as an artifact for additional use
    model_filename = "rf_model_integration.pkl"
    joblib.dump(model, model_filename)
    mlflow.log_artifact(model_filename)

    wandb.log_artifact(model_filename)
    wandb.finish()

if __name__ == "__main__":
    # Start an MLFlow run and execute the main function
    with mlflow.start_run():
        main()
