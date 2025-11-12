# Import necessary libraries
import mlflow  # MLFlow for experiment tracking and model management
import mlflow.sklearn  # MLFlow's Scikit-learn integration
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Initialize MLFlow
# Mention that MLFlow is particularly strong in model versioning and deployment.
mlflow.set_experiment("mlops-intro")

# Step 2: Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Step 3: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model and log it with MLFlow

with mlflow.start_run(run_name="RF_Model_Original_Dataset"):
    # Train the model
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Log the accuracy with MLFlow
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the model with MLFlow
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Save the model with joblib and log it as an artifact with MLFlow
    model_filename = "rf_model_mlflow.pkl"
    joblib.dump(model, model_filename)
    mlflow.log_artifact(model_filename)

    # Log model parameters for reproducibility
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("random_state", 42)


# Step 5: Finish WandB and MLFlow runs
# TODO: Ensure both MLFlow and WandB runs are properly closed.
mlflow.end_run()
