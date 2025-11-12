# Import necessary libraries
import wandb  # Weights and Biases for experiment tracking
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Initialize a new WandB run
wandb.init(project="mlops-intro", name="lesson-7-advanced-versioning")

# Step 2: Load the original dataset
iris = datasets.load_iris()
X_original = iris.data
y = iris.target 

# Step 3: Log the original dataset to WandB
artifact_original = wandb.Artifact(name="iris-dataset-original", type="dataset")
artifact_original.add(wandb.Table(data=X_original, columns=iris.feature_names))
wandb.log_artifact(artifact_original)

# Step 4: Preprocess the dataset (Cleaned Version)
scaler = StandardScaler()
X_cleaned = scaler.fit_transform(X_original)

# Log the cleaned dataset as a new versioned artifact to WandB.
artifact_cleaned = wandb.Artifact(name="iris-dataset-cleaned", type="dataset")
artifact_cleaned.add(wandb.Table(data=X_cleaned, columns=iris.feature_names))
wandb.log_artifact(artifact_cleaned)

# Step 5: Feature Engineering (Feature-Engineered Version)
X_feature_engineered = np.c_[X_cleaned, X_cleaned[:, 0] * X_cleaned[:, 1]]
feature_names = iris.feature_names + ["feature_0_1_product"]

# Log the feature-engineered dataset as another versioned artifact to WandB.
artifact_feature_eng = wandb.Artifact(name="iris-dataset-feature-eng", type="dataset")
artifact_feature_eng.add(wandb.Table(data=X_feature_engineered, columns=feature_names))
wandb.log_artifact(artifact_feature_eng) 

# Step 6: Train and Version Multiple Models

# Define common hyperparameters
hyperparameters = {
    "n_estimators": 100,
    "max_depth": 3,
    "random_state": 42
}
wandb.config.update(hyperparameters)

# Train and log model on original dataset
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_original, y, test_size=0.2, random_state=42)
model_original = RandomForestClassifier(**hyperparameters)
model_original.fit(X_train_orig, y_train)
accuracy_orig = accuracy_score(y_test, model_original.predict(X_test_orig))

# Log the accuracy for the original model
wandb.log({"accuracy_original": accuracy_orig})

# Save and log the original model as an artifact
model_filename_orig = "rf_model_original.pkl"
joblib.dump(model_original, model_filename_orig)

model_artifact_orig = wandb.Artifact(name="rf_model_original", type="model")
model_artifact_orig.add_file(model_filename_orig)
wandb.log_artifact(model_artifact_orig) 

# Train and log model on cleaned dataset
X_train_cleaned, X_test_cleaned, _, _ = train_test_split(X_cleaned, y, test_size=0.2, random_state=42)
model_cleaned = RandomForestClassifier(**hyperparameters)
model_cleaned.fit(X_train_cleaned, y_train)
accuracy_cleaned = accuracy_score(y_test, model_cleaned.predict(X_test_cleaned))

wandb.log({"accuracy_cleaned": accuracy_cleaned})

# Save and log the cleaned model as an artifact
model_filename_cleaned = "rf_model_cleaned.pkl"
joblib.dump(model_cleaned, model_filename_cleaned)

model_artifact_cleaned = wandb.Artifact(name="rf_model_cleaned", type="model")
model_artifact_cleaned.add_file(model_filename_cleaned)
wandb.log_artifact(model_artifact_cleaned) 

# Train and log model on feature-engineered dataset
X_train_feat_eng, X_test_feat_eng, _, _ = train_test_split(X_feature_engineered, y, test_size=0.2, random_state=42)
model_feat_eng = RandomForestClassifier(**hyperparameters)
model_feat_eng.fit(X_train_feat_eng, y_train)
accuracy_feat_eng = accuracy_score(y_test, model_feat_eng.predict(X_test_feat_eng))

wandb.log({"accuracy_feature_eng": accuracy_feat_eng})

# Save and log the feature-engineered model as an artifact
model_filename_feat_eng = "rf_model_feature_eng.pkl"
joblib.dump(model_feat_eng, model_filename_feat_eng)

model_artifact_feat_eng = wandb.Artifact(name="rf_model_feature_eng", type="model")
model_artifact_feat_eng.add_file(model_filename_feat_eng)
wandb.log_artifact(model_artifact_feat_eng) 

# Step 7: Hyperparameter Tuning with WandB Sweeps
# TODO: Introduce hyperparameter sweeps with WandB.

# Define the sweep configuration
sweep_config = {
    "method": "grid",  # grid search
    "metric": {
        "name": "accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "n_estimators": {
            "values": [50, 100, 200]
        },
        "max_depth": {
            "values": [3, 5, 7]
        },
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="mlops-intro")

# Function to run the sweep
def train():
    # Initialize a new WandB run for each configuration
    wandb.init()

    # Use the hyperparameters selected by the sweep
    hyperparameters = {
        "n_estimators": wandb.config.n_estimators,
        "max_depth": wandb.config.max_depth,
        "random_state": 42
    }

    # Train the model with these hyperparameters
    model_sweep = RandomForestClassifier(**hyperparameters)
    model_sweep.fit(X_train_orig, y_train)
    accuracy_sweep = accuracy_score(y_test, model_sweep.predict(X_test_orig))
    
    # Log the accuracy for this sweep run
    wandb.log({"accuracy": accuracy_sweep})

# TODO: Execute the sweep
wandb.agent(sweep_id, train, count=9)

# Step 8: Finish the WandB run
wandb.finish()
