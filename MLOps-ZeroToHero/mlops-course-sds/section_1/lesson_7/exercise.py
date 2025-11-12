# Import necessary libraries
import os
os.environ['WANDB_TIMEOUT'] = '60'
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

# Step 2: Load the original dataset
iris = datasets.load_iris()
X_original = iris.data
y = iris.target 
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_original, y, test_size=0.2, random_state=42)


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

# TODO: Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="my-first-wandb-mlops")

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
