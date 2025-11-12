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
wandb.init(project="first-project-mlops-course", name="lesson-6-versioning-multiple")

# Step 2: Load the original dataset
iris = datasets.load_iris()
X_original = iris.data
y = iris.target 

# Step 3: Log the original dataset to WandB
# TODO: Log the original dataset as an artifact to WandB.
artifact_original = wandb.Artifact(name="iris-dataset-original", type="dataset")
artifact_original.add(wandb.Table(data=X_original, columns=iris.feature_names))
wandb.log_artifact(artifact_original)

# Step 4: Preprocess the dataset (Cleaned Version)
scaler = StandardScaler()
X_cleaned = scaler.fit_transform(X_original)

# TODO: Log the cleaned dataset as a new versioned artifact to WandB.
artifact_cleaned = wandb.Artifact(name="iris-dataset-cleaned", type="dataset")
artifact_cleaned.add(wandb.Table(data=X_cleaned, columns=iris.feature_names))
wandb.log_artifact(artifact_cleaned)

# Step 5: Feature Engineering (Feature-Engineered Version)
X_feature_engineered = np.c_[X_cleaned, X_cleaned[:, 0] * X_cleaned[:, 1]]
feature_names = iris.feature_names + ["feature_0_1_product"]

# TODO: Log the feature-engineered dataset as another versioned artifact to WandB.
artifact_feature_engineered = wandb.Artifact(name="iris-dataset-feature-engineered", type="dataset")
artifact_feature_engineered.add(wandb.Table(data=X_feature_engineered, columns=feature_names))
wandb.log_artifact(artifact_feature_engineered)

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

wandb.log({"accuracy_original": accuracy_orig})

model_filename_orig = "rf_model_original.pkl"
joblib.dump(model_original, model_filename_orig)

# TODO: Log the original model as an artifact to WandB.
artifact_original = wandb.Artifact(name="rf_model_original", type="model")
artifact_original.add_file(model_filename_orig)
wandb.log_artifact(artifact_original)



# Train and log model on cleaned dataset
X_train_cleaned, X_test_cleaned, _, _ = train_test_split(X_cleaned, y, test_size=0.2, random_state=42)
model_cleaned = RandomForestClassifier(**hyperparameters)
model_cleaned.fit(X_train_cleaned, y_train)
accuracy_cleaned = accuracy_score(y_test, model_cleaned.predict(X_test_cleaned))

wandb.log({"accuracy_cleaned": accuracy_cleaned})

model_filename_cleaned = "rf_model_cleaned.pkl"
joblib.dump(model_cleaned, model_filename_cleaned)

# TODO: Log the cleaned model as an artifact to WandB.
artifact_cleaned = wandb.Artifact(name="rf_model_cleaned", type="model")
artifact_cleaned.add_file(model_filename_cleaned)
wandb.log_artifact(artifact_cleaned)


# Homework: Do the same for the feature-engineered dataset and compare the findings.

# Step 7: Finish the WandB run
wandb.finish()
