# Import necessary libraries
import wandb  # Weights and Biases for experiment tracking
import numpy as np
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Initialize a WandB project
# TODO: Initialize a new project in WandB by specifying the project name and run name.
wandb.init(project="my-first-wandb-mlops", name="lesson-4-logging-experiments")
# Step 2: Load and preprocess the data
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Solution

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Solution

# Step 3: Define hyperparameters and log them to WandB
# TODO: Define a dictionary of hyperparameters and log them using `wandb.config.update()`.
wandb.config.update({"n_estimators": 100, "max_depth": 3, "random_state": 42})

# Step 4: Train the model

model = RandomForestClassifier(
    n_estimators=wandb.config.n_estimators,
    max_depth=wandb.config.max_depth,
    random_state=wandb.config.random_state
)
model.fit(X_train, y_train)  # Solution

# Step 5: Evaluate the model and log metrics

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)  # Solution

# TODO: Log the accuracy to WandB using `wandb.log()`.
wandb.log({"accuracy": accuracy})


# Step 6: Save the model and log it to WandB
model_filename = "random_forest_model.pkl"
joblib.dump(model, model_filename)

# TODO: Log the model chekcpoint to WandB as an artifact using wandb.save().
artifact = wandb.Artifact(name="random_forest_model.pkl", type="model")
artifact.add_file(model_filename)
wandb.log_artifact(artifact)

# Step 7: Finish the WandB run
# TODO: End the WandB run
wandb.finish()



