# Import necessary libraries
import wandb  # Weights and Biases for experiment tracking
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Step 1: Initialize a new WandB run
wandb.init(project="mlops-intro", name="lesson-5-visualizing-experiments") 

# Step 2: Load and preprocess the data
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  

# Step 3: Define hyperparameters and log them to WandB
hyperparameters = {
    "n_estimators": 100,
    "max_depth": 3,
    "random_state": 42
}
wandb.config.update(hyperparameters) 

# Step 4: Train the model
model = RandomForestClassifier(
    n_estimators=wandb.config.n_estimators,
    max_depth=wandb.config.max_depth,
    random_state=wandb.config.random_state
)
model.fit(X_train, y_train)  

# Step 5: Evaluate the model and log additional metrics
# TODO: Predict on the test data and calculate accuracy, precision, and recall.
# HINT: Use `model.predict()` to make predictions and `accuracy_score()`, `precision_score()`, and `recall_score()` to calculate metrics.

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

# TODO: Log the accuracy, precision, and recall to WandB using `wandb.log()`
wandb.log({"accuracy": accuracy, "precision": precision, "recall": recall}) 

# Step 6: End the WandB run
wandb.finish() 