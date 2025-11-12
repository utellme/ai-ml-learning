import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log the accuracy to MLFlow
    mlflow.log_metric("accuracy", accuracy)

    # Log the model to MLFlow
    mlflow.sklearn.log_model(model, "random_forest_model")

if __name__ == "__main__":
    # Start an MLFlow run and execute the main function
    with mlflow.start_run():
        main()
