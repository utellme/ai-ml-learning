import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def train_model(input_path):
    # Load the data
    df = pd.read_csv(input_path)
    X = df[['size', 'num_rooms']]
    y = df['price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # Log the model and metrics to MLFlow
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print(f"Model trained with MSE: {mse}")

def main():
    input_path = "data/engineered_house_prices.csv"
    train_model(input_path)

if __name__ == "__main__":
    main()
