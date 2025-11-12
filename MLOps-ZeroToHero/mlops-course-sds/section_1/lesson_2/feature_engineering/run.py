import pandas as pd
from sklearn.preprocessing import StandardScaler

def feature_engineering(input_path, output_path):
    # Load the cleaned data
    df = pd.read_csv(input_path)
    
    # Example feature engineering: Scaling numerical features
    scaler = StandardScaler()
    df[['size', 'num_rooms']] = scaler.fit_transform(df[['size', 'num_rooms']])
    
    # Save the engineered features
    df.to_csv(output_path, index=False)
    print(f"Engineered features saved to {output_path}")

def main():
    input_path = "data/cleaned_house_prices.csv"
    output_path = "data/engineered_house_prices.csv"
    feature_engineering(input_path, output_path)

if __name__ == "__main__":
    main()
