import pandas as pd

def clean_data(input_path, output_path):
    # Load the data
    df = pd.read_csv(input_path)
    
    # Example cleaning steps
    df = df.dropna()  # Drop rows with missing values
    df = df[df['price'] > 0]  # Remove rows where price is not positive
    
    # Save the cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

def main():
    input_path = "data/house_prices.csv"
    output_path = "data/cleaned_house_prices.csv"
    clean_data(input_path, output_path)

if __name__ == "__main__":
    main()
