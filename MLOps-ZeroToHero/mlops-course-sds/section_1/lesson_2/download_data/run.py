import pandas as pd
import requests

def download_data(url, output_path):
    # Download the dataset from the given URL
    response = requests.get(url)
    with open(output_path, 'wb') as f:
        f.write(response.content)

def main():
    url = "https://example.com/house-prices-dataset.csv"  # Replace with a real dataset URL
    output_path = "data/house_prices.csv"
    download_data(url, output_path)
    print(f"Data downloaded to {output_path}")

if __name__ == "__main__":
    main()
