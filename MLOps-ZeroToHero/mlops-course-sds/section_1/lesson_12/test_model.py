import requests
import json
import pandas as pd
import numpy as np

# URL of the MLflow model server
url = "http://localhost:5005/invocations"

# Sample data (Iris dataset features)
sample_data = pd.DataFrame({
    "sepal length (cm)": [5.1, 4.9, 6.3],
    "sepal width (cm)": [3.5, 3.0, 2.5],
    "petal length (cm)": [1.4, 1.4, 4.9],
    "petal width (cm)": [0.2, 0.2, 1.5]
})

# Prepare the payload
payload = {
    "dataframe_split": sample_data.to_dict(orient="split")
}

# Set the headers
headers = {
    "Content-Type": "application/json",
}

# Send the POST request
try:
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    # Check the response
    if response.status_code == 200:
        print("Success!")
        predictions = response.json()
        print("Predictions:")
        for i, pred in enumerate(predictions):
            print(f"Sample {i+1}: {pred}")
    else:
        print("Error:")
        print("Status Code:", response.status_code)
        print("Response:", response.text)
except requests.exceptions.RequestException as e:
    print("Error sending request:", e)

# Print the payload for verification
print("\nSent payload:")
print(json.dumps(payload, indent=2))