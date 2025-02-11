import requests
import json

# Define the input data with 30 features
data = {
    'features': [1, 0, 0, 1, 34, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 56.95, 1889.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

# Send a POST request to the API
response = requests.post('http://127.0.0.1:5000/predict', json=data)

# Print the response
print("Prediction:", response.json())