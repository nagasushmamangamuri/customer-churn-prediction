from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('models/churn_model.pkl')
print("Model loaded successfully.")

# Load the column names used during training
column_names_path = 'data/column_names.txt'
if not os.path.exists(column_names_path):
    raise FileNotFoundError(f"Column names file not found: {column_names_path}")

with open(column_names_path, 'r') as f:
    expected_columns = f.read().splitlines()


# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json(force=True)
        print("Received input data:", data)  # Debugging: Print input data

        # Convert the input data into a DataFrame
        input_data = pd.DataFrame([data['features']])
        print("Input data as DataFrame:", input_data)  # Debugging: Print DataFrame

        # Assign the correct column names to the input data
        input_data.columns = expected_columns
        print("Input data with column names:", input_data)  # Debugging: Print DataFrame with column names

        # Make a prediction
        prediction = model.predict(input_data)
        print("Prediction:", prediction)  # Debugging: Print prediction

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        print("Error:", str(e))  # Debugging: Print the error
        return jsonify({'error': str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)