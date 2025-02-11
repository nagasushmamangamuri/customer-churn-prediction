import joblib

# Load the trained model
model = joblib.load('models/churn_model.pkl')

# Check the number of features the model expects
print("Number of features expected by the model:", model.n_features_in_)