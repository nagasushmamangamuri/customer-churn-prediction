import pandas as pd
import os

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

# Load the raw dataset
df = pd.read_csv(r"C:\Users\nagas\Downloads\archive (5)\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())  # Fixed warning

# Convert categorical variables to numerical
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop irrelevant columns
df.drop(columns=['customerID'], inplace=True)

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Save the cleaned dataset
df.to_csv('data/cleaned_churn_data.csv', index=False)
print("Cleaned dataset saved to 'data/cleaned_churn_data.csv'.")