import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Load the dataset
df = pd.read_csv(r"C:\Users\nagas\Downloads\archive (5)\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Convert categorical variables to numerical
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Debug: Check columns before dropping
print("Columns in DataFrame before dropping:", df.columns)

# Drop irrelevant columns (if they exist)
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)
else:
    print("Column 'customerID' not found in DataFrame.")

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Save the cleaned data
df.to_csv('data/cleaned_churn_data.csv', index=False)
print("Data cleaned and preprocessed successfully.")

# Exploratory Data Analysis (EDA)
sns.set_theme(style="whitegrid")

# Save churn distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, hue='Churn', palette='Set2', legend=False)
plt.title('Churn Distribution')
plt.savefig('plots/churn_distribution.png')  # Save the plot
plt.close()  # Close the plot
print("Churn distribution plot saved.")

# Save correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('plots/correlation_heatmap.png')  # Save the plot
plt.close()  # Close the plot
print("Correlation heatmap saved.")

# Save tenure vs churn boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='tenure', data=df, hue='Churn', palette='Set3', legend=False)
plt.title('Tenure vs Churn')
plt.savefig('plots/tenure_vs_churn.png')  # Save the plot
plt.close()  # Close the plot
print("Tenure vs churn boxplot saved.")

# Split the data into features (X) and target (y)
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Data split into training and testing sets.")
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Model training completed.")

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
joblib.dump(rf_model, 'models/churn_model.pkl')
print("Model saved as 'churn_model.pkl'")