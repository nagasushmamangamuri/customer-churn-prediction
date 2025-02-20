# Customer Churn Prediction

This project involves predicting customer churn using machine learning algorithms to analyze historical customer data and predict the likelihood of customers leaving the service.

## Project Overview

- **Objective**: Predict customer churn based on customer data.
- **Goal**: Enable businesses to proactively intervene and retain customers at risk of leaving.
- **Algorithms used**: Logistic Regression, Random Forest, and XGBoost.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

## 1. Introduction

- **Goal**: Analyze customer data to predict churn (whether a customer will leave the service).
- **Techniques**:
  - Data cleaning and preprocessing.
  - Feature engineering and transformation.
  - Model selection and training (Logistic Regression, Random Forest, XGBoost).
  - Model evaluation and performance analysis.

## 2. Installation

- **Clone the Repository**: 
  ```bash
  git clone https://github.com/nagasushmamangamuri/customer-churn-prediction.git
  cd customer-churn-prediction
  ```

- **Install Dependencies**: 
  - Ensure Python 3.x is installed.
  - Install required libraries using:
    ```bash
    pip install -r requirements.txt
    ```

## 3. Dataset

- **Customer Data**: The dataset includes customer demographics, subscription details, and service usage patterns.
- **Key Features**:
  - `CustomerID`: Unique identifier for each customer.
  - `Gender`: Gender of the customer.
  - `Age`: Age of the customer.
  - `Tenure`: Duration of subscription in months.
  - `Services`: Types of services subscribed (e.g., phone, internet).
  - `Churn`: Target variable (1 = Churned, 0 = Not Churned).

- **Dataset Location**: `data/` folder.

## 4. Data Preprocessing

- **Handling Missing Values**: Missing values are handled with imputation or removal.
- **Encoding Categorical Variables**: Convert variables like `Gender` and `Services` into numeric format.
- **Feature Scaling**: Numerical features are standardized for better model performance.
- **Feature Selection**: Important features are selected using correlation and feature importance techniques.

Example of Data Preprocessing:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('data/customer_churn.csv')

# Data Preprocessing steps
data.fillna(method='ffill', inplace=True)  # Handle missing values

X = data.drop('Churn', axis=1)
y = data['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 5. Modeling

- **Algorithms Used**:
  - **Logistic Regression**: Simple binary classification.
  - **Random Forest**: Ensemble method using multiple decision trees.
  - **XGBoost**: Gradient boosting algorithm for improved performance.

Example of Training Random Forest Model:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

## 6. Evaluation

- **Metrics** Used to Evaluate Models:
  - **Accuracy**: Percentage of correct predictions.
  - **Precision**: Percentage of correct positive predictions.
  - **Recall**: Percentage of actual positives correctly identified.
  - **F1-Score**: Harmonic mean of precision and recall.

Example Evaluation Code:

```python
from sklearn.metrics import classification_report

# Print classification report
print(classification_report(y_test, y_pred))
```

## 7. Results

- **Model Performance**:
  - **Logistic Regression**: Accuracy = 0.80, Precision = 0.78, Recall = 0.83, F1-Score = 0.80
  - **Random Forest**: Accuracy = 0.82, Precision = 0.80, Recall = 0.85, F1-Score = 0.82
  - **XGBoost**: Accuracy = 0.84, Precision = 0.83, Recall = 0.86, F1-Score = 0.84

- **Best Model**: XGBoost with the highest F1-Score.

## 8. Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/nagasushmamangamuri/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the churn prediction script:
   ```bash
   python churn_prediction.py
   ```

This will load the dataset, preprocess the data, train the model, and output the evaluation results.

## 9. Contributing

- **How to Contribute**: 
  - Fork the repository.
  - Make improvements or fix bugs.
  - Submit a pull request with your changes.
  
- **Issues and Suggestions**: If you encounter any issues or have suggestions, open an issue in the repository.

## 10. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
