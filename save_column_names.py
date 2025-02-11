import pandas as pd
import os

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

# Load the cleaned dataset
df = pd.read_csv('data/cleaned_churn_data.csv')

# Extract the column names (excluding the target column 'Churn')
column_names = [col for col in df.columns if col != 'Churn']

# Save the column names to a file
with open('data/column_names.txt', 'w') as f:
    f.write('\n'.join(column_names))

print("Column names saved to 'data/column_names.txt'.")
