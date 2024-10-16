import pandas as pd
import numpy as np
import sys
import os

# Add the src folder to the system path if needed
# sys.path.append(os.path.abspath('../src'))

from src.logger import logging
from sklearn.preprocessing import LabelEncoder

logging.info('importing necessary libraries for preprocess.py')

def handle_missing_values(df):
    # Impute Item_Weight with median
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())
    
    # Impute Outlet_Size with the first mode safely
    if not df['Outlet_Size'].mode().empty:  # Check if mode is available
        df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])
    else:
        logging.warning("Outlet_Size column has no mode value.")
    
    logging.info("Missing values handled")
    return df

def encode_categorical(df):
    le = LabelEncoder()
    categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
                        'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    logging.info("Categorical variables encoded")
    return df

def preprocess_data(df):
    df = handle_missing_values(df)
    df = encode_categorical(df)
    return df

if __name__ == "__main__":
    train_df = pd.read_csv('C:/Users/pc/OneDrive/Documents/Main_Projects/Sales_Store_Predictions/data/Train.csv')
    test_df = pd.read_csv('C:/Users/pc/OneDrive/Documents/Main_Projects/Sales_Store_Predictions/data/Test.csv')
    
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    
    # Ensure that the 'processed' folder exists
    processed_dir = 'C:/Users/pc/OneDrive/Documents/Main_Projects/Sales_Store_Predictions/data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    train_df.to_csv(f'{processed_dir}/train_processed')
    test_df.to_csv(f'{processed_dir}/test_processed')