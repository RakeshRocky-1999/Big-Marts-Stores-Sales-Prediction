import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sys
# Add the src folder to the system path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logger import logging

# Define a class for the custom transformations such as feature engineering
class FeatureEngineering:
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['Outlet_Age'] = 2024 - X['Outlet_Establishment_Year']
        X.drop(columns=['Item_Identifier', 'Outlet_Establishment_Year'], inplace=True)
        logging.info("Feature engineering applied: 'Outlet_Age' created and unnecessary columns dropped.")
        return X

# Build the complete preprocessing pipeline
def build_preprocessing_pipeline():
    # Imputers for missing values
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    # Transformers for scaling and encoding
    numeric_transformer = Pipeline(steps=[
        ('imputer', num_imputer),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', cat_imputer),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Select numerical and categorical columns
    num_features = ['Item_Weight', 'Outlet_Age','Item_Visibility','Item_MRP']  # Outlet_Age is created in feature engineering
    cat_features = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type','Item_Type','Outlet_Identifier']
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])
    
    # Full pipeline including feature engineering and preprocessing
    pipeline = Pipeline(steps=[
        ('feature_engineering', FeatureEngineering()),
        ('preprocessor', preprocessor)
    ])
    
    logging.info("Preprocessing pipeline created")
    return pipeline

# Save preprocessing pipeline as a .pkl file
def save_preprocessor(pipeline,directory='C:/Users/pc/OneDrive/Documents/Main_Projects/Sales_Store_Predictions/data/processed', filename='preprocessor.pkl'):
    filepath = os.path.join(directory,filename)
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)
    logging.info(f"Preprocessing pipeline saved as {filename}")

if __name__ == "__main__":
    # Load the data
    train_df = pd.read_csv('C:/Users/pc/OneDrive/Documents/Main_Projects/Sales_Store_Predictions/data/Train.csv')
    test_df = pd.read_csv('C:/Users/pc/OneDrive/Documents/Main_Projects/Sales_Store_Predictions/data/Test.csv')

    # Separate the target variable
    X_train = train_df.drop(columns='Item_Outlet_Sales')
    y_train = train_df['Item_Outlet_Sales']
    
    # Build the preprocessing pipeline
    preprocessor_pipeline = build_preprocessing_pipeline()
    
    # Fit the pipeline to the training data and transform the data
    X_train_processed = pd.DataFrame(preprocessor_pipeline.fit_transform(X_train).toarray())

    # Save the preprocessing pipeline
    save_preprocessor(preprocessor_pipeline)

    
    # Save processed data
    train_processed_path = 'C:/Users/pc/OneDrive/Documents/Main_Projects/Sales_Store_Predictions/data/processed/train_processed.csv'
    output_path = 'C:/Users/pc/OneDrive/Documents/Main_Projects/Sales_Store_Predictions/data/processed/output_feature.csv'
    
    X_train_processed.to_csv(train_processed_path, index=False)
    y_train.to_csv(output_path, index=False)

    logging.info(f"Processed data saved at {train_processed_path}, and {output_path}")