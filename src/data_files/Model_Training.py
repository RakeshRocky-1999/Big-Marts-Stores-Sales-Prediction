import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

# Add the src folder to the system path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logger import logging

# Load preprocessed data
def load_data(processed_data_path, output_feature_path):
    X = pd.read_csv(processed_data_path)
    y = pd.read_csv(output_feature_path)
    if y.shape[1] == 1:  # If there's only one column, select it as a Series
        y = y.iloc[:, 0]
    logging.info(f"Data loaded from {processed_data_path} and {output_feature_path}")
    return X, y

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2

# Train and validate models
def train_models(X_train, y_train, X_test, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "AdaBoost": AdaBoostRegressor()
    }
    
    best_model = None
    best_r2 = -np.inf  # Initialize with the worst r2 score

    for model_name, model in models.items():
        logging.info(f"Training {model_name}")
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        mae, rmse, r2 = evaluate_model(model, X_test, y_test)
        logging.info(f"{model_name} Performance: MAE={mae}, RMSE={rmse}, R2={r2}")

        # Update the best model if this one has a higher R2 score
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            logging.info(f"New best model: {model_name} with R2: {best_r2}")

    return best_model

# Save the best model as a .pkl file
def save_model(model, directory='C:/Users/pc/OneDrive/Documents/Main_Projects/Sales_Store_Predictions/data/processed', filename='best_model.pkl'):
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Best model saved as {filename}")

if __name__ == "__main__":
    # Paths to the preprocessed data and output feature files
    train_processed_path = 'C:/Users/pc/OneDrive/Documents/Main_Projects/Sales_Store_Predictions/data/processed/train_processed.csv'
    output_feature_path = 'C:/Users/pc/OneDrive/Documents/Main_Projects/Sales_Store_Predictions/data/processed/output_feature.csv'
    
    # Load data
    X, y = load_data(train_processed_path, output_feature_path)
    
    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
    logging.info("Data split into training and validation sets")
    
    # Train models and get the best model
    best_model = train_models(X_train, y_train, X_test, y_test)
    
    # Save the best model
    save_model(best_model)