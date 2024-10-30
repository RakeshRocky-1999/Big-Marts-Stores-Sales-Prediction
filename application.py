from flask import Flask, render_template, request  # type: ignore
import pandas as pd
import pickle
from src.logger import logging
from src.data_files.preprocess_engeneering import FeatureEngineering
import os

application = Flask(__name__)

# Get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load preprocessor and model
def load_preprocessor_and_model(preprocessor_path, model_path):
    logging.info("Loading preprocessor and model from disk.")
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logging.info("Preprocessor and model loaded successfully.")
    return preprocessor, model

# Load unique values for categorical columns from training data
def load_unique_values(train_data_path):
    logging.info(f"Loading training data from: {train_data_path}")
    train_df = pd.read_csv(train_data_path)
    
    # Categorical columns with unique values (used to populate dropdowns)
    categorical_columns = ['Item_Identifier', 'Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Type', 'Outlet_Identifier']
    
    unique_values = {col: train_df[col].unique().tolist() for col in categorical_columns}
    logging.info("Unique values loaded for categorical columns.")
    return unique_values

@application.route('/', methods=['GET', 'POST'])
def index():
    logging.info("Reached index route.")
    
    # Relative paths to the training data and model files
    train_data_path = os.path.join(BASE_DIR, 'data', 'Train.csv')
    preprocessor_path = os.path.join(BASE_DIR, 'data', 'processed', 'preprocessor.pkl')
    model_path = os.path.join(BASE_DIR, 'data', 'processed', 'best_model.pkl')
    
    logging.info("Loading unique values...")
    # Load categorical options from the training data
    unique_values = load_unique_values(train_data_path)
    logging.info(f"Loaded unique values: {unique_values}")
    
    if request.method == 'POST':
        logging.info("POST method triggered.")
        
        # Get form data including 'Outlet_Establishment_Year' and 'Item_Identifier'
        form_data = {
            'Item_Identifier': request.form['Item_Identifier'],
            'Item_Weight': request.form['Item_Weight'],
            'Item_Fat_Content': request.form['Item_Fat_Content'],
            'Item_Visibility': request.form['Item_Visibility'],
            'Item_MRP': request.form['Item_MRP'],
            'Item_Type': request.form['Item_Type'],
            'Outlet_Size': request.form['Outlet_Size'],
            'Outlet_Location_Type': request.form['Outlet_Location_Type'],
            'Outlet_Type': request.form['Outlet_Type'],
            'Outlet_Identifier': request.form['Outlet_Identifier'],
            'Outlet_Establishment_Year': request.form['Outlet_Establishment_Year']
        }
        
        logging.info(f"Form data received: {form_data}")
        
        # Convert form data to a DataFrame
        test_df = pd.DataFrame([form_data])
        
        # Convert numeric values to appropriate types
        test_df['Item_Weight'] = test_df['Item_Weight'].astype(float)
        test_df['Item_Visibility'] = test_df['Item_Visibility'].astype(float)
        test_df['Item_MRP'] = test_df['Item_MRP'].astype(float)
        test_df['Outlet_Establishment_Year'] = test_df['Outlet_Establishment_Year'].astype(int)  # Integer type
        
        # Load the preprocessor and model
        preprocessor, model = load_preprocessor_and_model(preprocessor_path, model_path)

        # Preprocess test data
        test_processed = pd.DataFrame(preprocessor.transform(test_df).toarray())
        logging.info(f"Processed test data: {test_processed.head()}")

        # Make prediction
        prediction = model.predict(test_processed)[0]
        logging.info(f"Prediction made: {prediction}")

        return render_template('index.html', prediction=prediction, unique_values=unique_values) 
    
    return render_template('index.html', unique_values=unique_values) 

if __name__ == '__main__':
    application.run(debug=True,port=8888)

