import pandas as pd
import pickle
from preprocess_engeneering import FeatureEngineering

def load_preprocessor_and_model(preprocessor_path, model_path):
    # Load preprocessor
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Load trained model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return preprocessor, model


def preprocess_test_data(test_df, preprocessor):
    # Apply preprocessing on test data
    test_processed = pd.DataFrame(preprocessor.transform(test_df).toarray())
    return test_processed

def predict_on_test(test_path, preprocessor, model, output_path):
    # Load test data
    test_df = pd.read_csv(test_path)
    
    # Preprocess the test data
    test_processed = preprocess_test_data(test_df, preprocessor)
    
    # Make predictions
    predictions = model.predict(test_processed)
    
    # Save predictions
    pd.DataFrame(predictions, columns=["Predicted_Sales"]).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    test_path='C:\\Users\\pc\\OneDrive\\Documents\\Main_Projects\\Sales_Store_Predictions\\data\\Test.csv'
    model_path = 'C:\\Users\\pc\\OneDrive\\Documents\\Main_Projects\\Sales_Store_Predictions\\data\processed\\best_model.pkl'
    preprocessor_path = 'C:\\Users\\pc\\OneDrive\\Documents\\Main_Projects\\Sales_Store_Predictions\\data\\processed\\preprocessor.pkl'
    output_path = 'C:\\Users\\pc\\OneDrive\\Documents\\Main_Projects\\Sales_Store_Predictions\\data\\processed\\test_predicted.csv'
    
    preprocessor, model=load_preprocessor_and_model(preprocessor_path, model_path)
    predict_on_test(test_path, preprocessor, model, output_path)