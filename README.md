
# Stores Sales Prediction


![NewTab-GoogleChrome2024-11-0510-45-56-ezgif com-crop](https://github.com/user-attachments/assets/2bb81bb1-6f7a-4fe2-aa51-abaa4c164a35)


# Table of Contents

1. Project Overview
2. Problem Statement
3. Technologies Used
4. Dataset Information
5. Installation and Setup
6. Project Structure
7. Data Preprocessing
8. Model Training
9. Prediction
10. API Integration
11. Logging
12. Cloud Database
13. System Architecture
14. Model Latency & Optimization
15. Future Improvements
16. License
17. Contact
18. How to use/usage
19. Demo

## Project Overview

The Stores Sales Prediction project aims to predict the sales of different Big Mart stores based on historical data. This project is implemented using machine learning techniques for data exploration, data cleaning, feature engineering, model building, and testing.

## Problem Statement

Shopping malls and Big Marts track individual item sales to forecast future demand and optimize inventory management. The objective of this project is to predict the sales for each product in various Big Mart outlets based on the provided dataset.

## Technologies Used
- Language: Python
- Frameworks: Flask (for API), AWS Elastic 
- Database: Astra Cassandra (NoSQL)
- Libraries: Scikit-learn, Pandas, NumPy, Matplotlib,seaborn, Pickle
- Version Control: Git & GitHub


## Dataset Information

The dataset used for this project consists of sales data of Big Mart stores. The dataset is split into training and test sets:

- Train Dataset: 8,523 rows with target variable (`Item_Outlet_Sales`).
- Test Dataset: 5,681 rows, where the sales need to be predicted.
The dataset is stored in Astra Cassandra database.

**Dataset link:**

https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data
# Features

- **Item_Identifier:** Unique product ID
- **Item_Weight:** Weight of the product
- **Item_Fat_Content:** Whether the product is low fat or not
- **Item_Visibility:** Display area percentage allocated to the product
- **Item_Type:** Product category
- **Item_MRP:** Maximum Retail Price of the product
- **Outlet_Identifier:** Unique store ID
- **Outlet_Establishment_Year:** Year of store establishment
- **Outlet_Size:** Size of the store (small, medium, high)
- **Outlet_Location_Type:** Type of city where store is located
- **Outlet_Type:** Whether the outlet is a grocery store or supermarket
- **Item_Outlet_Sales:** Sales of the product (target variable)
## Installation and Setup

1. Clone the repository:

```bash

git clone https://github.com/RakeshRocky-1999/Sales_Store_Prediction.git
cd Sales_Store_Prediction
```
GitHub link:
https://github.com/RakeshRocky-1999/Big-Marts-Stores-Sales-Prediction


2. Set up the Python virtual environment:

```bash

python -m venv bigmarts_env
conda activate bigmarts_env
```

3. Install the required dependencies:

```bash

pip install -r requirements.txt

```
4. Database setup: 

The data is imported from `Astra Cassandra`. Ensure the database credentials and configuration are correctly set up in the project.
    
## Project Structure

```bash

Sales_Store_Prediction/
│
├── data/                     # Contains Train.csv, Test.csv, processed files
│   ├── Train.csv
│   ├── Test.csv
│   ├── processed/
│       ├── preprocess.pkl
│       ├── best_model.pkl
│       ├── train_proceed.csv
│       ├── Output_feature.csv
│       ├── test_predicted.csv
│
├── src/
│   ├── logger.py              # Logging setup
│   ├── cassandra_database.ipynb # Importing data from Astra Cassandra DB
    ├──train_df.csv              #the dataset imported from cassandra
│   └── data_files/
│       ├── preprocess_engineering.py # Data preprocessing
│       ├── Model_Training.py  # Model training script
│       ├── predict.py         # Prediction script
│
├── notebooks/                 # Jupyter notebooks
│   ├── EDA.ipynb              # Exploratory data analysis
│   └── model_building_training.ipynb
│
├── templates/                 # HTML templates
│   └── index.html
├── .gitignore                 # Files to be ignored by Git
├── requirements.txt           # Required libraries
├── application.py             # Main Flask application
├── setup.py                   # Package setup file
├──LICENSE                     # MIT license
└── README.md                  # Project documentation

```
## Data Preprocessing

The preprocessing pipeline includes the following steps:

- Handling missing values using SimpleImputer.
- Scaling numeric data using StandardScaler.
- Encoding categorical variables using OneHotEncoder.
- Feature engineering: Created new features like Outlet_Age.

The preprocessing pipeline is saved as `preprocess.pkl` and applied to both train and test datasets.
## Model Training

Different machine learning models were trained, including:

- Linear Regression(Lasso , Ridge)
- Decision Tree regressor
- Random Forest regressor
- AdaBoost regressor
- KNeighborsRegressor

The model with the best performance (RandomForest) is saved as `best_model.pkl`.
## Prediction

Predictions for the test dataset are saved as `test_predicted.csv`. The prediction script uses the pre-trained model and preprocessing pipeline to make predictions based on new inputs.

## Logging

Logging is implemented using Python’s logging library. Logs are generated for data loading, preprocessing, model training, and API requests.
## Cloud Database
The dataset is imported from Astra Cassandra, a NoSQL cloud database, ensuring scalability and availability of the data.
## Project Workflow
**1. Exploratory Data Analysis (EDA)**

The notebook `EDA.ipynb` contains data exploration and analysis to understand the dataset and its key features.

**2. Preprocessing and Feature Engineering**

Feature engineering and preprocessing scripts are in `src/data_files/preprocess_engineering.py`.
Outputs of this step include `train_proceed.csv`, `Output_feature.csv`, and `preprocess.pkl`.

**3. Model Training**

The script src/data_files/Model_Training.py is used to train the machine learning models, including RandomForest, and save the best model as `best_model.pkl`.

**4. Prediction**

The `src/data_files/predict.py` script loads the trained model and preprocessor to generate predictions on test data, saving results to `test_predicted.csv`.



##  Model Latency & Optimization
The model response time was measured for various inputs, and the system was optimized at multiple levels:

Code optimization for faster predictions.
Cloud architecture optimizations to ensure minimal latency for API responses.
## Future Improvements

- Implement advanced machine learning models like `XGBoost` or `LightGBM`.
- Fine-tune hyperparameters to improve model performance.
- Expand the system for real-time data ingestion from a streaming source.
- Further optimize the preprocessing pipeline using tools like DVC for experiment tracking.

## License

License
This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License - see the LICENSE file for details.

## Contact

For any questions or suggestions, feel free to reach out at [rakeshrak8484@gmail.com].
## How to Use/Usage
Run the Flask app locally:

```bash
python application.py
```
check with url in same laptop  local browser 

```bash
http://127.0.0.1:8000/

```
Use the web form to input values for the features, and the app will return predicted sales based on the trained model.

## Demo
Demo `Gif` of the project how it works and predicts sales

![SalesStorePrediction-GoogleChrome2024-11-0114-54-51-ezgif com-effects](https://github.com/user-attachments/assets/71542512-6c53-4c44-9406-21d38e464a4c)
