import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from data_cleaning import clean_polling_data
from modeling import encode_data, select_features, preprocess_data  # Ensure preprocess_data is imported


# Step 1: Load the saved model
def load_model():
    models_dir = os.path.join('..', 'models')  # Corrected file path
    model_path = os.path.join(models_dir, 'random_forest_model.pkl')
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please ensure the model has been trained and saved.")
        return None
    
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

# Step 2: Load new data (or use existing cleaned data)
def load_new_data(file_path):
    print(f"looking for csv file at: {os.path.abspath(file_path)}")
    # Clean the new data using the cleaning function
    data = clean_polling_data(file_path)
    return data

# Step 3: Prepare the data for prediction
def prepare_data(data):
    # Preprocess, encode data and select features just like we did for training
    data_preprocessed = preprocess_data(data)  # Preprocess the data to clean and handle missing values
    data_encoded = encode_data(data_preprocessed)  # Encode the data for categorical variables
    features, _ = select_features(data_encoded)  # Select features as done during model training
    
    return features

# Step 4: Make predictions
def make_predictions(model, features):
    if model is None or features.empty:
        print("Error: Model or features data is not available.")
        return None
    
    predictions = model.predict(features)
    return predictions

# Main execution
if __name__ == "__main__":
    # Step 1: Load the trained model
    model = load_model()

    # Step 2: Load the new data to predict
    new_data_path = r'C:\Users\keith\desktop\ep\Election-Prediction\src\data\polling_data\cleaned_polls.csv'
    new_data = load_new_data(new_data_path)

    # Step 3: Prepare the data for prediction
    features = prepare_data(new_data)

    # Step 4: Make predictions
    predictions = make_predictions(model, features)

    if predictions is not None:
        print("Predictions:")
        print(predictions)
