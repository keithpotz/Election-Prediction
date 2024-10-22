import os
import sys
# Add the src directory to the system path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Print sys.path to verify module lookup paths
print("Python module search paths:")
for path in sys.path:
    print(path)

# Print the current script's directory to verify its location
print(f"Current script directory: {os.path.dirname(os.path.abspath(__file__))}")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from data_cleaning import clean_polling_data
from modeling import encode_data, select_features, preprocess_data  # Ensure preprocess_data is imported


# Step 1: Load the saved model
def load_model():
    models_dir = os.path.join(os.path.dirname(__file__))  # Corrected file path
    model_path = os.path.join(models_dir, 'random_forest_model.pkl')
    features_path = os.path.join(models_dir, 'training_features.pki')

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please ensure the model has been trained and saved.")
        return None, None
    
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    trained_features = joblib.load(features_path)
    return model, trained_features

# Step 2: Load new data (or use existing cleaned data)6
def load_new_data(file_path):
    print(f"looking for csv file at: {os.path.abspath(file_path)}")
    # Clean the new data using the cleaning function
    data = clean_polling_data(file_path)
    return data

# Step 3: Prepare the data for prediction
def prepare_data(data, trained_features):
    # Preprocess the new data
    data_preprocessed = preprocess_data(data)

    # Encode the new data (categorical variables)
    data_encoded = encode_data(data_preprocessed)

    # Identify missing columns
    missing_cols = set(trained_features) - set(data_encoded.columns)
    extra_cols = set(data_encoded.columns) - set(trained_features)

    # Create a DataFrame with the missing columns, filled with zeros
    missing_cols_df = pd.DataFrame(0, index=data_encoded.index, columns=list(missing_cols))

    # Concatenate the encoded data and missing columns DataFrame
    data_encoded = pd.concat([data_encoded, missing_cols_df], axis=1)

    # Drop extra columns that were not in the training data
    data_encoded = data_encoded[trained_features]

    print(f"Prepared data shape for prediction: {data_encoded.shape}")
    return data_encoded


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
    model, trained_features = load_model()

    if model is not None and trained_features is not None:
    # Step 2: Load the new data to predict
     new_data_path = r'YOURPATH'
    new_data = load_new_data(new_data_path)

    # Step 3: Prepare the data for prediction
    features = prepare_data(new_data, trained_features)

    # Step 4: Make predictions
    predictions = make_predictions(model, features)

    if predictions is not None:
        print("Predictions:")
        print(predictions)
