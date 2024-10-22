import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Step 1: Load data from PostgreSQL
def load_data():
    engine = create_engine("postgresql://postgres:Shogun12@localhost:5432/election_db")  # Replace with your connection string
    data = pd.read_sql_table('polling_data', con=engine)
    return data

# Step 2: Preprocess data (handle missing values and clean data)
def preprocess_data(data):
    print("Missing values before cleanup:")
    print(data.isnull().sum())

    # Ensure 'sample_size' exists and convert it to numeric, filling non-numeric values with 0
    if 'sample_size' in data.columns:
        data['sample_size'] = pd.to_numeric(data['sample_size'], errors='coerce').fillna(0)
    else:
        raise ValueError("Missing required 'sample_size' column")

    # Ensure 'pct' exists and convert it to numeric, filling non-numeric values with 0
    if 'pct' not in data.columns:
        raise ValueError("Missing required 'pct' column")
    data['pct'] = pd.to_numeric(data['pct'], errors='coerce').fillna(0)

    # Drop rows with missing values in essential columns
    data.dropna(subset=['candidate_name', 'candidate_id'], inplace=True)

    # Fill missing values in other important columns
    data.fillna({
        'modeldate': '1970-01-01',  # Default fallback date
        'contestdate': '1970-01-01',  # Default fallback date
        'race': 'Unknown',  # Fill missing 'race' with 'Unknown'
        'pct': 0,  # Fill missing 'pct_estimate' with 0 if needed
    }, inplace=True)

    # Convert 'modeldate' and 'contestdate' to datetime format
    if 'modeldate' in data.columns:
        data['modeldate'] = pd.to_datetime(data['modeldate'], errors='coerce').fillna(pd.Timestamp('1970-01-01'))
    if 'contestdate' in data.columns:
        data['contestdate'] = pd.to_datetime(data['contestdate'], errors='coerce').fillna(pd.Timestamp('1970-01-01'))

    print("Missing values after cleanup:")
    print(data.isnull().sum())

    return data

# Step 3: Encode categorical variables (race, state, candidate_name)
def encode_data(data):
    # List of categorical columns that need to be encoded
    columns_to_encode = ['race', 'state', 'candidate_name']
    
    # Check if these columns exist in the data before encoding
    available_columns = [col for col in columns_to_encode if col in data.columns]
    
    if available_columns:
        # One-hot encode the available categorical columns
        data = pd.get_dummies(data, columns=available_columns, drop_first=True)
    else:
        print(f"Warning: None of the columns {columns_to_encode} are available for encoding.")
    
    # Label encode the 'party' column to keep it a 3-letter designator either DEM or REP
    if 'party' in data.columns:
        party_mapping = {'DEM': 0, 'REP': 1}
        data['party'] = data['party'].map(party_mapping)
    
    # Clean up column names to replace "__" with "_" (if needed)
    data.columns = data.columns.str.replace('__', '_')

    print(f"Data shape after encoding: {data.shape}")

    return data


# Step 4: Select features and the target variable for the model
def select_features(data):

    if 'pct' not in data.columns:
        print("Error: 'pct' column not found in data. ")
        return None, None
    
    # Drop columns that are not used as features (example: dates and the target)
    columns_to_drop = ['pct', 'modeldate', 'contestdate']
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    features = data.drop(columns = columns_to_drop)

     # Adjust based on actual columns
    target = data['pct']  # The target we're trying to predict (percentage estimate)

    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    
    return features, target

# Step 5: Split the data into training and testing sets
def split_data(features, target):
    #Ensure there is data to split
    if features.empty or target.empty:
        print("Error: Features or target data is empty.")
        return None, None, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Ensure that all column names are strings
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    print("X_train columns: ", X_train.columns)  # Debugging print
    print("X_test columns: ", X_test.columns)    # Debugging print
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

# Step 6: Train a Forest regression model
def train_model(X_train, y_train):

    if X_train is None or y_train is None:
        print("Error: Training data is not available.")
        return None
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    #Save the model
    models_dir =os.path.join(os.path.dirname(__file__),'../src/models')
    os.makedirs(models_dir, exist_ok=True)

    model_path=os.path.join(models_dir, 'random_forest_model.pkl')
    joblib.dump(rf,model_path)
    print(f"model have been saved to {model_path}")


    return rf

# Step 7: Evaluate the model's performance
def evaluate_model(model, X_test, y_test):

    if model is None or X_test is None or y_test is None:
        print("Error: Model or test data is not available.")
        return 
    
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")

# Main execution: Load, preprocess, train, and evaluate
if __name__ == "__main__":
    # Step 1: Load data
    data = load_data()
    
    # Step 2: Preprocess data
    data = preprocess_data(data)
    
    # Step 3: Encode categorical variables
    data = encode_data(data)
    
    # Step 4: Select features and target
    features, target = select_features(data)
    
    # Step 5: Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(features, target)
    
    # Step 6: Train the model
    model = train_model(X_train, y_train)
    
    # Step 7: Evaluate the model
    evaluate_model(model, X_test, y_test)
