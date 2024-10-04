import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load data from PostgreSQL
def load_data():
    engine = create_engine("postgresql://postgres:Shogun12@localhost:5432/election_db")  # Replace with your connection string
    data = pd.read_sql_table('polling_data', con=engine)
    return data

# Step 2: Preprocess data (handle missing values and clean data)
def preprocess_data(data):
    print("Missing values before cleanup:")
    print(data.isnull().sum())

    # Handle missing values by dropping rows with missing values (adjust as needed)
    data.dropna(inplace=True)

    print("Missing values after cleanup:")
    print(data.isnull().sum())
    
    return data

# Step 3: Encode categorical variables (race, state, candidate_name)
def encode_data(data):
    # One-hot encode categorical variables (race, state, candidate_name)
    data = pd.get_dummies(data, columns=['race', 'state', 'candidate_name'], drop_first=True)

    # Clean up column names to replace "__" with "_" (if needed)
    data.columns = data.columns.str.replace('__', '_')

    return data

# Step 4: Select features and the target variable for the model
def select_features(data):
    # Drop columns that are not used as features (example: dates and the target)
    features = data.drop(columns=['pct_estimate', 'modeldate', 'contestdate'])  # Adjust based on actual columns
    target = data['pct_estimate']  # The target we're trying to predict (percentage estimate)
    
    return features, target

# Step 5: Split the data into training and testing sets
def split_data(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Ensure that all column names are strings
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    print("X_train columns: ", X_train.columns)  # Debugging print
    print("X_test columns: ", X_test.columns)    # Debugging print

    return X_train, X_test, y_train, y_test

# Step 6: Train a logistic regression model
def train_model(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    rf.fit(X_train, y_train)
    return rf

# Step 7: Evaluate the model's performance
def evaluate_model(model, X_test, y_test):
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
