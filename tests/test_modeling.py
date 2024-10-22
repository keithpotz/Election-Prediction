import pytest
import pandas as pd
import os
import sys
# Adjust the path to ensure 'src' can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modeling import preprocess_data, encode_data, select_features, split_data, train_model

# Sample valid data for testing
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'race': ['DEM', 'REP', 'DEM'],
        'state': ['California', 'Texas', 'Florida'],
        'modeldate': ['2020-01-01', '2020-02-01', '2020-03-01'],
        'candidate_name': ['Joe Biden', 'Donald Trump', 'Joe Biden'],
        'candidate_id': [1, 2, 1],
        'pct_estimate': [50, 45, 55],
        'pct_trend_adjusted': [51, 46, 56],
        'poll_id': [101, 102, 103],
        'sample_size': [1000, 1200, 1100],
        'contestdate': ['2020-11-03', '2020-11-03', '2020-11-03'],
        'pct':[52.5, 44.5, 51.2]  # Add pct to test data
    })
    return data

# Sample data with missing values
@pytest.fixture
def missing_data():
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'race': ['DEM', None, 'DEM'],
        'state': ['California', 'Texas', 'Florida'],
        'modeldate': ['2020-01-01', '2020-02-01', None],
        'candidate_name': ['Joe Biden', 'Donald Trump', None],
        'candidate_id': [1, 2, None],
        'pct_estimate': [50, None, 55],
        'pct_trend_adjusted': [51, 46, 56],
        'poll_id': [101, 102, 103],
        'sample_size': [1000, 1200, 1100],
        'contestdate': ['2020-11-03', '2020-11-03', '2020-11-03'],
        'pct': [None, None, 51.2]
    })
    return data

# Sample data with invalid types
@pytest.fixture
def invalid_data():
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'race': ['DEM', 'REP', 'DEM'],
        'state': ['California', 'Texas', 'Florida'],
        'modeldate': ['InvalidDate', 'InvalidDate', 'InvalidDate'],
        'candidate_name': ['Joe Biden', 'Donald Trump', 'Joe Biden'],
        'candidate_id': ['One', 'Two', 'One'],  # Invalid types for candidate_id
        'pct_estimate': ['Fifty', 'Forty-Five', 'Fifty-Five'],  # Invalid types for pct_estimate
        'pct_trend_adjusted': [51, 46, 56],
        'poll_id': [101, 102, 103],
        'sample_size': [1000, 1200, 1100],
        'contestdate': ['2020-11-03', '2020-11-03', '2020-11-03']
    })
    return data



# Test preprocessing step
def test_preprocess_data(sample_data, missing_data):
    # Test valid data
    data = preprocess_data(sample_data)
    assert not data.isnull().values.any(), "There should be no missing values after preprocessing"
    assert 'pct' in data.columns
    assert 'sample_size' in data.columns  # Ensure 'sample_size' is included
    assert data['pct'].notna().all()
    assert data['sample_size'].notna().all()  # Ensure no missing values for 'sample_size'

    # Test missing data: Ensure the missing rows are removed or filled
    data = preprocess_data(missing_data)
    assert not data.isnull().values.any(), "Missing values should be handled by dropping or filling."
    assert 'pct' in data.columns
    assert 'sample_size' in data.columns  # Ensure 'sample_size' is handled
    assert data['pct'].isna().sum() == 0
    assert data['sample_size'].isna().sum() == 0  # Ensure 'sample_size' has no missing values



# Test encoding step
def test_encode_data(sample_data):
    data = preprocess_data(sample_data)  # Preprocess before encoding
    data = encode_data(data)
    assert 'race_REP' in data.columns, "Expected one-hot encoded column for 'race_REP'"
    assert 'pct' in data.columns

# Test feature selection
def test_select_features(sample_data):
    data = preprocess_data(sample_data)
    data = encode_data(data)
    features, target = select_features(data)
    assert 'pct_estimate' not in features.columns, "Target variable should not be part of features"
    assert len(features) == len(target), "Features and target should have the same number of rows"
    assert 'pct' in features.columns

# Test data splitting
def test_split_data(sample_data):
    data = preprocess_data(sample_data)
    data = encode_data(data)
    features, target = select_features(data)
    X_train, X_test, y_train, y_test = split_data(features, target)
    assert len(X_train) > 0 and len(X_test) > 0, "Training and test sets should not be empty"
    assert len(y_train) == len(X_train), "Mismatch between training features and labels"
    assert len(y_test) == len(X_test), "Mismatch between test features and labels"
    assert 'pct' in X_train.columns
    assert 'pct' in X_test.columns

# Test model training
def test_train_model(sample_data):
    data = preprocess_data(sample_data)
    data = encode_data(data)
    features, target = select_features(data)
    X_train, X_test, y_train, y_test = split_data(features, target)
    model = train_model(X_train, y_train)
    assert model is not None, "Model should be trained successfully"
    

# Test handling invalid data
def test_invalid_data_handling(invalid_data):
    with pytest.raises(ValueError):
        data = preprocess_data(invalid_data)
        data = encode_data(data)

# Test handling empty dataset
def test_empty_data_handling():
    empty_data = pd.DataFrame()  # Empty dataset
    with pytest.raises(ValueError):
        preprocess_data(empty_data)

# Test handling edge cases like outliers
def test_outliers_handling(sample_data):
    # Add outliers to the sample data
    sample_data.loc[3] = [4, 'DEM', 'New York', '2020-04-01', 'Joe Biden', 1, 999, 999, 104, 500, '2020-11-03', 95.0]
    
    data = preprocess_data(sample_data)
    data = encode_data(data)
    features, target = select_features(data)
    X_train, X_test, y_train, y_test = split_data(features, target)
    model = train_model(X_train, y_train)
    
    # Ensure model still trains with outliers
    assert model is not None, "Model should train even with outliers in the dataset."
