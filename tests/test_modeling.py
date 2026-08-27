import pytest
import pandas as pd
import os
import sys
# Adjust the path to ensure 'src' can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modeling import (
    preprocess_data, encode_data, select_features, split_data, train_model,
    NATIONAL_CONDITION_COLS,
)

# Sample valid data matching the current featured_data.csv schema
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'year':                 [2016, 2016, 2020, 2020, 2024, 2024],
        'state':                ['California', 'Texas', 'California', 'Texas', 'California', 'Texas'],
        'state_po':             ['CA', 'TX', 'CA', 'TX', 'CA', 'TX'],
        'county_name':          ['Alameda', 'Harris', 'Alameda', 'Harris', 'Alameda', 'Harris'],
        'county_fips':          [6001, 48201, 6001, 48201, 6001, 48201],
        'party':                ['DEMOCRAT', 'REPUBLICAN', 'DEMOCRAT', 'REPUBLICAN', 'DEMOCRAT', 'REPUBLICAN'],
        'candidatevotes':       [500000, 400000, 520000, 410000, 540000, 420000],
        'totalvotes':           [900000, 900000, 930000, 930000, 960000, 960000],
        'pct':                  [55.5, 44.5, 55.9, 44.1, 56.3, 43.8],
        'gdp_growth':           [2.1, 2.1, -3.4, -3.4, 2.8, 2.8],
        'unemployment':         [4.9, 4.9, 8.1, 8.1, 4.1, 4.1],
        'inflation':            [2.1, 2.1, 1.4, 1.4, 2.6, 2.6],
        'consumer_confidence':  [93.8, 93.8, 81.8, 81.8, 70.5, 70.5],
        'is_incumbent':         [0, 1, 1, 0, 0, 1],
        'incumbent_on_ballot':  [0, 1, 1, 0, 0, 1],
        'consecutive_terms':    [2, 1, 1, 1, 1, 2],
        'state_lean':           [20.1, -8.3, 20.1, -8.3, 20.1, -8.3],
        'county_lean':          [55.0, -20.0, 55.0, -20.0, 55.0, -20.0],
        'turnout_delta':        [1.2, 0.8, -0.5, -0.9, 2.0, 1.5],
        'prior_midterm_house_margin': [-7.17, -7.17, 7.45, 7.45, -3.26, -3.26],
    })

# Sample data with missing values that should be dropped by preprocess_data
@pytest.fixture
def missing_data(sample_data):
    data = sample_data.copy()
    data.loc[1, 'turnout_delta'] = None
    data.loc[3, 'state_lean']    = None
    return data


# Test preprocessing step
def test_preprocess_data(sample_data, missing_data):
    data = preprocess_data(sample_data)
    assert not data.isnull().values.any(), "There should be no missing values after preprocessing"
    assert 'pct' in data.columns
    assert data['pct'].notna().all()

    # Rows with missing values should be dropped, not filled
    data = preprocess_data(missing_data)
    assert not data.isnull().values.any(), "Missing values should be handled by dropping rows"
    assert len(data) == len(missing_data) - 2

def test_preprocess_data_missing_column_raises(sample_data):
    for col in ['candidatevotes', 'totalvotes', 'pct']:
        with pytest.raises(ValueError):
            preprocess_data(sample_data.drop(columns=[col]))


# Test encoding step
def test_encode_data(sample_data):
    data = preprocess_data(sample_data)
    data = encode_data(data)
    assert 'party_REPUBLICAN' in data.columns, "Expected one-hot encoded column for 'party_REPUBLICAN'"
    assert 'state_Texas' in data.columns, "Expected one-hot encoded column for 'state_Texas'"
    assert 'pct' in data.columns

# Test feature selection
def test_select_features(sample_data):
    data = preprocess_data(sample_data)
    data = encode_data(data)
    features, target = select_features(data)
    assert 'pct' not in features.columns, "Target variable 'pct' should not be part of features"
    for col in ['county_name', 'state_po', 'county_fips', 'candidatevotes']:
        assert col not in features.columns, f"Identifier column '{col}' should not be part of features"
    for col in NATIONAL_CONDITION_COLS:
        assert col not in features.columns, (
            f"National-conditions column '{col}' is election-year-constant and should be "
            "excluded from the per-county tree model (see fundamentals.py)"
        )
    assert len(features) == len(target), "Features and target should have the same number of rows"

def test_select_features_missing_pct_raises(sample_data):
    data = preprocess_data(sample_data)
    data = encode_data(data).drop(columns=['pct'])
    with pytest.raises(ValueError):
        select_features(data)

# Test data splitting
def test_split_data(sample_data):
    data = preprocess_data(sample_data)
    data = encode_data(data)
    features, target = select_features(data)
    X_train, X_test, y_train, y_test = split_data(features, target)
    assert len(X_train) > 0 and len(X_test) > 0, "Training and test sets should not be empty"
    assert len(y_train) == len(X_train), "Mismatch between training features and labels"
    assert len(y_test) == len(X_test), "Mismatch between test features and labels"
    assert 'pct' not in X_train.columns, "Target 'pct' should not be in training features"
    assert 'pct' not in X_test.columns, "Target 'pct' should not be in test features"

# Test model training — writes to a temp dir so it never touches the real src/models/
def test_train_model(sample_data, tmp_path):
    data = preprocess_data(sample_data)
    data = encode_data(data)
    features, target = select_features(data)
    X_train, X_test, y_train, y_test = split_data(features, target)
    mid, lo, hi = train_model(X_train, y_train, models_dir=str(tmp_path))
    assert mid is not None and lo is not None and hi is not None, "All three models should train successfully"
    for fname in ['xgb_model_mid.pkl', 'xgb_model_lo.pkl', 'xgb_model_hi.pkl', 'training_features.pkl']:
        assert (tmp_path / fname).exists(), f"{fname} should be saved to models_dir"

# Test handling empty dataset
def test_empty_data_handling():
    empty_data = pd.DataFrame()  # Empty dataset
    with pytest.raises(ValueError):
        preprocess_data(empty_data)

# Test handling edge cases like outliers
def test_outliers_handling(sample_data, tmp_path):
    outlier_row = sample_data.iloc[0].to_dict()
    outlier_row['candidatevotes'] = 999999999
    outlier_row['totalvotes']     = 999999999
    outlier_row['pct']            = 100.0
    sample_data = pd.concat([sample_data, pd.DataFrame([outlier_row])], ignore_index=True)

    data = preprocess_data(sample_data)
    data = encode_data(data)
    features, target = select_features(data)
    X_train, X_test, y_train, y_test = split_data(features, target)
    mid, lo, hi = train_model(X_train, y_train, models_dir=str(tmp_path))

    # Ensure models still train with outliers
    assert mid is not None and lo is not None and hi is not None, "Models should train even with outliers in the dataset."
