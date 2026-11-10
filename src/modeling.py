import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/polling_data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

RECENCY_WEIGHTS = {
    2000: 1.0, 2004: 1.5, 2008: 2.0, 2012: 2.5,
    2016: 3.0, 2020: 4.0, 2024: 5.0,
}

XGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)

def load_data():
    path = os.path.join(DATA_DIR, 'featured_data.csv')
    return pd.read_csv(path)

def preprocess_data(data):
    if data.empty:
        raise ValueError("Input data is empty.")
    for col in ['candidatevotes', 'totalvotes', 'pct']:
        if col not in data.columns:
            raise ValueError(f"Missing required '{col}' column")

    print("Missing values before cleanup:")
    print(data.isnull().sum())

    data['candidatevotes'] = pd.to_numeric(data['candidatevotes'], errors='coerce').fillna(0)
    data['totalvotes'] = pd.to_numeric(data['totalvotes'], errors='coerce').fillna(0)
    data['pct'] = pd.to_numeric(data['pct'], errors='coerce').fillna(0)
    # National conditions aren't tree inputs (see NATIONAL_CONDITION_COLS), so a gap in
    # one of them (e.g. a failed FRED download) mustn't drop rows.
    data = data.dropna(subset=[c for c in data.columns if c not in NATIONAL_CONDITION_COLS])

    print("Missing values after cleanup:")
    print(data.isnull().sum())
    return data

def encode_data(data):
    # One-hot encode state and party
    data = pd.get_dummies(data, columns=['state', 'party'], drop_first=True)
    data.columns = data.columns.str.replace('__', '_')
    print(f"Data shape after encoding: {data.shape}")
    return data

# These are national, election-year-constant values (identical for every county in a
# given year, and incumbent_on_ballot/consecutive_terms don't even vary by party row) —
# only 7 distinct values ever exist across 2000-2024, nowhere near enough for a
# per-county tree model to learn a real functional relationship. They're modeled
# separately as a national swing in fundamentals.fundamentals_adjustment() instead.
NATIONAL_CONDITION_COLS = [
    'gdp_growth', 'unemployment', 'inflation', 'consumer_confidence',
    'incumbent_on_ballot', 'consecutive_terms', 'prior_midterm_house_margin',
]

def select_features(data):
    if 'pct' not in data.columns:
        raise ValueError("Missing required 'pct' column")

    # Same-election values leak the answer: that year's turnout (totalvotes, turnout_delta)
    # isn't known before the election, and `year` can't be extrapolated by a tree model.
    drop_cols = ['pct', 'county_name', 'state_po', 'county_fips', 'candidatevotes',
                 'year', 'totalvotes', 'turnout_delta']
    drop_cols += NATIONAL_CONDITION_COLS
    drop_cols = [c for c in drop_cols if c in data.columns]
    features = data.drop(columns=drop_cols)
    target = data['pct']

    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    return features, target

def split_data(features, target):
    if features is None or features.empty or target is None or target.empty:
        print("Error: Features or target data is empty.")
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test

def recency_sample_weight(X_train):
    if 'year' not in X_train.columns:
        return None
    return X_train['year'].map(RECENCY_WEIGHTS).fillna(1.0).values

def train_model(X_train, y_train, models_dir=None):
    if X_train is None or y_train is None:
        print("Error: Training data is not available.")
        return None, None, None

    models_dir = models_dir or MODELS_DIR
    weights = recency_sample_weight(X_train)

    mid = XGBRegressor(objective='reg:squarederror', **XGB_PARAMS)
    mid.fit(X_train, y_train, sample_weight=weights)

    lo = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.025, **XGB_PARAMS)
    lo.fit(X_train, y_train, sample_weight=weights)

    hi = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.975, **XGB_PARAMS)
    hi.fit(X_train, y_train, sample_weight=weights)

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(mid, os.path.join(models_dir, 'xgb_model_mid.pkl'))
    joblib.dump(lo, os.path.join(models_dir, 'xgb_model_lo.pkl'))
    joblib.dump(hi, os.path.join(models_dir, 'xgb_model_hi.pkl'))
    joblib.dump(X_train.columns.tolist(), os.path.join(models_dir, 'training_features.pkl'))
    print(f"Models saved to {models_dir}")
    return mid, lo, hi

def evaluate_model(model, X_test, y_test):
    if model is None or X_test is None or y_test is None:
        print("Error: Model or test data is not available.")
        return

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score:    {r2:.4f}")

def evaluate_honest(data=None):
    """
    Leave-one-election-out test: each presidential year is predicted by a model trained
    on the other years, with features that only use elections before the predicted one.
    Compared against the naive "the county votes exactly as it did last time".
    Scored on Democratic and Republican county vote shares.
    """
    data = data if data is not None else load_data()
    d = encode_data(preprocess_data(data.copy()))
    X, y = select_features(d)
    X.columns = X.columns.astype(str)
    years = data.loc[X.index, 'year']
    rows = []
    for yr in sorted(years.unique()):
        tr, te = years != yr, years == yr
        m = XGBRegressor(objective='reg:squarederror', **XGB_PARAMS)
        m.fit(X[tr], y[tr], sample_weight=years[tr].map(RECENCY_WEIGHTS).fillna(1.0).values)
        pred = pd.Series(m.predict(X[te]), index=X[te].index)
        two = data.loc[X[te].index, 'party'].isin(['DEMOCRAT', 'REPUBLICAN'])
        actual = y[te][two]
        # naive baseline: last election's share for the same county and party
        sub = data.loc[X[te].index][two]
        naive = np.where(sub['party'] == 'DEMOCRAT',
                         50 + sub['county_last_margin'] / 2, 50 - sub['county_last_margin'] / 2)
        rows.append({'year': int(yr), 'rows': int(two.sum()),
                     'model_rmse': float(np.sqrt(((pred[two] - actual) ** 2).mean())),
                     'naive_rmse': float(np.sqrt(((naive - actual.values) ** 2).mean()))})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    data = load_data()
    data = preprocess_data(data)
    data = encode_data(data)
    features, target = select_features(data)
    X_train, X_test, y_train, y_test = split_data(features, target)
    mid_model, lo_model, hi_model = train_model(X_train, y_train)
    evaluate_model(mid_model, X_test, y_test)
    print("\nHonest leave-one-election-out test (county Dem/Rep vote share RMSE, pts):")
    print(evaluate_honest(load_data()).round(2).to_string(index=False))
