import pandas as pd
import requests
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data/polling_data')
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Party holding the White House going into each election
INCUMBENT_PARTY = {
    1976: 'REPUBLICAN', 1980: 'DEMOCRAT',  1984: 'REPUBLICAN',
    1988: 'REPUBLICAN', 1992: 'REPUBLICAN', 1996: 'DEMOCRAT',
    2000: 'DEMOCRAT',   2004: 'REPUBLICAN', 2008: 'REPUBLICAN',
    2012: 'DEMOCRAT',   2016: 'DEMOCRAT',   2020: 'REPUBLICAN',
    2024: 'DEMOCRAT',
}

# Is the sitting president actually on the ballot?
INCUMBENT_ON_BALLOT = {
    2000: False,  # Gore (VP successor to Clinton)
    2004: True,   # Bush running for re-election
    2008: False,  # McCain (not sitting president)
    2012: True,   # Obama running for re-election
    2016: False,  # Clinton vs Trump (Obama not running)
    2020: True,   # Trump running for re-election
    2024: False,  # Harris (Biden withdrew late)
}

# How many consecutive terms has the incumbent party held the WH entering this election?
CONSECUTIVE_TERMS = {
    2000: 2,   # Democrats: Clinton 2 terms
    2004: 1,   # Republicans: Bush 1st term
    2008: 2,   # Republicans: Bush 2 terms
    2012: 1,   # Democrats: Obama 1st term
    2016: 2,   # Democrats: Obama 2 terms
    2020: 1,   # Republicans: Trump 1st term
    2024: 1,   # Democrats: Biden 1 term (Harris running)
}

ELECTION_YEARS = [2000, 2004, 2008, 2012, 2016, 2020, 2024]


def _fetch_fred_monthly(series_id, api_key, start='1999-01-01'):
    """Fetch a full FRED monthly series. Returns a tidy DataFrame with year/month/value."""
    try:
        params = {
            'series_id':        series_id,
            'api_key':          api_key,
            'file_type':        'json',
            'observation_start': start,
        }
        r = requests.get(FRED_BASE, params=params, timeout=15)
        r.raise_for_status()
        obs = r.json().get('observations', [])
        df  = pd.DataFrame(obs)
        df['date']  = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['year']  = df['date'].dt.year
        df['month'] = df['date'].dt.month
        return df
    except Exception as e:
        print(f"  FRED {series_id} fetch failed: {e}")
        return pd.DataFrame()


def load_gdp_growth():
    gdp = pd.read_csv(os.path.join(DATA_DIR, 'GDP.csv'))
    gdp['observation_date'] = pd.to_datetime(gdp['observation_date'])
    gdp['year']  = gdp['observation_date'].dt.year
    gdp['month'] = gdp['observation_date'].dt.month
    q3 = gdp[gdp['month'] == 7].set_index('year')['GDP']
    growth = {}
    for year in range(1976, 2025, 4):
        if year in q3.index and (year - 1) in q3.index:
            growth[year] = round((q3[year] - q3[year - 1]) / q3[year - 1] * 100, 2)
    return growth


def load_unemployment():
    unemp = pd.read_csv(os.path.join(DATA_DIR, 'unemployment.csv'))
    unemp['observation_date'] = pd.to_datetime(unemp['observation_date'])
    unemp['year']  = unemp['observation_date'].dt.year
    unemp['month'] = unemp['observation_date'].dt.month
    oct = unemp[unemp['month'] == 10].set_index('year')['UNRATE']
    return oct.to_dict()


def load_cpi_inflation(api_key):
    """YoY CPI change for October of each election year."""
    df = _fetch_fred_monthly('CPIAUCSL', api_key)
    if df.empty:
        return {}
    oct = df[df['month'] == 10].set_index('year')['value']
    inflation = {}
    for y in ELECTION_YEARS:
        if y in oct.index and (y - 1) in oct.index and oct[y - 1] != 0:
            inflation[y] = round((oct[y] - oct[y - 1]) / oct[y - 1] * 100, 2)
    return inflation


def load_consumer_confidence(api_key):
    """October University of Michigan Consumer Sentiment (UMCSENT) for each election year."""
    df = _fetch_fred_monthly('UMCSENT', api_key)
    if df.empty:
        return {}
    oct = df[df['month'] == 10].set_index('year')['value']
    return {y: float(oct[y]) if y in oct.index else None for y in ELECTION_YEARS}


def compute_state_lean(state_df):
    lean = {}
    for state, sdf in state_df.groupby('state'):
        margins = []
        for year, ydf in sdf.groupby('year'):
            total = ydf['totalvotes'].max()
            dem   = ydf[ydf['party'] == 'DEMOCRAT']['candidatevotes'].sum()
            rep   = ydf[ydf['party'] == 'REPUBLICAN']['candidatevotes'].sum()
            if total > 0:
                margins.append((dem - rep) / total * 100)
        if margins:
            lean[state] = round(sum(margins) / len(margins), 2)
    return lean


def compute_county_lean(county_df):
    lean = {}
    for fips, cdf in county_df.groupby('county_fips'):
        margins = []
        for year, ydf in cdf.groupby('year'):
            total = ydf['totalvotes'].max()
            dem   = ydf[ydf['party'] == 'DEMOCRAT']['candidatevotes'].sum()
            rep   = ydf[ydf['party'] == 'REPUBLICAN']['candidatevotes'].sum()
            if total > 0:
                margins.append((dem - rep) / total * 100)
        if margins:
            lean[fips] = round(sum(margins) / len(margins), 2)
    return lean


def compute_turnout_delta(county_df):
    """Percent change in county totalvotes from the prior election cycle."""
    tv = (
        county_df[['county_fips', 'year', 'totalvotes']]
        .drop_duplicates(subset=['county_fips', 'year'])
        .sort_values(['county_fips', 'year'])
        .copy()
    )
    tv['prev_tv'] = tv.groupby('county_fips')['totalvotes'].shift(1)
    tv['turnout_delta'] = (
        (tv['totalvotes'] - tv['prev_tv'])
        / tv['prev_tv'].replace(0, float('nan'))
        * 100
    ).round(2)

    delta_map = tv.set_index(['county_fips', 'year'])['turnout_delta']
    idx = pd.MultiIndex.from_arrays([county_df['county_fips'], county_df['year']])
    county_df = county_df.copy()
    county_df['turnout_delta'] = delta_map.reindex(idx).values
    return county_df


def build_features():
    from config import FRED_API_KEY as api_key

    print("Loading county election data...")
    county = pd.read_csv(os.path.join(DATA_DIR, 'countypres_2000-2024.csv'))
    county = county[county['mode'] == 'TOTAL'].copy()

    county['party'] = county['party'].str.upper().apply(
        lambda x: x if x in ['DEMOCRAT', 'REPUBLICAN'] else 'OTHER'
    )
    county = county[pd.to_numeric(county['totalvotes'], errors='coerce') > 0]
    county['candidatevotes'] = pd.to_numeric(county['candidatevotes'], errors='coerce').fillna(0)
    county['totalvotes']     = pd.to_numeric(county['totalvotes'],     errors='coerce')
    county = county.dropna(subset=['totalvotes'])
    county['pct'] = (county['candidatevotes'] / county['totalvotes'] * 100).round(2)

    print("Loading GDP & unemployment...")
    county['gdp_growth']   = county['year'].map(load_gdp_growth())
    county['unemployment'] = county['year'].map(load_unemployment())

    print("Fetching CPI inflation from FRED...")
    county['inflation'] = county['year'].map(load_cpi_inflation(api_key))

    print("Fetching consumer confidence from FRED...")
    county['consumer_confidence'] = county['year'].map(load_consumer_confidence(api_key))

    print("Computing incumbency flags...")
    county['incumbent_party']     = county['year'].map(INCUMBENT_PARTY)
    county['is_incumbent']        = (county['party'] == county['incumbent_party']).astype(int)
    county['incumbent_on_ballot'] = county['year'].map(INCUMBENT_ON_BALLOT).astype(int)
    county['consecutive_terms']   = county['year'].map(CONSECUTIVE_TERMS)

    print("Computing turnout delta...")
    county = compute_turnout_delta(county)

    print("Computing state partisan lean...")
    state_df = pd.read_csv(os.path.join(DATA_DIR, 'complete_data.csv'))
    state_df['party'] = state_df['party_simplified'].str.upper()
    county['state_lean'] = county['state'].map(compute_state_lean(state_df))

    print("Computing county partisan lean...")
    county['county_lean'] = county['county_fips'].map(compute_county_lean(county))

    keep = [
        'year', 'state', 'state_po', 'county_name', 'county_fips',
        'party', 'candidatevotes', 'totalvotes', 'pct',
        'gdp_growth', 'unemployment', 'inflation', 'consumer_confidence',
        'is_incumbent', 'incumbent_on_ballot', 'consecutive_terms',
        'state_lean', 'county_lean', 'turnout_delta',
    ]
    county = county[keep].dropna(subset=['gdp_growth', 'unemployment', 'state_lean', 'county_lean'])

    save_path = os.path.join(DATA_DIR, 'featured_data.csv')
    county.to_csv(save_path, index=False)
    print(f"\nFeatured data saved → {save_path}")
    print(f"Shape: {county.shape}")
    print(county[['year', 'state', 'party', 'pct', 'inflation', 'consumer_confidence',
                  'incumbent_on_ballot', 'consecutive_terms', 'turnout_delta']].head())
    return county


if __name__ == '__main__':
    build_features()
