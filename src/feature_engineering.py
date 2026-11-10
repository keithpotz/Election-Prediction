import pandas as pd
import requests
import os
import io

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


def _load_house_raw():
    """
    MEDSL's Dataverse export of house_1976_2024.csv double-encodes each data row:
    the whole row is wrapped in one quoted field, with any original embedded quotes
    backslash-escaped. Unwrap it before handing to pandas' normal CSV parser.
    """
    path = os.path.join(DATA_DIR, 'house_1976_2024.csv')
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    fixed = [lines[0]]
    for line in lines[1:]:
        inner = line.rstrip('\n')[1:-1].replace('\\"', '"')
        fixed.append(inner + '\n')
    return pd.read_csv(io.StringIO(''.join(fixed)), low_memory=False)


def normalize_party(series):
    """Bucket a raw party column down to DEMOCRAT/REPUBLICAN/OTHER."""
    return series.str.upper().apply(lambda x: x if x in ['DEMOCRAT', 'REPUBLICAN'] else 'OTHER')

def load_house_races():
    """
    All US House general-election, non-special-election rows (regular contests only —
    specials are excluded so this represents 'the normal election that cycle', matching
    the convention used everywhere else this data feeds a national margin/seat count).
    """
    house = _load_house_raw()
    house = house[
        (house['office'] == 'US HOUSE') & (house['stage'] == 'GEN') &
        (house['mode'] == 'TOTAL') & (~house['special'])
    ].copy()
    house['party']          = normalize_party(house['party'])
    house['candidatevotes'] = pd.to_numeric(house['candidatevotes'], errors='coerce').fillna(0)
    return house

def national_house_margin_by_year(house=None):
    """National two-party House popular vote margin (Dem - Rep, pct points) per election year."""
    house = house if house is not None else load_house_races()
    national = house.groupby(['year', 'party'])['candidatevotes'].sum().unstack(fill_value=0)
    dem, rep = national.get('DEMOCRAT', 0), national.get('REPUBLICAN', 0)
    return ((dem - rep) / (dem + rep) * 100).round(2)

def load_prior_midterm_house_margin():
    """
    National two-party US House popular vote margin (Dem - Rep, pct points) from the
    midterm election two years before each presidential election year.
    """
    margin = national_house_margin_by_year()
    return {
        pres_year: margin[pres_year - 2]
        for pres_year in ELECTION_YEARS if (pres_year - 2) in margin.index
    }


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


def load_county_results():
    """
    County presidential results 2000-2024, one row per (year, county, party) with
    candidatevotes, totalvotes and pct. MEDSL's 2024 file leaves `mode` blank for 8 states
    (NC, WI, UT, ID, RI, WA, WV, WY); most of those rows are already county totals, but
    North Carolina's are split by voting method, so any state-year without a 'TOTAL' row
    has its rows summed per county and candidate. (Filtering to mode == 'TOTAL' used to
    silently drop those 8 states' 2024 results.)
    """
    raw = pd.read_csv(os.path.join(DATA_DIR, 'countypres_2000-2024.csv'))
    raw['mode'] = raw['mode'].astype('string').str.upper()
    has_total = raw.groupby(['year', 'state'])['mode'].transform(lambda m: (m == 'TOTAL').any())
    county = pd.concat([raw[raw['mode'] == 'TOTAL'], raw[~has_total]], ignore_index=True)
    county['candidatevotes'] = pd.to_numeric(county['candidatevotes'], errors='coerce').fillna(0)
    county['totalvotes'] = pd.to_numeric(county['totalvotes'], errors='coerce')
    county['party'] = normalize_party(county['party'].fillna('OTHER'))
    county = county.dropna(subset=['county_fips', 'totalvotes'])
    county = county[county['totalvotes'] > 0]
    county = (county.groupby(['year', 'state', 'state_po', 'county_fips', 'county_name', 'party'], as_index=False)
                    .agg(candidatevotes=('candidatevotes', 'sum'), totalvotes=('totalvotes', 'first')))
    county['pct'] = (county['candidatevotes'] / county['totalvotes'] * 100).round(2)
    return county

def _county_margins(county):
    """(county_fips, year) -> Dem minus Rep, % of all votes, plus that year's turnout."""
    w = county.pivot_table(index=['county_fips', 'year'], columns='party', values='candidatevotes',
                           aggfunc='sum', fill_value=0)
    tv = county.groupby(['county_fips', 'year'])['totalvotes'].first()
    m = ((w.get('DEMOCRAT', 0) - w.get('REPUBLICAN', 0)) / tv * 100).rename('margin')
    return pd.concat([m, tv.rename('totalvotes')], axis=1).reset_index()

def _state_margins():
    """(state, year) -> statewide Dem minus Rep, % of all votes, 1976-2024."""
    from data_cleaning import load_and_merge
    pres = load_and_merge(os.path.join(DATA_DIR, 'complete_data.csv'), os.path.join(DATA_DIR, '2024president.csv'))
    pres['party'] = pres['party_simplified'].str.upper()
    tot = pres.groupby(['state', 'year'])['totalvotes'].max()
    d = pres[pres['party'] == 'DEMOCRAT'].groupby(['state', 'year'])['votes'].sum()
    r = pres[pres['party'] == 'REPUBLICAN'].groupby(['state', 'year'])['votes'].sum()
    return ((d - r) / tot * 100).rename('margin').reset_index()

def prior_features(county_m, state_m, target_year):
    """
    Partisan features for an election in `target_year`, built ONLY from elections before
    it (so a backtest never sees the result it's predicting):
      county_lean        average county margin over earlier elections (2000 on)
      county_last_margin the county's margin in the previous election
      prev_totalvotes    the county's turnout in the previous election
      state_lean         average statewide margin over earlier elections (1976 on)
    Returns (county features, {state: state_lean}), or None when there's no earlier data.
    """
    past = county_m[county_m['year'] < target_year]
    if past.empty:
        return None
    last_year = past['year'].max()
    g = past.groupby('county_fips')
    out = pd.DataFrame({'county_lean': g['margin'].mean()})
    last = past[past['year'] == last_year].set_index('county_fips')
    out['county_last_margin'] = last['margin']
    out['prev_totalvotes'] = last['totalvotes']
    sl = state_m[state_m['year'] < target_year].groupby('state')['margin'].mean()
    return out.reset_index(), sl

def county_profiles(target_year, county=None):
    """One row per county with its prior features for predicting `target_year`."""
    county = county if county is not None else load_county_results()
    feats, sl = prior_features(_county_margins(county), _state_margins(), target_year)
    names = county.drop_duplicates('county_fips', keep='last')[['county_fips', 'state', 'state_po', 'county_name']]
    prof = feats.merge(names, on='county_fips', how='left')
    prof['state_lean'] = prof['state'].map(sl)
    return prof.dropna(subset=['county_lean', 'county_last_margin', 'state_lean'])

def build_features():
    from config import FRED_API_KEY as api_key

    print("Loading county election data...")
    county = load_county_results()
    cm, sm = _county_margins(county), _state_margins()

    print("Computing prior-only partisan features...")
    parts = []
    for y in sorted(county['year'].unique()):
        res = prior_features(cm, sm, y)
        if res is None:
            continue                        # 2000: no earlier county data
        feats, sl = res
        cy = county[county['year'] == y].merge(feats, on='county_fips', how='inner')
        cy['state_lean'] = cy['state'].map(sl)
        parts.append(cy)
    county = pd.concat(parts, ignore_index=True)

    print("Loading national conditions...")
    county['gdp_growth']   = county['year'].map(load_gdp_growth())
    county['unemployment'] = county['year'].map(load_unemployment())
    county['inflation'] = county['year'].map(load_cpi_inflation(api_key))
    county['consumer_confidence'] = county['year'].map(load_consumer_confidence(api_key))
    county['incumbent_party']     = county['year'].map(INCUMBENT_PARTY)
    county['is_incumbent']        = (county['party'] == county['incumbent_party']).astype(int)
    county['incumbent_on_ballot'] = county['year'].map(INCUMBENT_ON_BALLOT).astype(int)
    county['consecutive_terms']   = county['year'].map(CONSECUTIVE_TERMS)
    county['prior_midterm_house_margin'] = county['year'].map(load_prior_midterm_house_margin())

    keep = [
        'year', 'state', 'state_po', 'county_name', 'county_fips',
        'party', 'candidatevotes', 'totalvotes', 'pct',
        'gdp_growth', 'unemployment', 'inflation', 'consumer_confidence',
        'is_incumbent', 'incumbent_on_ballot', 'consecutive_terms', 'prior_midterm_house_margin',
        'state_lean', 'county_lean', 'county_last_margin', 'prev_totalvotes',
    ]
    county = county[keep].dropna(subset=['state_lean', 'county_lean', 'county_last_margin'])

    save_path = os.path.join(DATA_DIR, 'featured_data.csv')
    county.to_csv(save_path, index=False)
    print(f"\nFeatured data saved -> {save_path}")
    print(f"Shape: {county.shape}; rows per year: {county.groupby('year').size().to_dict()}")
    return county


if __name__ == '__main__':
    build_features()
