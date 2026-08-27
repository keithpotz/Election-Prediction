import requests
import pandas as pd
import os
from datetime import datetime

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

APPROVAL_CSV_LOCAL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data/polling_data/Silver Bulletin Trump approval polls - Sheet1.csv"
)
APPROVAL_CSV_LIVE = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vS-FKWVTTFtJT6u56e0bqdfoMcXvDO1"
    "DUChsJ3jQAMB2lZk2SMqVfmg7dGjclTYkYWz-Pm5lfcLPjp4/pub?output=csv"
)
GENERIC_BALLOT_CSV_LIVE = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRsvXNCZ0ubJr8D_yNcU5q6C0_HBa35K7"
    "oDK03KpO7Ca43UwdXaIdvVLWoXEmHHph0EREz5430Hm5yZ/pub?output=csv"
)

def fetch_gdp_growth(api_key):
    """
    Fetch the most recent annual real GDP growth rate from FRED.
    Series: A191RL1A225NBEA (Real GDP, percent change from preceding period, annual)
    """
    try:
        params = {
            'series_id':  'A191RL1A225NBEA',
            'api_key':    api_key,
            'file_type':  'json',
            'sort_order': 'desc',
            'limit':      5,
        }
        r = requests.get(FRED_BASE, params=params, timeout=10)
        r.raise_for_status()
        for obs in r.json().get('observations', []):
            if obs['value'] != '.':
                return float(obs['value']), obs['date']
    except Exception as e:
        print(f"FRED GDP fetch failed: {e}")
    return None, None

def fetch_unemployment(api_key):
    """
    Fetch the most recent monthly unemployment rate from FRED.
    Series: UNRATE
    """
    try:
        params = {
            'series_id':  'UNRATE',
            'api_key':    api_key,
            'file_type':  'json',
            'sort_order': 'desc',
            'limit':      1,
        }
        r = requests.get(FRED_BASE, params=params, timeout=10)
        r.raise_for_status()
        observations = r.json().get('observations', [])
        if observations and observations[0]['value'] != '.':
            obs = observations[0]
            return float(obs['value']), obs['date']
    except Exception as e:
        print(f"FRED unemployment fetch failed: {e}")
    return None, None

def fetch_cpi_inflation(api_key):
    """
    Fetch the most recent YoY CPI inflation rate from FRED.
    Series: CPIAUCSL (monthly index level, YoY % change computed here)
    """
    try:
        params = {
            'series_id':  'CPIAUCSL',
            'api_key':    api_key,
            'file_type':  'json',
            'sort_order': 'desc',
            'limit':      15,  # buffer for occasional gaps/unpublished months
        }
        r = requests.get(FRED_BASE, params=params, timeout=10)
        r.raise_for_status()
        obs = [o for o in r.json().get('observations', []) if o['value'] != '.']
        if len(obs) < 2:
            return None, None

        by_date = {o['date']: float(o['value']) for o in obs}
        latest_date, latest_value = obs[0]['date'], float(obs[0]['value'])
        year_ago_date = (pd.Timestamp(latest_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        year_ago_value = by_date.get(year_ago_date)

        if year_ago_value:
            return round((latest_value - year_ago_value) / year_ago_value * 100, 2), latest_date
    except Exception as e:
        print(f"FRED CPI fetch failed: {e}")
    return None, None

def fetch_consumer_confidence(api_key):
    """
    Fetch the most recent University of Michigan Consumer Sentiment reading from FRED.
    Series: UMCSENT
    """
    try:
        params = {
            'series_id':  'UMCSENT',
            'api_key':    api_key,
            'file_type':  'json',
            'sort_order': 'desc',
            'limit':      1,
        }
        r = requests.get(FRED_BASE, params=params, timeout=10)
        r.raise_for_status()
        observations = r.json().get('observations', [])
        if observations and observations[0]['value'] != '.':
            obs = observations[0]
            return float(obs['value']), obs['date']
    except Exception as e:
        print(f"FRED consumer confidence fetch failed: {e}")
    return None, None

def _weighted_recent_avg(df, value_col, days=30):
    """Shared logic: weight-averaged value over the most recent `days`, falling back to all rows."""
    df = df.copy()
    df['enddate'] = pd.to_datetime(df['enddate'], errors='coerce')
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df['weight']  = pd.to_numeric(df['weight'], errors='coerce')
    df = df.dropna(subset=['enddate', value_col, 'weight'])

    cutoff = df['enddate'].max() - pd.Timedelta(days=days)
    recent = df[df['enddate'] >= cutoff]
    if recent.empty:
        recent = df

    weight_sum = recent['weight'].sum()
    if weight_sum == 0:
        # every poll in the window has zero weight — a 0/0 division would silently
        # return NaN here, which then passes `is not None` checks and is truthy in
        # Python, letting a NaN slip all the way to a crashing int(nan) at the UI.
        # Raise instead so callers' existing except-and-fallback logic handles it.
        raise ValueError("No polls with nonzero weight in the averaging window")

    weighted_avg = (recent[value_col] * recent['weight']).sum() / weight_sum
    latest_date  = recent['enddate'].max().strftime('%Y-%m-%d')
    return round(weighted_avg, 1), len(recent), latest_date

def fetch_approval_rating():
    """
    Compute a weighted average presidential approval rating from Silver Bulletin's
    live poll database (falls back to the local manually-downloaded CSV if offline).
    Returns (approval_pct, source_note) or (None, None) on failure.
    """
    for source_label, source in [("Silver Bulletin (live)", APPROVAL_CSV_LIVE),
                                  ("Silver Bulletin (local)", APPROVAL_CSV_LOCAL)]:
        try:
            df = pd.read_csv(source)
            df = df[df['subgroup'] == 'All polls']
            avg, n, latest_date = _weighted_recent_avg(df, 'adjusted_approve')
            return avg, f"{source_label} · {n} polls · latest: {latest_date}"
        except Exception as e:
            print(f"Approval fetch from {source_label} failed: {e}")
    return None, None

def fetch_generic_ballot():
    """
    Compute a weighted average national generic congressional ballot margin (Dem - Rep)
    from Silver Bulletin's live poll database.
    Returns (dem_minus_rep_margin, source_note) or (None, None) on failure.
    """
    try:
        df = pd.read_csv(GENERIC_BALLOT_CSV_LIVE)
        df = df[df['subgroup'] == 'All polls'].copy()
        avg, n, latest_date = _weighted_recent_avg(df, 'adjusted_net')
        return avg, f"Silver Bulletin generic ballot · {n} polls · latest: {latest_date}"
    except Exception as e:
        print(f"Generic ballot fetch failed: {e}")
        return None, None

def fetch_all(api_key):
    """Fetch all live economic and polling indicators. Returns a dict with values and metadata."""
    gdp,        gdp_date          = fetch_gdp_growth(api_key)
    unemp,      unemp_date        = fetch_unemployment(api_key)
    inflation,  inflation_date    = fetch_cpi_inflation(api_key)
    confidence, confidence_date   = fetch_consumer_confidence(api_key)
    approval,   approval_source   = fetch_approval_rating()
    generic_ballot, generic_ballot_source = fetch_generic_ballot()

    return {
        'gdp_growth':      gdp,
        'gdp_date':        gdp_date,
        'unemployment':    unemp,
        'unemp_date':      unemp_date,
        'inflation':       inflation,
        'inflation_date':  inflation_date,
        'consumer_confidence':      confidence,
        'consumer_confidence_date': confidence_date,
        'approval':        approval,
        'approval_source': approval_source,
        'generic_ballot':        generic_ballot,
        'generic_ballot_source': generic_ballot_source,
        'fetched_at':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
