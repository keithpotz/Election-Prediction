import requests
import pandas as pd
import os
from datetime import datetime

FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"
APPROVAL_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data/polling_data/Silver Bulletin Trump approval polls - Sheet1.csv"
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

def fetch_approval_rating():
    """
    Compute a weighted average approval rating from the Silver Bulletin CSV.
    Filters to the 'All polls' subgroup and uses the most recent 30 days.
    Returns (approval_pct, source_note) or (None, None) on failure.
    """
    try:
        df = pd.read_csv(APPROVAL_CSV)
        df = df[df['subgroup'] == 'All polls'].copy()
        df['enddate']          = pd.to_datetime(df['enddate'], errors='coerce')
        df['adjusted_approve'] = pd.to_numeric(df['adjusted_approve'], errors='coerce')
        df['weight']           = pd.to_numeric(df['weight'], errors='coerce')
        df = df.dropna(subset=['enddate', 'adjusted_approve', 'weight'])

        # Most recent 30 days; fall back to all if nothing in window
        cutoff = df['enddate'].max() - pd.Timedelta(days=30)
        recent = df[df['enddate'] >= cutoff]
        if recent.empty:
            recent = df

        weighted_avg = (recent['adjusted_approve'] * recent['weight']).sum() / recent['weight'].sum()
        latest_date  = recent['enddate'].max().strftime('%Y-%m-%d')
        source       = f"Silver Bulletin · {len(recent)} polls · latest: {latest_date}"
        return round(weighted_avg, 1), source

    except Exception as e:
        print(f"Silver Bulletin approval fetch failed: {e}")
        return None, None

def fetch_all(api_key):
    """Fetch all live economic indicators. Returns a dict with values and metadata."""
    gdp,    gdp_date          = fetch_gdp_growth(api_key)
    unemp,  unemp_date        = fetch_unemployment(api_key)
    approval, approval_source = fetch_approval_rating()

    return {
        'gdp_growth':      gdp,
        'gdp_date':        gdp_date,
        'unemployment':    unemp,
        'unemp_date':      unemp_date,
        'approval':        approval,
        'approval_source': approval_source,
        'fetched_at':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
