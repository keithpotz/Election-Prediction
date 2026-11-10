import requests
import pandas as pd
import os
from datetime import datetime

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

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

def fetch_approval_rating():
    """Presidential approval, averaged by us from Wikipedia's poll list (see national_polls)."""
    from national_polls import fetch_approval_rating as f
    return f()

def fetch_generic_ballot():
    """Generic ballot from non-market-affiliated aggregators (see national_polls)."""
    from national_polls import fetch_generic_ballot as f
    return f()

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
