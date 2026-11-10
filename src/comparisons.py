"""
How our forecast compares with (1) other forecasters and (2) individual pollsters.

Other forecasters: each race's rating from every forecaster in Wikipedia's predictions
tables (Cook, Sabato, Inside Elections, The Economist, FiftyPlusOne, Split Ticket, Fox),
excluding those with prediction-market ties, next to our probability converted to the same
rating scale. Plus each forecaster's projected Senate total.

Individual pollsters: each pollster's polls from the last 60 days compared with our
polling average of the same race (before lean corrections), averaged across races. A
positive lean means the pollster shows Democrats doing better than the rest of the field.
"""
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'polling_data')
RECENT_DAYS = 60

# Our probability -> the forecasters' rating words (boundaries halfway between the
# probabilities those words imply in grade_2026.EXPERT_RATING_TO_PROB)
RATING_BANDS = [(0.55, 'Tossup'), (0.65, 'Tilt'), (0.775, 'Lean'), (0.91, 'Likely'), (1.01, 'Safe')]
RATING_VALUE = {'Tossup': 0, 'Tilt': 3, 'Lean': 7, 'Likely': 15, 'Safe': 30}

def our_rating(p_dem_side, dem_side_party='DEMOCRAT'):
    fav = max(p_dem_side, 1 - p_dem_side)
    word = next(w for cut, w in RATING_BANDS if fav < cut)
    if word == 'Tossup':
        return word, 0
    side = ('I' if dem_side_party == 'INDEPENDENT' else 'D') if p_dem_side >= 0.5 else 'R'
    return f"{word} {side}", RATING_VALUE[word] * (1 if side != 'R' else -1)

def forecaster_table(forecast, office):
    """Race x forecaster ratings (non-market-tied), plus ours and where we stand out."""
    path = os.path.join(DATA_DIR, f'{office}_forecaster_ratings_2026.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    fr = pd.read_csv(path)
    fr = fr[~fr['market_tied']]
    wide = fr.pivot_table(index='state', columns='forecaster', values='rating', aggfunc='first')
    vals = fr.pivot_table(index='state', columns='forecaster', values='value', aggfunc='first')
    ours = forecast[office].set_index('state')
    rows = []
    for state in ours.index:
        p = float(ours.loc[state, 'dem_side_win_prob'])
        word, val = our_rating(p, ours.loc[state, 'dem_side_party'])
        others = vals.loc[state].dropna() if state in vals.index else pd.Series(dtype=float)
        rows.append({'state': state, 'ours': word, 'our_value': val, 'our_prob': p,
                     'forecasters_avg': float(others.mean()) if len(others) else np.nan,
                     'n_forecasters': int(len(others)),
                     **({f: wide.loc[state, f] for f in wide.columns} if state in wide.index else {})})
    t = pd.DataFrame(rows)
    t['gap'] = t['our_value'] - t['forecasters_avg']     # + = we're more Democratic than the forecasters
    return t.sort_values('gap', key=lambda s: -s.abs())

def forecaster_totals(office='senate'):
    path = os.path.join(DATA_DIR, f'{office}_forecaster_totals_2026.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    t = pd.read_csv(path)
    return t[~t['market_tied']].sort_values('dem')

def pollster_leans(annotated, poll_avgs, as_of=None, min_polls=3):
    """Each pollster's average distance from our (uncorrected) race averages, recent polls only."""
    as_of = pd.Timestamp(as_of or pd.Timestamp.today().normalize())
    p = annotated[annotated['end_date'] >= as_of - pd.Timedelta(days=RECENT_DAYS)].copy()
    avg = poll_avgs.set_index(['office', 'state'])['raw_poll_avg']
    p['race_avg'] = [avg.get((o, s), np.nan) for o, s in zip(p['office'], p['state'])]
    p = p.dropna(subset=['race_avg'])
    p['lean'] = p['margin'] - p['race_avg']
    p['name'] = p['pollster'].str.replace(r'\[[^\]]*\]', '', regex=True).str.strip()
    g = p.groupby('name').agg(polls=('lean', 'size'), races=('state', 'nunique'), lean=('lean', 'mean'),
                              sponsor=('sponsor', lambda s: s.dropna().iloc[0] if s.notna().any() else None),
                              rated_as=('matched_pollster', lambda s: s.dropna().iloc[0] if s.notna().any() else None))
    g = g[g['polls'] >= min_polls].copy()
    g['lean'] = g['lean'].round(1)
    return g.sort_values('lean').reset_index()
