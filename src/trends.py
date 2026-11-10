"""
Data behind the Trends tab:
  national mood   approval average over time, generic-ballot log, economy (FRED)
  races           polls vs fundamentals, our odds vs expert ratings, partisan vs
                  nonpartisan polls, day-to-day movers, close races' averages over time
  history         approval vs midterm seat change, House seats vs national vote
  polling volume  2026 polls per week
app.py caches these; dashboard_2026.render_trends draws them.
"""
import glob
import os

import numpy as np
import pandas as pd
import requests

from national_polls import approval_average, APPROVAL_WINDOW_DAYS

# Gallup job approval at its last reading before each midterm, and the president's
# party's seat change in that election. Source: UCSB American Presidency Project,
# "Seats in Congress Gained/Lost by the President's Party in Mid-Term Elections"
# (https://www.presidency.ucsb.edu/node/332343, updated Dec 2022).
MIDTERM_HISTORY = pd.DataFrame([
    (1946, 'Truman',     'D', 27, -56, -12),
    (1950, 'Truman',     'D', 41, -28,  -6),
    (1954, 'Eisenhower', 'R', 62, -18,  -1),
    (1958, 'Eisenhower', 'R', 57, -48, -13),
    (1962, 'Kennedy',    'D', 61,  -4,   3),
    (1966, 'Johnson',    'D', 44, -47,  -4),
    (1970, 'Nixon',      'R', 58, -12,   2),
    (1974, 'Ford',       'R', 53, -48,  -5),
    (1978, 'Carter',     'D', 45, -15,  -3),
    (1982, 'Reagan',     'R', 42, -26,   1),
    (1986, 'Reagan',     'R', 64,  -5,  -8),
    (1990, 'Bush',       'R', 57,  -8,  -1),
    (1994, 'Clinton',    'D', 48, -54,  -8),
    (1998, 'Clinton',    'D', 65,   4,   0),
    (2002, 'Bush',       'R', 67,   8,   2),
    (2006, 'Bush',       'R', 37, -30,  -6),
    (2010, 'Obama',      'D', 45, -63,  -6),
    (2014, 'Obama',      'D', 41, -13,  -9),
    (2018, 'Trump',      'R', 44, -41,   2),
    (2022, 'Biden',      'D', 40,  -9,   1),
], columns=['year', 'president', 'party', 'approval', 'house_change', 'senate_change'])

def rep_house_seats_2024():
    """Republican House seats won in 2024 (MIT Election Lab results): the baseline for
    the president's party's 2026 seat change."""
    import midterms as M
    return int(M.national_house_seats_by_year(M.load_house_races()).loc[2024, 'REPUBLICAN'])

def approval_series(polls):
    """Daily approve/disapprove averages, from the first day with a full 30-day window.
    Days with no polls in the window are NaN, so charts show a break, not a bridge."""
    polls = polls.dropna(subset=['end_date'])
    days = pd.date_range(polls['end_date'].min() + pd.Timedelta(days=APPROVAL_WINDOW_DAYS),
                         polls['end_date'].max(), freq='D')
    rows = []
    for d in days:
        try:
            rows.append({'date': d,
                         'approve': approval_average(polls, d)['approve'],
                         'disapprove': approval_average(polls, d, col='disapprove')['approve']})
        except ValueError:      # a 30-day stretch with no polls
            rows.append({'date': d, 'approve': np.nan, 'disapprove': np.nan})
    s = pd.DataFrame(rows)
    if len(s):
        s['net'] = s['approve'] - s['disapprove']
    return s

def approval_trend_line(polls, bandwidth_days=10, max_gap_days=45):
    """A smooth trend for charts: each day's value is a Gaussian-weighted average of polls
    around it (weights also grow with sample size). Unlike the 30-day average this doesn't
    jump when a poll enters or leaves the window, which makes stair-steps in months with
    only one or two polls. Days with no poll within max_gap_days are NaN (a break).
    Chart-only: the published approval number is still approval_average()."""
    p = polls.dropna(subset=['end_date', 'approve']).sort_values('end_date')
    if p.empty:
        return pd.DataFrame(columns=['date', 'approve', 'disapprove', 'net'])
    t = p['end_date'].values.astype('datetime64[D]').astype(float)
    size = np.sqrt(p['sample_size'].fillna(1000).clip(300, 5000).values / 1000)
    days = pd.date_range(p['end_date'].min(), p['end_date'].max(), freq='D')
    rows = []
    for d in days:
        dt = t - d.to_datetime64().astype('datetime64[D]').astype(float)
        if np.abs(dt).min() > max_gap_days:
            rows.append({'date': d, 'approve': np.nan, 'disapprove': np.nan})
            continue
        w = np.exp(-0.5 * (dt / bandwidth_days) ** 2) * size
        row = {'date': d}
        for col in ('approve', 'disapprove'):
            v = p[col].values
            ok = ~np.isnan(v)
            row[col] = float(np.average(v[ok], weights=w[ok])) if w[ok].sum() > 1e-9 else np.nan
        rows.append(row)
    s = pd.DataFrame(rows)
    s['net'] = s['approve'] - s['disapprove']
    return s

def approval_seat_fit(history=MIDTERM_HISTORY, chamber='house'):
    """Least-squares line: president's-party seat change ~ approval. Returns (slope,
    intercept, residual sd)."""
    y = history[f'{chamber}_change']
    slope, intercept = np.polyfit(history['approval'], y, 1)
    resid = y - (intercept + slope * history['approval'])
    return float(slope), float(intercept), float(resid.std(ddof=2))

def polls_per_week(polls):
    """Poll counts by week (ending Sunday) and office, since the start of 2026."""
    p = polls[polls['end_date'] >= '2026-01-01']
    weekly = p.groupby([pd.Grouper(key='end_date', freq='W'), 'office']).size().unstack(fill_value=0)
    return weekly.reindex(columns=[c for c in ('senate', 'governor', 'house') if c in weekly.columns])


# ── National mood ─────────────────────────────────────────────────────────────
FORECASTS_2026 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'forecasts', '2026')

def generic_ballot_history(log, topline):
    """Daily generic-ballot values: non-market-tied aggregators from the log, plus our
    average (the forecast input) from the published topline log, which goes back further."""
    rows = []
    if log is not None and len(log):
        rows.append(log[~log['market_tied'].astype(bool)][['date', 'aggregator', 'margin']])
    if topline is not None and len(topline):
        t = topline.assign(date=pd.to_datetime(topline['published_at']).dt.normalize())
        rows.append(t.groupby('date', as_index=False)['generic_ballot'].last()
                    .rename(columns={'generic_ballot': 'margin'}).assign(aggregator='Our average'))
    if not rows:
        return pd.DataFrame(columns=['date', 'aggregator', 'margin'])
    h = pd.concat(rows, ignore_index=True)
    h['date'] = pd.to_datetime(h['date']).dt.normalize()
    return h.drop_duplicates(['date', 'aggregator'], keep='first').sort_values('date')

# FRED series for the economy panel: (id, label, unit, transform)
ECONOMY_SERIES = [
    ('UNRATE',   'Unemployment rate', '%', None),
    ('CPIAUCSL', 'Inflation (CPI, year over year)', '%', 'yoy'),
    ('UMCSENT',  'Consumer sentiment (University of Michigan)', 'index', None),
    ('GASREGW',  'Regular gas, national average', '$/gal', None),
]
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

def fred_series(series_id, api_key, start='2021-01-01', transform=None):
    """One FRED series as a (date, value) frame; 'yoy' turns an index into % change from a year earlier."""
    fetch_start = (pd.Timestamp(start) - pd.DateOffset(months=13)).date().isoformat() if transform == 'yoy' else start
    r = requests.get(FRED_BASE, timeout=15, params={'series_id': series_id, 'api_key': api_key, 'file_type': 'json',
                                                    'observation_start': fetch_start})
    r.raise_for_status()
    obs = pd.DataFrame(r.json()['observations'])
    obs = obs[obs['value'] != '.']
    s = pd.Series(obs['value'].astype(float).values, index=pd.to_datetime(obs['date']))
    if transform == 'yoy':
        s = (s / s.shift(12) - 1) * 100
    s = s[s.index >= pd.Timestamp(start)].dropna()
    return pd.DataFrame({'date': s.index, 'value': s.values})

def economy(api_key, start='2021-01-01'):
    """{series_id: frame} for every series that fetches; failures are skipped."""
    out = {}
    for sid, _, _, transform in ECONOMY_SERIES:
        try:
            out[sid] = fred_series(sid, api_key, start, transform)
        except Exception as e:
            print(f"FRED {sid} failed: {e}")
    return out


# ── Races ─────────────────────────────────────────────────────────────────────
def polls_vs_fundamentals(forecast, office):
    """Polled races with the fundamentals prediction next to the polling average."""
    f = forecast[office].dropna(subset=['poll_avg', 'fund_pred']).copy()
    f['gap'] = f['poll_avg'] - f['fund_pred']
    return f

def ours_vs_experts(forecast, office):
    import comparisons as CMP
    t = CMP.forecaster_table(forecast, office)
    return t[t['n_forecasters'] > 0]

def partisan_poll_split(annotated, forecast, office, days=90, max_margin=10):
    """Average raw margin of Democratic-sponsored, nonpartisan, and Republican-sponsored
    polls in each competitive race over the last `days`."""
    close = forecast[office].loc[lambda d: d['forecast_margin'].abs() < max_margin, 'state']
    a = annotated[(annotated['office'] == office) & annotated['state'].isin(close)]
    a = a[a['end_date'] > annotated['end_date'].max() - pd.Timedelta(days=days)]
    group = a['sponsor'].map({'DEMOCRAT': 'Democratic-sponsored', 'REPUBLICAN': 'Republican-sponsored'}).fillna('Nonpartisan')
    g = a.assign(group=group).groupby(['state', 'group'])['margin'].agg(['mean', 'size']).reset_index()
    has_partisan = g[g['group'] != 'Nonpartisan']['state'].unique()
    return g[g['state'].isin(has_partisan)]

def race_history(offices=('senate', 'governor')):
    """Each race's published win probability for every forecast day on disk."""
    frames = []
    for office in offices:
        for path in sorted(glob.glob(os.path.join(FORECASTS_2026, f'*_{office}_races.csv'))):
            day = os.path.basename(path)[:10]
            d = pd.read_csv(path, usecols=['state', 'dem_side_win_prob', 'forecast_margin'])
            frames.append(d.assign(office=office, date=pd.Timestamp(day)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def movers(history, days=7, n=10):
    """Races whose Democratic-side win probability moved most between the earliest
    snapshot in the last `days` days and the latest one."""
    if history.empty:
        return history
    last = history['date'].max()
    window = history[history['date'] >= last - pd.Timedelta(days=days)]
    first = window['date'].min()
    a = window[window['date'] == first].set_index(['office', 'state'])['dem_side_win_prob']
    b = window[window['date'] == last].set_index(['office', 'state'])['dem_side_win_prob']
    m = pd.DataFrame({'before': a, 'after': b}).dropna()
    m['change'] = m['after'] - m['before']
    m = m.loc[m['change'].abs().sort_values(ascending=False).index].head(n).reset_index()
    return m.assign(start=first, end=last)

def close_race_averages(annotated, forecast, max_margin=5, start='2026-07-01'):
    """Polling average every other day (house-effect corrected, before the forecast's lean
    correction) for every race the forecast has within `max_margin` points."""
    from polls_average import race_average
    days = pd.date_range(start, annotated['end_date'].max(), freq='2D')     # every other day keeps this to a few seconds
    rows = []
    for office in ('senate', 'governor'):
        f = forecast[office]
        for state in f.loc[f['forecast_margin'].abs() < max_margin, 'state']:
            g = annotated[(annotated['office'] == office) & (annotated['state'] == state)]
            for d in days:
                avg = race_average(g, d)
                rows.append({'office': office, 'state': state, 'date': d,
                             'poll_avg': avg['poll_avg'] if avg else np.nan})
    return pd.DataFrame(rows)


# ── History ───────────────────────────────────────────────────────────────────
def house_seats_votes():
    """National House vote margin and Democratic seats won, by election year."""
    import midterms as M
    t = M.house_seats_votes_table()
    t.index.name = 'year'
    return t.reset_index()
