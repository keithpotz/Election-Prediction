"""
District-by-district 2026 House forecast for all 435 seats (roadmap Phase 2).

District data comes from Wikipedia's "2026 United States House of Representatives
elections" page: every district's 2026 Cook PVI (on the lines in use this November,
including the nine states that redrew mid-decade), incumbent, party, renomination or
retirement status, and general-election candidates.

Model: district margin = a*PVI + b*incumbency + c, shifted 1:1 by the change in the
national House vote since 2024 (uniform swing). a, b, c are calibrated on 2024 results
in the 41 states that did NOT redraw for 2026, so the 2026 PVI describes the same lines
those 2024 results were cast on. The calibration's leave-one-state-out residual sets
each district's own uncertainty, and the generic-ballot polling error is shared by
every district in a simulation.

Caveat: Cook's 2026 PVI averages the 2020 and 2024 presidential results, so it includes
the same day's 2024 presidential vote. The calibrated coefficients are slightly
flattered by that. Out-of-sample district error is reported anyway.

    python src/house_districts.py
"""
import io
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from race_status_2026 import _fetch_html, DATA_DIR
from polls_2026 import _flatten, _sanitize
from feature_engineering import load_house_races, national_house_margin_by_year
from midterms import _last_name, GENERIC_BALLOT_ERROR_SD, HOUSE_MAJORITY, HOUSE_SEATS_TOTAL, N_SIMS

HOUSE_WIKI_URL = "https://en.wikipedia.org/wiki/2026_United_States_House_of_Representatives_elections"
DISTRICTS_CSV = os.path.join(DATA_DIR, 'house_districts_2026.csv')
MAX_CACHE_AGE_HOURS = 24
REDISTRICTED_FOR_2026 = {'ALABAMA', 'CALIFORNIA', 'FLORIDA', 'LOUISIANA', 'NORTH CAROLINA',
                         'OHIO', 'TENNESSEE', 'TEXAS', 'UTAH'}
RUNNING = ('renominated', 'advanced to general', 'running', 're-elected')
FEATURES = ['pvi', 'incumbency', 'inc_strength']
STRENGTH_CLIP = 60      # uncontested races (+/-100) shouldn't dominate

def _margin(votes):
    d, r = votes.get('DEMOCRAT', 0), votes.get('REPUBLICAN', 0)
    return (d - r) / (d + r) * 100 if d + r > 0 else np.nan

def inc_strength(prev_margin, pvi, incumbency):
    """How far a running incumbent beat (or trailed) their district's lean last time:
    previous margin minus 2 x Cook PVI (PVI is on a half-margin scale). Zero for open seats
    or when the previous race was on different lines."""
    if incumbency == 0 or prev_margin is None or pd.isna(prev_margin) or pd.isna(pvi):
        return 0.0
    return float(np.clip(prev_margin, -STRENGTH_CLIP, STRENGTH_CLIP) - 2 * pvi)

def _party(text):
    t = str(text)
    if 'Democratic' in t or 'DFL' in t:
        return 'DEMOCRAT'
    if 'Republican' in t:
        return 'REPUBLICAN'
    return None

def _parse_pvi(text):
    t = str(text).upper()
    if t.startswith('EVEN'):
        return 0.0
    m = re.match(r'^(D|R)\+(\d+)', t)
    return (float(m.group(2)) if m.group(1) == 'D' else -float(m.group(2))) if m else np.nan

def _parse_location(text):
    t = re.sub(r'\[[^\]]*\]', '', str(text)).strip()
    m = re.match(r'^(.*?)\s+(at-large|\d+)$', t, flags=re.I)
    if not m:
        return None, None
    return m.group(1).upper(), 0 if m.group(2).lower() == 'at-large' else int(m.group(2))

def _candidates(text):
    """'▌Jerry Carl (Republican) ▌Clyde Jones Jr. (Democratic)' -> [(name, party), ...]"""
    out = []
    for name, party in re.findall(r'▌?\s*([^▌()]+?)\s*\(([^)]+)\)', str(text)):
        out.append((name.strip(), _party(party) or party.strip().upper()))
    return out

def scrape_house_districts():
    html = _fetch_html(HOUSE_WIKI_URL)
    tables = [_flatten(t) for t in pd.read_html(io.StringIO(_sanitize(html)))]
    tables = [t for t in tables if any('PVI' in c for c in t.columns) and any(c.startswith('Candidates') for c in t.columns)]
    rows = []
    for t in tables:
        t = t.set_axis(['location', 'pvi', 'member', 'party', 'first_elected', 'status', 'candidates'], axis=1)
        rows.append(t)
    raw = pd.concat(rows, ignore_index=True)

    out = []
    for loc, g in raw.groupby('location', sort=False):
        state, district = _parse_location(loc)
        if state is None:
            continue
        running = g[g['status'].str.lower().str.contains('|'.join(RUNNING), na=False)]
        inc = running.iloc[0] if len(running) else None
        cands = _candidates(g['candidates'].iloc[0])
        parties = {p for _, p in cands}
        out.append({
            'state': state, 'district': district,
            'pvi': _parse_pvi(g['pvi'].iloc[0]),
            'incumbent': re.sub(r'Redistricted from.*$', '', str(inc['member'])).strip() if inc is not None else None,
            'incumbent_party': _party(inc['party']) if inc is not None else None,
            'incumbent_running': inc is not None,
            'held_by': _party(g['party'].iloc[0]),       # party of the (first-listed) current member
            'already_decided': bool(g['status'].str.contains('re-elected', case=False, na=False).any()),
            'has_dem': 'DEMOCRAT' in parties, 'has_rep': 'REPUBLICAN' in parties,
            'dem_candidate': next((n for n, p in cands if p == 'DEMOCRAT'), None),
            'rep_candidate': next((n for n, p in cands if p == 'REPUBLICAN'), None),
            'status': '; '.join(g['status'].astype(str)),
        })
    df = pd.DataFrame(out)
    if len(df) != HOUSE_SEATS_TOTAL:
        raise ValueError(f"Parsed {len(df)} House districts, expected {HOUSE_SEATS_TOTAL}. Wikipedia layout may have changed.")
    return df

def load_house_districts(refresh=False):
    exists = os.path.exists(DISTRICTS_CSV)
    fresh = exists and (time.time() - os.path.getmtime(DISTRICTS_CSV)) < MAX_CACHE_AGE_HOURS * 3600
    if fresh and not refresh:
        return pd.read_csv(DISTRICTS_CSV)
    try:
        df = scrape_house_districts()
    except Exception as e:
        if not exists:
            raise
        print(f"House district scrape failed, using stale cache: {e}")
        return pd.read_csv(DISTRICTS_CSV)
    df.to_csv(DISTRICTS_CSV, index=False)
    return df


# ── Calibration on 2024 ────────────────────────────────────────────────────────
def calibration_data(districts=None):
    """2024 contested D-vs-R district results in states whose lines didn't change for 2026."""
    districts = districts if districts is not None else load_house_districts()
    cal = _district_results(load_house_races(), 2024, REDISTRICTED_FOR_2026)
    cal = cal.merge(districts[['state', 'district', 'pvi']], on=['state', 'district'])
    return _add_strength(cal.dropna(subset=['pvi']), LINES_CHANGED_2022_TO_2024)

LINES_CHANGED_2022_TO_2024 = {'ALABAMA', 'GEORGIA', 'LOUISIANA', 'NEW YORK', 'NORTH CAROLINA'}

def _district_results(house, year, exclude_states):
    """Contested D-vs-R results for `year`, with incumbency from the cycle before."""
    rows = []
    for (state, dist), g in house[house['year'] == year].groupby(['state', 'district']):
        if state in exclude_states:
            continue
        votes = g.groupby('party')['candidatevotes'].sum()
        d, r = votes.get('DEMOCRAT', 0), votes.get('REPUBLICAN', 0)
        if d <= 0 or r <= 0:
            continue
        prev = house[(house['year'] == year - 2) & (house['state'] == state) & (house['district'] == dist)]
        incumbency = 0
        prev_margin = _margin(prev.groupby('party')['candidatevotes'].sum()) if len(prev) else np.nan
        if len(prev):
            w = prev.loc[prev['candidatevotes'].idxmax()]
            if _last_name(w['candidate']) in {_last_name(c) for c in g['candidate']} and w['party'] in ('DEMOCRAT', 'REPUBLICAN'):
                incumbency = 1 if w['party'] == 'DEMOCRAT' else -1
        rows.append({'state': state, 'district': int(dist), 'margin': (d - r) / (d + r) * 100,
                     'incumbency': incumbency, 'prev_margin': prev_margin})
    return pd.DataFrame(rows)

def _add_strength(df, lines_changed):
    """inc_strength per row; zero where the previous race was on different lines."""
    df = df.copy()
    df['inc_strength'] = [0.0 if st in lines_changed else inc_strength(pm, pv, inc)
                          for st, pm, pv, inc in zip(df['state'], df['prev_margin'], df['pvi'], df['incumbency'])]
    return df

def out_of_year_check(model, districts):
    """
    Applies the 2024-calibrated model to 2022 results in states whose lines were the
    same in 2022, 2024, and 2026. This is a truer error estimate than the 2024 fit,
    which is flattered because Cook's PVI already contains the 2024 presidential vote.
    """
    env = national_house_margin_by_year()
    t = _district_results(load_house_races(), 2022, LINES_CHANGED_2022_TO_2024 | REDISTRICTED_FOR_2026)
    t = t.merge(districts[['state', 'district', 'pvi']], on=['state', 'district']).dropna(subset=['pvi'])
    # 2020 -> 2022 crossed the post-census redistricting nearly everywhere, so the previous
    # race isn't the same district: incumbent strength is set to zero for this check.
    t['inc_strength'] = 0.0
    pred = model.predict(t[FEATURES]) + (env[2022] - env[2024])
    err = t['margin'] - pred
    return {'n_districts': int(len(t)), 'sd': float(err.std(ddof=1)), 'bias': float(err.mean()),
            'winner_accuracy': float(((pred > 0) == (t['margin'] > 0)).mean())}

def calibrate(cal=None, districts=None):
    """Fits the district model on 2024. District uncertainty is the larger of the
    leave-one-state-out error and the 2022 out-of-year error."""
    cal = cal if cal is not None else calibration_data(districts)
    model = LinearRegression().fit(cal[FEATURES], cal['margin'])
    resid = []
    for state in cal['state'].unique():
        test = cal['state'] == state
        m = LinearRegression().fit(cal.loc[~test, FEATURES], cal.loc[~test, 'margin'])
        resid.extend(cal.loc[test, 'margin'] - m.predict(cal.loc[test, FEATURES]))
    resid = np.array(resid)
    oyc = out_of_year_check(model, districts if districts is not None else load_house_districts())
    stats = {'district_sd': max(float(resid.std(ddof=1)), oyc['sd']),
             'in_year_sd': float(resid.std(ddof=1)), 'out_of_year': oyc,
             'n_districts': int(len(cal)),
             'coefficients': {k: round(float(v), 2) for k, v in zip(FEATURES, model.coef_)},
             'intercept': round(float(model.intercept_), 2)}
    return model, stats


# ── 2026 forecast ─────────────────────────────────────────────────────────────
def add_2026_strength(districts, house=None):
    """Incumbent strength from each district's 2024 result, for states whose lines
    didn't change for 2026."""
    house = house if house is not None else load_house_races()
    h24 = house[house['year'] == 2024]
    prev = {(st, int(d)): _margin(g.groupby('party')['candidatevotes'].sum()) for (st, d), g in h24.groupby(['state', 'district'])}
    out = districts.copy()
    out['prev_margin'] = [prev.get((st, int(d))) for st, d in zip(out['state'], out['district'])]
    return _add_strength(out, REDISTRICTED_FOR_2026)


def forecast_house(generic_ballot_margin, districts=None, model_bundle=None, n_sims=N_SIMS, seed=42,
                   shared_sd=GENERIC_BALLOT_ERROR_SD, env_2024=None):
    districts = (districts if districts is not None else load_house_districts()).copy()
    model, stats = model_bundle or calibrate(calibration_data(districts), districts)
    env_2024 = env_2024 if env_2024 is not None else float(national_house_margin_by_year()[2024])

    inc = np.where(districts['incumbent_running'].astype(bool) & ~districts['already_decided'].astype(bool),
                   np.where(districts['incumbent_party'] == 'DEMOCRAT', 1, -1), 0)
    districts['incumbency'] = inc
    districts['pvi'] = districts['pvi'].fillna(0.0)
    if 'inc_strength' not in districts:
        districts = add_2026_strength(districts)
    districts['predicted_margin'] = model.predict(districts[FEATURES]) + (generic_ballot_margin - env_2024)

    # Races with only one major party on the ballot, or already settled, aren't simulated.
    fixed = pd.Series(np.nan, index=districts.index)
    fixed[districts['has_dem'] & ~districts['has_rep']] = 1.0
    fixed[districts['has_rep'] & ~districts['has_dem']] = 0.0
    decided = districts['already_decided'].astype(bool)
    fixed[decided] = (districts.loc[decided, 'incumbent_party'] == 'DEMOCRAT').astype(float)
    districts['uncontested'] = fixed.notna()

    rng = np.random.default_rng(seed)
    national = rng.normal(0, shared_sd, (n_sims, 1))
    local = rng.normal(0, stats['district_sd'], (n_sims, len(districts)))
    dem_win = (districts['predicted_margin'].values + national + local) > 0
    dem_win = np.where(fixed.notna().values, fixed.fillna(0).values.astype(bool), dem_win)

    districts['dem_win_prob'] = dem_win.mean(axis=0).round(3)
    districts['predicted_winner'] = np.where(districts['dem_win_prob'] >= 0.5, 'DEMOCRAT', 'REPUBLICAN')
    seats = dem_win.sum(axis=1)
    return {
        'districts':        districts,
        'dem_seats_median': int(np.median(seats)),
        'dem_seats_expected': float(seats.mean()),
        'dem_seats_lo':     int(np.percentile(seats, 5)),
        'dem_seats_hi':     int(np.percentile(seats, 95)),
        'dem_control_prob': round(float((seats >= HOUSE_MAJORITY).mean()), 3),
        'dem_seats_dist':   pd.Series(seats).value_counts(normalize=True).sort_index(),
        'calibration':      stats,
        'n_uncontested':    int(fixed.notna().sum()),
    }

if __name__ == '__main__':
    d = load_house_districts(refresh=True)
    print(f"{len(d)} districts; incumbents running: {d['incumbent_running'].sum()}; "
          f"one-party ballots: {int((d['has_dem'] ^ d['has_rep']).sum())}; decided: {int(d['already_decided'].sum())}")
    model, stats = calibrate(calibration_data(d), d)
    print("Calibration (2024, unchanged-line states):", stats)
    from data_sources import fetch_generic_ballot
    gb, _ = fetch_generic_ballot()
    for g in (0.0, gb):
        f = forecast_house(g, d, (model, stats))
        print(f"\nGeneric ballot D{g:+.1f}: median {f['dem_seats_median']} D seats "
              f"(90%: {f['dem_seats_lo']}-{f['dem_seats_hi']}), Dem control {f['dem_control_prob']:.0%}")
