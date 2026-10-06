"""
Scrapes 2026 general-election polls for every Senate and governor race from each race's
Wikipedia page (roadmap Phase 1). RealClearPolitics blocks scripted requests (403) and
FiveThirtyEight's live feed is gone, while Wikipedia's race pages carry a maintained
poll table with pollster, dates, sample size, voter screen, and each candidate's share.

How the actual matchup is picked on a page that also lists hypothetical matchups:
  1. The general-election "Source of poll aggregation" table (columns tagged (D)/(R))
     names the real nominees, when the page has one.
  2. Otherwise the first poll table with one Democratic and one Republican column is
     used. Wikipedia lists the actual matchup first, ahead of "Hypothetical polling".
The chosen candidates are stored on every row so a wrong pick is easy to spot.

Run as a script to refresh src/data/polls_2026/polls_2026.csv:
    python src/polls_2026.py
"""
import io
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd

from race_status_2026 import _fetch_html, load_or_scrape_race_status

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLLS_DIR = os.path.join(BASE_DIR, 'data', 'polls_2026')
POLLS_CSV = os.path.join(POLLS_DIR, 'polls_2026.csv')
COVERAGE_CSV = os.path.join(POLLS_DIR, 'coverage_2026.csv')

WIKI = "https://en.wikipedia.org/wiki/"
PARTY_TAG = re.compile(r'\((D|DFL|R|I|L|G)\)')
DEM_TAGS, REP_TAGS = {'D', 'DFL'}, {'R'}

def race_url(office, state, special=False):
    name = state.title().replace(' ', '_')
    if office == 'senate':
        kind = 'special_election' if special else 'election'
        return f"{WIKI}2026_United_States_Senate_{kind}_in_{name}"
    return f"{WIKI}2026_{name}_gubernatorial_election"

def _flatten(table):
    table = table.copy()
    table.columns = [
        ' '.join(str(x) for x in c if 'Unnamed' not in str(x)).strip() if isinstance(c, tuple) else str(c)
        for c in table.columns
    ]
    # Wikipedia's two-row headers often repeat the same text in both rows.
    table.columns = [re.sub(r'^(.+) \1$', r'\1', c) for c in table.columns]
    return table

def _party_columns(columns):
    out = {}
    for c in columns:
        m = PARTY_TAG.search(c)
        if m:
            out[c] = m.group(1)
    return out

def _cand_name(col):
    """Normalized full candidate name from a column header like 'Dan S. Sullivan (R)[x]'.
    Full names, not last names: Alaska 2026 has two Republicans named Dan Sullivan."""
    name = PARTY_TAG.sub('', re.sub(r'\[[^\]]*\]', '', col))
    return ' '.join(re.findall(r'[a-z]+', name.lower()))

def _pick_sides(party_cols):
    """(dem_side_column, rep_column, dem_side_party) or None. An independent stands in
    for the Democratic side only when no Democrat is listed (e.g. Nebraska's Osborn)."""
    dems = [c for c, p in party_cols.items() if p in DEM_TAGS]
    reps = [c for c, p in party_cols.items() if p in REP_TAGS]
    inds = [c for c, p in party_cols.items() if p == 'I']
    if len(reps) != 1:
        return None
    if len(dems) == 1:
        return dems[0], reps[0], 'DEMOCRAT'
    if not dems and len(inds) == 1:
        return inds[0], reps[0], 'INDEPENDENT'
    return None

def _pct(x):
    m = re.search(r'(\d+(?:\.\d+)?)\s*%', str(x))
    return float(m.group(1)) if m else np.nan

def _sample(x):
    s = str(x)
    m = re.search(r'([\d,]+)', s)
    pop = re.search(r'\((LV|RV|A|V)\)', s)
    return (float(m.group(1).replace(',', '')) if m else np.nan), (pop.group(1) if pop else None)

def _end_date(text):
    s = re.sub(r'\[[^\]]*\]', '', str(text)).strip()
    years = re.findall(r'\b(20\d\d)\b', s)
    if not years:
        return pd.NaT
    year = years[-1]
    body = re.sub(r',?\s*20\d\d', '', s)
    parts = [p.strip() for p in re.split(r'\s*[–—−-]\s*', body) if p.strip()]
    last = parts[-1]
    if re.fullmatch(r'\d+', last):                      # "October 1–4" -> month from start
        last = f"{parts[0].split()[0]} {last}"
    return pd.to_datetime(f"{last} {year}", errors='coerce')

def _sanitize(html):
    # Some pages carry malformed span attributes (e.g. colspan="2"") that crash read_html.
    return re.sub(r'''(colspan|rowspan)\s*=\s*["']?(\d+)[^\s>]*''', r'\1="\2"', html)

def parse_race_page(html, office, state, url):
    tables = [_flatten(t) for t in pd.read_html(io.StringIO(_sanitize(html)))]
    poll_tables = [t for t in tables if any(c.startswith('Poll source') for c in t.columns)]
    agg_tables = [t for t in tables if any(c.startswith('Source of poll aggregation') for c in t.columns)]

    nominees = None
    for t in agg_tables:
        sides = _pick_sides(_party_columns(t.columns))
        if sides:
            nominees = (_cand_name(sides[0]), _cand_name(sides[1]), sides[2])
            break

    chosen = []
    if nominees is None:
        # No aggregate table to name the nominees: take the D-vs-R table with the most
        # recent poll. Hypothetical matchups stop being polled once nominees are set.
        candidates = []
        for t in poll_tables:
            sides = _pick_sides(_party_columns(t.columns))
            date_col = next((c for c in t.columns if c.startswith('Date')), None)
            if sides and date_col:
                latest = t[date_col].map(_end_date).max()
                candidates.append((latest if pd.notna(latest) else pd.Timestamp.min, t, sides))
        if candidates:
            _, t, sides = max(candidates, key=lambda x: x[0])
            chosen.append((t, sides))
    for t in poll_tables:
        if nominees is None:
            break
        # Find the nominees' columns by full name, even when the table has extra
        # same-party candidates (top-four primaries, ranked-choice tables).
        by_name = {_cand_name(c): c for c in _party_columns(t.columns)}
        if nominees[0] in by_name and nominees[1] in by_name:
            chosen.append((t, (by_name[nominees[0]], by_name[nominees[1]], nominees[2])))

    rows = []
    for t, (d_col, r_col, d_party) in chosen:
        src = next(c for c in t.columns if c.startswith('Poll source'))
        date_col = next((c for c in t.columns if c.startswith('Date')), None)
        size_col = next((c for c in t.columns if c.startswith('Sample size')), None)
        rcv_col = next((c for c in t.columns if 'RCV round' in c), None)
        for _, r in t.iterrows():
            d, rep = _pct(r[d_col]), _pct(r[r_col])
            if pd.isna(r[src]) or pd.isna(d) or pd.isna(rep):
                continue                                   # event rows ("X withdraws") etc.
            n, pop = _sample(r[size_col]) if size_col else (np.nan, None)
            rows.append({
                'office':        office,
                'state':         state,
                'pollster':      re.sub(r'\[[^\]]*\]', '', str(r[src])).strip(),
                'end_date':      _end_date(r[date_col]) if date_col else pd.NaT,
                'sample_size':   n,
                'population':    pop,
                'dem_candidate': PARTY_TAG.sub('', d_col).strip(),
                'rep_candidate': PARTY_TAG.sub('', r_col).strip(),
                'dem_side_party': d_party,
                'dem_pct':       d,
                'rep_pct':       rep,
                'margin':        d - rep,
                'rcv_round':     pd.to_numeric(r[rcv_col], errors='coerce') if rcv_col else np.nan,
                'source_url':    url,
            })
    return pd.DataFrame(rows), nominees

def _dedupe_versions(df):
    """A pollster often reports one poll several ways (LV and RV, with/without leaners).
    Keep one version per pollster per end date, preferring likely voters."""
    if df.empty:
        return df
    pref = {'LV': 0, 'V': 1, 'RV': 2, 'A': 3}
    df = df.assign(_p=df['population'].map(pref).fillna(4))
    # Ranked-choice polls list each elimination round; the last round is the head-to-head.
    df = df.sort_values(['rcv_round', '_p'], ascending=[False, True], na_position='last')
    df = df.drop_duplicates(['office', 'state', 'pollster', 'end_date'], keep='first')
    return df.drop(columns='_p')

def scrape_all(pause=1.0):
    """Every 2026 Senate and governor race. Returns (polls, coverage)."""
    races = []
    for office in ('senate', 'governor'):
        status = load_or_scrape_race_status(office=office)
        for _, r in status.iterrows():
            races.append((office, r['state'], bool(r.get('is_special', False))))

    all_polls, coverage = [], []
    for office, state, special in races:
        url = race_url(office, state, special)
        try:
            polls, nominees = parse_race_page(_fetch_html(url), office, state, url)
            err = None
        except Exception as e:
            polls, nominees, err = pd.DataFrame(), None, str(e)[:120]
        polls = _dedupe_versions(polls)
        all_polls.append(polls)
        coverage.append({
            'office': office, 'state': state, 'special': special, 'n_polls': len(polls),
            'dem_candidate': polls['dem_candidate'].iloc[0] if len(polls) else None,
            'rep_candidate': polls['rep_candidate'].iloc[0] if len(polls) else None,
            'nominees_from_aggregate_table': nominees is not None,
            'latest_poll': polls['end_date'].max() if len(polls) else None,
            'error': err, 'url': url,
        })
        time.sleep(pause)  # be polite to Wikipedia

    polls = pd.concat([p for p in all_polls if len(p)], ignore_index=True)
    polls['scraped_at'] = datetime.now().isoformat(timespec='seconds')
    return polls.sort_values(['office', 'state', 'end_date']), pd.DataFrame(coverage)

def refresh_polls():
    polls, coverage = scrape_all()
    os.makedirs(POLLS_DIR, exist_ok=True)
    polls.to_csv(POLLS_CSV, index=False)
    coverage.to_csv(COVERAGE_CSV, index=False)
    return polls, coverage

def load_polls():
    return pd.read_csv(POLLS_CSV, parse_dates=['end_date'])

if __name__ == '__main__':
    polls, coverage = refresh_polls()
    pd.set_option('display.width', 220)
    print(coverage.drop(columns=['url']).to_string())
    print(f"\n{len(polls)} polls across {(coverage['n_polls'] > 0).sum()} of {len(coverage)} races.")
