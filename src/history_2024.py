"""
2024 polls and results, added to the historical record (FiveThirtyEight's archive ends in
2022, so 2024's polling miss wasn't in our ratings, corrections, or backtests).

Built from Wikipedia's 2024 race pages, which keep each race's poll table and certified
results with citations to official sources:
  - every Senate and governor race: general-election polls from the final 60 days
    (the archive's window), matched to the two leading candidates, plus the result
  - national presidential polls vs. the national popular vote: a stand-in for the 2024
    generic ballot, since no list of individual 2024 generic-ballot polls is available

Saved in the archive's own column format (data/polling_data/history_2024.csv), so
pollster_ratings.load_archive() picks it up and every rating, correction and backtest
includes 2024.

    python src/history_2024.py
"""
import hashlib
import io
import os
import re
import time

import numpy as np
import pandas as pd

from race_status_2026 import _fetch_html
from polls_2026 import _flatten, _sanitize, _pct, _sample, _end_date, parse_race_page, _dedupe_versions
from pollster_ratings import load_ratings, match_pollster, sponsor_party
from names import _name_score
from states import STATE_ABBREV

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(BASE_DIR, 'data', 'polling_data', 'history_2024.csv')
WIKI = "https://en.wikipedia.org/wiki/"
ELECTION_DAY = pd.Timestamp('2024-11-05')
WINDOW_DAYS = 60
GOVERNORS_2024 = ['DELAWARE', 'INDIANA', 'MISSOURI', 'MONTANA', 'NEW HAMPSHIRE', 'NORTH CAROLINA',
                  'NORTH DAKOTA', 'UTAH', 'VERMONT', 'WASHINGTON', 'WEST VIRGINIA']
PRES_URL = f"{WIKI}Nationwide_opinion_polling_for_the_2024_United_States_presidential_election"

def _url(office, state):
    name = state.title().replace(' ', '_')
    return (f"{WIKI}2024_United_States_Senate_election_in_{name}" if office == 'senate'
            else f"{WIKI}2024_{name}_gubernatorial_election")

def general_result(html):
    """(dem_side_name, dem_side_party, dem_pct, rep_name, rep_pct) from the race's general-
    election results table: the results table with the most votes that has a Republican
    and a non-Republican candidate."""
    best, best_total = None, -1
    for t in pd.read_html(io.StringIO(_sanitize(html))):
        t = _flatten(t)
        cand = next((c for c in t.columns if c.endswith('Candidate')), None)
        party = next((c for c in t.columns if c.startswith('Party') and c != 'Party'), None) or \
                next((c for c in t.columns if c.startswith('Party')), None)
        votes = next((c for c in t.columns if c == 'Votes' or c.endswith(' Votes')), None)
        pct = next((c for c in t.columns if c == '%' or c.endswith(' %')), None)
        if not (cand and party and votes and pct):
            continue
        rows = []
        for _, r in t.iterrows():
            v = pd.to_numeric(str(r[votes]).replace(',', ''), errors='coerce')
            p = _pct(r[pct])
            if pd.notna(v) and pd.notna(p) and isinstance(r[cand], str) and 'total' not in str(r[cand]).lower():
                rows.append((re.sub(r'\s*\(.*?\)', '', r[cand]).strip(), str(r[party]), v, p))
        reps = [x for x in rows if 'Republican' in x[1]]
        others = [x for x in rows if 'Republican' not in x[1] and 'Write' not in x[0]]
        total = sum(x[2] for x in rows)
        if reps and others and total > best_total:
            rep = max(reps, key=lambda x: x[2])
            dem = max(others, key=lambda x: x[2])
            dparty = 'DEM' if ('Democratic' in dem[1] or 'DFL' in dem[1]) else 'IND'
            best, best_total = (dem[0], dparty, dem[3], rep[0], rep[3]), total
    return best

def _pollster_id(name, ratings):
    m = match_pollster(name, ratings)
    if m is not None:
        return int(m['pollster_rating_id']), bool(m.get('aapor_roper', False))
    clean = re.sub(r'\[[^\]]*\]|\((D|R|I)\)', '', name).strip().lower()
    return 900000 + int(hashlib.md5(clean.encode()).hexdigest()[:6], 16) % 99999, False

def _archive_rows(polls, office, state_po, result, ratings, start_id):
    dem_name, dparty, d_act, rep_name, r_act = result
    rows = []
    type_simple = {'senate': 'Sen-G', 'governor': 'Gov-G', 'president': 'Pres-G'}[office]
    for k, (_, p) in enumerate(polls.iterrows()):
        pid, aapor = _pollster_id(p['pollster'], ratings)
        sp = sponsor_party(p['pollster'])
        rows.append({
            'poll_id': start_id + k, 'question_id': start_id + k,
            'race_id': None, 'cycle': 2024, 'location': state_po, 'type_simple': type_simple,
            'race': f"2024_{type_simple}_{state_po}",
            'pollster': re.sub(r'\[[^\]]*\]|\s*\((D|R|I)\)', '', p['pollster']).strip(),
            'pollster_rating_id': pid, 'aapor_roper': aapor,
            'partisan': {'DEMOCRAT': 'DEM', 'REPUBLICAN': 'REP'}.get(sp),
            'polldate': p['end_date'].strftime('%Y-%m-%d'), 'electiondate': ELECTION_DAY.strftime('%Y-%m-%d'),
            'time_to_election': int((ELECTION_DAY - p['end_date']).days),
            'samplesize': p['sample_size'],
            'cand1_name': dem_name, 'cand1_party': dparty, 'cand1_pct': p['dem_pct'], 'cand1_actual': d_act,
            'cand2_name': rep_name, 'cand2_party': 'REP', 'cand2_pct': p['rep_pct'], 'cand2_actual': r_act,
            'margin_poll': p['dem_pct'] - p['rep_pct'], 'margin_actual': d_act - r_act,
        })
    return rows

def scrape_races(pause=1.0):
    from midterms import senate_races_table
    ratings = load_ratings()
    races = senate_races_table()
    senate_states = sorted(races.loc[races['year'] == 2024, 'state'].unique())
    out, log, next_id = [], [], 9_000_000
    for office, states in (('senate', senate_states), ('governor', GOVERNORS_2024)):
        for state in states:
            url = _url(office, state)
            try:
                html = _fetch_html(url)
                result = general_result(html)
                if result is None:
                    raise ValueError("no general-election results table")
                polls, _ = parse_race_page(html, office, state, url)
                polls = _dedupe_versions(polls)
                if len(polls):
                    polls = polls[polls.apply(lambda p: _name_score(p['dem_candidate'], result[0]) >= 0.8
                                              and _name_score(p['rep_candidate'], result[3]) >= 0.8, axis=1)]
                    polls = polls[(polls['end_date'] <= ELECTION_DAY) &
                                  (polls['end_date'] >= ELECTION_DAY - pd.Timedelta(days=WINDOW_DAYS))]
                rows = _archive_rows(polls, office, STATE_ABBREV[state], result, ratings, next_id)
                next_id += 1000
                out.extend(rows)
                log.append({'office': office, 'state': state, 'polls': len(rows), 'dem': result[0],
                            'rep': result[3], 'actual_margin': round(result[2] - result[4], 2), 'error': None})
            except Exception as e:
                log.append({'office': office, 'state': state, 'polls': 0, 'error': str(e)[:100]})
            time.sleep(pause)
    return pd.DataFrame(out), pd.DataFrame(log)

def scrape_national_presidential(ratings=None):
    """2024 national Harris-vs-Trump polls in the final 60 days, vs the popular vote."""
    ratings = ratings if ratings is not None else load_ratings()
    pres = pd.read_csv(os.path.join(BASE_DIR, 'data', 'polling_data', '2024president.csv'))
    pres['party'] = pres['party_simplified'].str.upper()
    pres = pres[pres['mode'].astype(str).str.upper() == 'TOTAL'] if 'mode' in pres else pres
    votes = pres.groupby('party')['votes'].sum()
    total = pres.drop_duplicates('state')['totalvotes'].sum()
    d_act, r_act = votes['DEMOCRAT'] / total * 100, votes['REPUBLICAN'] / total * 100
    html = _fetch_html(PRES_URL)
    rows = []
    for t in pd.read_html(io.StringIO(_sanitize(html))):
        t = _flatten(t)
        src = next((c for c in t.columns if c.startswith('Poll source')), None)
        date = next((c for c in t.columns if c.startswith('Date')), None)
        harris = next((c for c in t.columns if 'Harris' in c), None)
        trump = next((c for c in t.columns if 'Trump' in c), None)
        size = next((c for c in t.columns if c.startswith('Sample')), None)
        if not (src and date and harris and trump):
            continue
        for _, r in t.iterrows():
            d, rep = _pct(r[harris]), _pct(r[trump])
            end = _end_date(r[date])
            if pd.isna(d) or pd.isna(rep) or pd.isna(end) or pd.isna(r[src]):
                continue
            n, pop = _sample(r[size]) if size else (np.nan, None)
            rows.append({'pollster': str(r[src]), 'end_date': end, 'dem_pct': d, 'rep_pct': rep,
                         'sample_size': n, 'population': pop})
    polls = pd.DataFrame(rows)
    polls = polls[(polls['end_date'] <= ELECTION_DAY) & (polls['end_date'] >= ELECTION_DAY - pd.Timedelta(days=WINDOW_DAYS))]
    polls = polls.assign(_lv=(polls['population'] != 'LV').astype(int)).sort_values('_lv') \
                 .drop_duplicates(['pollster', 'end_date']).drop(columns='_lv')
    return _archive_rows(polls, 'president', 'US', ('Kamala Harris', 'DEM', d_act, 'Donald Trump', r_act),
                         ratings, 9_900_000)

def build():
    races, log = scrape_races()
    races.to_csv(OUT_CSV, index=False)          # saved first so a national-poll failure keeps the races
    national = pd.DataFrame(scrape_national_presidential())
    df = pd.concat([races, national], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    return df, log

if __name__ == '__main__':
    pd.set_option('display.width', 200)
    df, log = build()
    print(log.to_string(index=False))
    print(f"\n{len(df)} polls saved ({(df['type_simple'] == 'Pres-G').sum()} national presidential) -> {OUT_CSV}")
