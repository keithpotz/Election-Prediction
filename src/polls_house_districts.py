"""
2026 House district polls from Wikipedia's state-level House election pages
("2026 United States House of Representatives elections in <State>").

Each state page has a section per district (headings like "District 7" or "At-large").
Poll tables are assigned to the district whose heading most recently precedes them,
then matched to that district's general-election nominees from house_districts.py.
Tables for primaries or hypothetical matchups don't contain both nominees and are
skipped automatically.

    python src/polls_house_districts.py
"""
import io
import os
import re
import time

import pandas as pd
from bs4 import BeautifulSoup

from race_status_2026 import _fetch_html
from polls_2026 import _flatten, _sanitize, _party_columns, _pct, _sample, _end_date, _cand_name, PARTY_TAG
from names import _name_score
from pollster_ratings import sponsor_party

WIKI = "https://en.wikipedia.org/wiki/"
RATINGS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'polls_2026', 'house_ratings_2026.csv')

def state_urls(state):
    """Multi-district states use '...elections in X'; at-large states '...election in X'."""
    name = state.title().replace(' ', '_')
    return [f"{WIKI}2026_United_States_House_of_Representatives_elections_in_{name}",
            f"{WIKI}2026_United_States_House_of_Representatives_election_in_{name}"]

def _district_of(heading_text):
    t = heading_text.strip().lower()
    if t.startswith('at-large') or t.startswith('at large'):
        return 0
    m = re.match(r'district\s+(\d+)', t)
    return int(m.group(1)) if m else None

def main_articles(html):
    """{district: url} for districts whose section points to a dedicated article
    ("Main article: 2026 New York's 13th congressional district election"); their
    polls usually live on that page rather than the state page."""
    soup = BeautifulSoup(_sanitize(html), 'lxml')
    content = soup.find('div', class_='mw-parser-output') or soup
    district, out = None, {}
    for el in content.find_all(['h2', 'h3', 'div']):
        if el.name in ('h2', 'h3'):
            d = _district_of(el.get_text())
            if d is not None or el.name == 'h2':
                district = d
        elif district is not None and 'hatnote' in (el.get('class') or []) and 'Main article' in el.get_text():
            a = el.find('a', href=True)
            if a and a['href'].startswith('/wiki/') and 'congressional_district_election' in a['href']:
                out.setdefault(district, 'https://en.wikipedia.org' + a['href'])
    return out

def tables_by_district(html, only_district=None):
    """[(district, DataFrame)] for every poll table, in page order. With only_district,
    every poll table on the page is assigned to that district (dedicated district pages)."""
    soup = BeautifulSoup(_sanitize(html), 'lxml')
    content = soup.find('div', class_='mw-parser-output') or soup
    district, out = only_district, []
    for el in content.find_all(['h2', 'h3', 'table']):
        if only_district is not None and el.name != 'table':
            continue
        if el.name in ('h2', 'h3'):
            d = _district_of(el.get_text())
            # Big states nest districts under regional h2 sections; an unrelated h2
            # (e.g. "References") ends the current district, an h3 subsection doesn't.
            if d is not None or el.name == 'h2':
                district = d
            continue
        if district is None or 'wikitable' not in (el.get('class') or []):
            continue
        try:
            t = _flatten(pd.read_html(io.StringIO(str(el)))[0])
        except ValueError:
            continue
        if any(c.startswith('Poll source') for c in t.columns):
            out.append((district, t))
    return out

def ratings_by_district(html, state):
    """Per-district forecaster ratings ('Source | Ranking | As of' tables under each
    district heading), excluding market-tied forecasters."""
    from independence import is_market_tied
    from race_status_2026 import _rating_to_number
    soup = BeautifulSoup(_sanitize(html), 'lxml')
    content = soup.find('div', class_='mw-parser-output') or soup
    district, rows = None, []
    for el in content.find_all(['h2', 'h3', 'table']):
        if el.name in ('h2', 'h3'):
            d = _district_of(el.get_text())
            if d is not None or el.name == 'h2':
                district = d
            continue
        if district is None or 'wikitable' not in (el.get('class') or []):
            continue
        try:
            t = _flatten(pd.read_html(io.StringIO(str(el)))[0])
        except ValueError:
            continue
        if not ({'Source', 'Ranking'} <= set(t.columns)):
            continue
        for _, r in t.iterrows():
            src = re.sub(r'\[[^\]]*\]', '', str(r['Source'])).strip()
            rows.append({'state': state, 'district': district, 'source': src, 'ranking': r['Ranking'],
                         'value': _rating_to_number(r['Ranking']), 'market_tied': is_market_tied(src)})
    return rows

def parse_state(html, state, nominees, url, only_district=None):
    """nominees: {district: (dem_name, rep_name)} for this state."""
    rows = []
    for district, t in tables_by_district(html, only_district):
        if district not in nominees:
            continue
        d_name, r_name = nominees[district]
        cols = _party_columns(t.columns)
        d_col = next((c for c, p in cols.items() if p in ('D', 'DFL') and _name_score(_cand_name(c), d_name) >= 0.8), None)
        r_col = next((c for c, p in cols.items() if p == 'R' and _name_score(_cand_name(c), r_name) >= 0.8), None)
        if d_col is None or r_col is None:
            continue
        src = next(c for c in t.columns if c.startswith('Poll source'))
        date_col = next((c for c in t.columns if c.startswith('Date')), None)
        size_col = next((c for c in t.columns if c.startswith('Sample size')), None)
        for _, r in t.iterrows():
            d, rep = _pct(r[d_col]), _pct(r[r_col])
            if pd.isna(r[src]) or pd.isna(d) or pd.isna(rep):
                continue
            n, pop = _sample(r[size_col]) if size_col else (None, None)
            pollster = re.sub(r'\[[^\]]*\]', '', str(r[src])).strip()
            rows.append({
                'office': 'house', 'state': state, 'district': district, 'pollster': pollster,
                'end_date': _end_date(r[date_col]) if date_col else pd.NaT,
                'sample_size': n, 'population': pop,
                'dem_candidate': PARTY_TAG.sub('', d_col).strip(), 'rep_candidate': PARTY_TAG.sub('', r_col).strip(),
                'dem_side_party': 'DEMOCRAT', 'dem_pct': d, 'rep_pct': rep, 'margin': d - rep,
                'sponsor': sponsor_party(pollster), 'source': 'wikipedia', 'source_url': url,
            })
    return pd.DataFrame(rows)

def scrape_district_polls(districts, pause=1.0, retries=2):
    """
    districts: house_districts.load_house_districts() output. Only states with at least
    one contested district are fetched. Returns (polls, errors) where errors lists states
    whose page couldn't be loaded after retries.
    """
    contested = districts[districts['has_dem'].astype(bool) & districts['has_rep'].astype(bool)]
    frames, errors, ratings = [], [], []
    for state, g in contested.groupby('state'):
        nominees = {int(r['district']): (r['dem_candidate'], r['rep_candidate']) for _, r in g.iterrows()
                    if isinstance(r['dem_candidate'], str) and isinstance(r['rep_candidate'], str)}
        last_err = None
        for url in state_urls(state):
            for attempt in range(retries + 1):
                try:
                    html = _fetch_html(url)
                    frames.append(parse_state(html, state, nominees, url))
                    ratings.extend(ratings_by_district(html, state))
                    for district, sub_url in main_articles(html).items():
                        if district in nominees:
                            time.sleep(pause)
                            try:
                                frames.append(parse_state(_fetch_html(sub_url), state, nominees, sub_url, district))
                            except Exception as e:
                                errors.append({'office': 'house', 'state': state, 'error': str(e)[:150], 'url': sub_url})
                    last_err = None
                    break
                except Exception as e:
                    last_err = (url, str(e)[:150])
                    if '404' in str(e):
                        break                       # wrong page name; try the next URL form
                    time.sleep(pause * 3)
            if last_err is None:
                break
        if last_err:
            errors.append({'office': 'house', 'state': state, 'error': last_err[1], 'url': last_err[0]})
        time.sleep(pause)
    polls = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()
    r = pd.DataFrame(ratings)
    if len(r):
        os.makedirs(os.path.dirname(RATINGS_CSV), exist_ok=True)
        r.to_csv(RATINGS_CSV, index=False)
    return polls, pd.DataFrame(errors)

def load_house_ratings():
    """(state, district) -> consensus rating (Dem-positive scale), non-market-tied sources only."""
    if not os.path.exists(RATINGS_CSV):
        return pd.DataFrame(columns=['state', 'district', 'consensus_rating', 'n_raters'])
    r = pd.read_csv(RATINGS_CSV)
    r = r[~r['market_tied'] & r['value'].notna()]
    return (r.groupby(['state', 'district'])['value'].agg(consensus_rating='mean', n_raters='size')
             .round(2).reset_index())

if __name__ == '__main__':
    from house_districts import load_house_districts
    polls, errors = scrape_district_polls(load_house_districts())
    print(f"{len(polls)} district polls across {polls.groupby(['state', 'district']).ngroups if len(polls) else 0} districts")
    print("errors:", errors.to_dict('records'))
    print(polls.groupby(['state', 'district']).size().sort_values(ascending=False).head(15).to_string())
