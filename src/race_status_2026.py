"""
Scrapes real 2026 Senate race status (open seats, expert consensus ratings) from
Wikipedia's "2026 United States Senate elections" Predictions table.

Why Wikipedia and not Ballotpedia: Ballotpedia is blocked by an AWS WAF bot-challenge
(confirmed via direct request — returns 202/challenge, no content, no way around it
without solving a JS challenge). Wikipedia's equivalent page is plain HTTP, no bot-gate,
and its Predictions table already aggregates 12 professional forecasters (Cook, Inside
Elections, Sabato, etc.) with consistent "Safe/Likely/Lean/Tilt/Tossup" ratings.

Why not per-state raw polling pages: individual "20XX United States Senate election in
X" pages have their own opinion-polling tables, but each is structured differently
(mixed primary/general polls, inconsistent columns) — scraping ~15 of them reliably is
much more fragile than this one well-structured summary table.
"""
import os
import re
import io
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/polling_data')
STATUS_CSV = os.path.join(DATA_DIR, 'senate_race_status_2026.csv')

SENATE_WIKI_URL = "https://en.wikipedia.org/wiki/2026_United_States_Senate_elections"

# Wikipedia's bot policy blocks pandas'/urllib's default generic User-Agent (403);
# needs a descriptive one, per https://meta.wikimedia.org/wiki/User-Agent_policy.
_USER_AGENT = "ElectionPredictionApp/1.0 (personal project; contact: keithpotz@gmail.com)"

def _fetch_html(url):
    r = requests.get(url, headers={'User-Agent': _USER_AGENT}, timeout=20)
    r.raise_for_status()
    r.encoding = 'utf-8'
    return r.text

RATING_SCALE = {
    'SAFE': 30, 'SOLID': 30,
    'LIKELY': 15,
    'LEAN': 7,
    'TILT': 3,
    'TOSSUP': 0, 'TOSS-UP': 0,
}

def _rating_to_number(text):
    """'Lean D (flip)' -> +7, 'Solid R' -> -30, 'Tossup' -> 0, unparseable -> NaN."""
    if not isinstance(text, str):
        return float('nan')
    text = text.upper().replace('(FLIP)', '').strip()
    m = re.match(r'([A-Z-]+)\s*(D|R)?$', text)
    if not m:
        return float('nan')
    label, party = m.group(1), m.group(2)
    magnitude = RATING_SCALE.get(label)
    if magnitude is None:
        return float('nan')
    if magnitude == 0 or party is None:
        return 0.0
    return magnitude if party == 'D' else -magnitude

def _parse_incumbent_status(text):
    """'Mitch McConnell (retiring)' -> ('Mitch McConnell', True); 'Tom Cotton' -> ('Tom Cotton', False)."""
    text = str(text).strip()
    m = re.match(r'^(.*?)\s*\((.+)\)\s*$', text)
    if not m:
        return text, False
    name, status = m.group(1).strip(), m.group(2).lower()
    open_seat = 'retir' in status or 'renomination' in status or 'running for' in status
    return name, open_seat

def _find_predictions_table(tables):
    for t in tables:
        cols = [' '.join(str(x) for x in c) if isinstance(c, tuple) else str(c) for c in t.columns]
        joined = ' '.join(cols).lower()
        if 'cook' in joined and 'state' in joined:
            return t
    raise ValueError("Could not find the Senate Predictions table on the Wikipedia page — its layout may have changed.")

def scrape_senate_race_status(source=SENATE_WIKI_URL):
    """
    Fetches and parses the Predictions table. Returns a DataFrame with columns:
    state (uppercased, matching the rest of this codebase's convention), incumbent,
    open_seat (bool), consensus_rating (numeric, Dem-Rep margin-style: positive=D lean).
    """
    html = _fetch_html(source) if source.startswith('http') else source
    tables = pd.read_html(io.StringIO(html))
    table = _find_predictions_table(tables)
    table.columns = [
        '_'.join(str(x) for x in c if 'Unnamed' not in str(x)) if isinstance(c, tuple) else str(c)
        for c in table.columns
    ]

    state_col     = next(c for c in table.columns if c.endswith('_State') or c == 'State')
    incumbent_col = next(c for c in table.columns if 'Senator' in c)
    rating_cols   = [c for c in table.columns if c.startswith('Ratings_')]

    rows = []
    for _, r in table.iterrows():
        state = str(r[state_col]).strip()
        if state.lower().startswith('overall') or not state:
            continue
        name, open_seat = _parse_incumbent_status(r[incumbent_col])
        ratings = [_rating_to_number(r[c]) for c in rating_cols]
        ratings = [v for v in ratings if pd.notna(v)]
        consensus = round(sum(ratings) / len(ratings), 2) if ratings else float('nan')
        rows.append({
            'state':            re.sub(r'\s*\(special\)', '', state).upper(),
            'is_special':       '(special)' in state.lower(),
            'incumbent':        name,
            'open_seat':        open_seat,
            'consensus_rating': consensus,
        })
    df = pd.DataFrame(rows)

    # Canary: if Wikipedia's table structure silently changed (column renamed, rating
    # phrasing changed), _rating_to_number/_parse_incumbent_status degrade to NaN/False
    # for every row rather than raising — that would look like valid "no rating" data
    # rather than a broken scrape. Catch it here instead.
    non_null_consensus = df['consensus_rating'].notna().mean() if len(df) else 0
    if non_null_consensus < 0.5:
        raise ValueError(
            f"Only {non_null_consensus:.0%} of scraped Senate races have a parseable "
            "consensus rating — Wikipedia's table structure may have changed. Check "
            "_find_predictions_table()/_rating_to_number() against the current page."
        )
    return df

def load_or_scrape_race_status(refresh=False):
    """Reads the cached CSV; scrapes and caches it if missing or refresh=True."""
    if not refresh and os.path.exists(STATUS_CSV):
        return pd.read_csv(STATUS_CSV)
    df = scrape_senate_race_status()
    df.to_csv(STATUS_CSV, index=False)
    return df

if __name__ == '__main__':
    df = load_or_scrape_race_status(refresh=True)
    print(df.to_string())
