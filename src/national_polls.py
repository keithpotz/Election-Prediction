"""
National inputs without prediction-market-affiliated sources: the generic
congressional ballot and presidential approval.

Independence policy: no data comes from organizations with a business tie to a
prediction/betting market. As checked 2026-10-06:
  Decision Desk HQ   runs live Polymarket and Kalshi odds displays in its election coverage
  RealClearPolitics  added Polymarket odds to its 2026 forecast maps (July 2026)
  Silver Bulletin    Nate Silver is a Polymarket adviser (since 2024)
  VoteHub            data partnership with Kalshi (Midterms Hub, July 2026)
Re-check these before relying on any new source; add names to MARKET_TIED.

Generic ballot: Wikipedia carries only aggregators' averages for 2026 (no table of
individual polls), so this averages the aggregators in Wikipedia's table that are NOT
market-tied (currently FiftyPlusOne and NBC News).

Approval: Wikipedia lists the individual approval polls by month, so the average is
computed here (recency and sample-size weighted, like the race averages). Wikipedia's
page skips many months (Aug 2025 - Aug 2026 is nearly empty), so the UC Santa Barbara
American Presidency Project's poll table (academic, no market ties) fills in, minus any
poll Wikipedia already lists.
"""
import io
import json
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from race_status_2026 import _fetch_html
from polls_2026 import _flatten, _sanitize, _pct, _sample, POLLS_DIR

HOUSE_URL = "https://en.wikipedia.org/wiki/2026_United_States_House_of_Representatives_elections"
APPROVAL_URL = "https://en.wikipedia.org/wiki/Opinion_polling_on_the_second_Donald_Trump_administration"
UCSB_APPROVAL_URL = "https://www.presidency.ucsb.edu/statistics/data/donald-j-trump-2nd-term-public-approval"
CACHE_JSON = os.path.join(POLLS_DIR, 'national_polls.json')
APPROVAL_CSV = os.path.join(POLLS_DIR, 'approval_polls.csv')
# One row per aggregator per refresh day. Wikipedia keeps only today's generic-ballot
# averages, so this log is the only record of how they move.
GENERIC_BALLOT_LOG = os.path.join(POLLS_DIR, 'generic_ballot_log.csv')
MAX_CACHE_AGE_HOURS = 24
APPROVAL_WINDOW_DAYS = 30
APPROVAL_HALF_LIFE_DAYS = 10

def _clean(name):
    return re.sub(r'\[[^\]]*\]', '', str(name)).strip()

from independence import MARKET_TIED, is_market_tied as _is_market_tied  # noqa: E402,F401

def scrape_generic_ballot():
    tables = [_flatten(t) for t in pd.read_html(io.StringIO(_sanitize(_fetch_html(HOUSE_URL))))]
    t = next(t for t in tables if any(c.startswith('Source of poll aggregation') for c in t.columns))
    src = next(c for c in t.columns if c.startswith('Source of poll aggregation'))
    dem = next(c for c in t.columns if c.startswith('Democrat'))
    rep = next(c for c in t.columns if c.startswith('Republican'))
    rows = []
    for _, r in t.iterrows():
        name = _clean(r[src])
        d, rp = _pct(r[dem]), _pct(r[rep])
        if not name or name.lower() in ('average', 'nan') or pd.isna(d) or pd.isna(rp):
            continue
        rows.append({'aggregator': name, 'dem': d, 'rep': rp, 'margin': round(d - rp, 2),
                     'market_tied': _is_market_tied(name)})
    agg = pd.DataFrame(rows)
    clean = agg[~agg['market_tied']]
    if clean.empty:
        raise ValueError("No non-market-tied generic ballot aggregators found on Wikipedia")
    return {'margin': round(float(clean['margin'].mean()), 2), 'aggregators': agg.to_dict('records'),
            'used': clean['aggregator'].tolist(), 'excluded': agg[agg['market_tied']]['aggregator'].tolist()}

def _approval_date(text, year, month):
    """Wikipedia's approval tables omit the year ('Sep 29 – Oct 2'); the month section
    heading supplies it. Ranges are resolved to their end date."""
    s = re.sub(r'\[[^\]]*\]', '', str(text)).strip()
    end = re.split(r'\s*[–—−-]\s*', s)[-1]
    if re.fullmatch(r'\d+', end):
        end = f"{s.split()[0]} {end}"
    d = pd.to_datetime(f"{end} {year}", errors='coerce')
    if pd.notna(d) and d.month == 12 and month == 1:      # Dec 30 – Jan 2 crossing a year boundary
        d = d.replace(year=year - 1)
    return d

def scrape_approval_polls():
    soup = BeautifulSoup(_sanitize(_fetch_html(APPROVAL_URL)), 'lxml')
    content = soup.find('div', class_='mw-parser-output') or soup
    rows, ym, in_national = [], None, False
    for el in content.find_all(['h2', 'h4', 'table']):
        text = el.get_text().strip()
        if el.name == 'h2':
            in_national = text.lower().startswith('nationwide job approval')
            ym = None
            continue
        if el.name == 'h4':
            m = re.match(r'([A-Za-z]+) (\d{4})$', text)
            ym = (int(m.group(2)), pd.to_datetime(m.group(1), format='%B').month) if m else None
            continue
        if not in_national or ym is None or 'wikitable' not in (el.get('class') or []):
            continue
        try:
            t = _flatten(pd.read_html(io.StringIO(str(el)))[0])
        except ValueError:
            continue
        src = next((c for c in t.columns if 'Poll source' in c), None)
        date = next((c for c in t.columns if c.endswith('Date')), None)
        size = next((c for c in t.columns if 'Sample' in c), None)
        app = next((c for c in t.columns if c.endswith('Approve')), None)
        dis = next((c for c in t.columns if c.endswith('Disapprove')), None)
        if not (src and date and app):
            continue
        for _, r in t.iterrows():
            a = _pct(r[app])
            if pd.isna(a) or pd.isna(r[src]):
                continue
            n, pop = _sample(r[size]) if size else (np.nan, None)
            rows.append({'pollster': _clean(r[src]), 'end_date': _approval_date(r[date], *ym),
                         'sample_size': n, 'population': pop, 'approve': a,
                         'disapprove': _pct(r[dis]) if dis else np.nan})
    df = pd.DataFrame(rows).dropna(subset=['end_date'])
    return df.drop_duplicates(['pollster', 'end_date', 'approve']).sort_values('end_date')

UCSB_NAMES = [   # (keyword in UCSB's lowercase source label, pollster name), first match wins
    ('gallup', 'Gallup'), ('norc', 'AP-NORC'), ('ssrs', 'CNN/SSRS'), ('marist', 'Marist'),
    ('reuters', 'Reuters/Ipsos'), ('wapo', 'Washington Post/Ipsos'), ('ipso', 'Ipsos'),
    ('pew', 'Pew Research Center'), ('verasight', 'Verasight'), ('focaldata', 'Focaldata'),
    ('economist', 'The Economist/YouGov'), ('cbs', 'CBS News/YouGov'), ('marquette', 'Marquette Law School'),
    ('quinnipiac', 'Quinnipiac University'), ('fox', 'Fox News'), ('nbc', 'NBC News'),
]

def _ucsb_pollster(label):
    low = str(label).lower()
    return next((name for key, name in UCSB_NAMES if key in low), str(label).strip().title())

def scrape_ucsb_approval():
    """The American Presidency Project's table of Trump approval polls (end date, approve,
    disapprove, source). Rows whose approve + disapprove passes 102 are entry errors and
    are skipped."""
    t = pd.read_html(io.StringIO(_fetch_html(UCSB_APPROVAL_URL)))[0]
    t.columns = [str(c).strip().lower() for c in t.columns]
    df = pd.DataFrame({
        'pollster': t['source'].map(_ucsb_pollster),
        'end_date': pd.to_datetime(t['end date'], format='%m/%d/%Y', errors='coerce'),
        'sample_size': np.nan, 'population': None,
        'approve': pd.to_numeric(t['approving'], errors='coerce'),
        'disapprove': pd.to_numeric(t['disapproving'], errors='coerce'),
    }).dropna(subset=['end_date', 'approve'])
    return df[df['approve'] + df['disapprove'].fillna(0) <= 102]

IGNORED_NAME_WORDS = {'news', 'the', 'poll', 'law', 'school', 'research', 'center', 'university'}

def _same_poll(a, b):
    """Same pollster (a name word in common) and end dates within 3 days."""
    words = lambda n: {w for w in re.findall(r'[a-z]+', str(n).lower()) if len(w) > 2} - IGNORED_NAME_WORDS
    return abs((a['end_date'] - b['end_date']).days) <= 3 and bool(words(a['pollster']) & words(b['pollster']))

def combine_approval(wiki, ucsb):
    """Wikipedia's polls plus the UCSB polls Wikipedia doesn't have."""
    wiki = wiki.assign(source='wikipedia')
    if ucsb is None or ucsb.empty:
        return wiki
    extra = [u for _, u in ucsb.iterrows()
             if not any(_same_poll(u, w) for _, w in
                        wiki[(wiki['end_date'] - u['end_date']).abs() <= pd.Timedelta(days=3)].iterrows())]
    if not extra:
        return wiki
    out = pd.concat([wiki, pd.DataFrame(extra).assign(source='ucsb')], ignore_index=True)
    return out.sort_values('end_date').reset_index(drop=True)

def approval_average(polls, as_of=None, col='approve'):
    """Recency/sample-weighted average of the last 30 days, one pollster's polls
    sharing one pollster's worth of weight (same idea as the race averages).
    col='disapprove' averages disapproval the same way; the result keeps the 'approve' key."""
    as_of = pd.Timestamp(as_of or polls['end_date'].max())
    p = polls.dropna(subset=[col])
    p = p[(p['end_date'] <= as_of) & (p['end_date'] > as_of - pd.Timedelta(days=APPROVAL_WINDOW_DAYS))]
    if p.empty:
        raise ValueError("No approval polls in the averaging window")
    age = (as_of - p['end_date']).dt.days.clip(lower=0)
    w = (0.5 ** (age / APPROVAL_HALF_LIFE_DAYS)) * np.sqrt(p['sample_size'].fillna(1000).clip(300, 5000) / 1000)
    w = w / np.sqrt(p.groupby('pollster')['pollster'].transform('size'))
    return {'approve': round(float(np.average(p[col], weights=w)), 1), 'n_polls': int(len(p)),
            'latest_poll': p['end_date'].max().date().isoformat()}

def refresh():
    gb = scrape_generic_ballot()
    approval_polls = scrape_approval_polls()
    try:
        approval_polls = combine_approval(approval_polls, scrape_ucsb_approval())
    except Exception as e:
        print(f"UCSB approval table unavailable, using Wikipedia only: {e}")
        approval_polls = approval_polls.assign(source='wikipedia')
    os.makedirs(POLLS_DIR, exist_ok=True)
    approval_polls.to_csv(APPROVAL_CSV, index=False)
    data = {'fetched_at': datetime.now().isoformat(timespec='seconds'), 'generic_ballot': gb,
            'approval': approval_average(approval_polls)}
    with open(CACHE_JSON, 'w') as f:
        json.dump(data, f, indent=2)
    log_generic_ballot(data)
    return data

def log_generic_ballot(data):
    """Append today's aggregator values to the log, replacing any earlier rows from today."""
    day = data['fetched_at'][:10]
    gb = data['generic_ballot']
    rows = pd.DataFrame([{'date': day, 'aggregator': a['aggregator'], 'dem': a.get('dem'), 'rep': a.get('rep'),
                          'margin': a['margin'], 'market_tied': a['market_tied']} for a in gb['aggregators']]
                        + [{'date': day, 'aggregator': 'Our average', 'margin': gb['margin'], 'market_tied': False}])
    if os.path.exists(GENERIC_BALLOT_LOG):
        old = pd.read_csv(GENERIC_BALLOT_LOG)
        rows = pd.concat([old[old['date'] != day], rows], ignore_index=True)
    rows.to_csv(GENERIC_BALLOT_LOG, index=False)

def load_generic_ballot_log():
    return pd.read_csv(GENERIC_BALLOT_LOG, parse_dates=['date']) if os.path.exists(GENERIC_BALLOT_LOG) else pd.DataFrame()

def load(refresh_if_stale=True):
    """Cached national inputs; re-scraped when older than a day, stale cache kept on failure."""
    exists = os.path.exists(CACHE_JSON)
    fresh = exists and (time.time() - os.path.getmtime(CACHE_JSON)) < MAX_CACHE_AGE_HOURS * 3600
    if refresh_if_stale and not fresh:
        try:
            return refresh()
        except Exception as e:
            if not exists:
                raise
            print(f"National polls refresh failed, using cached copy: {e}")
    with open(CACHE_JSON) as f:
        return json.load(f)

def load_approval_polls():
    return pd.read_csv(APPROVAL_CSV, parse_dates=['end_date']) if os.path.exists(APPROVAL_CSV) else pd.DataFrame()

def fetch_generic_ballot():
    """(Dem-minus-Rep margin, source note) or (None, None)."""
    try:
        gb = load()['generic_ballot']
        return gb['margin'], f"Average of {', '.join(gb['used'])} (market-tied aggregators excluded)"
    except Exception as e:
        print(f"Generic ballot unavailable: {e}")
        return None, None

def fetch_approval_rating():
    """(approve %, source note) or (None, None)."""
    try:
        a = load()['approval']
        return a['approve'], f"Our average of {a['n_polls']} polls (Wikipedia, UCSB Presidency Project) · latest: {a['latest_poll']}"
    except Exception as e:
        print(f"Approval unavailable: {e}")
        return None, None

if __name__ == '__main__':
    d = refresh()
    print(json.dumps({k: v for k, v in d.items()}, indent=2))
    print(load_approval_polls().tail(8).to_string(index=False))
