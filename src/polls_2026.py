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
    pop = re.search(r'(?:\(|\s)(LV|RV|A|V)\)?\s*$', s)
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

def _rcv_round(x):
    """'3', '3[j]' -> 3. 'BA' (the ballot before ranking) and blanks -> NaN."""
    m = re.match(r'\s*(\d+)', str(x))
    return float(m.group(1)) if m else np.nan

def _sanitize(html):
    # Some pages carry malformed span attributes (e.g. colspan="2"") that crash read_html.
    return re.sub(r'''(colspan|rowspan)\s*=\s*["']?(\d+)[^\s>]*''', r'\1="\2"', html)

# A non-ranked-choice table whose other candidates from the nominees' parties (or any
# other Democrat/Republican) average this much (Alaska governor first-choice polls: two
# more Republicans splitting ~20%; Idaho Senate: a withdrawn Democrat splitting the
# anti-Risch vote with independent Achilles), so its D-minus-R margin would mislead.
SPLIT_FIELD_PCT = 10.0
# A row where the two nominees together have less than this isn't a head-to-head either
# (an early ranked-choice round that still includes other candidates).
MIN_TWO_WAY_PCT = 60.0

def _split_field(t, d_col, r_col):
    if any('RCV round' in c for c in t.columns):
        return False
    pc = _party_columns(t.columns)
    splitting = {pc[d_col], pc[r_col]} | DEM_TAGS | REP_TAGS
    others = [c for c, p in pc.items() if c not in (d_col, r_col) and p in splitting]
    return bool(others) and t[others].apply(lambda col: col.map(_pct)).mean().sum() >= SPLIT_FIELD_PCT

def parse_race_page(html, office, state, url, known=None):
    """known: (dem_side_name, rep_name, dem_side_party) from the confirmed nominee list,
    used when the page has no polling-aggregate table naming the nominees."""
    tables = [_flatten(t) for t in pd.read_html(io.StringIO(_sanitize(html)))]
    poll_tables = [t for t in tables if any(c.startswith('Poll source') for c in t.columns)]
    agg_tables = [t for t in tables if any(c.startswith('Source of poll aggregation') for c in t.columns)]

    nominees = None
    for t in agg_tables:
        sides = _pick_sides(_party_columns(t.columns))
        if sides:
            nominees = (_cand_name(sides[0]), _cand_name(sides[1]), sides[2])
            break
    if nominees is None and known is not None:
        nominees = (_cand_name(known[0]), _cand_name(known[1]), known[2])

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
            d_col, r_col = by_name[nominees[0]], by_name[nominees[1]]
            if not _split_field(t, d_col, r_col):
                chosen.append((t, (d_col, r_col, nominees[2])))

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
            if d + rep < MIN_TWO_WAY_PCT:
                continue                                   # other candidates still in the count
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
                'rcv_round':     _rcv_round(r[rcv_col]) if rcv_col else np.nan,
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

    known = {}
    nominees_csv = os.path.join(POLLS_DIR, 'nominees_2026.csv')
    if os.path.exists(nominees_csv):
        n = pd.read_csv(nominees_csv)
        known = {(r['office'], r['state']): (r['dem_candidate'], r['rep_candidate'], r['dem_side_party'])
                 for _, r in n.iterrows() if isinstance(r['dem_candidate'], str) and r.get('confirmed_by_candidate_list')}

    all_polls, coverage = [], []
    for office, state, special in races:
        url = race_url(office, state, special)
        polls, nominees, err = pd.DataFrame(), None, None
        for attempt in range(3):           # transient failures shouldn't silently zero out a race
            try:
                polls, nominees = parse_race_page(_fetch_html(url), office, state, url, known.get((office, state)))
                err = None
                break
            except Exception as e:
                err = str(e)[:120]
                if '404' in err:
                    break
                time.sleep(pause * 3)
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

# ── Official candidate lists (nominee confirmation) ────────────────────────────
PARTY_WORDS = {'Democratic': 'DEMOCRAT', 'Democratic–Farmer–Labor': 'DEMOCRAT', 'DFL': 'DEMOCRAT',
               'Republican': 'REPUBLICAN', 'Independent': 'INDEPENDENT'}

def scrape_candidate_lists():
    """
    (office, state) -> [(name, party)] from the Candidates column of Wikipedia's Senate
    and governor overview pages. These list the general-election candidates with party
    labels, which is a firmer source of nominees than inferring them from poll tables.
    """
    from race_status_2026 import SENATE_WIKI_URL, GOVERNOR_WIKI_URL
    out = {}
    for office, url in (('senate', SENATE_WIKI_URL), ('governor', GOVERNOR_WIKI_URL)):
        tables = [_flatten(t) for t in pd.read_html(io.StringIO(_sanitize(_fetch_html(url))))]
        table = next(t for t in tables if any('Candidates' in c for c in t.columns) and len(t) >= 30)
        state_col = next(c for c in table.columns if c.endswith('State'))
        cand_col = next(c for c in table.columns if 'Candidates' in c)
        for _, r in table.iterrows():
            state = re.sub(r'\s*\(special\)|\[[^\]]*\]', '', str(r[state_col])).strip().upper()
            cands = [(n.strip(), PARTY_WORDS.get(pt.strip(), pt.strip().upper()))
                     for n, pt in re.findall(r'▌?\s*([^▌()\[\]]+?)\s*\(([^)]+)\)', str(r[cand_col]))]
            if cands:
                out[(office, state)] = cands
    return out

def primary_votes(html):
    """{candidate cell text: votes} from every results table on a race page (tables with
    Candidate and Votes columns). Cells may include a running mate's name too."""
    out = {}
    for t in pd.read_html(io.StringIO(_sanitize(html))):
        t = _flatten(t)
        cand = next((c for c in t.columns if c.endswith('Candidate')), None)
        votes = next((c for c in t.columns if c.endswith('Votes') and 'First choice' not in c or c == 'Votes'), None)
        if not cand or not votes:
            continue
        for _, r in t.iterrows():
            v = pd.to_numeric(str(r[votes]).replace(',', ''), errors='coerce')
            if pd.notna(v) and isinstance(r[cand], str):
                out[r[cand]] = max(out.get(r[cand], 0), float(v))
    return out

def resolve_nominees(cands, poll_names, fallback=None, votes=None):
    """
    Picks (dem_side, republican, dem_side_party) from an official candidate list.
    The Dem side is the lone Democrat, or the independent most present in polls when no
    Democrat runs (Osborn, Achilles). When a party has several candidates on the general
    ballot (Alaska's top-four system), the one appearing most in recent polls is used.
    poll_names: candidate names seen in recent polls of this race (with repeats).
    """
    from names import _name_score, _norm
    def presence(name):
        return sum(1 for n in poll_names if _name_score(n, name) >= 0.8)
    def primary(name):
        # Results cells can carry a running mate ("Bernadette Wilson Mike Shower")
        key = _norm(name)
        return max((v for c, v in (votes or {}).items() if _norm(c).startswith(key)), default=0.0)
    def pick(names):
        if not names:
            return None
        if len(names) == 1:
            return names[0]
        ranked = sorted(names, key=presence, reverse=True)
        if presence(ranked[0]) > presence(ranked[1]):
            return ranked[0]
        # Polls don't settle it (e.g. Alaska's top-four general ballot): use the primary leader
        ranked = sorted(names, key=primary, reverse=True)
        return ranked[0] if primary(ranked[0]) > primary(ranked[1]) else None
    dems = [n for n, p in cands if p == 'DEMOCRAT']
    reps = [n for n, p in cands if p == 'REPUBLICAN']
    inds = [n for n, p in cands if p == 'INDEPENDENT']
    rep_name = pick(reps)
    if dems:
        dem_name, party = pick(dems), 'DEMOCRAT'
    else:
        dem_name, party = pick([n for n in inds if presence(n)]), 'INDEPENDENT'
    if dem_name and rep_name:
        return dem_name, rep_name, party
    # Ambiguous list: only accept the poll-table matchup if both of its candidates are
    # actually on the official list (never a stale hypothetical, e.g. a candidate who
    # is running for a different office).
    if fallback and all(any(_name_score(f, n) >= 0.8 for n, _ in cands) for f in fallback[:2]):
        return fallback
    return None

POLL_SOURCE_NOTE = ("Polls: Wikipedia race and state House pages, compiled from pollsters' own releases. "
                    "No prediction-market-affiliated data sources are used.")
DROPPED_CSV = os.path.join(POLLS_DIR, 'dropped_polls.csv')
SCRAPE_ERRORS_CSV = os.path.join(POLLS_DIR, 'scrape_errors.csv')

def refresh_polls():
    """
    Every 2026 Senate, governor, and House district poll from Wikipedia, matched to the
    confirmed general-election nominees. Polls of other matchups (hypotheticals,
    withdrawn candidates) are set aside in dropped_polls.csv; pages that failed to load
    are logged in scrape_errors.csv so the data checks can flag them.
    """
    from pollster_ratings import sponsor_party
    from names import _name_score
    from house_districts import load_house_districts
    from polls_house_districts import scrape_district_polls

    wiki, coverage = scrape_all()
    wiki['source'] = 'wikipedia'
    wiki['sponsor'] = wiki['pollster'].map(sponsor_party)
    wiki['district'] = None

    # Nominees: official candidate lists, with poll-table matchups as fallback
    fallback = {(r['office'], r['state'], None): (r['dem_candidate'], r['rep_candidate'],
                 wiki[(wiki['office'] == r['office']) & (wiki['state'] == r['state'])]['dem_side_party'].iloc[0])
                for _, r in coverage[coverage['n_polls'] > 0].iterrows()}
    seen = {}
    for _, w in wiki[wiki['end_date'] >= '2026-06-01'].iterrows():
        seen.setdefault((w['office'], w['state'], None), []).extend([w['dem_candidate'], w['rep_candidate']])
    try:
        official = scrape_candidate_lists()
    except Exception as e:
        print(f"Official candidate lists unavailable, using poll-table nominees: {e}")
        official = {}
    nominees, confirmed = dict(fallback), set()
    for (office, state), cands in official.items():
        key = (office, state, None)
        resolved = resolve_nominees(cands, seen.get(key, []), fallback.get(key))
        if resolved is None:
            # Several same-party candidates and no poll signal: try the race page's primary results
            try:
                special = bool(coverage[(coverage['office'] == office) & (coverage['state'] == state)]['special'].any())
                votes = primary_votes(_fetch_html(race_url(office, state, special)))
                resolved = resolve_nominees(cands, seen.get(key, []), fallback.get(key), votes)
            except Exception as e:
                print(f"Primary results unavailable for {office} {state}: {e}")
        if resolved:
            nominees[key] = resolved
            confirmed.add((office, state))
        else:
            nominees.pop(key, None)       # list contradicts the poll tables: use no polls

    def matches(w):
        n = nominees.get((w['office'], w['state'], None))
        return (n is not None and _name_score(w['dem_candidate'], n[0]) >= 0.8
                and _name_score(w['rep_candidate'], n[1]) >= 0.8)
    ok = wiki.apply(matches, axis=1) if len(wiki) else pd.Series(dtype=bool)
    dropped = wiki[~ok].assign(reason=lambda d: "matchup isn't the confirmed nominees: "
                               + d['dem_candidate'] + ' vs ' + d['rep_candidate'])
    wiki = wiki[ok]

    # House district polls
    districts = load_house_districts()
    house, house_errors = scrape_district_polls(districts)
    if len(house):
        house = house.drop_duplicates(['state', 'district', 'pollster', 'end_date', 'margin'])

    polls = pd.concat([wiki, house], ignore_index=True).sort_values(['office', 'state', 'end_date'])
    polls['scraped_at'] = datetime.now().isoformat(timespec='seconds')
    errors = pd.concat([coverage[coverage['error'].notna()][['office', 'state', 'error', 'url']],
                        house_errors], ignore_index=True)

    os.makedirs(POLLS_DIR, exist_ok=True)
    polls.to_csv(POLLS_CSV, index=False)
    coverage.to_csv(COVERAGE_CSV, index=False)
    dropped.to_csv(DROPPED_CSV, index=False)
    errors.to_csv(SCRAPE_ERRORS_CSV, index=False)
    pd.DataFrame([{'office': k[0], 'state': k[1], 'dem_candidate': v[0], 'rep_candidate': v[1],
                   'dem_side_party': v[2], 'confirmed_by_candidate_list': (k[0], k[1]) in confirmed}
                  for k, v in nominees.items() if k[2] is None]
                 + [{'office': o, 'state': st, 'dem_candidate': None, 'rep_candidate': None, 'dem_side_party': None,
                     'confirmed_by_candidate_list': False}
                    for (o, st) in official if (o, st) not in confirmed]).to_csv(
        os.path.join(POLLS_DIR, 'nominees_2026.csv'), index=False)
    print(f"{len(wiki)} Senate/governor polls ({len(dropped)} non-nominee matchups set aside), "
          f"{len(house)} House district polls, {len(errors)} page load failures")
    return polls, coverage

# Polls Wikipedia hasn't added yet, entered by hand from the pollster's own release (the
# source_url must point to it). Merged in at load time, so a new row counts without a
# re-scrape; once Wikipedia lists the same poll, the scraped copy wins.
MANUAL_CSV = os.path.join(POLLS_DIR, 'manual_polls.csv')
MANUAL_COLUMNS = ['office', 'state', 'district', 'pollster', 'end_date', 'sample_size', 'population',
                  'dem_candidate', 'rep_candidate', 'dem_side_party', 'dem_pct', 'rep_pct', 'source_url',
                  'entered_at', 'note']

def _pollster_key(name):
    return re.sub(r'[^a-z0-9]', '', PARTY_TAG.sub('', str(name)).lower())

def load_manual_polls():
    if not os.path.exists(MANUAL_CSV):
        return pd.DataFrame(columns=MANUAL_COLUMNS)
    from pollster_ratings import sponsor_party
    m = pd.read_csv(MANUAL_CSV, parse_dates=['end_date'], comment='#')
    if m.empty:
        return m
    m['state'] = m['state'].str.upper()
    m['margin'] = m['dem_pct'] - m['rep_pct']
    m['source'] = 'manual'
    m['sponsor'] = m['pollster'].map(sponsor_party)
    m['scraped_at'] = m['entered_at']
    m['rcv_round'] = None
    return m

def merge_manual(scraped, manual):
    """Adds hand-entered polls that the scrape doesn't already have (same race and pollster,
    end date within 3 days). Senate and governor polls must match the nominees on file."""
    if manual.empty:
        return scraped
    keep = []
    nominees_path = os.path.join(POLLS_DIR, 'nominees_2026.csv')
    nominees = pd.read_csv(nominees_path) if os.path.exists(nominees_path) else pd.DataFrame()
    from names import _name_score
    for _, m in manual.iterrows():
        same = scraped[(scraped['office'] == m['office']) & (scraped['state'] == m['state'])]
        if m['office'] == 'house':
            same = same[pd.to_numeric(same['district'], errors='coerce') == m['district']]
        same = same[(same['pollster'].map(_pollster_key) == _pollster_key(m['pollster']))
                    & ((same['end_date'] - m['end_date']).abs() <= pd.Timedelta(days=3))]
        if len(same):
            continue                     # Wikipedia has it now
        if m['office'] != 'house' and len(nominees):
            n = nominees[(nominees['office'] == m['office']) & (nominees['state'] == m['state'])]
            if n.empty or not isinstance(n['dem_candidate'].iloc[0], str) or \
                    _name_score(m['dem_candidate'], n['dem_candidate'].iloc[0]) < 0.8 or \
                    _name_score(m['rep_candidate'], n['rep_candidate'].iloc[0]) < 0.8:
                print(f"Manual poll skipped, matchup isn't the nominees: {m['office']} {m['state']} {m['pollster']}")
                continue
        keep.append(m)
    if not keep:
        return scraped
    # All-empty columns dropped before concat (pandas deprecates them steering dtypes); concat refills them as NaN.
    add = pd.DataFrame(keep).reindex(columns=scraped.columns).dropna(axis=1, how='all')
    return pd.concat([scraped, add], ignore_index=True).sort_values(['office', 'state', 'end_date'])

def load_polls(include_house=False, manual=True):
    """Senate and governor polls; House district polls too when include_house=True.
    Includes hand-entered polls from manual_polls.csv unless manual=False."""
    p = pd.read_csv(POLLS_CSV, parse_dates=['end_date'])
    if manual:
        p = merge_manual(p, load_manual_polls())
    return p if include_house or 'office' not in p else p[p['office'] != 'house']

if __name__ == '__main__':
    polls, coverage = refresh_polls()
    pd.set_option('display.width', 220)
    print(coverage.drop(columns=['url']).to_string())
    print(f"\n{len(polls)} polls across {(coverage['n_polls'] > 0).sum()} of {len(coverage)} races.")
