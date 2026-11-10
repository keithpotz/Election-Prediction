"""
Rasmussen Reports posts its 2026 race polls on its own site, often days before Wikipedia
lists them (and some never get listed). This reads each race page linked from the 2026
index, pulls the nominees' numbers, sample and field dates out of the write-up, and
checks whether our poll data already has that poll.

    python src/rasmussen_check.py          # report only
    python src/rasmussen_check.py --add    # also append the missing polls to manual_polls.csv

Each page shows only the newest Rasmussen poll in that race. A row is added only when
both nominees' numbers, the sample and the field dates were all read; anything else is
listed for entering by hand. Then run the daily update (or "Refresh polls" in the app).
"""
import argparse
import html as H
import os
import re
from datetime import datetime

import pandas as pd

from polls_2026 import MANUAL_COLUMNS, MANUAL_CSV, POLLS_DIR, _pollster_key, load_polls
from race_status_2026 import _fetch_html

BASE = 'https://www.rasmussenreports.com'
INDEX_URL = BASE + '/public_content/politics/elections/election_2026'
POLLSTER = 'Rasmussen Reports (R)'      # how Wikipedia lists it in most 2026 races
MONTHS = 'January|February|March|April|May|June|July|August|September|October|November|December'
FIELD_DATES = re.compile(rf'conducted on ({MONTHS}) (\d{{1,2}})(?:\s*[-–]\s*(?:({MONTHS}) )?(\d{{1,2}}))?,? (\d{{4}})')
SAMPLE = re.compile(r'survey of ([\d,]+) (?:[A-Z][\w.]* )*?(Likely|Registered) Voters')
PUBLISHED = re.compile(rf'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), ({MONTHS}) (\d{{1,2}}), (\d{{4}})')
SAME_POLL_DAYS = 3

def race_pages():
    """(office, state, url) for every Senate and governor page on the 2026 index."""
    links = sorted(set(re.findall(r'href="([^"]*/election_2026/election_2026_([a-z_]+?)_(senate|governor))"',
                                  _fetch_html(INDEX_URL))))
    return [(office, slug.replace('_', ' ').upper(), href if href.startswith('http') else BASE + href)
            for href, slug, office in links]

def page_text(url):
    raw = re.sub(r'<script.*?</script>|<style.*?</style>', ' ', _fetch_html(url), flags=re.S)
    return re.sub(r'\s+', ' ', H.unescape(re.sub(r'<[^>]+>', ' ', raw)))

def _pct_for(text, last):
    """The candidate's topline number: the first 'NN% ... for/favor/support <Last>' or
    '<Last> ... NN%' in the survey's headline sentence."""
    m = re.search(rf'(\d{{1,2}})% (?:of [^.%]*?)?(?:would )?(?:vote for|favor|support|prefer|choose)s? (?:\w+ )*?{re.escape(last)}\b',
                  text)
    if m:
        return int(m.group(1))
    m = re.search(rf'\b{re.escape(last)}\b[^.%]{{0,60}}?(\d{{1,2}})%', text)
    return int(m.group(1)) if m else None

def _last(name):
    tokens = [t for t in str(name).split() if t.lower().strip(',.') not in {'jr', 'sr', 'ii', 'iii', 'iv'}]
    return tokens[-1].strip(',') if tokens else str(name)

def read_poll(office, state, url, nominees):
    """What the page says, as a manual_polls row, plus a list of anything it couldn't read."""
    text = page_text(url)
    nom = nominees.get((office, state))
    problems = []
    row = {'office': office, 'state': state, 'district': None, 'pollster': POLLSTER, 'source_url': url}
    if nom is None:
        return row, ['no confirmed nominees on file for this race']
    row.update({'dem_candidate': nom['dem_candidate'], 'rep_candidate': nom['rep_candidate'],
                'dem_side_party': nom['dem_side_party']})
    # The topline is in the sentence after "survey finds"; fall back to the whole lead
    lead = text[text.find('survey finds'):][:600] if 'survey finds' in text else text[:3000]
    row['dem_pct'] = _pct_for(lead, _last(nom['dem_candidate']))
    row['rep_pct'] = _pct_for(lead, _last(nom['rep_candidate']))
    if row['dem_pct'] is None or row['rep_pct'] is None:
        problems.append("couldn't find both nominees' numbers (different candidates, or an unusual write-up)")
    m = FIELD_DATES.search(text)
    if m:
        month0, day0, month1, day1, year = m.groups()
        row['end_date'] = pd.Timestamp(f"{month1 or month0} {day1 or day0} {year}")
        row['start_date'] = pd.Timestamp(f"{month0} {day0} {year}")
    else:
        row['end_date'] = None
        problems.append("couldn't read the field dates")
    m = SAMPLE.search(text)
    if m:
        row['sample_size'] = int(m.group(1).replace(',', ''))
        row['population'] = 'LV' if m.group(2) == 'Likely' else 'RV'
    else:
        row['sample_size'], row['population'] = None, None
        problems.append("couldn't read the sample size")
    m = PUBLISHED.search(text)
    row['published'] = pd.Timestamp(f"{m.group(1)} {m.group(2)} {m.group(3)}") if m else None
    return row, problems

def have_it(polls, row):
    """Whether our poll data (scraped plus manual) already has this poll."""
    if row.get('end_date') is None:
        return False
    p = polls[(polls['office'] == row['office']) & (polls['state'] == row['state'])
              & (polls['pollster'].map(_pollster_key) == _pollster_key(POLLSTER))]
    return bool(((p['end_date'] - row['end_date']).abs() <= pd.Timedelta(days=SAME_POLL_DAYS)).any())

def check():
    nominees = pd.read_csv(os.path.join(POLLS_DIR, 'nominees_2026.csv'))
    nominees = {(r['office'], r['state']): r for _, r in nominees.iterrows()}
    polls = load_polls()
    results = []
    for office, state, url in race_pages():
        try:
            row, problems = read_poll(office, state, url, nominees)
        except Exception as e:      # one bad page shouldn't stop the rest
            row, problems = {'office': office, 'state': state, 'source_url': url, 'end_date': None}, [f"fetch failed: {e}"]
        results.append({**row, 'problems': problems, 'have': have_it(polls, row)})
    return results

def manual_row(r):
    span = (f"fielded {r['start_date']:%b %d}" + (f"-{r['end_date']:%b %d}" if r['end_date'] != r['start_date'] else "")
            if r.get('start_date') is not None else "")
    pub = f"published {r['published']:%b %d}" if r.get('published') is not None else ""
    return {**{c: r.get(c) for c in MANUAL_COLUMNS},
            'end_date': r['end_date'].strftime('%Y-%m-%d'),
            'entered_at': datetime.now().isoformat(timespec='seconds'),
            'note': '; '.join(x for x in ("Read by rasmussen_check.py", span, pub) if x)}

def add_rows(rows):
    """Append to manual_polls.csv, keeping its comment header."""
    with open(MANUAL_CSV, 'a', encoding='utf-8', newline='') as fh:
        pd.DataFrame(rows, columns=MANUAL_COLUMNS).to_csv(fh, header=False, index=False)

def _line(r):
    race = f"{r['state'].title()} {r['office']}"
    if r.get('dem_pct') is None or r.get('rep_pct') is None or r.get('end_date') is None:
        return race
    return (f"{race}: {_last(r['dem_candidate'])} {r['dem_pct']}, {_last(r['rep_candidate'])} {r['rep_pct']}, "
            f"ended {r['end_date']:%b %d}, n={r['sample_size']} {r['population']}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--add', action='store_true', help='append cleanly read missing polls to manual_polls.csv')
    args = ap.parse_args()
    results = check()
    have = [r for r in results if r['have']]
    missing = [r for r in results if not r['have'] and not r['problems']]
    unread = [r for r in results if not r['have'] and r['problems']]
    print(f"Rasmussen 2026 race pages: {len(results)}. Already in our data: {len(have)}.")
    for r in have:
        print(f"  ok       {_line(r)}")
    for r in missing:
        print(f"  MISSING  {_line(r)}\n           {r['source_url']}")
    for r in unread:
        print(f"  CHECK    {_line(r)}: {'; '.join(r['problems'])}\n           {r['source_url']}")
    if missing and args.add:
        add_rows([manual_row(r) for r in missing])
        print(f"Added {len(missing)} poll(s) to {os.path.relpath(MANUAL_CSV)}. Refresh polls to use them.")
    elif missing:
        print("Run with --add to append the MISSING polls to manual_polls.csv.")
