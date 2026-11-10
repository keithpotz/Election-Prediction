"""
Data-quality checks on the 2026 poll file, run after every refresh.

Bad inputs shouldn't be able to move the forecast quietly. Each check produces rows
with a severity:
  warning  likely a data problem worth a look before publishing
  info     worth knowing, usually fine

    python src/poll_checks.py
"""
import os

import numpy as np
import pandas as pd

from polls_2026 import POLLS_DIR, DROPPED_CSV, SCRAPE_ERRORS_CSV, load_polls
from polls_average import annotate_polls, race_average

CHECKS_CSV = os.path.join(POLLS_DIR, 'data_checks.csv')
OUTLIER_PTS = 15        # corrected margin this far from the race average
STALE_DAYS = 60
RECENT_DAYS = 60

def _race(office, state, district=None):
    label = f"{office.title()} · {state.title()}"
    return label + (f"-{int(district)}" if pd.notna(district) and district is not None else '')

def run_checks(as_of=None):
    as_of = pd.Timestamp(as_of or pd.Timestamp.today().normalize())
    polls = load_polls(include_house=True)
    polls['district'] = polls.get('district')
    issues = []

    def add(sev, check, race, detail):
        issues.append({'severity': sev, 'check': check, 'race': race, 'detail': detail})

    # 1. Missing fields
    for col, label in (('end_date', 'end date'), ('sample_size', 'sample size')):
        miss = polls[polls[col].isna()]
        for (o, s, d), g in miss.groupby(['office', 'state', 'district'], dropna=False):
            add('info', f'missing {label}', _race(o, s, d), f"{len(g)} poll(s): {', '.join(g['pollster'].head(4))}")

    # 2. Outliers and staleness, per race
    annotated = annotate_polls(polls)
    for (o, s, d), g in annotated.groupby(['office', 'state', 'district'], dropna=False):
        avg = race_average(g, as_of)
        race = _race(o, s, d)
        if avg is None:
            if g['end_date'].notna().any():
                add('info', 'stale race', race, f"latest poll {g['end_date'].max():%Y-%m-%d}; too old to use, so the "
                                                "forecast relies on fundamentals")
            continue
        recent = g[g['end_date'] >= as_of - pd.Timedelta(days=RECENT_DAYS)]
        dev = (recent['adjusted_margin'] - avg['poll_avg']).abs()
        for _, p in recent[dev > OUTLIER_PTS].iterrows():
            add('warning', 'outlier poll', race,
                f"{p['pollster']} ({p['end_date']:%b %d}): corrected margin {p['adjusted_margin']:+.1f} vs race "
                f"average {avg['poll_avg']:+.1f}, from {p.get('source', '?')}")
        if (as_of - pd.Timestamp(avg['latest_poll'])).days > STALE_DAYS:
            add('info', 'stale race', race, f"latest poll {avg['latest_poll']}")

    # 3. Pages that failed to load: their races silently have fewer (or no) polls
    if os.path.exists(SCRAPE_ERRORS_CSV):
        errs = pd.read_csv(SCRAPE_ERRORS_CSV)
        for _, e in errs.iterrows():
            add('warning', 'page failed to load', _race(e['office'], e['state']), f"{e['error']} ({e['url']})")

    # 4. Nominee problems: most recent polls of a race don't match the nominees on file
    if os.path.exists(DROPPED_CSV):
        dropped = pd.read_csv(DROPPED_CSV, parse_dates=['end_date'])
        recent_drop = dropped[dropped['end_date'] >= as_of - pd.Timedelta(days=RECENT_DAYS)]
        kept = polls[polls['end_date'] >= as_of - pd.Timedelta(days=RECENT_DAYS)]
        for (o, s_), g in recent_drop.groupby(['office', 'state']):
            k = kept[(kept['office'] == o) & (kept['state'] == s_)]
            if len(g) > len(k):
                add('warning', 'nominee mismatch', _race(o, s_),
                    f"{len(g)} recent polls don't match the nominees on file ({len(k)} do). Latest: "
                    f"{g.sort_values('end_date').iloc[-1]['reason'][:140]}")

    # 5. Nominees inferred without an aggregate table to confirm them
    nom_csv = os.path.join(POLLS_DIR, 'nominees_2026.csv')
    if os.path.exists(nom_csv):
        nom = pd.read_csv(nom_csv)
        cov_csv = os.path.join(POLLS_DIR, 'coverage_2026.csv')
        from_agg = set()
        if os.path.exists(cov_csv):
            cov = pd.read_csv(cov_csv)
            from_agg = set(zip(cov.loc[cov['nominees_from_aggregate_table'], 'office'],
                               cov.loc[cov['nominees_from_aggregate_table'], 'state']))
        unconfirmed = nom[~nom['confirmed_by_candidate_list'] &
                          ~nom.apply(lambda r: (r['office'], r['state']) in from_agg, axis=1)]
        for _, r in unconfirmed[unconfirmed['dem_candidate'].isna()].iterrows():
            add('warning', 'nominees unresolved', _race(r['office'], r['state']),
                "Several same-party candidates are on the general ballot and recent polls don't settle which "
                "matchup to use, so this race's polls are left out (fundamentals only)")
        unconfirmed = unconfirmed[unconfirmed['dem_candidate'].notna()]
        for _, r in unconfirmed.iterrows():
            add('warning', 'nominees unconfirmed', _race(r['office'], r['state']),
                f"{r['dem_candidate']} vs {r['rep_candidate']} inferred from poll tables; not confirmed by "
                "Wikipedia's candidate list or a poll-aggregator table")

    out = pd.DataFrame(issues, columns=['severity', 'check', 'race', 'detail'])
    out = out.sort_values(['severity', 'check', 'race'], key=lambda c: c.map({'warning': 0, 'info': 1}) if c.name == 'severity' else c)
    out.to_csv(CHECKS_CSV, index=False)
    return out

def load_checks():
    return pd.read_csv(CHECKS_CSV) if os.path.exists(CHECKS_CSV) else None

if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 140)
    c = run_checks()
    print(c.groupby(['severity', 'check']).size().to_string())
    print()
    print(c[c['severity'] == 'warning'].to_string(index=False))
