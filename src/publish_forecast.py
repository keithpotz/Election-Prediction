"""
Snapshot today's 2026 forecast (every Senate, governor, and House race) into
forecasts/2026/ so it can be committed to git.

The git commit is the timestamp: once pushed, GitHub's history shows exactly what this
forecast said on a given day, which makes the track record verifiable (no after-the-fact
edits). Run once per forecast update, then commit the new files:

    python src/polls_2026.py          # refresh polls first
    python src/publish_forecast.py
    git add forecasts src/data/polls_2026 && git commit -m "Forecast YYYY-MM-DD"
"""
import os
import json
from datetime import datetime

import pandas as pd

from data_sources import fetch_generic_ballot
from forecast_2026 import build_forecast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, '..', 'forecasts', '2026')
MODEL_VERSION = '2026.10.06b'  # bump with every methodology change, and log it in forecasts/CHANGELOG.md

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if k != 'dem_seats_dist'}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, 'item'):
        return x.item()
    return x

def publish(generic_ballot=None):
    source = 'manual'
    if generic_ballot is None:
        generic_ballot, source = fetch_generic_ballot()
        if generic_ballot is None:
            raise RuntimeError("Couldn't fetch the live generic ballot; pass one in explicitly.")

    f = build_forecast(generic_ballot)
    now = datetime.now()
    stamp = now.strftime('%Y-%m-%d')
    os.makedirs(OUT_DIR, exist_ok=True)

    summary = {
        'published_at':   now.isoformat(timespec='seconds'),
        'model_version':  MODEL_VERSION,
        'days_to_election': f['days_to_election'],
        'generic_ballot_dem_margin': float(generic_ballot),
        'generic_ballot_source': source,
        **_jsonable(f['summary']),
    }
    with open(os.path.join(OUT_DIR, f'{stamp}_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    f['senate'].to_csv(os.path.join(OUT_DIR, f'{stamp}_senate_races.csv'), index=False)
    f['governor'].to_csv(os.path.join(OUT_DIR, f'{stamp}_governor_races.csv'), index=False)
    house_cols = ['state', 'district', 'pvi', 'incumbent', 'incumbent_party', 'incumbent_running',
                  'dem_candidate', 'rep_candidate', 'uncontested', 'predicted_margin',
                  'dem_win_prob', 'predicted_winner']
    f['house'][house_cols].to_csv(os.path.join(OUT_DIR, f'{stamp}_house_districts.csv'), index=False)

    # One line per publish: the topline history, for charting the forecast over time.
    # Read-append-write so rows from older model versions (fewer columns) stay aligned.
    log_path = os.path.join(OUT_DIR, 'topline_log.csv')
    s = f['summary']
    row = pd.DataFrame([{
        'published_at': summary['published_at'], 'model_version': MODEL_VERSION,
        'generic_ballot': generic_ballot,
        'house_dem_seats': s['house']['dem_seats_median'], 'house_dem_control_prob': s['house']['dem_control_prob'],
        'senate_dem_seats_median': s['senate']['dem_seats_median'],
        'senate_dem_control_prob': s['senate']['dem_control_prob'],
        'dem_governorships_median': s['governor']['dem_governorships_median'],
    }])
    log = pd.concat([pd.read_csv(log_path), row], ignore_index=True) if os.path.exists(log_path) else row
    log.to_csv(log_path, index=False)

    print(json.dumps({k: summary[k] for k in ('published_at', 'model_version', 'generic_ballot_dem_margin')}, indent=2))
    print(f"Senate Dem control {s['senate']['dem_control_prob']:.0%}, House Dem control {s['house']['dem_control_prob']:.0%}, "
          f"Dem governorships (median) {s['governor']['dem_governorships_median']}")
    print(f"Wrote forecast files to {os.path.abspath(OUT_DIR)}")
    return summary

if __name__ == '__main__':
    publish()
