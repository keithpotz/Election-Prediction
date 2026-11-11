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
import gzip
import json
import shutil
from datetime import datetime

import pandas as pd

from data_sources import fetch_generic_ballot
from forecast_2026 import build_forecast
from polls_2026 import POLL_SOURCE_NOTE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, '..', 'forecasts', '2026')
MODEL_VERSION = '2026.11.12'  # bump with every methodology change, and log it in forecasts/CHANGELOG.md

# The scraped and fitted inputs a forecast is built from. The daily refresh overwrites
# them, so each snapshot keeps a gzipped copy in forecasts/2026/inputs/<date>/ (about
# 100 KB a day). With them, any published forecast can be rebuilt exactly later, e.g. to
# test a method change on what the forecast actually knew that day.
DATA_DIR = os.path.join(BASE_DIR, 'data')
SNAPSHOT_INPUTS = [
    os.path.join('polls_2026', 'polls_2026.csv'),
    os.path.join('polls_2026', 'manual_polls.csv'),
    os.path.join('polls_2026', 'nominees_2026.csv'),
    os.path.join('polls_2026', 'house_ratings_2026.csv'),
    os.path.join('polling_data', 'house_districts_2026.csv'),
    os.path.join('polling_data', 'senate_race_status_2026.csv'),
    os.path.join('polling_data', 'governor_race_status_2026.csv'),
    os.path.join('polling_data', 'pollster_ratings.csv'),
    os.path.join('polling_data', 'poll_error_model.json'),
    os.path.join('..', '..', 'forecasts', 'calibration', 'calibration_report.json'),
]

def save_inputs(stamp, out_dir=None, data_dir=None):
    """Gzipped copies of SNAPSHOT_INPUTS, under the same relative paths, plus a manifest.
    Missing files are listed in the manifest rather than failing the publish."""
    data_dir = data_dir or DATA_DIR
    dest = os.path.join(out_dir or OUT_DIR, 'inputs', stamp)
    saved, missing = [], []
    for rel in SNAPSHOT_INPUTS:
        src = os.path.normpath(os.path.join(data_dir, rel))
        if not os.path.exists(src):
            missing.append(rel.replace(os.sep, '/'))
            continue
        name = os.path.basename(src) + '.gz'
        sub = 'calibration' if 'calibration' in rel else os.path.basename(os.path.dirname(src))
        os.makedirs(os.path.join(dest, sub), exist_ok=True)
        with open(src, 'rb') as fi, gzip.open(os.path.join(dest, sub, name), 'wb') as fo:
            shutil.copyfileobj(fi, fo)
        saved.append(f'{sub}/{name}')
    with open(os.path.join(dest, 'manifest.json'), 'w') as fh:
        json.dump({'model_version': MODEL_VERSION, 'saved': saved, 'missing': missing}, fh, indent=2)
    return dest

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if k != 'dem_seats_dist'}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, 'item'):
        return x.item()
    return x

def _data_warnings():
    """Open data-quality warnings at publish time, recorded with the snapshot."""
    try:
        from poll_checks import load_checks
        c = load_checks()
        if c is None:
            return None
        w = c[c['severity'] == 'warning']
        return [f"[{r['check']}] {r['race']}: {r['detail']}" for _, r in w.iterrows()]
    except Exception:
        return None

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
        'poll_data_sources': POLL_SOURCE_NOTE,
        'data_warnings': _data_warnings(),
        'generic_ballot_bias_removed': f['generic_ballot_bias_removed'],
        'national_environment_used': f['national_environment'],
        **_jsonable(f['summary']),
    }
    with open(os.path.join(OUT_DIR, f'{stamp}_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    f['senate'].to_csv(os.path.join(OUT_DIR, f'{stamp}_senate_races.csv'), index=False)
    f['governor'].to_csv(os.path.join(OUT_DIR, f'{stamp}_governor_races.csv'), index=False)
    house_cols = ['state', 'district', 'pvi', 'incumbent', 'incumbent_party', 'incumbent_running',
                  'dem_candidate', 'rep_candidate', 'uncontested', 'predicted_margin',
                  'dem_win_prob', 'predicted_winner', 'consensus_rating', 'n_polls', 'poll_avg',
                  'poll_weight', 'race_sd', 'shared_sd']
    f['house'][house_cols].to_csv(os.path.join(OUT_DIR, f'{stamp}_house_districts.csv'), index=False)
    save_inputs(stamp)

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
