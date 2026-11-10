"""
One-command daily update: refresh every input, check the data, publish the snapshot.

    python src/daily_update.py              # full refresh + publish
    python src/daily_update.py --no-publish # refresh + checks only

Steps:
  1. Race ratings and status (Senate, governor) and House district list from Wikipedia
  2. Polls (Senate, governor, House districts) and national inputs from Wikipedia,
     matched to confirmed nominees; no prediction-market-affiliated sources
  3. Data-quality checks (warnings are logged and included in the snapshot), plus a
     report of Rasmussen polls on its own site that aren't in our data yet
  4. Publish today's forecast snapshot to forecasts/2026/

Each run appends a line to forecasts/2026/update_log.csv so gaps in the daily record
are visible. Safe to schedule (e.g. Windows Task Scheduler) once a day.
"""
import os
import sys
import time
import traceback
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_CSV = os.path.join(BASE_DIR, '..', 'forecasts', '2026', 'update_log.csv')

def _log(row):
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(LOG_CSV, mode='a', header=not os.path.exists(LOG_CSV), index=False)

def run(publish=True):
    started = time.time()
    row = {'started_at': datetime.now().isoformat(timespec='seconds'), 'status': 'failed', 'step': '',
           'polls': None, 'warnings': None, 'senate_dem_control': None, 'house_dem_control': None,
           'seconds': None, 'error': ''}
    try:
        row['step'] = 'race status'
        from race_status_2026 import load_or_scrape_race_status
        for office in ('senate', 'governor'):
            load_or_scrape_race_status(refresh=True, office=office)
        from house_districts import load_house_districts
        load_house_districts(refresh=True)

        row['step'] = 'polls'
        from national_polls import refresh as refresh_national
        refresh_national()
        from polls_2026 import refresh_polls
        polls, _ = refresh_polls()
        row['polls'] = len(polls)

        row['step'] = 'data checks'
        from poll_checks import run_checks
        checks = run_checks()
        warnings = checks[checks['severity'] == 'warning']
        row['warnings'] = len(warnings)
        for _, w in warnings.iterrows():
            print(f"  WARNING [{w['check']}] {w['race']}: {w['detail']}")

        # Report only: Rasmussen polls on its own site that our data doesn't have yet.
        # A failure here (site down, page change) never stops the update.
        try:
            from rasmussen_check import check as rasmussen_check, _line
            for r in [r for r in rasmussen_check() if not r['have']]:
                print(f"  NOTE [Rasmussen not in our data] {_line(r)}. Run python src/rasmussen_check.py --add")
        except Exception as e:
            print(f"  NOTE Rasmussen check skipped: {type(e).__name__}: {e}")

        if publish:
            row['step'] = 'publish'
            from publish_forecast import publish as publish_snapshot
            summary = publish_snapshot()
            env = summary.get('environment_check') or {}
            if env.get('flag'):
                print(f"  WARNING [environment check] race polls imply D{env['implied_by_race_polls']:+.1f} vs the "
                      f"generic-ballot environment D{env['environment_used']:+.1f}")
                row['warnings'] = (row['warnings'] or 0) + 1
            row['step'] = 'share cards'
            from forecast_2026 import build_forecast
            from share_cards import generate_all
            from publish_forecast import MODEL_VERSION
            out_dir, n_cards = generate_all(build_forecast(summary['generic_ballot_dem_margin']), MODEL_VERSION)
            print(f"  {n_cards} share cards -> {out_dir}")
            row['senate_dem_control'] = summary['senate']['dem_control_prob']
            row['house_dem_control'] = summary['house']['dem_control_prob']
        row['status'], row['step'] = 'ok', 'done'
    except Exception as e:
        row['error'] = f"{type(e).__name__}: {e}"[:300]
        traceback.print_exc()
    finally:
        row['seconds'] = round(time.time() - started)
        _log(row)
    print(f"\nDaily update {row['status']} ({row['seconds']}s): {row['polls']} polls, {row['warnings']} data warnings"
          + (f", Senate D {row['senate_dem_control']:.0%}, House D {row['house_dem_control']:.0%}"
             if row['senate_dem_control'] is not None else ''))
    return row['status'] == 'ok'

if __name__ == '__main__':
    sys.exit(0 if run(publish='--no-publish' not in sys.argv) else 1)
