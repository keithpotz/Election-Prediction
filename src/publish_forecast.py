"""
Snapshot today's 2026 midterm forecast into forecasts/2026/ so it can be committed to git.

The git commit is the timestamp: once pushed, GitHub's history shows exactly what this
forecast said on a given day, which makes the track record verifiable (no after-the-fact
edits). Run once per forecast update, then commit the new files:

    python src/publish_forecast.py
    git add forecasts && git commit -m "Forecast YYYY-MM-DD"
"""
import os
import json
from datetime import datetime

from data_sources import fetch_generic_ballot
from midterms import (
    build_house_swing_model, build_senate_race_model, load_race_status,
    predict_house_2026, predict_senate_2026,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, '..', 'forecasts', '2026')
MODEL_VERSION = '2026.10.06'  # bump with every methodology change, and log it in forecasts/CHANGELOG.md

def publish(generic_ballot=None):
    source = 'manual'
    if generic_ballot is None:
        generic_ballot, source = fetch_generic_ballot()
        if generic_ballot is None:
            raise RuntimeError("Couldn't fetch the live generic ballot; pass one in explicitly.")

    house = predict_house_2026(generic_ballot, model_and_std=build_house_swing_model())
    senate = predict_senate_2026(generic_ballot, model_bundle=build_senate_race_model(),
                                 race_status_df=load_race_status())
    if senate['warnings'] or senate['skipped_states']:
        raise RuntimeError(f"Not publishing a forecast with data problems: "
                           f"{senate['warnings']} skipped={senate['skipped_states']}")

    now = datetime.now()
    stamp = now.strftime('%Y-%m-%d')
    os.makedirs(OUT_DIR, exist_ok=True)

    summary = {
        'published_at':           now.isoformat(timespec='seconds'),
        'model_version':          MODEL_VERSION,
        'generic_ballot_dem_margin': float(generic_ballot),
        'generic_ballot_source':  source,
        'house': {k: float(v) if hasattr(v, 'item') else v for k, v in house.items()},
        'senate': {
            'dem_control_prob': senate['dem_control_prob'],
            'dem_seats_median': senate['dem_seats_median'],
            'dem_seats_90pct':  [senate['dem_seats_lo'], senate['dem_seats_hi']],
            'holdovers':        senate['holdovers'],
            'projected_flips':  senate['flipped_states'],
            'backtest':         senate['error_stats'],
        },
    }
    with open(os.path.join(OUT_DIR, f'{stamp}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Expert consensus is saved alongside so the model can be graded against it too.
    races = senate['seats']
    races.to_csv(os.path.join(OUT_DIR, f'{stamp}_senate_races.csv'), index=False)

    # One line per publish: the topline history, for charting the forecast over time.
    log_path = os.path.join(OUT_DIR, 'topline_log.csv')
    new_log = not os.path.exists(log_path)
    with open(log_path, 'a') as f:
        if new_log:
            f.write('published_at,model_version,generic_ballot,house_dem_seats,house_dem_control_prob,'
                    'senate_dem_seats_median,senate_dem_control_prob\n')
        f.write(f"{summary['published_at']},{MODEL_VERSION},{generic_ballot},{house['dem_seats']},"
                f"{house['dem_control_prob']},{senate['dem_seats_median']},{senate['dem_control_prob']}\n")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote forecast files to {os.path.abspath(OUT_DIR)}")
    return summary

if __name__ == '__main__':
    publish()
