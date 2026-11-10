"""
Grade every published 2026 forecast snapshot against the actual results (roadmap Phase 4).

Results come from official state results, entered in forecasts/2026/results_2026.csv
(one row per race, with a link to the official source). Wikipedia race pages post these
quickly with citations to the state election offices, and nothing here depends on a
prediction-market-affiliated source.

Every snapshot is graded, not just the final one, so the record shows how the forecast
did at each point in the campaign. Each is compared with:
  polls only         our corrected polling average alone
  fundamentals only  the structural model alone
  experts            professional forecasters' consensus rating, converted to a
                     probability (see EXPERT_RATING_TO_PROB, a stated assumption)

    python src/grade_2026.py --template   # create results_2026.csv to fill in
    python src/grade_2026.py              # grade all snapshots, write forecasts/GRADES_2026.md
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_DIR = os.path.join(BASE_DIR, '..', 'forecasts', '2026')
RESULTS_CSV = os.path.join(FORECAST_DIR, 'results_2026.csv')
GRADES_MD = os.path.join(BASE_DIR, '..', 'forecasts', 'GRADES_2026.md')
GRADES_CSV = os.path.join(FORECAST_DIR, 'grades_2026.csv')
PROB_FLOOR = 0.005

# Consensus ratings (Safe 30, Likely 15, Lean 7, Tilt 3, Tossup 0, averaged across
# forecasters) mapped to win probabilities. Assumption, stated up front: roughly the
# historical hit rates of published ratings (Safe ~97%, Likely ~85%, Lean ~70%).
EXPERT_RATING_TO_PROB = [(0, 0.5), (3, 0.6), (7, 0.7), (15, 0.85), (30, 0.97)]
RESULT_COLUMNS = ['office', 'state', 'district', 'dem_side_candidate', 'rep_candidate',
                  'dem_side_pct', 'rep_pct', 'winner_party', 'source_url', 'entered_at']
# Only a party in winner_party decides a race. Anything else (e.g. "RUNOFF") is still pending:
# it isn't a loss for either side, so it's left out of grading and election-night calls.
WINNER_PARTIES = ('DEMOCRAT', 'REPUBLICAN', 'INDEPENDENT')
# The final forecast is the last snapshot published on or before election day; a snapshot
# from a run after polls closed isn't graded.
ELECTION_DATE = '2026-11-03'

def is_decided(winner_party):
    return winner_party.astype('string').str.strip().str.upper().isin(WINNER_PARTIES).fillna(False).astype(bool)

def expert_prob(rating):
    if rating is None or pd.isna(rating):
        return np.nan
    xs, ys = zip(*EXPERT_RATING_TO_PROB)
    p = float(np.interp(abs(rating), xs, ys))
    return p if rating >= 0 else 1 - p

# ── Results file ──────────────────────────────────────────────────────────────
def graded_dates():
    dates = sorted(os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(FORECAST_DIR, '*_summary.json')))
    return [d for d in dates if d <= ELECTION_DATE]

def latest_snapshot_date():
    files = sorted(glob.glob(os.path.join(FORECAST_DIR, '*_summary.json')))
    return os.path.basename(files[-1]).split('_')[0] if files else None

def make_results_template():
    """One blank row per race in the latest snapshot. Won't overwrite entered results."""
    if os.path.exists(RESULTS_CSV):
        raise FileExistsError(f"{RESULTS_CSV} already exists; not overwriting entered results")
    d = latest_snapshot_date()
    rows = []
    for office in ('senate', 'governor'):
        f = pd.read_csv(os.path.join(FORECAST_DIR, f'{d}_{office}_races.csv'))
        for _, r in f.iterrows():
            rows.append({'office': office, 'state': r['state'], 'district': None,
                         'dem_side_candidate': r['dem_candidate'], 'rep_candidate': r['rep_candidate']})
    h = pd.read_csv(os.path.join(FORECAST_DIR, f'{d}_house_districts.csv'))
    for _, r in h.iterrows():
        rows.append({'office': 'house', 'state': r['state'], 'district': int(r['district']),
                     'dem_side_candidate': r['dem_candidate'], 'rep_candidate': r['rep_candidate']})
    out = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    out.to_csv(RESULTS_CSV, index=False)
    return RESULTS_CSV

def load_results():
    r = pd.read_csv(RESULTS_CSV)
    r = r[is_decided(r['winner_party'])].copy()
    r['winner_party'] = r['winner_party'].str.strip().str.upper()
    r['district'] = r['district'].astype('Int64')
    return r

# ── Grading ───────────────────────────────────────────────────────────────────
def _snapshot(date):
    races = []
    for office in ('senate', 'governor'):
        path = os.path.join(FORECAST_DIR, f'{date}_{office}_races.csv')
        if os.path.exists(path):
            f = pd.read_csv(path).assign(office=office, district=pd.NA)
            f['model_p'] = f['dem_side_win_prob']
            f['experts_p'] = f['consensus_rating'].map(expert_prob)
            if 'poll_sd' in f:
                f['polls_p'] = np.where(f['poll_avg'].notna(),
                                        norm.cdf(f['poll_avg'] / np.hypot(f['poll_sd'], f['poll_shared_sd'])), np.nan)
                f['fund_p'] = norm.cdf(f['fund_pred'] / np.hypot(f['fund_sd'], f['fund_shared_sd']))
            races.append(f)
    path = os.path.join(FORECAST_DIR, f'{date}_house_districts.csv')
    if os.path.exists(path):
        h = pd.read_csv(path).assign(office='house', dem_side_party='DEMOCRAT')
        h['model_p'] = h['dem_win_prob']
        if 'consensus_rating' in h:
            h['experts_p'] = h['consensus_rating'].map(expert_prob)
        races.append(h)
    out = pd.concat(races, ignore_index=True)
    out['district'] = pd.to_numeric(out['district'], errors='coerce').astype('Int64')
    return out

def _scores(df, col):
    d = df.dropna(subset=[col])
    if d.empty:
        return None
    p = d[col].clip(PROB_FLOOR, 1 - PROB_FLOOR)
    y = d['won'].astype(float)
    return {'races': int(len(d)), 'brier': float(((p - y) ** 2).mean()),
            'log_loss': float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
            'winners_right': float(((p >= 0.5) == (y == 1)).mean())}

def calibration(df, col='model_p'):
    p_fav = np.maximum(df[col], 1 - df[col])
    fav_won = np.where(df[col] >= 0.5, df['won'], ~df['won'].astype(bool))
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]
    labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-95%', '95-99%', '99%+']
    t = pd.DataFrame({'bucket': pd.cut(p_fav, bins, labels=labels, right=False), 'p': p_fav, 'won': fav_won.astype(float)})
    return t.groupby('bucket', observed=True).agg(races=('won', 'size'), forecast=('p', 'mean'), actual=('won', 'mean')).reset_index()

def grade_all(results=None):
    results = results if results is not None else load_results()
    rows, final = [], None
    for date in graded_dates():
        snap = _snapshot(date).merge(results[['office', 'state', 'district', 'winner_party']],
                                     on=['office', 'state', 'district'], how='inner')
        snap['won'] = snap['winner_party'] == snap['dem_side_party']
        for office, g in [('all', snap)] + list(snap.groupby('office')):
            for method, col in (('model', 'model_p'), ('polls only', 'polls_p'),
                                ('fundamentals only', 'fund_p'), ('experts', 'experts_p')):
                if col in g:
                    s = _scores(g, col)
                    if s:
                        rows.append({'snapshot': date, 'office': office, 'method': method, **s})
        final = snap
    return pd.DataFrame(rows), final

def chamber_outcomes(results):
    """Actual Senate and House control from the results (Dem-caucus seats). Control is reported
    once a party has clinched it, even if some races aren't entered yet."""
    out = {}
    with open(os.path.join(FORECAST_DIR, f'{graded_dates()[-1]}_summary.json')) as f:
        summ = json.load(f)
    sen = results[results['office'] == 'senate']
    if len(sen):
        dem = summ['senate']['holdovers']['DEMOCRAT'] + int((sen['winner_party'] == 'DEMOCRAT').sum())
        rep = summ['senate']['holdovers']['REPUBLICAN'] + int((sen['winner_party'] == 'REPUBLICAN').sum())
        out['senate_dem_seats'] = dem
        if dem >= 51 or rep >= 50:
            out['senate_dem_control'] = dem >= 51
    house = results[results['office'] == 'house']
    if len(house):
        dem = int((house['winner_party'] == 'DEMOCRAT').sum())
        rep = int((house['winner_party'] == 'REPUBLICAN').sum())
        out['house_dem_seats'] = dem
        if dem >= 218 or rep >= 218:
            out['house_dem_control'] = dem >= 218
    return out

def write_report(scores, final, results):
    def table(df, cols):
        """cols: [(header, column, formatter or None)]"""
        nl = '\n'
        head = '| ' + ' | '.join(c[0] for c in cols) + ' |' + nl + '|' + '|'.join('---' for _ in cols) + '|' + nl
        body = ''.join('| ' + ' | '.join((fmt or str)(r[col]) for _, col, fmt in cols) + ' |' + nl
                       for _, r in df.iterrows())
        return head + body
    pct = lambda v: f"{v:.0%}"
    f3 = lambda v: f"{v:.3f}"
    last = scores['snapshot'].max()
    fin = scores[(scores['snapshot'] == last)]
    # results holds decided races only; re-read the file so the report says what's still out.
    all_rows = pd.read_csv(RESULTS_CSV)
    pending = all_rows[~is_decided(all_rows['winner_party'])]
    md = [f"# 2026 Forecast Grades\n\n*Generated by `src/grade_2026.py` from official results in "
          f"`forecasts/2026/results_2026.csv` ({len(results)} races decided).*\n"]
    if not pending.empty:
        names = [f"{r['office'].title()} · {str(r['state']).title()}"
                 + ((f"-{int(r['district'])}" if r['district'] else '-AL') if pd.notna(r['district']) else '')
                 for _, r in pending.iterrows()]
        md.append(f"> **Preliminary: {len(pending)} of {len(all_rows)} races still pending** "
                  f"(uncalled or headed to a runoff). These numbers will change once they're in.\n>\n"
                  f"> Pending: {', '.join(names)}\n")
    md.append(f"## Final forecast ({last})\n")
    md.append(table(fin, [('Office', 'office', None), ('Method', 'method', None), ('Races', 'races', None),
                          ('Winners right', 'winners_right', pct), ('Brier', 'brier', f3), ('Log loss', 'log_loss', f3)]))
    md.append("\nBrier score and log loss: lower is better. The expert line converts consensus ratings to "
              "probabilities with an assumed scale (Safe ≈ 97%, Likely ≈ 85%, Lean ≈ 70%).\n")
    ch = chamber_outcomes(results)
    if ch:
        md.append("## Chamber control\n")
        for k, v in ch.items():
            md.append(f"- {k.replace('_', ' ')}: **{v}**")
    md.append("\n## How the forecast did over the campaign (model, all races)\n")
    over = scores[(scores['office'] == 'all') & (scores['method'] == 'model')]
    md.append(table(over, [('Snapshot', 'snapshot', None), ('Races', 'races', None),
                           ('Winners right', 'winners_right', pct), ('Brier', 'brier', f3)]))
    md.append("\n## Calibration of the final forecast\n")
    md.append(table(calibration(final), [('Forecast chance for favorite', 'bucket', str), ('Races', 'races', None),
                                         ('Average forecast', 'forecast', pct), ('Favorite won', 'actual', pct)]))
    misses = final[(final['model_p'] >= 0.5) != final['won']].copy()
    misses['conf'] = np.maximum(misses['model_p'], 1 - misses['model_p'])
    md.append("\n## Misses (final forecast), most confident first\n")
    if misses.empty:
        md.append("None.")
    else:
        misses = misses.sort_values('conf', ascending=False)
        for _, m in misses.iterrows():
            race = f"{m['office'].title()} · {m['state'].title()}" + (f"-{m['district']}" if pd.notna(m['district']) else '')
            md.append(f"- {race}: forecast {m['conf']:.0%} for the "
                      f"{'Democratic side' if m['model_p'] >= 0.5 else 'Republican'}, won by {m['winner_party'].title()}")
    with open(GRADES_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    scores.to_csv(GRADES_CSV, index=False)
    return GRADES_MD

if __name__ == '__main__':
    if '--template' in sys.argv:
        print("Created", make_results_template())
    else:
        results = load_results()
        scores, final = grade_all(results)
        print(scores[scores['snapshot'] == scores['snapshot'].max()].round(3).to_string(index=False))
        print("Wrote", write_report(scores, final, results))
