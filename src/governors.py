"""
Governor fundamentals model (roadmap Phase 2): the prior that 2026 governor polls are
blended with.

There's no governor results file in this project, so training races come from the
538 polls-vs-results archive, which records the actual result of every governor race
that was polled in its final ~60 days (most of them, 1998-2022). Unpolled blowouts are
missing, so the training set leans toward competitive races. That's acceptable, since
competitive races are the ones where the prior matters.

Same features as the Senate model (midterms.SENATE_FEATURES): state partisan lean,
incumbency (+1 D running / -1 R running / 0 open), national House environment, and the
incumbent's past over-performance. Governors run far ahead of or behind their state's
partisanship (Phil Scott in Vermont, Andy Beshear in Kentucky), so the lean coefficient
is learned separately here rather than borrowed from the Senate.

    python src/governors.py    # backtest + 2026 fundamentals
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import midterms as M
from pollster_ratings import prepare_polls, load_archive

FEATURES = M.SENATE_FEATURES
TRAIN_START_YEAR = 2000

def _state_names():
    races = M.senate_races_table()
    return dict(zip(races['state_po'], races['state']))

def archive_governor_races():
    """One row per regular governor race in the archive: cycle, state, actual D margin,
    winner party and name, and the two candidates' last names."""
    p = prepare_polls(load_archive())
    p = p[p['race'].str.split('_').str[1] == 'Gov-G']
    names = _state_names()
    rows = []
    for race, g in p.groupby('race'):
        r = g.iloc[0]
        d_first = r['cand1_party'] == 'DEM'
        dem_name, rep_name = (r['cand1_name'], r['cand2_name']) if d_first else (r['cand2_name'], r['cand1_name'])
        margin = r['actual_margin']
        rows.append({
            'cycle': int(r['cycle']), 'state_po': r['location'], 'state': names.get(r['location']),
            'margin': margin, 'winner': 'DEMOCRAT' if margin > 0 else 'REPUBLICAN',
            'winner_last': M._last_name(dem_name if margin > 0 else rep_name),
            'candidate_last_names': {M._last_name(dem_name), M._last_name(rep_name)},
        })
    return pd.DataFrame(rows).dropna(subset=['state']).sort_values(['state', 'cycle'])

def _previous(races, state, cycle):
    """Most recent earlier race for this governorship within 4 years (NH/VT have 2-year terms)."""
    prev = races[(races['state'] == state) & (races['cycle'] < cycle) & (races['cycle'] >= cycle - 4)]
    return prev.iloc[-1] if len(prev) else None

def governor_training_data():
    races = archive_governor_races()
    pres_lean = M.presidential_lean_by_year()
    house_margin = M.national_house_margin_by_year()
    rows = []
    for _, r in races[races['cycle'] >= TRAIN_START_YEAR].iterrows():
        if r['cycle'] % 2:          # odd-year races have no same-year House environment
            continue
        prev = _previous(races, r['state'], r['cycle'])
        if prev is None:
            continue
        running = prev['winner_last'] in r['candidate_last_names']
        incumbency = (1 if prev['winner'] == 'DEMOCRAT' else -1) if running else 0
        lean = M.pvi_style_lean(r['cycle'], r['state'], pres_lean)
        env = house_margin.get(r['cycle'])
        if lean is None or pd.isna(env):
            continue
        rows.append({'year': r['cycle'], 'state': r['state'], 'state_po': r['state_po'],
                     'margin': r['margin'], 'lean': lean, 'incumbency': incumbency,
                     'national_environment': env,
                     'incumbent_strength': M.incumbent_strength(prev['margin'], lean, incumbency)})
    return pd.DataFrame(rows)

def loco_predictions(train=None):
    train = (train if train is not None else governor_training_data()).copy()
    train['fund_pred'] = np.nan
    for year in train['year'].unique():
        test = train['year'] == year
        m = LinearRegression().fit(train.loc[~test, FEATURES], train.loc[~test, 'margin'])
        train.loc[test, 'fund_pred'] = m.predict(train.loc[test, FEATURES])
    return train

def evaluate(train=None):
    df = loco_predictions(train)
    resid = df['margin'] - df['fund_pred']
    year_miss = resid.groupby(df['year']).mean()
    return {
        'rmse':            float(np.sqrt((resid ** 2).mean())),
        'winner_accuracy': float(((df['fund_pred'] > 0) == (df['margin'] > 0)).mean()),
        'national_sd':     float(year_miss.std(ddof=1)),
        'state_sd':        float((resid - df['year'].map(year_miss)).std(ddof=1)),
        'n_races':         int(len(df)),
        'n_years':         int(df['year'].nunique()),
    }

def build_model():
    train = governor_training_data()
    model = LinearRegression().fit(train[FEATURES], train['margin'])
    return model, evaluate(train), M.presidential_lean_by_year()

def fundamentals_2026(generic_ballot_margin, model_bundle=None, race_status=None, archive_races=None):
    """Fundamentals-only margin for each 2026 governor race."""
    from race_status_2026 import load_or_scrape_race_status
    model, err, pres_lean = model_bundle or build_model()
    status = race_status if race_status is not None else load_or_scrape_race_status(office='governor')
    archive = archive_races if archive_races is not None else archive_governor_races()
    rows = []
    for _, s in status.iterrows():
        state = s['state']
        lean = M.pvi_style_lean(2026, state, pres_lean)
        party = s['last_election_party'] if isinstance(s['last_election_party'], str) else None
        if lean is None or party is None:
            rows.append({'state': state, 'fund_pred': np.nan, 'skip_reason': 'missing lean or incumbent party'})
            continue
        sign = 1 if party == 'DEMOCRAT' else -1
        incumbency = 0 if s['open_seat'] else sign
        prev = _previous(archive, state, 2026)
        strength = M.incumbent_strength(prev['margin'] if prev is not None else None, lean, incumbency)
        x = pd.DataFrame([[lean, incumbency, generic_ballot_margin, strength]], columns=FEATURES)
        rows.append({'state': state, 'incumbent_party': party, 'open_seat': bool(s['open_seat']),
                     'lean': round(lean, 1), 'fund_pred': float(model.predict(x)[0]),
                     'consensus_rating': s['consensus_rating'], 'skip_reason': None})
    return pd.DataFrame(rows), err

def validate_blend():
    from blend import _validate
    preds = loco_predictions().rename(columns={'year': 'cycle'})
    return _validate(preds[['cycle', 'state_po', 'fund_pred']], evaluate()['state_sd'], 'Gov-G')

if __name__ == '__main__':
    train = governor_training_data()
    model = LinearRegression().fit(train[FEATURES], train['margin'])
    print("Coefficients:", dict(zip(FEATURES, model.coef_.round(2))), "intercept", round(model.intercept_, 2))
    print("Backtest:", {k: round(v, 3) if isinstance(v, float) else v for k, v in evaluate(train).items()})
    summary, _ = validate_blend()
    print("\nGovernors: fundamentals vs polls vs blend (out of sample)")
    print(summary.to_string(index=False))
