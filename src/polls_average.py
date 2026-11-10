"""
Weighted, house-effect-corrected polling averages (roadmap Phase 1), plus a backtest of
the same averaging method against 2000-2022 Senate and governor results.

Each poll's margin (Dem side minus Rep) is first corrected for its pollster's house
effect. Campaign/party-sponsored polls (marked (D)/(R) on Wikipedia) are corrected by the
average lean of sponsored polls instead. Polls are then averaged with weights for:
  recency      half-life RECENCY_HALF_LIFE_DAYS; polls older than MAX_POLL_AGE_DAYS are dropped
  sample size  sqrt(n / 600), capped, so a 3,000-person poll counts ~2x a 750-person one
  quality      only bad track records are penalized. Grading showed accuracy barely
               persists across cycles except at the bottom (see pollster_ratings.py)
  volume       a pollster's polls share one pollster's worth of weight (1/sqrt(count))

Run as a script for the backtest:
    python src/polls_average.py
"""
import numpy as np
import pandas as pd

from pollster_ratings import (
    load_ratings, match_pollster, sponsor_party, prepare_polls, load_archive,
    compute_ratings, partisan_sponsor_effects,
)

RECENCY_HALF_LIFE_DAYS = 14
MAX_POLL_AGE_DAYS = 120
SPONSORED_WEIGHT = 0.5      # partisan polls count half, after their lean is corrected
# Corrections are applied at half strength. Backtest (final 3 weeks, 2000-2022 Senate+Gov):
# no correction 6.02 RMSE, full 6.07, half 6.03, all within noise. The archive has few
# sponsored polls (11%), but 46% of 2026 polls are sponsored and some races are lopsided,
# so a damped correction guards against that without leaning on an effect the backtest
# couldn't confirm.
CORRECTION_STRENGTH = 0.5
UNRATED_WEIGHT = 0.8        # pollsters with no track record

def _quality_weight(plus_minus):
    return float(np.exp(-0.5 * max(plus_minus, 0.0)))

def annotate_polls(polls, ratings=None, sponsor_effects=None):
    """Adds matched pollster, house-effect correction, quality weight, and adjusted margin."""
    ratings = ratings if ratings is not None else load_ratings()
    if sponsor_effects is None:
        sponsor_effects = partisan_sponsor_effects(prepare_polls(load_archive()))

    cache = {}
    rows = []
    for _, p in polls.iterrows():
        name = p['pollster']
        if name not in cache:
            m = match_pollster(name, ratings)
            cache[name] = (m['pollster'], float(m['house_effect']), float(m['plus_minus'])) if m is not None else (None, 0.0, None)
        matched, house, pm = cache[name]
        sponsor = p['sponsor'] if isinstance(p.get('sponsor'), str) else sponsor_party(name)
        if sponsor == 'INDEPENDENT':
            # An independent candidate's own poll leans toward them. They sit on the
            # Dem side of the margin when no Democrat is running (Osborn, Achilles).
            lean_toward_dem_side = p.get('dem_side_party') == 'INDEPENDENT'
            correction = sponsor_effects['DEMOCRAT'] if lean_toward_dem_side else 0.0
            qw = SPONSORED_WEIGHT
        elif sponsor:
            correction, qw = sponsor_effects[sponsor], SPONSORED_WEIGHT
        elif matched:
            correction, qw = house, _quality_weight(pm)
        else:
            correction, qw = 0.0, UNRATED_WEIGHT
        rows.append({'matched_pollster': matched, 'sponsor': sponsor,
                     'house_correction': correction, 'quality_weight': qw})
    added = pd.DataFrame(rows)
    out = pd.concat([polls.reset_index(drop=True).drop(columns=list(added.columns), errors='ignore'), added], axis=1)
    out['house_correction'] *= CORRECTION_STRENGTH
    out['adjusted_margin'] = out['margin'] - out['house_correction']
    return out

def _weights(p, as_of):
    """Each usable poll's weight before quality (recency, sample size, and the share of a
    pollster's polls) and with quality."""
    age = (as_of - p['end_date']).dt.days.clip(lower=0)
    recency = 0.5 ** (age / RECENCY_HALF_LIFE_DAYS)
    size = np.sqrt(p['sample_size'].fillna(600).clip(100, 3000) / 600)
    per_pollster = p.groupby('pollster')['pollster'].transform('size')
    base = recency * size / np.sqrt(per_pollster)
    return base, base * p['quality_weight']

def _usable(race_polls, as_of):
    p = race_polls.dropna(subset=['end_date'])
    return p[(p['end_date'] <= as_of) & (p['end_date'] >= as_of - pd.Timedelta(days=MAX_POLL_AGE_DAYS))]

def poll_shares(race_polls, as_of):
    """Each poll's share of its race's average as of a date (0 when too old to count)."""
    as_of = pd.Timestamp(as_of)
    p = _usable(race_polls, as_of)
    out = pd.Series(0.0, index=race_polls.index)
    if len(p):
        w = _weights(p, as_of)[1]
        if w.sum() > 0:
            out.loc[p.index] = w / w.sum()
    return out

def race_average(race_polls, as_of):
    """Weighted average of one race's annotated polls as of a date. None if no usable polls."""
    as_of = pd.Timestamp(as_of)
    p = _usable(race_polls, as_of)
    if p.empty:
        return None
    base, w = _weights(p, as_of)
    if w.sum() <= 0:
        return None
    # Effective number of polls: how many equally-weighted polls this average is worth,
    # scaled by average quality so that nine half-weight sponsored polls count as
    # about 4.5 polls, not nine.
    n_eff = base.sum() ** 2 / (base ** 2).sum() * (w.sum() / base.sum())
    return {
        'poll_avg':        float(np.average(p['adjusted_margin'], weights=w)),
        'raw_poll_avg':    float(np.average(p['margin'], weights=w)),
        'n_polls':         int(len(p)),
        'n_pollsters':     int(p['pollster'].nunique()),
        'effective_polls': float(n_eff),
        'latest_poll':     p['end_date'].max().date().isoformat(),
    }

def all_race_averages(polls, as_of=None):
    """One row per (office, state) with its polling average."""
    as_of = as_of or pd.Timestamp.today().normalize()
    annotated = annotate_polls(polls)
    rows = []
    for (office, state), g in annotated.groupby(['office', 'state']):
        avg = race_average(g, as_of)
        if avg:
            rows.append({'office': office, 'state': state,
                         'dem_candidate': g['dem_candidate'].iloc[-1],
                         'rep_candidate': g['rep_candidate'].iloc[-1],
                         'dem_side_party': g['dem_side_party'].iloc[-1], **avg})
    return pd.DataFrame(rows), annotated


# ── Backtest ──────────────────────────────────────────────────────────────────
def backtest_poll_averages(types=('Sen-G', 'Gov-G'), min_cycle=2000, days_before=0):
    """
    Rebuilds each historical race's final polling average with the method above and
    compares it to the result. House effects for each cycle come only from *other*
    cycles' ratings, so a pollster's 2016 miss can't be used to "correct" its 2016 polls.

    days_before: build the average as it stood this many days before the election (the
    archive covers roughly the final 60 days), to measure how much averages drift.

    Returns per-race errors and a summary, including the split between a miss shared by
    every race in a cycle (correlated, e.g. 2016/2020 polling misses) and the per-race miss.
    """
    polls = prepare_polls(load_archive())
    from pollster_ratings import add_herding
    polls = add_herding(polls)          # race-local, so computed once rather than per left-out cycle
    effects = partisan_sponsor_effects(polls)
    target = polls[polls['type_simple'].isin(types) & (polls['cycle'] >= min_cycle)]

    rows = []
    for cycle, cyc in target.groupby('cycle'):
        ratings = compute_ratings(polls[polls['cycle'] != cycle]).reset_index()
        house = ratings.set_index('pollster_rating_id')['house_effect']
        pm = ratings.set_index('pollster_rating_id')['plus_minus']
        c = cyc.copy()
        sponsored = c['partisan'].isin(['DEM', 'REP'])
        c['house_correction'] = np.where(
            sponsored, c['partisan'].map({'DEM': effects['DEMOCRAT'], 'REP': effects['REPUBLICAN']}),
            c['pollster_rating_id'].map(house).fillna(0.0))
        c['quality_weight'] = np.where(
            sponsored, SPONSORED_WEIGHT,
            c['pollster_rating_id'].map(pm).map(lambda v: _quality_weight(v) if pd.notna(v) else UNRATED_WEIGHT))
        c['margin'] = c['poll_margin']
        c['house_correction'] = c['house_correction'] * CORRECTION_STRENGTH
        c['adjusted_margin'] = c['margin'] - c['house_correction']
        c['end_date'] = pd.to_datetime(c['polldate'])
        c['pollster'] = c['pollster_rating_id']
        c['sample_size'] = c['samplesize']
        for race, g in c.groupby('race'):
            as_of = pd.to_datetime(g['electiondate'].iloc[0]) - pd.Timedelta(days=days_before)
            avg = race_average(g, as_of)
            if avg:
                rows.append({'cycle': cycle, 'race': race, 'type': g['type_simple'].iloc[0],
                             'location': g['location'].iloc[0], 'days_before': days_before,
                             'actual': g['actual_margin'].iloc[0], **avg})

    df = pd.DataFrame(rows)
    df['error'] = df['poll_avg'] - df['actual']
    df['raw_error'] = df['raw_poll_avg'] - df['actual']
    cycle_miss = df.groupby('cycle')['error'].mean()
    race_miss = df['error'] - df['cycle'].map(cycle_miss)
    summary = {
        'n_races':                   int(len(df)),
        'rmse_adjusted':             float(np.sqrt((df['error'] ** 2).mean())),
        'rmse_raw':                  float(np.sqrt((df['raw_error'] ** 2).mean())),
        'winner_accuracy':           float(((df['poll_avg'] > 0) == (df['actual'] > 0)).mean()),
        'cycle_shared_miss_sd':      float(cycle_miss.std(ddof=1)),
        'race_miss_sd':              float(race_miss.std(ddof=1)),
        'cycle_misses':              cycle_miss.round(2).to_dict(),
    }
    return df, summary

def fit_poll_error_model(df):
    """
    Fits the per-race error of a polling average (after removing the cycle-wide shared
    miss) as  error_sd = sqrt(floor + thin / n_eff + drift * days_to_election):
      floor  irreducible miss even with many polls on election eve
      thin   extra miss when an average rests on few polls
      drift  extra variance per day still to go, since opinion can move before election day
    `df` should pool backtests run at several days_before horizons.
    Least squares on squared residuals; coefficients floored at sensible minimums.
    """
    cycle_miss = df.groupby(['cycle', 'days_before'])['error'].transform('mean')
    e2 = (df['error'] - cycle_miss) ** 2
    X = np.column_stack([np.ones(len(df)), 1.0 / df['effective_polls'].clip(lower=0.5), df['days_before']])
    coef, *_ = np.linalg.lstsq(X, e2.values, rcond=None)
    return {'floor': float(max(coef[0], 1.0)), 'thin': float(max(coef[1], 0.0)),
            'drift': float(max(coef[2], 0.0))}

def poll_error_sd(effective_polls, model, days_to_election=0):
    """Per-race (non-shared) polling-average error."""
    return float(np.sqrt(model['floor'] + model['thin'] / max(effective_polls, 0.5)
                         + model.get('drift', 0.0) * max(days_to_election, 0)))

HORIZONS = (0, 14, 28, 42)

def load_poll_error_model(refresh=False):
    """The fitted error model, cached as JSON because fitting reruns the full
    multi-horizon backtest (a few minutes)."""
    import json, os
    from pollster_ratings import DATA_DIR
    path = os.path.join(DATA_DIR, 'poll_error_model.json')
    if not refresh and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    df, summaries = backtest_all_horizons()
    model = fit_poll_error_model(df)
    model['shared_miss_sd_by_days'] = {str(d): s['cycle_shared_miss_sd'] for d, s in summaries.items()}
    model['rmse_by_days'] = {str(d): s['rmse_adjusted'] for d, s in summaries.items()}
    with open(path, 'w') as f:
        json.dump(model, f, indent=2)
    return model

def backtest_all_horizons(**kw):
    frames, summaries = [], {}
    for d in HORIZONS:
        df, summ = backtest_poll_averages(days_before=d, **kw)
        frames.append(df)
        summaries[d] = summ
    return pd.concat(frames, ignore_index=True), summaries

if __name__ == '__main__':
    df, summaries = backtest_all_horizons()
    for d, summ in summaries.items():
        print(f"{d:2d} days out: {summ['n_races']} races, RMSE {summ['rmse_adjusted']:.2f}, "
              f"winners {summ['winner_accuracy']:.1%}, shared cycle miss sd {summ['cycle_shared_miss_sd']:.2f}")
    m = fit_poll_error_model(df)
    print("\nError model:", {k: round(v, 3) for k, v in m.items()})
    for d in HORIZONS:
        print(f"  {d:2d} days out:", {n: round(poll_error_sd(n, m, d), 2) for n in (1, 4, 16)})
