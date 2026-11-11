"""
Polls-primary forecast with fundamentals as the prior (roadmap Phase 2).

Each race has two estimates of the final margin:
  fundamentals  structural model (lean, incumbency, national environment), per-race sd
                from its leave-one-cycle-out backtest
  polls         weighted polling average, sd from polls_average's error model, which
                widens with fewer polls and more days to go
They're combined by inverse-variance weighting, the standard way to merge two noisy,
independent estimates. Weights follow the evidence instead of a hand-set schedule:
polls dominate a well-polled race near election day, fundamentals dominate an unpolled
or thinly polled one, and the shift happens on its own as polls accumulate and the
date nears.

Shared (national) errors are handled separately in the simulation, since they move
every race together and don't average out within a race.

Run as a script to validate the blend on 2000-2022 Senate and governor races:
    python src/blend.py
"""
import numpy as np
import pandas as pd

from polls_average import backtest_poll_averages, fit_poll_error_model, poll_error_sd, HORIZONS

# The cycle-wide shared polling miss, held flat at its election-eve size (2.7 pts).
# Polls alone do miss by more further out (3.6 pts at 4 weeks, 4.2 at 6), and until
# 2026.11.12 this grew to match. But the full model's own cycle-wide miss, after blending
# in fundamentals, stays about 2.7-3.0 pts at every horizon in the 2000-2022 backtest, so
# the growing version made early forecasts too uncertain (cycle misses at 28 days came in
# at about two-thirds of the assumed size). See forecasts/CHANGELOG.md, 2026.11.12.
SHARED_POLL_MISS_BASE = 2.7
SHARED_POLL_MISS_PER_DAY = 0.0

def shared_poll_miss_sd(days_to_election):
    return SHARED_POLL_MISS_BASE + SHARED_POLL_MISS_PER_DAY * max(days_to_election, 0)

def blend(fund_mean, fund_sd, poll_mean=None, poll_sd=None):
    """Inverse-variance blend. Returns (mean, race_sd, poll_weight)."""
    if poll_mean is None or poll_sd is None or pd.isna(poll_mean):
        return float(fund_mean), float(fund_sd), 0.0
    wp = 1.0 / poll_sd ** 2
    wf = 1.0 / fund_sd ** 2
    mean = (wp * poll_mean + wf * fund_mean) / (wp + wf)
    return float(mean), float(np.sqrt(1.0 / (wp + wf))), float(wp / (wp + wf))

def shared_sd(poll_weight, fund_national_sd, days_to_election):
    """A race's exposure to the national shared miss, weighted by what it relies on.
    Treated as fully correlated between the two sources (conservative)."""
    return poll_weight * shared_poll_miss_sd(days_to_election) + (1 - poll_weight) * fund_national_sd


# ── Validation ────────────────────────────────────────────────────────────────
def _validate(fund_preds, fund_race_sd, poll_type, horizons=HORIZONS):
    """
    fund_preds: DataFrame with cycle, state_po, fund_pred (out-of-sample predictions).
    Joins to historical polling averages of the same races at each horizon and scores
    fundamentals-only, polls-only, and blended forecasts against the actual result.
    The poll error model is refit leaving out the scored cycle's own races.
    """
    frames = [backtest_poll_averages(types=(poll_type,), days_before=d)[0] for d in horizons]
    polls = pd.concat(frames, ignore_index=True)
    polls = polls[polls['race'].str.split('_').str[1] == poll_type]   # regular races only
    joined = polls.merge(fund_preds, left_on=['cycle', 'location'], right_on=['cycle', 'state_po'])

    out = []
    for cycle in joined['cycle'].unique():
        model = fit_poll_error_model(polls[polls['cycle'] != cycle])
        for _, r in joined[joined['cycle'] == cycle].iterrows():
            psd = poll_error_sd(r['effective_polls'], model, r['days_before'])
            mean, _, w = blend(r['fund_pred'], fund_race_sd, r['poll_avg'], psd)
            out.append({**r.to_dict(), 'blend': mean, 'poll_weight': w})
    df = pd.DataFrame(out)

    rows = []
    for d, g in df.groupby('days_before'):
        row = {'days_before': d, 'races': len(g), 'mean_poll_weight': round(g['poll_weight'].mean(), 2)}
        for col, name in [('fund_pred', 'fundamentals'), ('poll_avg', 'polls'), ('blend', 'blend')]:
            err = g[col] - g['actual']
            row[f'{name}_rmse'] = round(float(np.sqrt((err ** 2).mean())), 2)
            row[f'{name}_winners'] = round(float(((g[col] > 0) == (g['actual'] > 0)).mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows), df

def validate_senate():
    import midterms as M
    train = M.senate_training_data()
    preds = M.senate_loco_predictions(train)
    po = M.senate_races_table()[['year', 'state', 'state_po']].drop_duplicates()
    preds = preds.merge(po, on=['year', 'state']).rename(columns={'year': 'cycle'})
    stats = M.evaluate_senate_model(train)
    return _validate(preds[['cycle', 'state_po', 'fund_pred']], stats['state_sd'], 'Sen-G')

if __name__ == '__main__':
    summary, _ = validate_senate()
    print("Senate, 2000-2022: fundamentals vs polls vs blend (out of sample)")
    print(summary.to_string(index=False))
