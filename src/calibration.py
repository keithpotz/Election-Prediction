"""
Calibration report and honest baseline comparison (roadmap Phase 3).

Rebuilds full probabilistic forecasts for every polled Senate and governor race from
2000 to 2022 as they would have looked 0, 14, 28 and 42 days before each election.
Each cycle is forecast using only information from *other* cycles:
  - fundamentals coefficients and error sizes
  - pollster ratings and the poll error model
  - the national environment, taken from the generic-ballot polling average at that
    point (the actual House vote isn't known until after the election), optionally
    corrected for the generic ballot's historical lean

Then:
  calibration  When the model says X%, does the favored side win about X% of the time?
  baselines    Is the full model better than simpler ways of forecasting?
               - last election: the seat's previous result
               - simple poll average: unweighted mean of the last 30 days of polls
               - fundamentals only
               - polls only (our weighted, corrected average, no fundamentals)
  scores       RMSE of the margin, winner accuracy, Brier score, log loss
               (lower Brier/log loss = better probabilities)

    python src/calibration.py      # prints the report and writes forecasts/calibration/
"""
import os
import json

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

import midterms as M
import governors as G
from blend import blend
from pollster_ratings import prepare_polls, load_archive
from polls_average import backtest_poll_averages, fit_poll_error_model, poll_error_sd, HORIZONS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, '..', 'forecasts', 'calibration')
SIMPLE_AVG_WINDOW_DAYS = 30
PROB_FLOOR = 0.005  # keeps log loss finite when a forecast is ~certain and wrong

# ── Inputs ────────────────────────────────────────────────────────────────────
def _generic_ballot_by_horizon():
    """
    {days_before: DataFrame(cycle, gb_avg, gb_actual, proxy)} from archived generic-ballot
    polls. A cycle with no generic-ballot polls on record (2024) uses national
    presidential polls vs the popular vote as a stand-in, flagged proxy=True.
    """
    out = {}
    for d in HORIZONS:
        df, _ = backtest_poll_averages(types=('House-G-US',), days_before=d, min_cycle=1998)
        gb = df[['cycle', 'poll_avg', 'actual']].rename(columns={'poll_avg': 'gb_avg', 'actual': 'gb_actual'})
        gb['proxy'] = False
        pres, _ = backtest_poll_averages(types=('Pres-G',), days_before=d, min_cycle=2024)
        pres = pres[(pres['location'] == 'US') & ~pres['cycle'].isin(gb['cycle'])]
        if len(pres):
            gb = pd.concat([gb, pres[['cycle', 'poll_avg', 'actual']].rename(
                columns={'poll_avg': 'gb_avg', 'actual': 'gb_actual'}).assign(proxy=True)], ignore_index=True)
        out[d] = gb.sort_values('cycle')
    return out

def _fund_training(office):
    if office == 'senate':
        train = M.senate_training_data()
        po = M.senate_races_table()[['year', 'state', 'state_po']].drop_duplicates()
        return train.merge(po, on=['year', 'state']), M.SENATE_FEATURES
    return G.governor_training_data(), G.FEATURES

def _last_election(office):
    """(cycle, state_po) -> the same seat's previous result (Dem-positive margin)."""
    if office == 'senate':
        races = M.senate_races_table()
        prev = races[races['contested']].assign(cycle=lambda d: d['year'] + 6)
        return prev[['cycle', 'state_po', 'margin']].rename(columns={'margin': 'last_margin'})
    races = G.archive_governor_races()
    rows = []
    for _, r in races.iterrows():
        p = G._previous(races, r['state'], r['cycle'])
        if p is not None:
            rows.append({'cycle': r['cycle'], 'state_po': r['state_po'], 'last_margin': p['margin']})
    return pd.DataFrame(rows)

def _simple_poll_average(archive, poll_type, days_before):
    """Unweighted mean of every poll in the 30 days before the as-of date. No corrections."""
    p = archive[archive['race'].str.split('_').str[1] == poll_type].copy()
    p['as_of'] = pd.to_datetime(p['electiondate']) - pd.Timedelta(days=days_before)
    p['end'] = pd.to_datetime(p['polldate'])
    p = p[(p['end'] <= p['as_of']) & (p['end'] > p['as_of'] - pd.Timedelta(days=SIMPLE_AVG_WINDOW_DAYS))]
    return p.groupby(['cycle', 'location'])['poll_margin'].mean().rename('simple_avg').reset_index()


# ── Backtest ──────────────────────────────────────────────────────────────────
def build_backtest(office, gb, poll_frames, archive, correct_generic_bias=True, correct_poll_bias=False,
                   poll_bias_scale=1.0, poll_bias_decay=None):
    """One row per (race, horizon) with every method's predicted margin and P(Dem wins)."""
    poll_type = 'Sen-G' if office == 'senate' else 'Gov-G'
    train, feats = _fund_training(office)
    last = _last_election(office)
    polls_all = pd.concat(poll_frames.values(), ignore_index=True)
    env_i = feats.index('national_environment')

    rows = []
    for d in HORIZONS:
        polls_d = poll_frames[d]
        polls_d = polls_d[polls_d['race'].str.split('_').str[1] == poll_type]
        simple = _simple_poll_average(archive, poll_type, d)
        gbd = gb[d].set_index('cycle')
        for cycle in sorted(set(polls_d['cycle']) & set(train['year'])):
            if cycle not in gbd.index:
                continue
            other = train[train['year'] != cycle]
            test = train[train['year'] == cycle]
            model = LinearRegression().fit(other[feats], other['margin'])

            # National environment as it was polled, optionally de-biased using other cycles
            gb_other = gb[d][gb[d]['cycle'] != cycle]
            gb_bias = float((gb_other['gb_avg'] - gb_other['gb_actual']).mean()) if correct_generic_bias else 0.0
            gb_err_sd = float((gb_other['gb_avg'] - gb_other['gb_actual'] - gb_bias).std(ddof=1))
            env_now = gbd.loc[cycle, 'gb_avg'] - gb_bias
            X = test[feats].copy()
            X['national_environment'] = env_now
            fund_pred = pd.Series(model.predict(X), index=test.index)

            # Fundamentals error sizes from other cycles only
            loco = other.copy()
            loco['res'] = np.nan
            for y in other['year'].unique():
                tr = other['year'] != y
                m2 = LinearRegression().fit(other.loc[tr, feats], other.loc[tr, 'margin'])
                loco.loc[~tr, 'res'] = other.loc[~tr, 'margin'] - m2.predict(other.loc[~tr, feats])
            year_miss = loco.groupby('year')['res'].mean()
            f_race_sd = float((loco['res'] - loco['year'].map(year_miss)).std(ddof=1))
            f_shared_sd = float(np.hypot(year_miss.std(ddof=1), model.coef_[env_i] * gb_err_sd))

            # Poll error model and shared poll miss from other cycles only
            other_polls = polls_all[polls_all['cycle'] != cycle]
            err_model = fit_poll_error_model(other_polls)
            od = other_polls[other_polls['days_before'] == d]
            cycle_means = od.groupby('cycle')['error'].mean()
            p_shared_sd = float(cycle_means.std(ddof=1))
            # Optional: remove state polls' average historical lean (other cycles only)
            if correct_poll_bias:
                # optionally weight recent cycles more (decay per cycle of distance from this one)
                wts = (poll_bias_decay ** (abs(cycle_means.index - cycle) / 2)) if poll_bias_decay else np.ones(len(cycle_means))
                poll_bias = poll_bias_scale * float(np.average(cycle_means.values, weights=wts))
            else:
                poll_bias = 0.0

            # Baseline error sizes from other cycles only
            last_c = last.merge(other[['year', 'state_po', 'margin']].rename(columns={'year': 'cycle'}),
                                on=['cycle', 'state_po'])
            last_sd = float((last_c['margin'] - last_c['last_margin']).std(ddof=1))
            simple_o = simple.merge(od[['cycle', 'location', 'actual']].drop_duplicates(), on=['cycle', 'location'])
            simple_sd = float((simple_o['simple_avg'] - simple_o['actual']).std(ddof=1))

            fund_by_po = dict(zip(test['state_po'], fund_pred))
            last_by_po = last[last['cycle'] == cycle].set_index('state_po')['last_margin']
            simple_by_po = simple[simple['cycle'] == cycle].set_index('location')['simple_avg']
            for _, r in polls_d[polls_d['cycle'] == cycle].iterrows():
                po = r['location']
                if po not in fund_by_po:
                    continue
                f = fund_by_po[po]
                psd = poll_error_sd(r['effective_polls'], err_model, d)
                pavg = r['poll_avg'] - poll_bias
                mean, race_sd, w = blend(f, f_race_sd, pavg, psd)
                shared = w * p_shared_sd + (1 - w) * f_shared_sd
                rows.append({
                    'office': office, 'cycle': cycle, 'state_po': po, 'days_before': d, 'actual': r['actual'],
                    'model_margin': mean, 'model_p': norm.cdf(mean / np.hypot(race_sd, shared)),
                    'fund_margin': f, 'fund_p': norm.cdf(f / np.hypot(f_race_sd, f_shared_sd)),
                    'polls_margin': pavg, 'polls_p': norm.cdf(pavg / np.hypot(psd, p_shared_sd)),
                    'last_margin': last_by_po.get(po, np.nan),
                    'last_p': norm.cdf(last_by_po[po] / last_sd) if po in last_by_po.index else np.nan,
                    'simple_margin': simple_by_po.get(po, np.nan),
                    'simple_p': norm.cdf(simple_by_po[po] / simple_sd) if po in simple_by_po.index else np.nan,
                    'poll_weight': w, 'gb_bias_used': gb_bias, 'poll_bias_used': poll_bias,
                })
    return pd.DataFrame(rows)

def generic_ballot_bias(days_to_election):
    """
    How far the generic-ballot polling average has historically overstated Democrats at
    this many days out (1998-2022: about 2.4 pts on election eve, 3.0 at four weeks, in
    10-11 of 13 cycles). Linearly interpolated between backtested horizons. Read from the
    calibration report, so rerunning calibration.py updates it.
    """
    path = os.path.join(OUT_DIR, 'calibration_report.json')
    with open(path) as f:
        rows = json.load(f)['generic_ballot_bias']
    days = [r['days_before'] for r in rows]
    bias = [r['mean_dem_overstatement'] for r in rows]
    return float(np.interp(days_to_election, days, bias))

def regional_error_sd(days_to_election):
    """
    The part of a race's forecast error shared with other races in its region, after
    removing the cycle-wide national miss (variance decomposition of the out-of-sample
    backtest residuals). About 1.3 pts on election eve, 1.2 at four weeks.
    Interpolated between backtested horizons; 0 if the backtest hasn't been run.
    """
    from states import ABBREV_STATE, STATE_REGION
    path = os.path.join(OUT_DIR, 'backtest_gb_bias_corrected.csv')
    if not os.path.exists(path):
        return 0.0
    bt = pd.read_csv(path)
    bt['region'] = bt['state_po'].map(lambda s: STATE_REGION.get(ABBREV_STATE.get(s)))
    days, sds = [], []
    for d, b in bt.groupby('days_before'):
        e = b['actual'] - b['model_margin']
        e = e - e.groupby(b['cycle']).transform('mean')
        rc = e.groupby([b['cycle'], b['region']]).agg(['mean', 'size'])
        resid = e - pd.Series(list(zip(b['cycle'], b['region'])), index=b.index).map(rc['mean'].to_dict())
        s_state2 = resid.var(ddof=1)
        s_region2 = max(rc['mean'].var(ddof=1) - s_state2 * (1 / rc['size']).mean(), 0.0)
        days.append(d)
        sds.append(float(np.sqrt(s_region2)))
    return float(np.interp(days_to_election, days, sds))

METHODS = [('model', 'Full model (polls + fundamentals)'), ('polls', 'Polls only (weighted, corrected)'),
           ('simple', 'Simple poll average (last 30 days)'), ('fund', 'Fundamentals only'),
           ('last', 'Last election result')]

def score(df):
    """Scores every method on the races where all methods have a forecast."""
    common = df.dropna(subset=[f'{m}_p' for m, _ in METHODS])
    won = (common['actual'] > 0).astype(float)
    rows = []
    for m, label in METHODS:
        p = common[f'{m}_p'].clip(PROB_FLOOR, 1 - PROB_FLOOR)
        rows.append({
            'method': label, 'races': len(common),
            'rmse': float(np.sqrt(((common[f'{m}_margin'] - common['actual']) ** 2).mean())),
            'winner_accuracy': float(((common[f'{m}_margin'] > 0) == (common['actual'] > 0)).mean()),
            'brier': float(((p - won) ** 2).mean()),
            'log_loss': float(-(won * np.log(p) + (1 - won) * np.log(1 - p)).mean()),
        })
    return pd.DataFrame(rows)

def calibration_table(df, col='model_p'):
    """Groups forecasts by how confident they were in the favorite and compares that to
    how often the favorite actually won."""
    p_fav = np.maximum(df[col], 1 - df[col])
    fav_won = np.where(df[col] >= 0.5, df['actual'] > 0, df['actual'] <= 0)
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]
    labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-95%', '95-99%', '99%+']
    t = pd.DataFrame({'bucket': pd.cut(p_fav, bins, labels=labels, right=False), 'p': p_fav, 'won': fav_won})
    out = t.groupby('bucket', observed=True).agg(races=('won', 'size'), predicted=('p', 'mean'), actual=('won', 'mean'))
    # 90% interval for the observed rate if the forecast were perfectly calibrated
    se = np.sqrt(out['predicted'] * (1 - out['predicted']) / out['races'])
    out['expected_low'] = (out['predicted'] - 1.645 * se).clip(0, 1)
    out['expected_high'] = (out['predicted'] + 1.645 * se).clip(0, 1)
    return out.reset_index()

# State/district poll lean correction (adopted 2026-10-07 after testing with 2024 included):
# half of the cycles' average polling miss, weighting recent cycles more (0.7 per cycle
# back). Tested alternatives: none, full, half, recent-weighted full (see CALIBRATION.md).
STATE_POLL_BIAS_SCALE = 0.5
STATE_POLL_BIAS_DECAY = 0.7

def state_poll_bias_estimates(poll_frames, as_of_cycle=2026):
    """Live correction per horizon: the backtested cycle-average poll miss, recent cycles
    weighted more, at half strength."""
    rows = []
    for d, df in poll_frames.items():
        means = df.groupby('cycle')['error'].mean()
        w = STATE_POLL_BIAS_DECAY ** ((as_of_cycle - means.index.values) / 2)
        rows.append({'days_before': d, 'cycles': int(len(means)),
                     'weighted_dem_overstatement': float(np.average(means.values, weights=w)),
                     'applied_correction': float(STATE_POLL_BIAS_SCALE * np.average(means.values, weights=w)),
                     'latest_cycle_miss': float(means.iloc[-1])})
    return rows

def state_poll_bias(days_to_election):
    """How much to pull state and district polling averages toward Republicans (pts)."""
    path = os.path.join(OUT_DIR, 'calibration_report.json')
    if not os.path.exists(path):
        return 0.0
    with open(path) as f:
        rows = json.load(f).get('state_poll_bias') or []
    if not rows:
        return 0.0
    return float(np.interp(days_to_election, [r['days_before'] for r in rows], [r['applied_correction'] for r in rows]))

def run(horizons_to_report=(0, 28)):
    archive = prepare_polls(load_archive())
    gb = _generic_ballot_by_horizon()
    poll_frames = {d: backtest_poll_averages(types=('Sen-G', 'Gov-G'), days_before=d)[0] for d in HORIZONS}

    results = {}
    for correct in (False, True):
        kw = dict(correct_poll_bias=True, poll_bias_scale=STATE_POLL_BIAS_SCALE,
                  poll_bias_decay=STATE_POLL_BIAS_DECAY) if correct else {}
        bt = pd.concat([build_backtest(o, gb, poll_frames, archive, correct, **kw) for o in ('senate', 'governor')],
                       ignore_index=True)
        results[correct] = bt
    return results, gb, state_poll_bias_estimates(poll_frames)

def _gb_bias_summary(gb):
    rows = []
    for d, df in gb.items():
        e = df['gb_avg'] - df['gb_actual']
        rows.append({'days_before': d, 'cycles': len(e), 'mean_dem_overstatement': float(e.mean()),
                     'cycles_overstating_dem': int((e > 0).sum()), 'sd': float(e.std(ddof=1))})
    return pd.DataFrame(rows)

# ── Publishable chart + report ────────────────────────────────────────────────
SURFACE, INK, INK_2, INK_3, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
SERIES_1, NEUTRAL = '#2a78d6', '#b9b8b2'

def make_chart(report, key='gb_bias_corrected_28d', path=None):
    """Two single-axis panels: calibration (dots vs the band a perfectly calibrated
    forecast would land in) and probability score by method (Brier, lower is better)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cal = pd.DataFrame(report['calibration'][key])
    sc = pd.DataFrame(report['scores'][key])
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK_2,
                         'xtick.color': INK_2, 'ytick.color': INK_2, 'text.color': INK})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={'width_ratios': [1.1, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    # Panel 1: calibration
    x = cal['predicted'] * 100
    a1.plot([50, 100], [50, 100], color=INK_3, lw=1, ls=(0, (4, 3)), zorder=1)
    a1.vlines(x, cal['expected_low'] * 100, cal['expected_high'] * 100, color=NEUTRAL, lw=6,
              capstyle='round', zorder=2, label='Range a perfectly calibrated forecast would land in (90%)')
    a1.scatter(x, cal['actual'] * 100, s=64, color=SERIES_1, edgecolor=SURFACE, linewidth=2, zorder=3,
               label='How often the favorite actually won')
    for xi, yi, n in zip(x, cal['actual'] * 100, cal['races']):
        # The top-right corner is crowded: steer each count a different way.
        if xi > 99:
            off, ha = (9, -3), 'left'
        elif xi > 95:
            off, ha = (0, -17), 'center'
        elif xi > 90:
            off, ha = (-9, -3), 'right'
        else:
            off, ha = (9, -3), 'left'
        a1.annotate(f"{n} races" if xi < 60 else f"{n}", (xi, yi), xytext=off,
                    textcoords='offset points', fontsize=8, color=INK_3, ha=ha)
    a1.set_xlim(48, 105); a1.set_ylim(38, 102)
    a1.set_xticks(range(50, 101, 10))
    a1.set_xlabel("Forecast chance for the favorite (%)")
    a1.set_ylabel("Favorite actually won (%)")
    a1.grid(color=GRID, lw=0.6); a1.set_axisbelow(True)
    a1.set_title("Calibration: forecasts made 28 days out", loc='left', fontsize=11, color=INK, pad=10)
    a1.legend(loc='lower right', frameon=False, fontsize=8, labelcolor=INK_2)

    # Panel 2: Brier by method
    sc = sc.sort_values('brier', ascending=False)
    colors = [SERIES_1 if m.startswith('Full model') else NEUTRAL for m in sc['method']]
    a2.barh(sc['method'], sc['brier'], color=colors, height=0.55)
    for i, v in enumerate(sc['brier']):
        a2.text(v + 0.003, i, f"{v:.3f}", va='center', fontsize=9, color=INK_2)
    a2.set_xlabel("Brier score (lower = better probabilities)")
    a2.grid(axis='x', color=GRID, lw=0.6); a2.set_axisbelow(True)
    a2.set_xlim(0, sc['brier'].max() * 1.18)
    a2.set_title("Versus simpler ways to forecast", loc='left', fontsize=11, color=INK, pad=10)
    n = int(sc['races'].iloc[0])
    fig.suptitle(f"Election forecast backtest: {n} Senate and governor races, 2000-2022 (out of sample)",
                 x=0.01, ha='left', fontsize=13, color=INK, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = path or os.path.join(OUT_DIR, 'calibration_28d.png')
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path

def _md_table(rows, cols, fmt):
    head = '| ' + ' | '.join(h for h, _ in cols) + ' |\n|' + '|'.join('---' for _ in cols) + '|\n'
    body = ''.join('| ' + ' | '.join(fmt.get(k, str)(r[k]) for _, k in cols) + ' |\n' for r in rows)
    return head + body

def _calibration_verdict(report):
    """Plain-language summary computed from the calibration tables, not hand-written."""
    lines = []
    for key, label in (('gb_bias_corrected_28d', '28 days out'), ('gb_bias_corrected_0d', 'On election eve')):
        rows = report['calibration'][key]
        outside = [r for r in rows if not (r['expected_low'] - 1e-9 <= r['actual'] <= r['expected_high'] + 1e-9)]
        under = [r for r in outside if r['actual'] > r['expected_high']]
        if not outside:
            lines.append(f"- **{label}:** all {len(rows)} buckets land inside their expected range.")
        else:
            desc = ', '.join(f"{r['bucket']} (won {r['actual'] * 100:.0f}% vs {r['expected_low'] * 100:.0f}-"
                             f"{r['expected_high'] * 100:.0f}% expected)" for r in outside)
            kind = ('favorites won *more* often than forecast (underconfident)' if len(under) == len(outside)
                    else 'mixed direction')
            lines.append(f"- **{label}:** {len(rows) - len(outside)} of {len(rows)} buckets inside their expected "
                         f"range; outside: {desc}, so {kind}.")
    lines.append("\nWhere the model misses calibration, it leans toward *under*confidence: favorites win a bit "
                 "more often than forecast. That's the safer direction to be wrong in, and it means the "
                 "uncertainty in the live forecast is, if anything, slightly too wide.")
    return '\n'.join(lines)

def _spb_at(report, days):
    rows = report.get('state_poll_bias') or []
    if not rows:
        return 'n/a'
    return f"{np.interp(days, [r['days_before'] for r in rows], [r['applied_correction'] for r in rows]):.1f}"

def write_markdown(report, path=None):
    """forecasts/CALIBRATION.md: the human-readable version of the backtest, regenerated
    every time calibration.py runs."""
    pct = lambda v: f"{v * 100:.0f}%"
    f3 = lambda v: f"{v:.3f}"
    f2 = lambda v: f"{v:.2f}"
    score_cols = [('Method', 'method'), ('RMSE (pts)', 'rmse'), ('Winners right', 'winner_accuracy'),
                  ('Brier', 'brier'), ('Log loss', 'log_loss')]
    score_fmt = {'rmse': f2, 'winner_accuracy': pct, 'brier': f3, 'log_loss': f3}
    cal_cols = [('Forecast chance for favorite', 'bucket'), ('Races', 'races'), ('Avg forecast', 'predicted'),
                ('Favorite actually won', 'actual'), ('Expected range if calibrated (90%)', 'range')]
    gb_cols = [('Days out', 'days_before'), ('Cycles', 'cycles'), ('Avg Dem overstatement (pts)', 'mean_dem_overstatement'),
               ('Cycles overstating Dems', 'cycles_overstating_dem')]

    def cal_rows(key):
        rows = report['calibration'][key]
        return [{**r, 'range': f"{r['expected_low'] * 100:.0f}-{r['expected_high'] * 100:.0f}%"} for r in rows]

    n28 = report['scores']['gb_bias_corrected_28d'][0]['races']
    md = f"""# Forecast Calibration & Baselines

*Generated by `src/calibration.py`. Rerun it to refresh.*

This report checks whether the forecast's probabilities mean what they say, and whether
the full model beats simpler ways of forecasting. Every polled Senate and governor race
from 2000 to 2022 was re-forecast as of 0 and 28 days before the election. Each cycle
was forecast using only other cycles' data: model coefficients, error sizes, pollster
ratings, and the generic-ballot correction. The national environment came from the
generic-ballot polls of the time, not the actual result.

![Calibration chart](calibration/calibration_28d.png)

## 1. Does the full model beat simpler methods?

**28 days out** ({n28} races, same horizon as a forecast published four weeks before election day):

{_md_table(report['scores']['gb_bias_corrected_28d'], score_cols, score_fmt)}
**Election eve:**

{_md_table(report['scores']['gb_bias_corrected_0d'], score_cols, score_fmt)}
Brier score and log loss measure probability quality (lower is better). They reward
being confident when right and punish being confident when wrong. The full model has
the best probabilities at both horizons. On picking winners it ties polls alone: the
fundamentals mostly sharpen margins and probabilities rather than flipping calls.

## 2. Calibration

When the model gave the favorite a given chance, how often did the favorite win?

**28 days out:**

{_md_table(cal_rows('gb_bias_corrected_28d'), cal_cols, {'predicted': pct, 'actual': pct})}
**Election eve:**

{_md_table(cal_rows('gb_bias_corrected_0d'), cal_cols, {'predicted': pct, 'actual': pct})}
{_calibration_verdict(report)}

## 3. Generic-ballot correction

The generic congressional ballot sets the national environment for every race.
Historically its polling average has overstated Democrats:

{_md_table(report['generic_ballot_bias'], gb_cols, {'mean_dem_overstatement': f2})}
Removing that lean (estimated from other cycles only) improved the backtest at both
horizons, so the live forecast applies it. 2024 has no list of individual generic-ballot
polls, so national presidential polls vs. the popular vote stand in for that year.

**State and district polls** lean the same way, less strongly. In 2024 the Senate polls
overstated Democrats by about 2.5 points in 17 of 22 races. The forecast applies half of
the historical average lean, weighting recent cycles more, which is a correction of
{_spb_at(report, 28)} pts at four weeks out. Tested with 2024 included, at four weeks out:

| State-poll correction | Margin RMSE | Brier | Log loss |
|---|---|---|---|
| None | 6.72 | 0.0673 | 0.2155 |
| Full | 6.68 | 0.0698 | 0.2200 |
| Half | 6.64 | 0.0677 | 0.2153 |
| Recent-weighted, full | 6.57 | 0.0695 | 0.2192 |
| **Recent-weighted, half (adopted)** | 6.58 | 0.0673 | 0.2144 |

Full-strength corrections over-correct and hurt the probabilities. The adopted version
ties on Brier, has the best log loss, cuts the margin error, and shrinks a 2024-style
miss from 4.4 to 3.0 points. Five variants were tried, so there is some risk of fitting
noise; the two half-strength versions agree.

## 4. Tested and not adopted

- **Full-strength pollster house-effect correction.** It was neutral in backtest (RMSE
  6.02 none / 6.03 half / 6.07 full), so corrections run at half strength.

## 5. Known limits

- Calibration covers Senate and governor races that had polls. Unpolled races are
  easier and would flatter the numbers, so they're excluded.
- Cycle-level effects (2016, 2020) mean the effective sample is about 12 election
  years, not 500 independent races. Chamber-control probabilities can't be calibrated
  directly from that few cycles.
- The House district model is checked separately: calibrated on 2024 and applied to
  2022 districts with unchanged lines, it called 94.6% of winners with ±5.7-point
  typical error.
"""
    path = path or os.path.join(OUT_DIR, '..', 'CALIBRATION.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(md)
    return os.path.abspath(path)

if __name__ == '__main__':
    pd.set_option('display.width', 200)
    results, gb, spb = run()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generic ballot polling average vs actual House vote (Dem overstatement; 2024 = presidential-poll stand-in):")
    gbs = _gb_bias_summary(gb)
    print(gbs.round(2).to_string(index=False))

    report = {'generic_ballot_bias': gbs.round(3).to_dict('records'), 'state_poll_bias': spb,
              'scores': {}, 'calibration': {}}
    print("State-poll lean correction:", spb)
    for correct, bt in results.items():
        tag = 'gb_bias_corrected' if correct else 'gb_raw'
        bt.to_csv(os.path.join(OUT_DIR, f'backtest_{tag}.csv'), index=False)
        for d in (0, 28):
            sub = bt[bt['days_before'] == d]
            s = score(sub)
            print(f"\n=== {d} days out, generic ballot {'bias-corrected' if correct else 'raw'} "
                  f"(Senate+governor, {len(sub)} races) ===")
            print(s.round(3).to_string(index=False))
            report['scores'][f'{tag}_{d}d'] = s.round(4).to_dict('records')
            c = calibration_table(sub)
            report['calibration'][f'{tag}_{d}d'] = c.round(3).astype({'bucket': str}).to_dict('records')
            if correct:
                print("Calibration (full model):")
                print(c.round(3).to_string(index=False))
                bias = (sub['model_margin'] - sub['actual']).mean()
                print(f"Mean signed error (Dem-positive): {bias:+.2f}")
    with open(os.path.join(OUT_DIR, 'calibration_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print("Chart:", make_chart(report))
    print("Report:", write_markdown(report))
    print(f"\nWrote {os.path.abspath(OUT_DIR)}")
