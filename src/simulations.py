"""
What the 20,000 joint simulations say beyond each race's own odds:

  tipping_point       Which race delivers the deciding seat (the 51st Democratic Senate
                      seat, the 218th House seat). The chamber is decided where these are.
  paths_to_control    The most common combinations of competitive-race outcomes in the
                      simulations a party wins the chamber.
  election_night      Election-night mode: re-weights the simulations by the races already
                      called, so the uncalled races and chamber odds update together. A call
                      in one race says something about the national error, which moves
                      every other race too.

forecast_2026.build_forecast keeps the simulation draws in f['sims'] for these.
"""
import numpy as np
import pandas as pd

from states import STATE_ABBREV

SENATE_DEM_CONTROL, SENATE_REP_CONTROL, SENATE_TOTAL = 51, 50, 100
HOUSE_MAJORITY, HOUSE_TOTAL = 218, 435

# Election night: a simulation that disagrees with a called race keeps this much weight
# per disagreement instead of being thrown out. Pure rejection runs out of simulations
# after a few surprises; this keeps the closest-matching ones in play.
CALL_MISMATCH_WEIGHT = 0.02
# Only calls in races the forecast had in doubt re-weight the simulations. Safe seats are
# called first on election night and almost always go as expected; the model's small upset
# chances there aren't precise enough (few such upsets to calibrate on) for 150 of them
# going to form to say anything real about the national error. They still count as seats.
INFORMATIVE_CALL_RANGE = (0.03, 0.97)


def tipping_point(margins, eligible, need):
    """margins: (n_sims, n_races) Democratic-side margins. eligible: which races count
    toward the Democratic total (not an independent's race). need: Democratic wins needed
    among these races for control. Returns each race's share of simulations in which it
    was the deciding seat: the need-th most Democratic race in that simulation."""
    n_sims, n_races = margins.shape
    eligible = np.asarray(eligible, dtype=bool)
    if need < 1 or need > eligible.sum():
        return np.zeros(n_races)
    m = np.where(eligible, margins, -np.inf)
    k = n_races - need                    # need-th largest = k-th smallest
    tp = np.argpartition(m, k, axis=1)[:, k]
    return np.bincount(tp, minlength=n_races) / n_sims


def paths_to_control(party_wins, held, control, labels, top=6, tossup=0.2, min_p=0.03):
    """How a party wins the chamber, from the simulations where it does. party_wins:
    (n_sims, n_races) bool, that party (or its side) wins each race. held: bool per race,
    the party holds the seat now. control: bool per sim.

    races   Every race that isn't nearly settled (won in min_p to 1 - min_p of all
            simulations): how often the party wins it overall and in its winning
            simulations. A big gap marks a must-win race.
    combos  The most common outcomes of the toss-up races (won in tossup to 1 - tossup of
            simulations) among the winning simulations. Exact combinations across every
            competitive race are too scattered to read; the toss-ups are what's in doubt."""
    n_ctrl = int(control.sum())
    if not n_ctrl:
        return {'control': 0.0, 'races': [], 'combos': []}
    labels, held = np.asarray(labels), np.asarray(held, dtype=bool)
    p = party_wins.mean(axis=0)
    p_ctrl = party_wins[control].mean(axis=0)
    comp = np.flatnonzero((p > min_p) & (p < 1 - min_p))
    races = sorted(({'race': str(labels[i]), 'held': bool(held[i]), 'p': round(float(p[i]), 3),
                     'p_when_control': round(float(p_ctrl[i]), 3)} for i in comp),
                   key=lambda r: -r['p_when_control'])
    tu = (p > tossup) & (p < 1 - tossup)
    combos = []
    if tu.any():
        keys, counts = np.unique(party_wins[control][:, tu], axis=0, return_counts=True)
        tl, th = labels[tu], held[tu]
        for i in np.argsort(-counts)[:top]:
            k = keys[i]
            combos.append({'wins': tl[k].tolist(), 'loses': tl[~k].tolist(),
                           'gains': tl[k & ~th].tolist(), 'losses': tl[~k & th].tolist(),
                           'share_of_wins': round(float(counts[i] / n_ctrl), 4)})
    return {'control': round(n_ctrl / party_wins.shape[0], 4), 'races': races, 'combos': combos,
            'tossups': labels[tu].tolist()}


def senate_extras(senate, sen_m, holdovers):
    """Tipping-point chances (a column on the races table) and paths to control."""
    dem_party = (senate['dem_side_party'] == 'DEMOCRAT').values
    tp = tipping_point(sen_m, dem_party, SENATE_DEM_CONTROL - holdovers['DEMOCRAT'])
    dem_side_win = sen_m > 0
    sen_d = holdovers['DEMOCRAT'] + (dem_side_win & dem_party).sum(axis=1)
    sen_r = SENATE_TOTAL - sen_d - (dem_side_win & ~dem_party).sum(axis=1)
    labels = [STATE_ABBREV.get(s, s) + (' (I)' if not d else '') for s, d in zip(senate['state'], dem_party)]
    rep_held = (senate['incumbent_party'] == 'REPUBLICAN').values
    paths = {
        'DEMOCRAT': paths_to_control(dem_side_win, ~rep_held, sen_d >= SENATE_DEM_CONTROL, labels),
        'REPUBLICAN': paths_to_control(~dem_side_win, rep_held, sen_r >= SENATE_REP_CONTROL, labels),
    }
    return tp, paths


def house_tipping_point(h_margins, fixed, fixed_dem):
    m = np.where(fixed, np.where(fixed_dem, 1e3, -1e3), h_margins)
    return tipping_point(m, np.ones(m.shape[1], dtype=bool), HOUSE_MAJORITY)


# ── Election night ───────────────────────────────────────────────────────────
def _sim_index(sims, office):
    if office == 'house':
        return {k: i for i, k in enumerate(sims['house']['keys'])}
    return {s: i for i, s in enumerate(sims[office]['states'])}


def calls_from_results(results, sims):
    """Called races (rows of results_2026.csv with a winner) matched to simulation columns.
    Returns a list of (office, column, dem_side_won)."""
    calls = []
    if results is None or results.empty:
        return calls
    idx = {o: _sim_index(sims, o) for o in ('senate', 'governor', 'house')}
    from grade_2026 import is_decided
    for _, r in results[is_decided(results['winner_party'])].iterrows():
        o = r['office']
        if o == 'house':
            key = (r['state'], int(r['district']))
            col = idx['house'].get(key)
            dem_side_won = r['winner_party'] == 'DEMOCRAT'
        else:
            col = idx[o].get(r['state'])
            if col is None:
                continue
            dem_side_won = r['winner_party'] == sims[o]['dem_side_party'][col]
        if col is not None:
            calls.append((o, col, bool(dem_side_won)))
    return calls


def _wins(sims, office):
    return sims['house']['dem_win'] if office == 'house' else sims[office]['margin'] > 0


def call_weights(sims, calls):
    """Simulation weights given the calls, and the effective number of simulations left."""
    n = sims['senate']['margin'].shape[0]
    logw = np.zeros(n)
    lo, hi = INFORMATIVE_CALL_RANGE
    for o, col, won in calls:
        x = _wins(sims, o)[:, col]
        if lo <= x.mean() <= hi:
            logw += np.where(x == won, 0.0, np.log(CALL_MISMATCH_WEIGHT))
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return w, float(1 / (w ** 2).sum())


def election_night(sims, calls):
    """Chamber odds and every race's updated odds, given the calls. Called races are
    fixed to their result; uncalled races use the re-weighted simulations."""
    w, ess = call_weights(sims, calls)
    wins = {o: _wins(sims, o).copy() for o in ('senate', 'governor', 'house')}
    for o, col, won in calls:
        wins[o][:, col] = won
    s = sims['senate']
    dem_party = np.asarray(s['dem_side_party']) == 'DEMOCRAT'
    sen_d = sims['senate_holdovers']['DEMOCRAT'] + (wins['senate'] & dem_party).sum(axis=1)
    sen_r = SENATE_TOTAL - sen_d - (wins['senate'] & ~dem_party).sum(axis=1)
    h_d = wins['house'].sum(axis=1)
    gov_d = wins['governor'].sum(axis=1)
    called = {o: {col: won for oo, col, won in calls if oo == o} for o in ('senate', 'governor', 'house')}
    race_probs = {o: w @ wins[o] for o in wins}

    def pct(x, q):
        order = np.argsort(x)
        cw = np.cumsum(w[order])
        return int(x[order][np.searchsorted(cw, q)])

    return {
        'ess': ess, 'weights': w, 'race_probs': race_probs, 'called': called,
        'senate': {'dem_control': float(w @ (sen_d >= SENATE_DEM_CONTROL)),
                   'rep_control': float(w @ (sen_r >= SENATE_REP_CONTROL)),
                   'dem_median': pct(sen_d, 0.5), 'dem_90': (pct(sen_d, 0.05), pct(sen_d, 0.95)),
                   'rep_median': pct(sen_r, 0.5)},
        'house': {'dem_control': float(w @ (h_d >= HOUSE_MAJORITY)),
                  'dem_median': pct(h_d, 0.5), 'dem_90': (pct(h_d, 0.05), pct(h_d, 0.95))},
        'governor': {'dem_median': pct(gov_d, 0.5), 'dem_90': (pct(gov_d, 0.05), pct(gov_d, 0.95))},
        'sen_d': sen_d, 'sen_r': sen_r, 'h_d': h_d,
    }


def race_swing(sims, night, office='senate', min_p=0.03):
    """For each uncalled race still in doubt (min_p to 1 - min_p): Democratic chamber-control
    chance if the Democratic side wins it minus if it loses, under the current weights."""
    w = night['weights']
    ctrl = night['sen_d'] >= SENATE_DEM_CONTROL if office == 'senate' else night['h_d'] >= HOUSE_MAJORITY
    wins = _wins(sims, office)
    rows = []
    for col in range(wins.shape[1]):
        if col in night['called'][office]:
            continue
        x = wins[:, col]
        pw, pl = w[x].sum(), w[~x].sum()
        if not min_p <= pw <= 1 - min_p:
            continue
        rows.append({'col': col, 'p_win': float(pw),
                     'if_win': float(w[x] @ ctrl[x] / pw), 'if_lose': float(w[~x] @ ctrl[~x] / pl)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out['swing'] = out['if_win'] - out['if_lose']
    return out.sort_values('swing', ascending=False)
