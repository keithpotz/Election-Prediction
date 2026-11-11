"""
The 2026 forecast for every race on the November 3 ballot (roadmap Phase 2):
35 Senate races, 36 governor races, and all 435 House districts.

Per race:
  Senate / governor   fundamentals prior blended with the house-effect-corrected
                      polling average (blend.py). Races without polls use
                      fundamentals alone.
  House               district model (house_districts.py) driven by the generic-ballot
                      polling average. District polls aren't included yet.

All races are simulated together. Every simulation draws ONE national error that
moves every race at once, scaled by how exposed each race is to it, plus independent
race-level errors. This is what makes "Democrats win the Senate" a real joint
probability rather than a product of separate coin flips.

    python src/forecast_2026.py
"""
import os
from datetime import date

import numpy as np
import pandas as pd

import midterms as M
import governors as G
import house_districts as H
from blend import blend, shared_sd, shared_poll_miss_sd
from polls_2026 import load_polls
from polls_average import all_race_averages, annotate_polls, race_average, load_poll_error_model, poll_error_sd
from race_status_2026 import load_or_scrape_race_status
from calibration import generic_ballot_bias, regional_error_sd, state_poll_bias
from states import REGIONS, STATE_REGION
import simulations as SIMS

ELECTION_DAY = date(2026, 11, 3)

# Unpolled races can be upset by things the fundamentals can't see (a scandal, a collapsing
# campaign, a strong local candidate). In each simulation, each contested unpolled race has
# this chance of being decided by a coin flip instead of its simulated margin, which caps
# its win probability at 1 - UNPOLLED_UPSET_RATE / 2 = 99.5%. Fit on the House district
# model's out-of-year results: no such upsets among 137 districts at 99%+ in 2022, but 3
# of 260 unpolled districts at 99%+ in 2026 (expected 0.2). The best fit to both years is
# 1.2%. See forecasts/CHANGELOG.md, 2026.11.12.
UNPOLLED_UPSET_RATE = 0.01

def days_to_election(today=None):
    return max((ELECTION_DAY - (today or date.today())).days, 0)

# The fundamentals models are fit on Democrat-vs-Republican races; when the main
# challenger is an independent (Osborn, Achilles), their prior is much less informative.
# Doubling its uncertainty lets the race's polls carry it. A judgment call: there's too
# little independent-candidate history to fit this.
INDEPENDENT_FUND_SD_MULTIPLIER = 2.0

def _nominee_names():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'polls_2026', 'nominees_2026.csv')
    if not os.path.exists(path):
        return {}
    n = pd.read_csv(path)
    return {(r['office'], r['state']): r for _, r in n.iterrows() if isinstance(r['dem_candidate'], str)}

def _blend_races(fund, poll_avgs, office, fund_race_sd, fund_national_sd, err_model, days, poll_lean=0.0):
    """Adds poll average, blended mean, race sd, and shared-error exposure to each race.
    poll_lean: state polls' historical Democratic lean, removed before blending
    (calibration.state_poll_bias); the raw average is kept in poll_avg."""
    polls = poll_avgs[poll_avgs['office'] == office].set_index('state')
    nominees = _nominee_names()
    out = []
    base_fund_sd = fund_race_sd
    for _, r in fund.iterrows():
        p = polls.loc[r['state']] if r['state'] in polls.index else None
        independent = p is not None and p['dem_side_party'] == 'INDEPENDENT'
        fund_race_sd = base_fund_sd * (INDEPENDENT_FUND_SD_MULTIPLIER if independent else 1.0)
        if p is not None:
            psd = poll_error_sd(p['effective_polls'], err_model, days)
            mean, race_sd, w = blend(r['fund_pred'], fund_race_sd, p['poll_avg'] - poll_lean, psd)
        else:
            psd, (mean, race_sd, w) = None, blend(r['fund_pred'], fund_race_sd)
        out.append({
            **r.to_dict(),
            'poll_avg':        None if p is None else round(float(p['poll_avg']), 2),
            'poll_avg_corrected': None if p is None else round(float(p['poll_avg']) - poll_lean, 2),
            'n_polls':         0 if p is None else int(p['n_polls']),
            'effective_polls': 0.0 if p is None else round(float(p['effective_polls']), 1),
            'latest_poll':     None if p is None else p['latest_poll'],
            # Candidate names from the polls, or from the confirmed nominee list when unpolled
            'dem_candidate':   p['dem_candidate'] if p is not None else nominees.get((office, r['state']), {}).get('dem_candidate'),
            'rep_candidate':   p['rep_candidate'] if p is not None else nominees.get((office, r['state']), {}).get('rep_candidate'),
            'dem_side_party':  p['dem_side_party'] if p is not None else
                               (nominees.get((office, r['state']), {}).get('dem_side_party') or 'DEMOCRAT'),
            'poll_weight':     round(w, 2),
            'forecast_margin': round(mean, 2),
            'race_sd':         race_sd,
            'shared_sd':       shared_sd(w, fund_national_sd, days),
            # Each input's own uncertainty, saved so the grading can score polls-only and
            # fundamentals-only forecasts alongside the blend
            'poll_sd':         psd,
            'poll_shared_sd':  shared_poll_miss_sd(days),
            'fund_sd':         fund_race_sd,
            'fund_shared_sd':  fund_national_sd,
        })
    return pd.DataFrame(out)

def house_poll_averages(polls, as_of):
    """District polling averages, keyed by (state, district). Same weighting and
    house-effect/sponsor corrections as Senate and governor polls."""
    house = polls[polls['office'] == 'house']
    if house.empty:
        return pd.DataFrame(columns=['state', 'district']), house
    annotated = annotate_polls(house)
    rows = []
    for (state, district), g in annotated.groupby(['state', 'district']):
        avg = race_average(g, as_of)
        if avg:
            rows.append({'state': state, 'district': int(district), **avg})
    return pd.DataFrame(rows), annotated

ENV_CHECK_THRESHOLD = 3.0

def environment_check(senate, governor, house, env_used, s_env_coef, g_env_coef, generic_poll):
    """
    An independent read on the national environment from our own race polls. In each
    polled race, how far the polls sit from the structural estimate (which assumed the
    generic-ballot environment), divided by that model's sensitivity to the national
    environment, says what environment the polls imply.

    Compared like-for-like with the generic-ballot POLL average (before its historical
    lean is removed), since race polls get no such correction. Senate and House races
    form the combined read; governors are shown separately, because the governor
    structural model is weak (governors run far from their state's lean), making its
    implied environment noisy.
    """
    parts = []
    for df, coef, label in ((senate, s_env_coef, 'Senate'), (governor, g_env_coef, 'Governor')):
        d = df[(df['effective_polls'] >= 2) & (df['dem_side_party'] == 'DEMOCRAT')]
        if len(d):
            parts.append((label, env_used + (d['poll_avg'] - d['fund_pred']) / coef, d['effective_polls']))
    h = house[(house['effective_polls'].fillna(0) >= 1) & house['poll_avg'].notna()]
    if len(h):
        parts.append(('House districts', env_used + (h['poll_avg'] - h['model_margin']), h['effective_polls']))
    rows = {label: {'implied': round(float(np.average(v, weights=w)), 2), 'races': int(len(v))}
            for label, v, w in parts}
    core = [(v, w) for label, v, w in parts if label != 'Governor']
    allv = pd.concat([v for v, _ in core]) if core else pd.Series(dtype=float)
    allw = pd.concat([w for _, w in core]) if core else pd.Series(dtype=float)
    implied = round(float(np.average(allv, weights=allw)), 2) if len(allv) else None
    return {'environment_used': round(env_used, 2), 'generic_ballot_poll': round(generic_poll, 2),
            'implied_by_race_polls': implied, 'by_office': rows,
            'gap': None if implied is None else round(implied - generic_poll, 2),
            'flag': implied is not None and abs(implied - generic_poll) > ENV_CHECK_THRESHOLD}

def load_inputs(today=None):
    """
    Everything the forecast needs that doesn't depend on the generic ballot: scraped
    race data, fitted models, error models, and polling averages. Slow (model fits and
    CSV parsing), so the dashboard builds this once and reruns build_forecast() cheaply
    as the generic-ballot input changes.
    """
    as_of = pd.Timestamp(today or date.today())
    poll_avgs, annotated = all_race_averages(load_polls(), as_of=as_of)
    house_avgs, house_annotated = house_poll_averages(load_polls(include_house=True), as_of)
    districts = H.load_house_districts()
    return {
        'today':        today or date.today(),
        'err_model':    load_poll_error_model(),
        'poll_avgs':    poll_avgs,
        'annotated_polls': annotated,
        'house_poll_avgs': house_avgs,
        'house_annotated_polls': house_annotated,
        'sen_bundle':   M.build_senate_race_model(),
        'sen_status':   M.load_race_status(),
        'sen_races':    M.senate_races_table(),
        'gov_bundle':   G.build_model(),
        'gov_status':   load_or_scrape_race_status(office='governor'),
        'gov_archive':  G.archive_governor_races(),
        'districts':    districts,
        'h_model':      H.calibrate(H.calibration_data(districts), districts),
        'house_ratings': __import__('polls_house_districts').load_house_ratings(),
        'env_2024':     float(M.national_house_margin_by_year()[2024]),
    }

def build_forecast(generic_ballot_poll_margin, today=None, n_sims=M.N_SIMS, seed=7, inputs=None, extra_poll_miss=0.0,
                   keep_sims=True):
    """
    generic_ballot_poll_margin: the generic-ballot polling average (Dem minus Rep). It's
    converted to an expected national House vote by removing the generic ballot's
    historical Democratic lean (see calibration.generic_ballot_bias). The backtest showed
    that correction improves forecasts, while a similar correction to state polls didn't.

    inputs: optional load_inputs() result, to skip rebuilding models on every call.
    extra_poll_miss: a what-if. Assume every poll (generic ballot and each race's polls)
    overstates Democrats by this many more points than the usual corrections remove.
    Negative values assume the polls understate Democrats.
    keep_sims: keep the simulation draws in f['sims'] (about 15 MB) for election-night
    mode (simulations.py). Off for quick what-if sweeps.
    """
    inputs = inputs or load_inputs(today)
    today = inputs['today']
    days = days_to_election(today)
    gb_bias = generic_ballot_bias(days)
    generic_ballot_margin = generic_ballot_poll_margin - gb_bias - extra_poll_miss
    err_model = inputs['err_model']
    poll_avgs = inputs['poll_avgs']
    poll_lean = state_poll_bias(days) + extra_poll_miss

    # ── Senate ──
    sen_bundle = inputs['sen_bundle']
    sen_fund = M.predict_senate_2026(generic_ballot_margin, model_bundle=sen_bundle,
                                     race_status_df=inputs['sen_status'], races=inputs['sen_races'], n_sims=200)
    if sen_fund['warnings'] or sen_fund['skipped_states']:
        raise RuntimeError(f"Senate data problems: {sen_fund['warnings']} skipped={sen_fund['skipped_states']}")
    s_err = sen_bundle[1]
    s_env = sen_bundle[0].coef_[M.SENATE_FEATURES.index('national_environment')]
    senate = _blend_races(
        sen_fund['seats'].rename(columns={'predicted_margin': 'fund_pred'}).drop(
            columns=['dem_win_prob', 'predicted_winner']),
        poll_avgs, 'senate', s_err['state_sd'],
        float(np.hypot(s_err['national_sd'], s_env * M.GENERIC_BALLOT_ERROR_SD)), err_model, days, poll_lean)

    # ── Governors ──
    gov_bundle = inputs['gov_bundle']
    gov_fund, g_err = G.fundamentals_2026(generic_ballot_margin, gov_bundle, inputs['gov_status'],
                                          archive_races=inputs['gov_archive'])
    if gov_fund['fund_pred'].isna().any():
        raise RuntimeError(f"Governor data problems: {gov_fund[gov_fund['fund_pred'].isna()]['state'].tolist()}")
    g_env = gov_bundle[0].coef_[G.FEATURES.index('national_environment')]
    governor = _blend_races(gov_fund.drop(columns=['skip_reason']), poll_avgs, 'governor', g_err['state_sd'],
                            float(np.hypot(g_err['national_sd'], g_env * M.GENERIC_BALLOT_ERROR_SD)),
                            err_model, days, poll_lean)

    # ── House ──
    h_model = inputs['h_model']
    house_pt = H.forecast_house(generic_ballot_margin, inputs['districts'], h_model, n_sims=200,
                                env_2024=inputs['env_2024'])
    house = house_pt['districts']
    house_fixed = house['uncontested'].values
    house_fixed_dem = (house['dem_win_prob'].values >= 0.5) & house_fixed

    # Blend district polls (where any exist) with the district model, exactly as for
    # Senate/governor races. Unpolled districts keep the model estimate.
    district_sd = h_model[1]['district_sd']
    hp = inputs.get('house_poll_avgs')
    house = house.drop(columns=[c for c in ('poll_avg', 'n_polls') if c in house], errors='ignore')
    if hp is not None and len(hp):
        house = house.merge(hp[['state', 'district', 'poll_avg', 'n_polls', 'effective_polls', 'latest_poll']],
                            on=['state', 'district'], how='left')
    else:
        house['poll_avg'], house['n_polls'], house['effective_polls'], house['latest_poll'] = np.nan, 0, 0.0, None
    house['n_polls'] = house['n_polls'].fillna(0).astype(int)
    hr = inputs.get('house_ratings')
    house = house.drop(columns=[c for c in ('consensus_rating', 'n_raters') if c in house], errors='ignore')
    if hr is not None and len(hr):
        house = house.merge(hr, on=['state', 'district'], how='left')
    else:
        house['consensus_rating'], house['n_raters'] = np.nan, 0
    blended = [blend(m, district_sd, pa - poll_lean if pd.notna(pa) else pa,
                     poll_error_sd(ne, err_model, days) if pd.notna(pa) else None)
               for m, pa, ne in zip(house['predicted_margin'], house['poll_avg'], house['effective_polls'].fillna(0))]
    house['model_margin'] = house['predicted_margin']
    house['predicted_margin'] = [b[0] for b in blended]
    house['race_sd'] = [b[1] for b in blended]
    house['poll_weight'] = [round(b[2], 2) for b in blended]
    # Each district's exposure to the national miss, weighted by what it relies on, as for
    # Senate and governor races. The district model's national error is the error in the
    # national environment itself (the generic ballot's miss on the House vote).
    house['shared_sd'] = [shared_sd(b[2], M.GENERIC_BALLOT_ERROR_SD, days) for b in blended]

    # ── Joint simulation: one national draw shared by every race ──
    rng = np.random.default_rng(seed)
    nat = rng.normal(0, 1, (n_sims, 1))
    # Regional errors: races in the same region also move together. Each race's total
    # uncertainty is unchanged (the regional part is carved out of its own race-level
    # error), so per-race probabilities stay calibrated; only the correlation changes.
    region_sd = regional_error_sd(days)
    region_names = list(REGIONS)
    region_shock = rng.normal(0, region_sd, (n_sims, len(region_names)))

    def regional(states):
        idx = np.array([region_names.index(STATE_REGION.get(s, 'South')) for s in states])
        return region_shock[:, idx]

    def own_sd(race_sd):
        return np.sqrt(np.maximum(np.asarray(race_sd, dtype=float) ** 2 - region_sd ** 2,
                                  0.25 * np.asarray(race_sd, dtype=float) ** 2))

    def sim(df, mean_col):
        local = rng.normal(0, 1, (n_sims, len(df))) * own_sd(df['race_sd'].values)
        return df[mean_col].values + nat * df['shared_sd'].values + regional(df['state']) + local

    sen_m = sim(senate, 'forecast_margin')
    gov_m = sim(governor, 'forecast_margin')
    h_local = rng.normal(0, 1, (n_sims, len(house))) * own_sd(house['race_sd'].values)
    h_m = house['predicted_margin'].values + nat * house['shared_sd'].values + regional(house['state']) + h_local
    h_dem = np.where(house_fixed, house_fixed_dem, h_m > 0)

    # Unpolled-race upsets (UNPOLLED_UPSET_RATE). A separate generator, so every other draw
    # is unchanged. For Senate and governor races the margin's sign is flipped at random,
    # since chamber counts and tipping points read the margins.
    up_rng = np.random.default_rng(seed + 1)
    def upset(n_races, unpolled):
        hit = (up_rng.random((n_sims, n_races)) < UNPOLLED_UPSET_RATE) & np.asarray(unpolled, dtype=bool)
        return hit, up_rng.random((n_sims, n_races)) < 0.5
    for m, df in ((sen_m, senate), (gov_m, governor)):
        hit, dem = upset(len(df), df['n_polls'].fillna(0).values == 0)
        m[hit] = np.where(dem, 1, -1)[hit] * np.maximum(np.abs(m[hit]), 0.01)
    hit, dem = upset(len(house), (house["n_polls"].fillna(0).values == 0) & ~np.asarray(house_fixed, dtype=bool))
    h_dem = np.where(hit, dem, h_dem)

    for df, m in ((senate, sen_m), (governor, gov_m)):
        df['dem_side_win_prob'] = (m > 0).mean(axis=0).round(3)
        df['forecast_winner'] = np.where(df['forecast_margin'] >= 0, df['dem_side_party'], 'REPUBLICAN')
        df['race_sd'] = df['race_sd'].round(2)
        df['shared_sd'] = df['shared_sd'].round(2)
    house['dem_win_prob'] = h_dem.mean(axis=0).round(3)
    house['predicted_winner'] = np.where(house['dem_win_prob'] >= 0.5, 'DEMOCRAT', 'REPUBLICAN')

    # Senate chamber: Dem-side wins by an independent (Osborn in Nebraska) don't count
    # toward the Democratic caucus.
    dem_caucus = (sen_m > 0) & (senate['dem_side_party'] == 'DEMOCRAT').values
    ind = (sen_m > 0) & (senate['dem_side_party'] != 'DEMOCRAT').values
    holdovers = sen_fund['holdovers']
    sen_d = holdovers['DEMOCRAT'] + dem_caucus.sum(axis=1)
    sen_i = ind.sum(axis=1)
    sen_r = M.SENATE_SEATS_TOTAL - sen_d - sen_i
    h_d = h_dem.sum(axis=1)

    # Which race decides each chamber, and the common paths to control (simulations.py)
    senate['tipping_point_prob'], sen_paths = SIMS.senate_extras(senate, sen_m, holdovers)
    senate['tipping_point_prob'] = senate['tipping_point_prob'].round(4)
    house['tipping_point_prob'] = SIMS.house_tipping_point(h_m, house_fixed, house_fixed_dem).round(4)
    sims = None
    if keep_sims:
        sims = {
            'senate': {'states': senate['state'].tolist(), 'dem_side_party': senate['dem_side_party'].tolist(),
                       'margin': sen_m.astype(np.float32)},
            'governor': {'states': governor['state'].tolist(), 'dem_side_party': governor['dem_side_party'].tolist(),
                         'margin': gov_m.astype(np.float32)},
            'house': {'keys': list(zip(house['state'], house['district'].astype(int))), 'dem_win': h_dem},
            'senate_holdovers': dict(holdovers),
        }

    return {
        'as_of':           (today or date.today()).isoformat(),
        'days_to_election': days,
        'generic_ballot_poll': float(generic_ballot_poll_margin),
        'generic_ballot_bias_removed': round(gb_bias, 2),
        'state_poll_lean_removed': round(poll_lean, 2),
        'national_environment': round(float(generic_ballot_margin), 2),
        'senate':          senate.sort_values('forecast_margin'),
        'governor':        governor.sort_values('forecast_margin'),
        'regional_error_sd': round(region_sd, 2),
        'house':           house,
        'sims':            sims,
        'summary': {
            'senate': {
                'dem_control_prob': round(float((sen_d >= M.SENATE_DEM_CONTROL_SEATS).mean()), 3),
                'rep_control_prob': round(float((sen_r >= 50).mean()), 3),   # VP Vance breaks 50-50
                'dem_seats_median': int(np.median(sen_d)),
                'dem_seats_90pct':  [int(np.percentile(sen_d, 5)), int(np.percentile(sen_d, 95))],
                'rep_seats_median': int(np.median(sen_r)),
                'rep_seats_90pct':  [int(np.percentile(sen_r, 5)), int(np.percentile(sen_r, 95))],
                'independent_seats_expected': round(float(sen_i.mean()), 2),
                'holdovers': holdovers,
                'dem_seats_dist':   pd.Series(sen_d).value_counts(normalize=True).sort_index(),
                'paths':            sen_paths,
            },
            'house': {
                'dem_control_prob': round(float((h_d >= M.HOUSE_MAJORITY).mean()), 3),
                'dem_seats_median': int(np.median(h_d)),
                'dem_seats_90pct':  [int(np.percentile(h_d, 5)), int(np.percentile(h_d, 95))],
                'dem_seats_dist':   pd.Series(h_d).value_counts(normalize=True).sort_index(),
                'calibration':      h_model[1],
            },
            'governor': {
                'dem_governorships_median': int(np.median((gov_m > 0).sum(axis=1))),
                'dem_governorships_90pct':  [int(np.percentile((gov_m > 0).sum(axis=1), 5)),
                                             int(np.percentile((gov_m > 0).sum(axis=1), 95))],
                'races': len(governor),
            },
            'environment_check': environment_check(senate, governor, house, generic_ballot_margin, s_env, g_env,
                                                   generic_ballot_poll_margin),
            'backtests': {'senate_fundamentals': s_err, 'governor_fundamentals': g_err,
                          'poll_error_model': {k: err_model[k] for k in ('floor', 'thin', 'drift')}},
        },
    }

if __name__ == '__main__':
    from data_sources import fetch_generic_ballot
    gb, src = fetch_generic_ballot()
    f = build_forecast(gb)
    pd.set_option('display.width', 250)
    s = f['summary']
    print(f"As of {f['as_of']} ({f['days_to_election']} days out), generic ballot poll D{gb:+.1f} "
          f"-> expected House vote D{f['national_environment']:+.1f} (historical poll lean removed)")
    print(f"SENATE  Dem control {s['senate']['dem_control_prob']:.0%} | Rep control {s['senate']['rep_control_prob']:.0%} | "
          f"median {s['senate']['dem_seats_median']} D seats {s['senate']['dem_seats_90pct']}")
    print(f"HOUSE   Dem control {s['house']['dem_control_prob']:.0%} | median {s['house']['dem_seats_median']} D seats {s['house']['dem_seats_90pct']}")
    print(f"GOVS    median {s['governor']['dem_governorships_median']} of {s['governor']['races']} races won by Dems {s['governor']['dem_governorships_90pct']}")
    cols = ['state', 'incumbent_party', 'fund_pred', 'poll_avg', 'n_polls', 'poll_weight', 'forecast_margin',
            'dem_side_win_prob', 'consensus_rating']
    print("\nSenate:\n", f['senate'][cols].round(2).to_string(index=False))
    print("\nGovernor:\n", f['governor'][cols].round(2).to_string(index=False))
