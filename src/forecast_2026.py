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
from datetime import date

import numpy as np
import pandas as pd

import midterms as M
import governors as G
import house_districts as H
from blend import blend, shared_sd, shared_poll_miss_sd
from polls_2026 import load_polls
from polls_average import all_race_averages, load_poll_error_model, poll_error_sd
from race_status_2026 import load_or_scrape_race_status

ELECTION_DAY = date(2026, 11, 3)

def days_to_election(today=None):
    return max((ELECTION_DAY - (today or date.today())).days, 0)

def _blend_races(fund, poll_avgs, office, fund_race_sd, fund_national_sd, err_model, days):
    """Adds poll average, blended mean, race sd, and shared-error exposure to each race."""
    polls = poll_avgs[poll_avgs['office'] == office].set_index('state')
    out = []
    for _, r in fund.iterrows():
        p = polls.loc[r['state']] if r['state'] in polls.index else None
        if p is not None:
            psd = poll_error_sd(p['effective_polls'], err_model, days)
            mean, race_sd, w = blend(r['fund_pred'], fund_race_sd, p['poll_avg'], psd)
        else:
            psd, (mean, race_sd, w) = None, blend(r['fund_pred'], fund_race_sd)
        out.append({
            **r.to_dict(),
            'poll_avg':        None if p is None else round(float(p['poll_avg']), 2),
            'n_polls':         0 if p is None else int(p['n_polls']),
            'effective_polls': 0.0 if p is None else round(float(p['effective_polls']), 1),
            'latest_poll':     None if p is None else p['latest_poll'],
            'dem_candidate':   None if p is None else p['dem_candidate'],
            'rep_candidate':   None if p is None else p['rep_candidate'],
            'dem_side_party':  'DEMOCRAT' if p is None else p['dem_side_party'],
            'poll_weight':     round(w, 2),
            'forecast_margin': round(mean, 2),
            'race_sd':         race_sd,
            'shared_sd':       shared_sd(w, fund_national_sd, days),
        })
    return pd.DataFrame(out)

def build_forecast(generic_ballot_margin, today=None, n_sims=M.N_SIMS, seed=7):
    days = days_to_election(today)
    err_model = load_poll_error_model()
    poll_avgs, _ = all_race_averages(load_polls(), as_of=pd.Timestamp(today or date.today()))

    # ── Senate ──
    sen_bundle = M.build_senate_race_model()
    sen_status = M.load_race_status()
    sen_fund = M.predict_senate_2026(generic_ballot_margin, model_bundle=sen_bundle, race_status_df=sen_status)
    if sen_fund['warnings'] or sen_fund['skipped_states']:
        raise RuntimeError(f"Senate data problems: {sen_fund['warnings']} skipped={sen_fund['skipped_states']}")
    s_err = sen_bundle[1]
    s_env = sen_bundle[0].coef_[M.SENATE_FEATURES.index('national_environment')]
    senate = _blend_races(
        sen_fund['seats'].rename(columns={'predicted_margin': 'fund_pred'}).drop(
            columns=['dem_win_prob', 'predicted_winner']),
        poll_avgs, 'senate', s_err['state_sd'],
        float(np.hypot(s_err['national_sd'], s_env * M.GENERIC_BALLOT_ERROR_SD)), err_model, days)

    # ── Governors ──
    gov_bundle = G.build_model()
    gov_fund, g_err = G.fundamentals_2026(generic_ballot_margin, gov_bundle,
                                          load_or_scrape_race_status(office='governor'))
    if gov_fund['fund_pred'].isna().any():
        raise RuntimeError(f"Governor data problems: {gov_fund[gov_fund['fund_pred'].isna()]['state'].tolist()}")
    g_env = gov_bundle[0].coef_[G.FEATURES.index('national_environment')]
    governor = _blend_races(gov_fund.drop(columns=['skip_reason']), poll_avgs, 'governor', g_err['state_sd'],
                            float(np.hypot(g_err['national_sd'], g_env * M.GENERIC_BALLOT_ERROR_SD)),
                            err_model, days)

    # ── House ──
    districts = H.load_house_districts()
    h_model = H.calibrate(H.calibration_data(districts), districts)
    house_pt = H.forecast_house(generic_ballot_margin, districts, h_model, n_sims=200)
    house = house_pt['districts']
    house_fixed = house['uncontested'].values
    house_fixed_dem = (house['dem_win_prob'].values >= 0.5) & house_fixed

    # ── Joint simulation: one national draw shared by every race ──
    rng = np.random.default_rng(seed)
    nat = rng.normal(0, 1, (n_sims, 1))

    def sim(df, mean_col):
        local = rng.normal(0, 1, (n_sims, len(df))) * df['race_sd'].values
        return df[mean_col].values + nat * df['shared_sd'].values + local

    sen_m = sim(senate, 'forecast_margin')
    gov_m = sim(governor, 'forecast_margin')
    h_local = rng.normal(0, h_model[1]['district_sd'], (n_sims, len(house)))
    h_m = house['predicted_margin'].values + nat * shared_poll_miss_sd(days) + h_local
    h_dem = np.where(house_fixed, house_fixed_dem, h_m > 0)

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

    return {
        'as_of':           (today or date.today()).isoformat(),
        'days_to_election': days,
        'generic_ballot':  float(generic_ballot_margin),
        'senate':          senate.sort_values('forecast_margin'),
        'governor':        governor.sort_values('forecast_margin'),
        'house':           house,
        'summary': {
            'senate': {
                'dem_control_prob': round(float((sen_d >= M.SENATE_DEM_CONTROL_SEATS).mean()), 3),
                'rep_control_prob': round(float((sen_r >= 50).mean()), 3),   # VP Vance breaks 50-50
                'dem_seats_median': int(np.median(sen_d)),
                'dem_seats_90pct':  [int(np.percentile(sen_d, 5)), int(np.percentile(sen_d, 95))],
                'independent_seats_expected': round(float(sen_i.mean()), 2),
                'holdovers': holdovers,
                'dem_seats_dist':   pd.Series(sen_d).value_counts(normalize=True).sort_index(),
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
    print(f"As of {f['as_of']} ({f['days_to_election']} days out), generic ballot D{gb:+.1f}")
    print(f"SENATE  Dem control {s['senate']['dem_control_prob']:.0%} | Rep control {s['senate']['rep_control_prob']:.0%} | "
          f"median {s['senate']['dem_seats_median']} D seats {s['senate']['dem_seats_90pct']}")
    print(f"HOUSE   Dem control {s['house']['dem_control_prob']:.0%} | median {s['house']['dem_seats_median']} D seats {s['house']['dem_seats_90pct']}")
    print(f"GOVS    median {s['governor']['dem_governorships_median']} of {s['governor']['races']} races won by Dems {s['governor']['dem_governorships_90pct']}")
    cols = ['state', 'incumbent_party', 'fund_pred', 'poll_avg', 'n_polls', 'poll_weight', 'forecast_margin',
            'dem_side_win_prob', 'consensus_rating']
    print("\nSenate:\n", f['senate'][cols].round(2).to_string(index=False))
    print("\nGovernor:\n", f['governor'][cols].round(2).to_string(index=False))
