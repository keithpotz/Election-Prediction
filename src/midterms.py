import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

from feature_engineering import (
    normalize_party, load_house_races, national_house_margin_by_year, compute_state_lean,
)
from race_status_2026 import load_or_scrape_race_status

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/polling_data')

HOUSE_SEATS_TOTAL = 435
HOUSE_MAJORITY    = 218


# ── House ──────────────────────────────────────────────────────────────────────
def _district_winners(house):
    idx = house.groupby(['year', 'state', 'district'])['candidatevotes'].idxmax()
    return house.loc[idx, ['year', 'state', 'district', 'party']]

def national_house_seats_by_year(house):
    winners = _district_winners(house)
    return winners.groupby(['year', 'party']).size().unstack(fill_value=0)

def build_house_swing_model():
    """
    Fits the classic 'seats-votes' relationship: national Dem House seat share as a
    linear function of the national Dem-Rep popular vote margin. Deliberately a simple
    1-feature linear fit rather than a complex model — there are only ~25 House election
    cycles in the historical record, too few to support anything more elaborate.
    """
    house  = load_house_races()
    margin = national_house_margin_by_year(house)
    seats  = national_house_seats_by_year(house)

    df = pd.DataFrame({'margin': margin}).join(seats)
    df['dem_seat_share'] = df.get('DEMOCRAT', 0) / (df.get('DEMOCRAT', 0) + df.get('REPUBLICAN', 0))
    df = df.dropna()

    X, y  = df[['margin']].values, df['dem_seat_share'].values
    model = LinearRegression().fit(X, y)
    resid_std = float(np.std(y - model.predict(X), ddof=1))
    return model, resid_std

def predict_house_2026(generic_ballot_margin, n_sims=20000, seed=42, model_and_std=None):
    """
    Projects the 2026 House popular-vote-to-seats outcome from a national generic
    ballot margin, using the historical swing model plus a Monte Carlo simulation
    (drawing from the model's historical residual distribution) for a control probability.

    model_and_std: optional pre-built (model, resid_std) from build_house_swing_model() —
    pass this in (e.g. cached via @st.cache_resource at the call site) to avoid reparsing
    house_1976_2024.csv and refitting the model on every call.
    """
    model, resid_std = model_and_std if model_and_std is not None else build_house_swing_model()
    point_share  = float(np.clip(model.predict([[generic_ballot_margin]])[0], 0, 1))
    dem_seats_pt = round(point_share * HOUSE_SEATS_TOTAL)

    rng  = np.random.default_rng(seed)
    sims = np.clip(point_share + rng.normal(0, resid_std, n_sims), 0, 1)
    dem_seats_sim = np.round(sims * HOUSE_SEATS_TOTAL)

    return {
        'dem_seats':        int(dem_seats_pt),
        'rep_seats':        HOUSE_SEATS_TOTAL - int(dem_seats_pt),
        'dem_control_prob': round(float((dem_seats_sim >= HOUSE_MAJORITY).mean()), 3),
        'dem_seat_lo':      int(np.percentile(dem_seats_sim, 5)),
        'dem_seat_hi':      int(np.percentile(dem_seats_sim, 95)),
    }


# ── Senate ─────────────────────────────────────────────────────────────────────
def load_senate_races():
    df = pd.read_csv(os.path.join(DATA_DIR, 'senate_1976_2024.csv'), sep='\t')
    df['stage']   = df['stage'].astype(str).str.upper()
    df['mode']    = df['mode'].astype(str).str.upper()
    df['special'] = df['special'].astype(str).str.upper() == 'TRUE'
    df = df[(df['office'] == 'US SENATE') & (df['stage'] == 'GEN') &
            (df['mode'] == 'TOTAL') & (~df['special'])].copy()
    df['party']          = normalize_party(df['party_simplified'])
    df['candidatevotes'] = pd.to_numeric(df['candidatevotes'], errors='coerce').fillna(0)
    return df

def senate_state_margins(senate_df):
    """
    One row per (year, state) race: Dem-Rep margin and the winning party. A handful of
    races have no candidate labeled DEMOCRAT/REPUBLICAN at all (a source-data gap —
    e.g. 2020 WYOMING has null party_simplified for every candidate) — those get NaN
    margin/winner rather than being silently misclassified.
    """
    agg = senate_df.groupby(['year', 'state', 'state_po', 'party'])['candidatevotes'] \
                    .sum().unstack(fill_value=0)
    dem, rep = agg.get('DEMOCRAT', 0), agg.get('REPUBLICAN', 0)
    total = dem + rep
    margin = ((dem - rep) / total.replace(0, np.nan) * 100).round(2)
    winner = pd.Series(np.where(dem >= rep, 'DEMOCRAT', 'REPUBLICAN'), index=agg.index)
    winner[total == 0] = np.nan
    out = pd.DataFrame({'margin': margin, 'winner': winner}).reset_index()
    return out

def get_2026_senate_seats(races=None):
    """
    Senate seats up in 2026 are the 'Class II' seats — those last elected in 2020
    (Senate terms are 6 years). Derived from the data itself rather than a hardcoded
    state list, so it can't go stale or rely on memorized rosters.

    races: optional pre-computed senate_state_margins(load_senate_races()) result, to
    avoid reloading the Senate CSV when the caller already has it (see predict_senate_2026).
    """
    races = races if races is not None else senate_state_margins(load_senate_races())
    return sorted(races.loc[races['year'] == 2020, 'state'].unique())

def build_senate_race_model():
    """
    Fits state Senate margin ~ presidential state lean + incumbent party + national
    House environment that year. Trained across all historical Senate races (not just
    Class II) to maximize sample size; a simple linear model given the small n.
    """
    races = senate_state_margins(load_senate_races())
    races = races.dropna(subset=['margin'])  # a few races had no major-party candidate on one side

    # incumbent party = whoever won this state's seat one cycle (6 years) earlier
    prev_winner = races.set_index(['year', 'state'])['winner']
    races['incumbent_dem'] = races.apply(
        lambda r: prev_winner.get((r['year'] - 6, r['state'])), axis=1
    )
    races = races.dropna(subset=['incumbent_dem'])
    races['incumbent_dem'] = (races['incumbent_dem'] == 'DEMOCRAT').astype(int)

    state_df = pd.read_csv(os.path.join(DATA_DIR, 'complete_data.csv'))
    state_df['party'] = state_df['party_simplified'].str.upper()
    pres_lean = compute_state_lean(state_df)
    races['state_pres_lean'] = races['state'].map(pres_lean)

    house_margin = national_house_margin_by_year()
    races['national_environment'] = races['year'].map(house_margin)

    races = races.dropna(subset=['state_pres_lean', 'national_environment'])

    X = races[['state_pres_lean', 'incumbent_dem', 'national_environment']].values
    y = races['margin'].values
    model = LinearRegression().fit(X, y)
    resid_std = float(np.std(y - model.predict(X), ddof=1))
    return model, resid_std, pres_lean

def load_race_status():
    """Real 2026 open-seat/consensus-rating data (scraped from Wikipedia); empty if unavailable."""
    try:
        df = load_or_scrape_race_status()
        return df.set_index('state')
    except Exception as e:
        print(f"2026 race status unavailable, falling back to class-based incumbency only: {e}")
        return pd.DataFrame(columns=['state', 'open_seat', 'consensus_rating']).set_index('state')

def predict_senate_2026(generic_ballot_margin, model_bundle=None, race_status_df=None):
    """
    Projects each of the ~33 Class II Senate races up in 2026. Scoped deliberately to
    just the contested seats (predicted winner, win probability, margin) rather than a
    full 100-seat chamber-control call — the other 67 seats' current holders aren't
    derivable from a historical general-election results file (mid-term appointments,
    resignations, and party switches aren't captured there), so asserting full chamber
    control from this data alone would be guessing.

    Open seats (real 2026 retirements/lost-renomination, from race_status_2026.py) don't
    get the model's incumbency-party bonus: since build_senate_race_model() fits a plain
    LinearRegression, evaluating it with incumbent_dem=0.5 gives exactly the average of
    the Dem-held and Rep-held versions — i.e. the incumbency term's contribution is
    zeroed out. This equivalence only holds because the model is linear; if it's ever
    swapped for something nonlinear, this would need to become a real trained "open
    seat" category instead (there isn't reliable historical open-seat ground truth in
    the vote-totals data alone to fit one today).

    model_bundle: optional pre-built (model, resid_std, pres_lean) from
    build_senate_race_model() — pass this in (cached via @st.cache_resource at the call
    site) to avoid reloading 3 CSVs and refitting on every call.
    race_status_df: optional pre-loaded load_race_status() result, to avoid re-reading/
    re-scraping on every call.
    """
    model, resid_std, pres_lean = model_bundle if model_bundle is not None else build_senate_race_model()
    races = senate_state_margins(load_senate_races())
    winner_2020 = races.set_index(['year', 'state'])['winner']
    race_status = race_status_df if race_status_df is not None else load_race_status()

    rows = []
    skipped = []
    for state in get_2026_senate_seats(races):
        incumbent_party = winner_2020.get((2020, state))
        state_lean      = pres_lean.get(state)
        if state_lean is None or pd.isna(incumbent_party):
            skipped.append(state)
            continue

        status      = race_status.loc[state] if state in race_status.index else None
        open_seat   = bool(status['open_seat']) if status is not None else False
        consensus   = float(status['consensus_rating']) if status is not None and pd.notna(status['consensus_rating']) else None

        incumbent_dem = 0.5 if open_seat else (1 if incumbent_party == 'DEMOCRAT' else 0)
        pred_margin = float(model.predict([[state_lean, incumbent_dem, generic_ballot_margin]])[0])
        dem_win_prob = float(norm.sf(0, loc=pred_margin, scale=resid_std)) if resid_std > 0 else float(pred_margin > 0)

        rows.append({
            'state':            state,
            'incumbent_party':  incumbent_party,
            'open_seat':        open_seat,
            'predicted_margin': round(pred_margin, 2),
            'predicted_winner': 'DEMOCRAT' if pred_margin >= 0 else 'REPUBLICAN',
            'dem_win_prob':     round(dem_win_prob, 3),
            'consensus_rating': consensus,
        })

    seat_df = pd.DataFrame(rows)
    flips = seat_df[seat_df['incumbent_party'] != seat_df['predicted_winner']]
    return {
        'seats':          seat_df,
        'dem_seats_won':  int((seat_df['predicted_winner'] == 'DEMOCRAT').sum()),
        'rep_seats_won':  int((seat_df['predicted_winner'] == 'REPUBLICAN').sum()),
        'net_seats_flipped': len(flips),
        'flipped_states':   sorted(flips['state'].tolist()),
        'skipped_states':   skipped,  # incumbency/lean undeterminable from source data
    }
