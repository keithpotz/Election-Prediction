import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from feature_engineering import normalize_party, load_house_races, national_house_margin_by_year
from data_cleaning import load_and_merge
from race_status_2026 import load_or_scrape_race_status

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/polling_data')

HOUSE_SEATS_TOTAL = 435
HOUSE_MAJORITY    = 218

# How far the final generic-ballot polling average typically misses the actual national
# House vote (2010-2022 misses run ~1-3 pts), plus some room for movement in the last
# month. A judgment call, not fitted — used for both House and Senate simulations.
GENERIC_BALLOT_ERROR_SD = 3.0

N_SIMS = 20000


# ── House ──────────────────────────────────────────────────────────────────────
def _district_winners(house):
    idx = house.groupby(['year', 'state', 'district'])['candidatevotes'].idxmax()
    return house.loc[idx, ['year', 'state', 'district', 'party']]

def national_house_seats_by_year(house):
    winners = _district_winners(house)
    return winners.groupby(['year', 'party']).size().unstack(fill_value=0)

# Before the 1994 realignment Democrats won far more seats per vote (southern
# conservative Democrats), which inflates any seats-votes line fit across it.
HOUSE_TRAIN_START_YEAR = 1996

def house_seats_votes_table():
    house  = load_house_races()
    margin = national_house_margin_by_year(house)
    seats  = national_house_seats_by_year(house)
    df = pd.DataFrame({'margin': margin}).join(seats)
    df['dem_seat_share'] = df.get('DEMOCRAT', 0) / (df.get('DEMOCRAT', 0) + df.get('REPUBLICAN', 0))
    return df.dropna()

def evaluate_house_model(df=None):
    """Leave-one-cycle-out backtest of the seats-votes line (seat share error, in seats)."""
    df = df if df is not None else house_seats_votes_table()
    df = df[df.index >= HOUSE_TRAIN_START_YEAR]
    errs = []
    for year in df.index:
        train = df.drop(year)
        m = LinearRegression().fit(train[['margin']], train['dem_seat_share'])
        errs.append((df.loc[year, 'dem_seat_share'] - m.predict(df.loc[[year], ['margin']])[0]) * HOUSE_SEATS_TOTAL)
    errs = np.array(errs)
    return {'rmse_seats': float(np.sqrt((errs ** 2).mean())), 'n_years': len(errs)}

def build_house_swing_model():
    """
    Fits the classic 'seats-votes' relationship: national Dem House seat share as a
    linear function of the national Dem-Rep popular vote margin. Deliberately a simple
    1-feature linear fit: there are only ~15 post-1994 cycles, too few for anything more.
    Returns (model, resid_std) where resid_std comes from the leave-one-cycle-out
    backtest (out-of-sample), not the in-sample fit.
    """
    df = house_seats_votes_table()
    df = df[df.index >= HOUSE_TRAIN_START_YEAR]
    model = LinearRegression().fit(df[['margin']].values, df['dem_seat_share'].values)
    resid_std = evaluate_house_model()['rmse_seats'] / HOUSE_SEATS_TOTAL
    return model, resid_std

def predict_house_2026(generic_ballot_margin, n_sims=N_SIMS, seed=42, model_and_std=None):
    """
    Projects the 2026 House from a national generic ballot margin. The simulation draws
    two independent errors: how far the generic-ballot polls miss the real national vote
    (GENERIC_BALLOT_ERROR_SD), and how far seats deviate from the historical
    votes-to-seats line (the model's residual).

    model_and_std: optional pre-built (model, resid_std) from build_house_swing_model() —
    pass this in (e.g. cached via @st.cache_resource at the call site) to avoid reparsing
    house_1976_2024.csv and refitting the model on every call.
    """
    model, resid_std = model_and_std if model_and_std is not None else build_house_swing_model()
    point_share  = float(np.clip(model.predict([[generic_ballot_margin]])[0], 0, 1))
    dem_seats_pt = round(point_share * HOUSE_SEATS_TOTAL)

    rng = np.random.default_rng(seed)
    sim_margin = generic_ballot_margin + rng.normal(0, GENERIC_BALLOT_ERROR_SD, n_sims)
    sim_share  = model.intercept_ + model.coef_[0] * sim_margin + rng.normal(0, resid_std, n_sims)
    dem_seats_sim = np.round(np.clip(sim_share, 0, 1) * HOUSE_SEATS_TOTAL)

    return {
        'dem_seats':        int(dem_seats_pt),
        'rep_seats':        HOUSE_SEATS_TOTAL - int(dem_seats_pt),
        'dem_control_prob': round(float((dem_seats_sim >= HOUSE_MAJORITY).mean()), 3),
        'dem_seat_lo':      int(np.percentile(dem_seats_sim, 5)),
        'dem_seat_hi':      int(np.percentile(dem_seats_sim, 95)),
    }


# ── Senate ─────────────────────────────────────────────────────────────────────
SENATE_SEATS_TOTAL = 100
# With a Republican VP (Vance) breaking ties, Democrats need 51 for control.
SENATE_DEM_CONTROL_SEATS = 51
# Holdover-seat winners the source data labels OTHER, mapped to the party they caucus
# with. Any other OTHER winner raises, so this can't silently go stale.
INDEPENDENT_CAUCUS = {
    (2024, 'MAINE'):   'DEMOCRAT',    # Angus King
    (2024, 'VERMONT'): 'DEMOCRAT',    # Bernie Sanders
}
# Major candidates with no party at all in the source file (party_simplified and
# party_detailed both blank).
PARTY_CORRECTIONS = {
    (2010, 'ALASKA',  'LISA MURKOWSKI'):     'REPUBLICAN',  # write-in campaign
    (2020, 'WYOMING', 'CYNTHIA M. LUMMIS'):  'REPUBLICAN',
    (2020, 'WYOMING', 'MERAV BEN DAVID'):    'DEMOCRAT',
    (2022, 'ALASKA',  'MURKOWSKI, LISA'):    'REPUBLICAN',
    (2022, 'ALASKA',  'TSHIBAKA, KELLY C.'): 'REPUBLICAN',
}
# Regular races absent from the source file entirely, needed for holdover counts.
MISSING_RACE_WINNERS = {
    (2022, 'CONNECTICUT'): 'DEMOCRAT',  # Richard Blumenthal
}
# Partisan composition entering 2026 (119th Congress, Dems incl. 2 independents).
# Only used as a consistency check on the data-derived seat counts.
CURRENT_SENATE = {'DEMOCRAT': 47, 'REPUBLICAN': 53}
# An appointed senator running for the first time gets half the elected-incumbent
# effect. Judgment call: appointees historically underperform elected incumbents,
# and there's no clean appointee ground truth in the vote data to fit this.
APPOINTED_INCUMBENCY_WEIGHT = 0.5

def _read_senate_raw():
    df = pd.read_csv(os.path.join(DATA_DIR, 'senate_1976_2024.csv'), sep='\t')
    df['stage']   = df['stage'].astype(str).str.upper()
    df['mode']    = df['mode'].astype(str).str.upper()
    df['special'] = df['special'].astype(str).str.upper() == 'TRUE'
    df = df[(df['office'] == 'US SENATE') & (df['mode'] == 'TOTAL')].copy()
    df['party'] = normalize_party(df['party_simplified'].fillna(''))
    # Some major-party candidates are simplified to OTHER even though party_detailed
    # says otherwise (e.g. Duckworth/Van Hollen 2022 'DEMOCRATIC', Mattingly 2000
    # 'REPUBLICAN (NOT IDENTIFIED ON BALLOT)').
    detailed = df['party_detailed'].fillna('').str.upper()
    df.loc[(df['party'] == 'OTHER') & detailed.str.startswith('DEMOCRAT'), 'party'] = 'DEMOCRAT'
    df.loc[(df['party'] == 'OTHER') & detailed.str.startswith('REPUBLICAN'), 'party'] = 'REPUBLICAN'
    for (year, state, cand), party in PARTY_CORRECTIONS.items():
        df.loc[(df['year'] == year) & (df['state'] == state) & (df['candidate'] == cand), 'party'] = party
    # Fusion voting (CT, NY): a candidate's votes on minor-party lines (Working Families,
    # Conservative, ...) belong to their major party, not OTHER.
    major = df[df['party'].isin(['DEMOCRAT', 'REPUBLICAN']) & (df['candidate'].fillna('') != '')]
    major_party = major.drop_duplicates(['year', 'state', 'special', 'stage', 'candidate'])                        .set_index(['year', 'state', 'special', 'stage', 'candidate'])['party']
    keys = pd.MultiIndex.from_frame(df[['year', 'state', 'special', 'stage', 'candidate']])
    fused = major_party.reindex(keys).values
    df['party'] = np.where(pd.notna(fused), fused, df['party'])
    df['candidatevotes'] = pd.to_numeric(df['candidatevotes'], errors='coerce').fillna(0)
    return df

def load_senate_races():
    """Regular (non-special) general-election rows."""
    df = _read_senate_raw()
    return df[(df['stage'] == 'GEN') & (~df['special'])].copy()

def _runoff_winners():
    """
    (election_year, state) -> winning party of a regular-seat runoff. Georgia's Jan 2021
    runoffs are dated 2021 in the source file, so odd years are folded back to the
    preceding even election year. Without this, a candidate who led the November round
    but lost the runoff (Perdue 2020) is wrongly treated as the seat's winner.
    """
    df = _read_senate_raw()
    df = df[df['stage'].str.contains('RUNOFF') & (~df['special'])]
    if df.empty:
        return {}
    df = df.assign(year=df['year'] - (df['year'] % 2))
    idx = df.groupby(['year', 'state'])['candidatevotes'].idxmax()
    return {(r['year'], r['state']): r['party'] for _, r in df.loc[idx].iterrows()}

def _last_name(name):
    tokens = [t for t in re.sub(r'[^A-Z ]', ' ', str(name).upper()).split()
              if t not in {'JR', 'SR', 'II', 'III', 'IV'}]
    return tokens[-1] if tokens else ''

def senate_races_table():
    """
    One row per regular (year, state) Senate race: Dem/Rep votes, two-party margin,
    winning party (runoff-corrected), winner's name, and the set of candidate last names
    on the ballot (used to tell whether the previous winner ran again).
    """
    df = load_senate_races()
    votes = df.groupby(['year', 'state', 'state_po', 'party'])['candidatevotes'].sum().unstack(fill_value=0)
    dem, rep = votes.get('DEMOCRAT', 0), votes.get('REPUBLICAN', 0)
    out = pd.DataFrame({'dem_votes': dem, 'rep_votes': rep}).reset_index()
    two_party = out['dem_votes'] + out['rep_votes']
    out['margin'] = ((out['dem_votes'] - out['rep_votes']) / two_party.replace(0, np.nan) * 100).round(2)
    out['contested'] = (out['dem_votes'] > 0) & (out['rep_votes'] > 0)

    top = df.loc[df.groupby(['year', 'state'])['candidatevotes'].idxmax(), ['year', 'state', 'party', 'candidate']]
    top = top.rename(columns={'party': 'winner', 'candidate': 'winner_name'})
    out = out.merge(top, on=['year', 'state'], how='left')

    for key, party in _runoff_winners().items():
        mask = (out['year'] == key[0]) & (out['state'] == key[1])
        out.loc[mask, 'winner'] = party

    names = df.groupby(['year', 'state'])['candidate'].apply(lambda s: {_last_name(n) for n in s})
    out['candidate_last_names'] = out.set_index(['year', 'state']).index.map(names).values
    return out

def presidential_lean_by_year():
    """
    {pres_year: {state: lean}}, where lean is the state's two-party presidential margin
    minus the national two-party margin (Dem-positive). The same idea as Cook PVI.
    """
    pres = load_and_merge(
        os.path.join(DATA_DIR, 'complete_data.csv'),
        os.path.join(DATA_DIR, '2024president.csv'),
    )
    if 'stage' in pres.columns:
        pres = pres[(pres['stage'] == 'GEN') | pres['stage'].isna()]
    pres['party'] = pres['party_simplified'].str.upper()
    pres = pres[pres['party'].isin(['DEMOCRAT', 'REPUBLICAN'])]
    by_state = pres.groupby(['year', 'state', 'party'])['votes'].sum().unstack(fill_value=0)
    state_margin = (by_state['DEMOCRAT'] - by_state['REPUBLICAN']) / (by_state['DEMOCRAT'] + by_state['REPUBLICAN']) * 100
    nat = by_state.groupby(level='year').sum()
    nat_margin = (nat['DEMOCRAT'] - nat['REPUBLICAN']) / (nat['DEMOCRAT'] + nat['REPUBLICAN']) * 100
    rel = state_margin - nat_margin.reindex(state_margin.index.get_level_values('year')).values
    return {y: grp.droplevel('year').to_dict() for y, grp in rel.groupby(level='year')}

def pvi_style_lean(year, state, pres_lean):
    """
    Partisan lean entering a race in `year`: 75% the most recent presidential election
    strictly before `year`, 25% the one before that (Cook PVI's weighting). Strictly
    before, so a presidential-year race never sees that same day's presidential result.
    """
    p1 = year - 2 if year % 4 == 2 else year - 4
    l1, l2 = pres_lean.get(p1, {}).get(state), pres_lean.get(p1 - 4, {}).get(state)
    if l1 is None:
        return None
    return 0.75 * l1 + 0.25 * l2 if l2 is not None else l1

SENATE_FEATURES = ['lean', 'incumbency', 'national_environment', 'incumbent_strength']
# Pre-2000 Senate races were far less nationalized (lots of split-ticket voting), so the
# model trains on the polarized era only. Backtested: this plus incumbent_strength cut
# out-of-sample error from ~15.9 to ~13.6 pts.
TRAIN_START_YEAR = 2000

def incumbent_strength(prev_margin, lean, incumbency):
    """How much the running incumbent beat (or trailed) their state's lean last time.
    Zero for open seats. Previous margin clipped to +/-60 so uncontested races (+/-100)
    don't dominate."""
    if incumbency == 0 or prev_margin is None or pd.isna(prev_margin):
        return 0.0
    return float(np.clip(prev_margin, -60, 60) - lean)

def senate_training_data():
    """
    Historical contested regular Senate races with:
      lean                  PVI-style state lean entering the race
      incumbency            +1 Dem incumbent running, -1 Rep incumbent running, 0 open
                            seat (the seat's previous winner is not on the ballot)
      national_environment  national House popular-vote margin that year
      incumbent_strength    see incumbent_strength()
    """
    races = senate_races_table()
    pres_lean = presidential_lean_by_year()
    house_margin = national_house_margin_by_year()

    prev = races.set_index(['year', 'state'])
    rows = []
    for _, r in races[races['contested'] & (races['year'] >= TRAIN_START_YEAR)].iterrows():
        key = (r['year'] - 6, r['state'])
        if key not in prev.index:
            continue
        p = prev.loc[key]
        if pd.isna(p['winner']) or p['winner'] not in ('DEMOCRAT', 'REPUBLICAN'):
            continue
        running = _last_name(p['winner_name']) in r['candidate_last_names']
        incumbency = (1 if p['winner'] == 'DEMOCRAT' else -1) if running else 0
        lean = pvi_style_lean(r['year'], r['state'], pres_lean)
        env = house_margin.get(r['year'])
        if lean is None or pd.isna(env):
            continue
        rows.append({'year': r['year'], 'state': r['state'], 'margin': r['margin'],
                     'lean': lean, 'incumbency': incumbency, 'national_environment': env,
                     'incumbent_strength': incumbent_strength(p['margin'], lean, incumbency)})
    return pd.DataFrame(rows)

def evaluate_senate_model(train=None):
    """
    Leave-one-cycle-out backtest: for each election year, fit on every other year and
    predict that year's races. Out-of-sample residuals give honest error estimates,
    split into a shared per-year national miss and a per-race state miss.
    """
    train = train if train is not None else senate_training_data()
    preds = pd.Series(index=train.index, dtype=float)
    for year in train['year'].unique():
        test = train['year'] == year
        m = LinearRegression().fit(train.loc[~test, SENATE_FEATURES], train.loc[~test, 'margin'])
        preds[test] = m.predict(train.loc[test, SENATE_FEATURES])
    resid = train['margin'] - preds
    year_miss = resid.groupby(train['year']).mean()
    state_miss = resid - train['year'].map(year_miss)
    return {
        'rmse':            float(np.sqrt((resid ** 2).mean())),
        'winner_accuracy': float(((preds > 0) == (train['margin'] > 0)).mean()),
        'national_sd':     float(year_miss.std(ddof=1)),
        'state_sd':        float(state_miss.std(ddof=1)),
        'n_races':         int(len(train)),
        'n_years':         int(train['year'].nunique()),
    }

def build_senate_race_model():
    """Fits the race model on all history; returns (model, error_stats, pres_lean)."""
    train = senate_training_data()
    model = LinearRegression().fit(train[SENATE_FEATURES], train['margin'])
    return model, evaluate_senate_model(train), presidential_lean_by_year()

def load_race_status():
    """Real 2026 open-seat/consensus-rating data (scraped from Wikipedia); empty if unavailable."""
    try:
        df = load_or_scrape_race_status()
        return df.set_index('state')
    except Exception as e:
        print(f"2026 race status unavailable, falling back to class-based incumbency only: {e}")
        return pd.DataFrame(columns=['state', 'open_seat', 'consensus_rating']).set_index('state')

def _seat_winner(races, year, state):
    hit = races[(races['year'] == year) & (races['state'] == state)]
    if hit.empty or pd.isna(hit['winner'].iloc[0]):
        return None
    winner = hit['winner'].iloc[0]
    if winner == 'OTHER':
        if (year, state) not in INDEPENDENT_CAUCUS:
            raise ValueError(f"{year} {state} Senate winner is labeled OTHER and isn't in INDEPENDENT_CAUCUS")
        winner = INDEPENDENT_CAUCUS[(year, state)]
    return winner

def senate_seats_2026(races, race_status):
    """
    Seats on the 2026 ballot (Class II seats last elected in 2020, plus any specials
    listed in the scraped race status) and the holdover seats not up this cycle.
    Specials are assumed to be for Class III seats (last elected 2022) — true of 2026's
    Florida and Ohio specials (the Rubio and Vance seats).
    """
    specials = []
    if 'is_special' in race_status.columns:
        specials = sorted(race_status.index[race_status['is_special'].astype(bool)])

    up = [(s, 2020) for s in sorted(races.loc[races['year'] == 2020, 'state'].unique())]
    up += [(s, 2022) for s in specials]

    holdovers = {'DEMOCRAT': 0, 'REPUBLICAN': 0}
    for (year, state), party in MISSING_RACE_WINNERS.items():
        if year in (2022, 2024) and not (year == 2022 and state in specials):
            holdovers[party] += 1
    for year in (2022, 2024):
        for state in races.loc[races['year'] == year, 'state'].unique():
            if year == 2022 and state in specials:
                continue
            w = _seat_winner(races, year, state)
            if w is not None:
                holdovers[w] += 1
    return up, holdovers

def predict_senate_2026(generic_ballot_margin, model_bundle=None, race_status_df=None,
                        n_sims=N_SIMS, seed=42):
    """
    Projects every 2026 Senate race and chamber control.

    Each simulation draws ONE shared national miss (the generic ballot polling error,
    scaled by the model's national coefficient, plus the backtest's per-year national
    miss) applied to every race, then an independent per-race miss. Races are therefore
    correlated, the way real election-night errors are: if the national environment is
    off, it's off everywhere at once.

    model_bundle: optional pre-built (model, error_stats, pres_lean) from
    build_senate_race_model(); race_status_df: optional load_race_status() result.
    Pass both in from cached wrappers to avoid refitting/re-scraping on every call.
    """
    model, err, pres_lean = model_bundle if model_bundle is not None else build_senate_race_model()
    race_status = race_status_df if race_status_df is not None else load_race_status()
    races = senate_races_table()
    up, holdovers = senate_seats_2026(races, race_status)

    rows, skipped, warnings = [], [], []
    for state, seat_year in up:
        status = race_status.loc[state] if state in race_status.index else None
        data_party = _seat_winner(races, seat_year, state)
        scraped_party = status.get('last_election_party') if status is not None else None
        scraped_party = scraped_party if isinstance(scraped_party, str) else None
        if scraped_party and data_party and scraped_party != data_party:
            warnings.append(f"{state}: data says {data_party}, Wikipedia says {scraped_party}; using Wikipedia")
        incumbent_party = scraped_party or data_party
        lean = pvi_style_lean(2026, state, pres_lean)
        if incumbent_party is None or lean is None:
            skipped.append(state)
            continue

        open_seat = bool(status['open_seat']) if status is not None else False
        appointed = bool(status.get('appointed', False)) if status is not None else False
        sign = 1 if incumbent_party == 'DEMOCRAT' else -1
        incumbency = 0 if open_seat else sign * (APPOINTED_INCUMBENCY_WEIGHT if appointed else 1)
        consensus = status['consensus_rating'] if status is not None else np.nan
        # An appointee didn't earn the previous result, so it says nothing about them.
        prev = races[(races['year'] == seat_year) & (races['state'] == state)]['margin']
        prev_margin = None if appointed or prev.empty else prev.iloc[0]
        strength = incumbent_strength(prev_margin, lean, incumbency)

        x = pd.DataFrame([[lean, incumbency, generic_ballot_margin, strength]], columns=SENATE_FEATURES)
        rows.append({
            'state':            state,
            'special':          seat_year != 2020,
            'incumbent_party':  incumbent_party,
            'open_seat':        open_seat,
            'lean':             round(lean, 1),
            'predicted_margin': float(model.predict(x)[0]),
            'consensus_rating': float(consensus) if pd.notna(consensus) else None,
        })
    seat_df = pd.DataFrame(rows)

    # Correlated simulation
    env_coef = model.coef_[SENATE_FEATURES.index('national_environment')]
    national_sd = float(np.hypot(err['national_sd'], env_coef * GENERIC_BALLOT_ERROR_SD))
    rng = np.random.default_rng(seed)
    national_miss = rng.normal(0, national_sd, (n_sims, 1))
    race_miss = rng.normal(0, err['state_sd'], (n_sims, len(seat_df)))
    sim_margins = seat_df['predicted_margin'].values + national_miss + race_miss
    dem_wins = sim_margins > 0
    dem_seats_sim = holdovers['DEMOCRAT'] + dem_wins.sum(axis=1)

    seat_df['dem_win_prob']     = dem_wins.mean(axis=0).round(3)
    seat_df['predicted_winner'] = np.where(seat_df['predicted_margin'] >= 0, 'DEMOCRAT', 'REPUBLICAN')
    seat_df['predicted_margin'] = seat_df['predicted_margin'].round(2)

    up_by_party = seat_df['incumbent_party'].value_counts()
    implied_current = {p: holdovers[p] + int(up_by_party.get(p, 0)) for p in holdovers}
    if not skipped and implied_current != CURRENT_SENATE:
        warnings.append(f"Seat counts {implied_current} don't match current Senate {CURRENT_SENATE}")

    flips = seat_df[seat_df['incumbent_party'] != seat_df['predicted_winner']]
    return {
        'seats':             seat_df,
        'dem_seats_won':     int((seat_df['predicted_winner'] == 'DEMOCRAT').sum()),
        'rep_seats_won':     int((seat_df['predicted_winner'] == 'REPUBLICAN').sum()),
        'net_seats_flipped': len(flips),
        'flipped_states':    sorted(flips['state'].tolist()),
        'skipped_states':    skipped,
        'holdovers':         holdovers,
        'dem_seats_median':  int(np.median(dem_seats_sim)),
        'dem_seats_lo':      int(np.percentile(dem_seats_sim, 5)),
        'dem_seats_hi':      int(np.percentile(dem_seats_sim, 95)),
        'dem_control_prob':  round(float((dem_seats_sim >= SENATE_DEM_CONTROL_SEATS).mean()), 3),
        'dem_seats_dist':    pd.Series(dem_seats_sim).value_counts(normalize=True).sort_index(),
        'error_stats':       err,
        'warnings':          warnings,
    }
