"""
Election-night mode. Record races as they're called; the chamber odds and every uncalled
race update from the same 20,000 simulations, re-weighted toward the ones that match the
calls (simulations.election_night). Calls are saved to forecasts/2026/results_2026.csv,
the same file grade_2026.py scores after the election.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import simulations as SIMS
from dashboard_2026 import (DEM, REP, IND, NEUTRAL, DEM_DEEP, REP_DEEP, STATE_ABBREV, _title_case, fmt_margin,
                            seat_bar, _seat_header)
from grade_2026 import RESULTS_CSV, RESULT_COLUMNS, make_results_template, is_decided

OFFICE_LABEL = {'senate': 'Senate', 'governor': 'Governor', 'house': 'House'}


def race_key(office, state, district=None):
    if office == 'house':
        d = int(district)
        return f"{STATE_ABBREV.get(state, state)}-{'AL' if d == 0 else d}"
    return f"{OFFICE_LABEL[office]} · {_title_case(state)}"


def load_results():
    # Blank columns would otherwise load as floats and refuse text
    r = pd.read_csv(RESULTS_CSV, dtype={'winner_party': 'object', 'source_url': 'object', 'entered_at': 'object'})
    r['district'] = pd.to_numeric(r['district'], errors='coerce')
    return r


def _row_mask(res, office, state, district):
    m = (res['office'] == office) & (res['state'] == state)
    if office == 'house':
        m &= res['district'] == district
    return m


def save_call(office, state, district, winner_party, dem_pct=None, rep_pct=None, source=''):
    res = load_results()
    m = _row_mask(res, office, state, district)
    if not m.any():
        raise KeyError(f"{office} {state} {district} isn't in {RESULTS_CSV}")
    res.loc[m, 'winner_party'] = winner_party
    res.loc[m, 'dem_side_pct'] = dem_pct
    res.loc[m, 'rep_pct'] = rep_pct
    res.loc[m, 'source_url'] = source or None
    res.loc[m, 'entered_at'] = datetime.now().isoformat(timespec='seconds')
    res.reindex(columns=RESULT_COLUMNS).to_csv(RESULTS_CSV, index=False)


def clear_call(office, state, district):
    res = load_results()
    m = _row_mask(res, office, state, district)
    for c in ('winner_party', 'dem_side_pct', 'rep_pct', 'source_url', 'entered_at'):
        res.loc[m, c] = None
    res.reindex(columns=RESULT_COLUMNS).to_csv(RESULTS_CSV, index=False)


def _races(f):
    """Every race with its forecast win chance (Democratic side), keyed the way calls are."""
    rows = []
    for office in ('senate', 'governor'):
        for _, r in f[office].iterrows():
            rows.append({'office': office, 'state': r['state'], 'district': np.nan, 'label': race_key(office, r['state']),
                         'dem_side_party': r['dem_side_party'], 'dem_candidate': r['dem_candidate'],
                         'rep_candidate': r['rep_candidate'], 'p': r['dem_side_win_prob']})
    for _, r in f['house'].iterrows():
        rows.append({'office': 'house', 'state': r['state'], 'district': int(r['district']),
                     'label': race_key('house', r['state'], r['district']), 'dem_side_party': 'DEMOCRAT',
                     'dem_candidate': r['dem_candidate'], 'rep_candidate': r['rep_candidate'], 'p': r['dem_win_prob']})
    return pd.DataFrame(rows)


def render_entry(races, res):
    called = res[is_decided(res['winner_party'])]
    called_keys = set(zip(called['office'], called['state'], called['district'].fillna(-1)))
    with st.expander("✍️ Record a call", expanded=True):
        c = st.columns([1, 2, 2])
        office = c[0].radio("Office", ['senate', 'governor', 'house'], format_func=OFFICE_LABEL.get, key='night_office')
        uncalled = np.array([(o, s, d if pd.notna(d) else -1) not in called_keys
                             for o, s, d in zip(races['office'], races['state'], races['district'])])
        open_races = races[(races['office'] == office).values & uncalled]
        open_races = open_races.loc[(open_races['p'] - 0.5).abs().sort_values().index]
        if open_races.empty:
            c[1].info("Every race for this office is called.")
            return
        pick = c[1].selectbox("Race (closest first)", open_races['label'].tolist(), key='night_race')
        r = open_races[open_races['label'] == pick].iloc[0]
        d_party = 'INDEPENDENT' if r['dem_side_party'] == 'INDEPENDENT' else 'DEMOCRAT'
        names = {d_party: f"{r['dem_candidate'] or 'Democrat'} ({d_party[0]})",
                 'REPUBLICAN': f"{r['rep_candidate'] or 'Republican'} (R)"}
        winner = c[2].radio("Winner", list(names), format_func=names.get, key='night_winner', horizontal=True)
        c2 = st.columns([1, 1, 3, 1])
        dem_pct = c2[0].number_input(f"{d_party.title()} %", 0.0, 100.0, value=None, step=0.1, key='night_dpct')
        rep_pct = c2[1].number_input("Republican %", 0.0, 100.0, value=None, step=0.1, key='night_rpct')
        source = c2[2].text_input("Source (link to the call or official results)", key='night_src')
        c2[3].write("")
        if c2[3].button("Record call", type='primary', key='night_save'):
            save_call(office, r['state'], r['district'] if office == 'house' else None, winner, dem_pct, rep_pct, source)
            st.rerun()
        st.caption(f"Our forecast gave {names[d_party]} a {r['p']:.0%} chance. Calls are saved to "
                   "forecasts/2026/results_2026.csv (the file the post-election grading reads), so you can also "
                   "edit that file directly. Vote shares are optional tonight; fill them in when results are final.")
    if len(called):
        with st.expander("↩️ Undo a call"):
            labels = [race_key(o, s, d) for o, s, d in zip(called['office'], called['state'], called['district'])]
            u = st.selectbox("Called race", labels, key='night_undo_pick')
            if st.button("Remove this call", key='night_undo'):
                row = called.iloc[labels.index(u)]
                clear_call(row['office'], row['state'], row['district'] if row['office'] == 'house' else None)
                st.rerun()


def _senate_segments(f, res, night):
    hold = f['sims']['senate_holdovers']
    s = res[(res['office'] == 'senate') & is_decided(res['winner_party'])]
    cd, cr = int((s['winner_party'] == 'DEMOCRAT').sum()), int((s['winner_party'] == 'REPUBLICAN').sum())
    ci = int((s['winner_party'] == 'INDEPENDENT').sum())
    open_n = len(f['senate']) - cd - cr - ci
    return [(hold['DEMOCRAT'], DEM_DEEP, 1.0, f"{hold['DEMOCRAT']} Democratic seats not up"),
            (cd, DEM, 1.0, f"{cd} called for Democrats"),
            (ci, IND, 1.0, f"{ci} called for an independent"),
            (open_n, NEUTRAL, 1.0, f"{open_n} not yet called"),
            (cr, REP, 1.0, f"{cr} called for Republicans"),
            (hold['REPUBLICAN'], REP_DEEP, 1.0, f"{hold['REPUBLICAN']} Republican seats not up")], cd, cr


def _house_segments(res):
    h = res[res['office'] == 'house']
    cd, cr = int((h['winner_party'] == 'DEMOCRAT').sum()), int((h['winner_party'] == 'REPUBLICAN').sum())
    return [(cd, DEM, 1.0, f"{cd} called for Democrats"), (435 - cd - cr, NEUTRAL, 1.0, f"{435 - cd - cr} not yet called"),
            (cr, REP, 1.0, f"{cr} called for Republicans")], cd, cr


def render_election_night(f, model_version):
    st.title("🗳️ Election Night")
    st.caption("Record races as they're called. Chamber odds and every uncalled race update together: a surprise in "
               "one state says something about how the polls missed, and that moves similar races too. This "
               "re-weights the forecast's 20,000 simulations toward the ones that match the calls so far.")
    if f.get('sims') is None:
        st.error("This forecast was built without its simulations. Restart the app.")
        return
    if not os.path.exists(RESULTS_CSV):
        st.info("No results file yet. Setting one up creates forecasts/2026/results_2026.csv with a blank row for "
                "every Senate, governor, and House race in the latest published forecast.")
        if st.button("Set up the results file", key='night_setup'):
            make_results_template()
            st.rerun()
        return
    res = load_results()
    races = _races(f)
    calls = SIMS.calls_from_results(res, f['sims'])
    night = SIMS.election_night(f['sims'], calls)
    n_called = int(is_decided(res['winner_party']).sum())

    s0, h0 = f['summary']['senate'], f['summary']['house']
    sn, hn = night['senate'], night['house']
    st.markdown(f"**{n_called} of {len(res)} races called.** "
                + (f"The odds below lean on about {night['ess']:,.0f} of the 20,000 simulations that best match the calls."
                   if calls else "No calls yet, so the numbers below are the forecast."))
    pending = res[res['winner_party'].notna() & ~is_decided(res['winner_party'])]
    if len(pending):
        st.caption("Still undecided (not counted for either party): " + ", ".join(
            f"{race_key(o, s, d)} ({w.title()})" for o, s, d, w in
            zip(pending['office'], pending['state'], pending['district'], pending['winner_party']))
            + ". Record the winner when it's decided; it stays in the race list until then.")
    if calls and night['ess'] < 300:
        st.warning("Few simulations match the calls so far, so tonight's results are outside what the forecast "
                   "expected. Treat the updated odds as rough.")

    st.subheader("Senate")
    seg, cd, cr = _senate_segments(f, res, night)
    _seat_header(f"{sn['dem_control']:.0%}", f"{sn['rep_control']:.0%}",
                 f"chance of Democratic control (forecast {s0['dem_control_prob']:.0%}) · projected {sn['dem_median']} seats",
                 f"chance of Republican control (forecast {s0['rep_control_prob']:.0%}) · projected {sn['rep_median']} seats")
    st.plotly_chart(seat_bar(seg, 100, 50.5, "51 for Democratic control"), width='stretch', key='night_sen_bar',
                    config={'displayModeBar': False})
    st.subheader("House")
    seg, hd, hr = _house_segments(res)
    _seat_header(f"{hn['dem_control']:.0%}", f"{1 - hn['dem_control']:.0%}",
                 f"chance of Democratic control (forecast {h0['dem_control_prob']:.0%}) · projected {hn['dem_median']} seats",
                 f"chance of Republican control · projected {435 - hn['dem_median']} seats")
    st.plotly_chart(seat_bar(seg, 435, 217.5, "218 for a majority"), width='stretch', key='night_house_bar',
                    config={'displayModeBar': False})
    g = night['governor']
    st.caption(f"Governors: Democrats projected to win {g['dem_median']} of 36 (90% range {g['dem_90'][0]}–{g['dem_90'][1]}).")

    st.divider()
    render_entry(races, res)

    st.subheader("Senate races that matter most now")
    sw = SIMS.race_swing(f['sims'], night, 'senate')
    if len(sw):
        states = f['sims']['senate']['states']
        sw = sw.head(8)
        st.dataframe(pd.DataFrame({
            'Race': [_title_case(states[c]) for c in sw['col']],
            'Dem-side win chance now': sw['p_win'] * 100,
            'Dem control if Dem side wins': sw['if_win'] * 100,
            'Dem control if Republican wins': sw['if_lose'] * 100,
        }), hide_index=True, width='stretch', column_config={
            c: st.column_config.ProgressColumn(color=DEM, format='%.0f%%', min_value=0, max_value=100)
            for c in ('Dem-side win chance now', 'Dem control if Dem side wins', 'Dem control if Republican wins')})
        st.caption("Uncalled races ranked by how much their result moves Democratic Senate control.")

    called = res[is_decided(res['winner_party'])].merge(races[['office', 'state', 'district', 'p', 'dem_side_party']],
                                                    on=['office', 'state'], how='left', suffixes=('', '_f'))
    called = called[(called['office'] != 'house') | (called['district'] == called['district_f'])]
    if len(called):
        st.subheader("Called races vs. our forecast")
        won = np.where(called['office'] == 'house', called['winner_party'] == 'DEMOCRAT',
                       called['winner_party'] == called['dem_side_party'])
        p_winner = np.where(won, called['p'], 1 - called['p'])
        right = p_winner >= 0.5
        brier = float(np.mean((called['p'] - won) ** 2))
        c = st.columns(3)
        c[0].metric("Favorite won", f"{int(right.sum())} of {len(called)}")
        c[1].metric("Brier score", f"{brier:.3f}", help="Mean squared error of our win probabilities (0 is perfect, "
                                                        "0.25 is a coin flip). Our 2000–2024 backtest scored 0.067.")
        c[2].metric("Upsets (winner had < 30%)", int((p_winner < 0.3).sum()))
        t = pd.DataFrame({
            'Race': [race_key(o, s, d) for o, s, d in zip(called['office'], called['state'], called['district'])],
            'Winner': called['winner_party'].str.title(),
            'Our chance for the winner': p_winner * 100,
            'Favorite won': np.where(right, '✓', '✗ upset'),
            'Called at': pd.to_datetime(called['entered_at']).dt.strftime('%H:%M'),
        }).iloc[::-1]
        st.dataframe(t, hide_index=True, width='stretch', height=min(35 * (len(t) + 1) + 3, 420), column_config={
            'Our chance for the winner': st.column_config.ProgressColumn(color=DEM, format='%.0f%%', min_value=0,
                                                                         max_value=100)})

    st.subheader("Uncalled races: forecast vs. now")
    office = st.radio("Office", ['senate', 'governor', 'house'], format_func=OFFICE_LABEL.get, horizontal=True,
                      key='night_view')
    probs = night['race_probs'][office]
    called_cols = night['called'][office]
    sub = races[races['office'] == office].reset_index(drop=True)
    if office == 'house':
        idx = {k: i for i, k in enumerate(f['sims']['house']['keys'])}
        cols = [idx[(s, int(d))] for s, d in zip(sub['state'], sub['district'])]
    else:
        idx = {s: i for i, s in enumerate(f['sims'][office]['states'])}
        cols = [idx[s] for s in sub['state']]
    sub['now'] = [probs[c] for c in cols]
    sub = sub[[c not in called_cols for c in cols]]
    sub = sub[(sub['p'].between(0.03, 0.97)) | (sub['now'].between(0.03, 0.97))]
    sub = sub.assign(change=sub['now'] - sub['p']).sort_values('now')
    st.dataframe(pd.DataFrame({
        'Race': sub['label'],
        'Dem side': sub['dem_candidate'].fillna('—'),
        'Republican': sub['rep_candidate'].fillna('—'),
        'Forecast': sub['p'] * 100,
        'Now': sub['now'] * 100,
        'Change': (sub['change'] * 100).map(lambda v: f"{v:+.0f} pts"),
    }), hide_index=True, width='stretch', height=min(35 * (len(sub) + 1) + 3, 520), column_config={
        c: st.column_config.ProgressColumn(color=DEM, format='%.0f%%', min_value=0, max_value=100)
        for c in ('Forecast', 'Now')})
    st.caption("Dem-side win chances for races still in doubt (between 3% and 97%). Model "
               f"{model_version}; the forecast column is the forecast loaded in this app.")
