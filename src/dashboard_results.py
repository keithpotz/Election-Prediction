"""
2026 results: how the forecast did. Reads forecasts/2026/results_2026.csv (never writes it)
and the published daily snapshots, and scores them the way grade_2026.py does.

The final forecast is the last snapshot published on or before election day. A snapshot
published after polls closed (the Nov 4 morning run) isn't graded.
Races whose winner isn't a party yet (e.g. "RUNOFF") are treated as pending.
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import grade_2026 as GR
from dashboard_2026 import (DEM, REP, IND, NEUTRAL, INK_2, GRID, STATE_ABBREV, _layout, _title_case, fmt_margin,
                            FORECASTS_DIR)

ELECTION_DATE, PARTIES = GR.ELECTION_DATE, GR.WINNER_PARTIES
PARTY_COLOR = {'DEMOCRAT': DEM, 'REPUBLICAN': REP, 'INDEPENDENT': IND}
OFFICE_LABEL = {'senate': 'Senate', 'governor': 'Governor', 'house': 'House'}
HOUSE_STATUS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'polling_data',
                                'house_districts_2026.csv')
POLLS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'polls_2026', 'polls_2026.csv')
TOPLINE_CSV = os.path.join(FORECASTS_DIR, '2026', 'topline_log.csv')
POLL_WINDOW_DAYS = 21          # pollster scorecard: polls ending in the final three weeks
MIN_POLLSTER_POLLS = 2
SENATE_TOTAL, SENATE_DEM_CONTROL, SENATE_REP_CONTROL, HOUSE_MAJORITY = 100, 51, 50, 218


def race_label(office, state, district=None):
    if office == 'house':
        d = int(district)
        return f"{STATE_ABBREV.get(state, state)}-{'AL' if d == 0 else d}"
    return f"{OFFICE_LABEL[office]} · {_title_case(state)}"


def _summary(date):
    with open(os.path.join(GR.FORECAST_DIR, f'{date}_summary.json')) as fh:
        return json.load(fh)


# ── Data ──────────────────────────────────────────────────────────────────────
def load_all_results():
    r = pd.read_csv(GR.RESULTS_CSV, dtype={'winner_party': 'object', 'source_url': 'object', 'entered_at': 'object'})
    r['district'] = pd.to_numeric(r['district'], errors='coerce').astype('Int64')
    r['winner_party'] = r['winner_party'].str.strip().str.upper()
    return r


def actual_margin(df):
    """Dem side minus Republican, only where both shares are entered and it was a real two-party race."""
    d, r = df['dem_side_pct'], df['rep_pct']
    ok = d.notna() & r.notna() & (d > 0) & (r > 0) & ((d + r) <= 100.5)
    return (d - r).where(ok)


def _held_house():
    h = pd.read_csv(HOUSE_STATUS_CSV)
    h['district'] = h['district'].astype('Int64')
    return h[['state', 'district', 'held_by']]


@st.cache_data(show_spinner=False)
def build(results_mtime):
    """Everything the page draws. Keyed on the results file's modified time, so new entries show on rerun."""
    allres = load_all_results()
    res = allres[allres['winner_party'].isin(PARTIES)].copy()
    dates = GR.graded_dates()
    final_date = dates[-1]
    keys = ['office', 'state', 'district']

    # Scores for every snapshot, on the races with results
    rows = []
    for date in dates:
        snap = GR._snapshot(date).merge(res[keys + ['winner_party']], on=keys, how='inner')
        snap['won'] = snap['winner_party'] == snap['dem_side_party']
        for office, g in [('all', snap)] + list(snap.groupby('office')):
            for method, col in (('Our forecast', 'model_p'), ('Polls only', 'polls_p'),
                                ('Fundamentals only', 'fund_p'), ('Experts', 'experts_p')):
                if col in g:
                    s = GR._scores(g, col)
                    if s:
                        rows.append({'snapshot': date, 'office': office, 'method': method, **s})
    scores = pd.DataFrame(rows)

    # Final forecast, race by race
    g = GR._snapshot(final_date).merge(
        res[keys + ['dem_side_candidate', 'rep_candidate', 'dem_side_pct', 'rep_pct', 'winner_party', 'source_url']],
        on=keys, how='inner', suffixes=('', '_res'))
    g['won'] = g['winner_party'] == g['dem_side_party']
    g['fc_margin'] = g['forecast_margin'].fillna(g['predicted_margin']) if 'predicted_margin' in g else g['forecast_margin']
    g['actual'] = actual_margin(g)
    if 'uncontested' in g:
        g.loc[g['uncontested'].eq(True).values, 'actual'] = np.nan
    g['error'] = g['actual'] - g['fc_margin']              # + means Democrats beat the forecast
    g['p_winner'] = np.where(g['won'], g['model_p'], 1 - g['model_p'])
    g['label'] = [race_label(o, s, d) for o, s, d in zip(g['office'], g['state'], g['district'])]
    held = g.merge(_held_house(), on=['state', 'district'], how='left')['held_by'].values
    g['held'] = np.where(g['office'] == 'house', held, g.get('incumbent_party'))
    g['flip'] = g['held'].notna() & (g['held'] != g['winner_party'])
    g['dem_name'] = g['dem_side_candidate'].fillna(g['dem_candidate'])
    g['rep_name'] = g['rep_candidate_res'].fillna(g['rep_candidate'])
    return {'all': allres, 'res': res, 'scores': scores, 'final': g, 'final_date': final_date,
            'summary': _summary(final_date)}


@st.cache_data(show_spinner=False)
def pollster_errors(results_mtime, final_date):
    """Each poll from the final three weeks against the result of its race."""
    p = pd.read_csv(POLLS_CSV, low_memory=False)
    p['end_date'] = pd.to_datetime(p['end_date'], errors='coerce')
    cutoff = pd.Timestamp(ELECTION_DATE) - pd.Timedelta(days=POLL_WINDOW_DAYS)
    p = p[(p['end_date'] >= cutoff) & (p['end_date'] <= pd.Timestamp(ELECTION_DATE)) & p['margin'].notna()]
    p = p[p['rcv_round'].isna()] if 'rcv_round' in p else p
    p['district'] = pd.to_numeric(p.get('district'), errors='coerce').astype('Int64')
    p.loc[p['office'] != 'house', 'district'] = pd.NA
    r = load_all_results()
    r = r[r['winner_party'].isin(PARTIES)].copy()
    r['actual'] = actual_margin(r)
    m = p.merge(r[['office', 'state', 'district', 'actual', 'winner_party']], on=['office', 'state', 'district'],
                how='inner').dropna(subset=['actual'])
    m['error'] = m['margin'] - m['actual']                  # + means the poll overstated the Dem side
    m['right_winner'] = np.sign(m['margin']) == np.sign(m['actual'])
    m['label'] = [race_label(o, s, d) for o, s, d in zip(m['office'], m['state'], m['district'])]
    return m


# ── Charts ────────────────────────────────────────────────────────────────────
def range_chart(lo, hi, median, actual, line_at, xmax, note, final=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[0, 0], mode='lines', line=dict(color=NEUTRAL, width=14),
                             hovertemplate=f"90% range {lo}–{hi}<extra></extra>", showlegend=False))
    fig.add_trace(go.Scatter(x=[median], y=[0], mode='markers', marker=dict(color=INK_2, size=14, symbol='line-ns',
                             line=dict(width=3, color=INK_2)), hovertemplate=f"Forecast median {median}<extra></extra>",
                             showlegend=False))
    if actual is not None:
        fig.add_trace(go.Scatter(x=[actual], y=[0], mode='markers+text', text=[f"{actual}"], textposition='top center',
                                 marker=dict(color=DEM, size=16, symbol='diamond' if final else 'diamond-open',
                                             line=dict(color='white' if final else DEM, width=1 if final else 2)),
                                 hovertemplate=f"{note}{'' if final else ' so far'}: {actual}<extra></extra>",
                                 showlegend=False))
    fig.add_vline(x=line_at, line_color=INK_2, line_dash='dot', line_width=1)
    fig.update_xaxes(range=[max(0, lo - (hi - lo) * 0.6), min(xmax, hi + (hi - lo) * 0.6)])
    fig.update_yaxes(visible=False, range=[-1, 1.2])
    _layout(fig, height=110)
    fig.update_layout(margin=dict(l=0, r=0, t=6, b=6))
    return fig


def calibration_chart(cal):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.5, 1], y=[0.5, 1], mode='lines', line=dict(color=INK_2, dash='dot'),
                             hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(
        x=cal['forecast'], y=cal['actual'], mode='markers+text', text=cal['races'].astype(str), textposition='top center',
        marker=dict(color=DEM, size=np.clip(np.sqrt(cal['races']) * 4, 8, 40), line=dict(color='white', width=1)),
        customdata=np.stack([cal['bucket'].astype(str), cal['races']], axis=1),
        hovertemplate="%{customdata[0]}: %{customdata[1]} races<br>Average forecast %{x:.0%} · favorite won "
                      "%{y:.0%}<extra></extra>", showlegend=False))
    fig.update_xaxes(title="Our chance for the favorite", tickformat='.0%', range=[0.48, 1.02])
    fig.update_yaxes(title="How often the favorite won", tickformat='.0%', range=[0.3, 1.05])
    return _layout(fig, height=380)


def margin_scatter(d, x_col, x_title):
    fig = go.Figure()
    lim = float(np.nanmax(np.abs(np.r_[d[x_col], d['actual']]))) * 1.05 if len(d) else 50
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode='lines', line=dict(color=INK_2, dash='dot'),
                             hoverinfo='skip', showlegend=False))
    for party in PARTIES:
        s = d[d['winner_party'] == party]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s[x_col], y=s['actual'], mode='markers', name=f"{party.title()} won",
            marker=dict(color=PARTY_COLOR[party], size=9, opacity=0.85, line=dict(color='white', width=0.5),
                        symbol=np.where((s['model_p'] >= 0.5) != s['won'], 'x', 'circle')),
            customdata=np.stack([s['label'], s[x_col].map(fmt_margin), s['actual'].map(fmt_margin)], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>" + x_title + ": %{customdata[1]}<br>Result: %{customdata[2]}"
                          "<extra></extra>"))
    fig.add_hline(y=0, line_color=GRID)
    fig.add_vline(x=0, line_color=GRID)
    fig.update_xaxes(title=f"{x_title} (Dem − Rep)", range=[-lim, lim])
    fig.update_yaxes(title="Result (Dem − Rep)", range=[-lim, lim])
    return _layout(fig, height=480, legend=dict(orientation='h', y=1.06))


def error_hist(errors, title):
    fig = go.Figure(go.Histogram(x=errors, xbins=dict(size=2), marker_color=NEUTRAL,
                                 hovertemplate="Miss %{x}: %{y} races<extra></extra>"))
    fig.add_vline(x=0, line_color=INK_2, line_dash='dot')
    if len(errors):
        fig.add_vline(x=float(np.mean(errors)), line_color=DEM if np.mean(errors) > 0 else REP, line_width=2,
                      annotation_text=f"average {fmt_margin(np.mean(errors))}", annotation_font_color=INK_2)
    fig.update_xaxes(title=title)
    fig.update_yaxes(title="Races")
    return _layout(fig, height=300, bargap=0.05)


def error_map(d, col, label):
    lim = max(float(np.nanmax(np.abs(d[col]))), 1) if len(d) else 10
    fig = go.Figure(go.Choropleth(
        locations=d['state'].map(STATE_ABBREV), z=d[col], locationmode='USA-states',
        colorscale=[[0, REP], [0.5, '#f2f1ec'], [1, DEM]], zmin=-lim, zmax=lim,
        marker_line_color='white', marker_line_width=1,
        hovertext=[f"<b>{_title_case(s)}</b><br>{label}: {fmt_margin(v)}<br>Polls {fmt_margin(pa)} · result "
                   f"{fmt_margin(a)}" for s, v, pa, a in zip(d['state'], d[col], d['poll_avg'], d['actual'])],
        hoverinfo='text', colorbar=dict(title='Dems beat<br>polls by', len=0.7, thickness=12)))
    fig.update_layout(geo=dict(scope='usa', bgcolor='rgba(0,0,0,0)', lakecolor='rgba(0,0,0,0)',
                               landcolor='rgba(128,128,128,0.12)', showland=True, subunitcolor='rgba(128,128,128,0.4)'))
    return _layout(fig, height=420)


def error_bars(d, col, label):
    d = d.sort_values(col)
    fig = go.Figure(go.Bar(
        x=d[col], y=d['label'], orientation='h', marker_color=np.where(d[col] > 0, DEM, REP),
        text=d[col].map(fmt_margin), textposition='outside', cliponaxis=False,
        hovertemplate="%{y}: " + label + " %{text}<extra></extra>"))
    fig.update_xaxes(title=f"{label} (positive: Democrats did better)")
    return _layout(fig, height=max(260, 22 * len(d) + 60))


def accuracy_over_time(scores):
    s = scores[scores['office'] == 'all']
    colors = {'Our forecast': DEM, 'Polls only': '#1baf7a', 'Fundamentals only': '#eb6834', 'Experts': INK_2}
    fig = go.Figure()
    for method, t in s.groupby('method'):
        fig.add_trace(go.Scatter(x=pd.to_datetime(t['snapshot']), y=t['brier'], mode='lines+markers', name=method,
                                 line=dict(color=colors.get(method, NEUTRAL), width=3 if method == 'Our forecast' else 2),
                                 hovertemplate=f"{method}<br>%{{x|%b %d}}: Brier %{{y:.3f}}<extra></extra>"))
    fig.update_yaxes(title="Brier score (lower is better)", rangemode='tozero')
    return _layout(fig, height=340, legend=dict(orientation='h', y=1.1))


def odds_over_time(log, outcomes):
    t = log.copy()
    t['published_at'] = pd.to_datetime(t['published_at'])
    t = t[t['published_at'] < pd.Timestamp(ELECTION_DATE) + pd.Timedelta(days=1)]
    t = t.groupby(t['published_at'].dt.date).last().reset_index(drop=True)
    fig = go.Figure()
    for col, name, dash, won in (('senate_dem_control_prob', 'Senate', 'solid', outcomes.get('senate')),
                                 ('house_dem_control_prob', 'House', 'dash', outcomes.get('house'))):
        tag = '' if won is None else (' (Dems won it)' if won else ' (Republicans won it)')
        fig.add_trace(go.Scatter(x=t['published_at'], y=t[col], mode='lines+markers', name=name + tag,
                                 line=dict(color=DEM, dash=dash, width=2),
                                 hovertemplate=f"{name}: %{{y:.0%}} Democratic control, %{{x|%b %d}}<extra></extra>"))
    fig.add_hline(y=0.5, line_color=INK_2, line_dash='dot')
    fig.update_yaxes(title="Chance of Democratic control", tickformat='.0%', range=[0, 1])
    return _layout(fig, height=320, legend=dict(orientation='h', y=1.1))


# ── Page ──────────────────────────────────────────────────────────────────────
def senate_control(dem, rep, done):
    """Who controls the Senate once it's clinched: Democrats need 51, Republicans 50 (VP Vance breaks ties)."""
    if dem >= SENATE_DEM_CONTROL:
        return 'Democrats'
    if rep >= SENATE_REP_CONTROL:
        return 'Republicans'
    return 'Neither caucus outright' if done else None


def house_control(dem, rep):
    if dem >= HOUSE_MAJORITY:
        return 'Democrats'
    if rep >= HOUSE_MAJORITY:
        return 'Republicans'
    return None


def _metric_row(d):
    c = st.columns(5)
    right = (d['model_p'] >= 0.5) == d['won']
    p = d['model_p'].clip(GR.PROB_FLOOR, 1 - GR.PROB_FLOOR)
    y = d['won'].astype(float)
    comp = d[d['model_p'].between(0.1, 0.9)]
    comp_right = ((comp['model_p'] >= 0.5) == comp['won'])
    c[0].metric("Winners called right", f"{int(right.sum())} of {len(d)}", f"{right.mean():.1%}", delta_color='off')
    c[1].metric("Competitive races right", f"{int(comp_right.sum())} of {len(comp)}",
                f"{comp_right.mean():.0%}" if len(comp) else None, delta_color='off',
                help="Races where our final forecast gave the favorite less than 90%.")
    c[2].metric("Brier score", f"{float(((p - y) ** 2).mean()):.3f}",
                help="Mean squared error of the win probabilities. 0 is perfect, 0.25 is a coin flip on every race. "
                     "Our 2000–2024 backtest scored 0.067.")
    c[3].metric("Log loss", f"{float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()):.3f}",
                help="Punishes confident misses harder than the Brier score. Lower is better.")
    c[4].metric("Upsets (winner had < 30%)", int((d['p_winner'] < 0.3).sum()))


def render_status(data):
    allres, res = data['all'], data['res']
    pending = allres[allres['winner_party'].notna() & ~allres['winner_party'].isin(PARTIES)]
    missing = allres[allres['winner_party'].isna()]
    by = res.groupby('office').size().reindex(['senate', 'governor', 'house']).fillna(0).astype(int)
    tot = allres.groupby('office').size().reindex(['senate', 'governor', 'house']).fillna(0).astype(int)
    st.markdown(f"**{len(res)} of {len(allres)} races have a winner entered** · "
                + " · ".join(f"{OFFICE_LABEL[o]} {by[o]}/{tot[o]}" for o in by.index)
                + f" · graded against the final forecast of {pd.Timestamp(data['final_date']):%b %d}")
    if len(pending):
        st.info("Not decided yet: " + ", ".join(
            f"{race_label(o, s, d)} ({w.title()})" for o, s, d, w in
            zip(pending['office'], pending['state'], pending['district'], pending['winner_party']))
            + ". These are left out of every score until a winner is entered.")
    if len(missing):
        st.caption(f"{len(missing)} races have no result yet. Uncalled races are usually safe seats, so the scores "
                   "can shift a little as they're added.")


def render_chambers(data):
    s, res = data['summary'], data['res']
    sen, house, gov = (res[res['office'] == o] for o in ('senate', 'house', 'governor'))
    n_sen, n_gov = len(GR._snapshot(data['final_date']).query("office == 'senate'")), s['governor']['races']
    hold = s['senate']['holdovers']
    sen_d = hold['DEMOCRAT'] + int((sen['winner_party'] == 'DEMOCRAT').sum())
    sen_r = hold['REPUBLICAN'] + int((sen['winner_party'] == 'REPUBLICAN').sum())
    sen_i = int((sen['winner_party'] == 'INDEPENDENT').sum())
    h_d, h_r = int((house['winner_party'] == 'DEMOCRAT').sum()), int((house['winner_party'] == 'REPUBLICAN').sum())
    g_d = int((gov['winner_party'] == 'DEMOCRAT').sum())
    sen_done, house_done, gov_done = len(sen) == n_sen, len(house) == 435, len(gov) == n_gov

    sen_ctrl, house_ctrl = senate_control(sen_d, sen_r, sen_done), house_control(h_d, h_r)
    outcomes = {k: v == 'Democrats' for k, v in (('senate', sen_ctrl), ('house', house_ctrl))
                if v in ('Democrats', 'Republicans')}

    st.subheader("Who won the chambers")
    c = st.columns(3)
    with c[0]:
        p = s['senate']['dem_control_prob']
        st.markdown("**Senate**")
        if sen_ctrl:
            st.metric("Control", sen_ctrl, f"we gave Democrats {p:.0%}", delta_color='off')
        else:
            st.metric("Control", "Not decided", f"{len(sen)} of {n_sen} races in", delta_color='off')
        st.caption(f"Democratic seats {sen_d}{' so far' if not sen_done else ''} · Republican {sen_r}"
                   + (f" · independent {sen_i}" if sen_i else "")
                   + f". Forecast median {s['senate']['dem_seats_median']} D (90% range "
                     f"{s['senate']['dem_seats_90pct'][0]}–{s['senate']['dem_seats_90pct'][1]}).")
        lo, hi = s['senate']['dem_seats_90pct']
        st.plotly_chart(range_chart(lo, hi, s['senate']['dem_seats_median'], sen_d, 50.5, 100, "Democratic seats",
                                    final=sen_done), width='stretch', key='res_sen_range',
                        config={'displayModeBar': False})
    with c[1]:
        p = s['house']['dem_control_prob']
        st.markdown("**House**")
        if house_ctrl:
            st.metric("Control", house_ctrl, f"we gave Democrats {p:.0%}", delta_color='off')
        else:
            st.metric("Control", "Not decided", f"{len(house)} of 435 seats in", delta_color='off')
        st.caption(f"Democratic seats {h_d}{' so far' if not house_done else ''} · Republican {h_r}. Forecast median "
                   f"{s['house']['dem_seats_median']} D (90% range {s['house']['dem_seats_90pct'][0]}–"
                   f"{s['house']['dem_seats_90pct'][1]}).")
        lo, hi = s['house']['dem_seats_90pct']
        st.plotly_chart(range_chart(lo, hi, s['house']['dem_seats_median'], h_d, 217.5, 435, "Democratic seats",
                                    final=house_done), width='stretch', key='res_house_range',
                        config={'displayModeBar': False})
    with c[2]:
        st.markdown("**Governors**")
        st.metric("Democratic wins", f"{g_d} of {n_gov}" if gov_done else f"{g_d} so far",
                  f"forecast median {s['governor']['dem_governorships_median']}", delta_color='off')
        lo, hi = s['governor']['dem_governorships_90pct']
        st.caption(f"Republican wins {int((gov['winner_party'] == 'REPUBLICAN').sum())}. 90% range {lo}–{hi}.")
        st.plotly_chart(range_chart(lo, hi, s['governor']['dem_governorships_median'], g_d, n_gov / 2, n_gov,
                                    "Democratic wins", final=gov_done), width='stretch', key='res_gov_range',
                        config={'displayModeBar': False})
    st.caption("Gray bar: the forecast's 90% range. Tick: the forecast median. Diamond: the actual seat count (hollow "
               "while races are still being entered). Control is called as soon as a party has clinched a majority, "
               "even with races still out.")
    return outcomes


def render_flips(g):
    st.subheader("Seats that changed hands")
    f = g[g['flip']]
    c = st.columns(3)
    for i, office in enumerate(('senate', 'house', 'governor')):
        o = g[g['office'] == office]
        fo = f[f['office'] == office]
        d_gain = int(((fo['winner_party'] == 'DEMOCRAT')).sum())
        r_gain = int(((fo['winner_party'] == 'REPUBLICAN')).sum())
        i_gain = int(((fo['winner_party'] == 'INDEPENDENT')).sum())
        net = d_gain - int((fo['held'] == 'DEMOCRAT').sum())
        c[i].metric(f"{OFFICE_LABEL[office]} flips", len(fo),
                    f"Democrats net {net:+d}" if len(o) else None, delta_color='normal' if net else 'off',
                    help=f"Gained by Democrats {d_gain}, by Republicans {r_gain}" + (f", by an independent {i_gain}"
                                                                                   if i_gain else '') + ".")
    if len(f):
        t = f.sort_values(['office', 'model_p'])
        st.dataframe(pd.DataFrame({
            'Race': t['label'],
            'Held by': t['held'].str.title(),
            'Won by': t['winner_party'].str.title(),
            'Winner': np.where(t['winner_party'] == 'REPUBLICAN', t['rep_name'], t['dem_name']),
            'Result': t['actual'].map(fmt_margin),
            'Our chance for the winner': t['p_winner'] * 100,
        }), hide_index=True, width='stretch', height=min(35 * (len(t) + 1) + 3, 420), column_config={
            'Our chance for the winner': st.column_config.ProgressColumn(color=DEM, format='%.0f%%', min_value=0,
                                                                         max_value=100)})
        st.caption("Held by: the party holding the seat going into the election. For House seats that's the "
                   "district's current holder; redistricting can make a few of these look like flips when they aren't.")


def render_scorecard(data):
    g, scores = data['final'], data['scores']
    st.subheader("Scorecard")
    office = st.radio("Races", ['all', 'senate', 'governor', 'house'], horizontal=True, key='res_score_office',
                      format_func=lambda o: 'All races' if o == 'all' else OFFICE_LABEL[o])
    d = g if office == 'all' else g[g['office'] == office]
    if d.empty:
        st.info("No results entered for this office yet.")
        return
    _metric_row(d)

    st.markdown("##### Against simpler approaches")
    fin = scores[(scores['snapshot'] == data['final_date']) & (scores['office'] == office)]
    if len(fin):
        t = fin[['method', 'races', 'winners_right', 'brier', 'log_loss']].sort_values('brier')
        st.dataframe(t.rename(columns={'method': 'Method', 'races': 'Races', 'winners_right': 'Winners right',
                                       'brier': 'Brier', 'log_loss': 'Log loss'}),
                     hide_index=True, width='stretch', column_config={
                         'Winners right': st.column_config.NumberColumn(format='percent'),
                         'Brier': st.column_config.NumberColumn(format='%.3f'),
                         'Log loss': st.column_config.NumberColumn(format='%.3f')})
        st.caption("Each line scores only the races it covers (polls only needs polls; experts need a consensus "
                   "rating), so race counts differ. Experts' ratings are converted with an assumed scale: Safe ≈ 97%, "
                   "Likely ≈ 85%, Lean ≈ 70%, Tilt ≈ 60%.")

    c = st.columns(2)
    with c[0]:
        st.markdown("##### Calibration")
        cal = GR.calibration(d)
        st.plotly_chart(calibration_chart(cal), width='stretch', key='res_cal')
        st.caption("If the forecast is honest, races we gave 70% should go to the favorite about 70% of the time, so "
                   "points should sit near the dotted line. Numbers are race counts; small buckets bounce around.")
    with c[1]:
        st.markdown("##### Accuracy over the campaign")
        st.plotly_chart(accuracy_over_time(scores), width='stretch', key='res_time')
        st.caption("Every published snapshot, scored on the same races (all offices). It shows whether the forecast "
                   "improved as election day got closer, and how it compared with the alternatives along the way.")


def render_margins(g):
    st.subheader("Margins: forecast vs. result")
    office = st.radio("Office", ['senate', 'governor', 'house'], format_func=OFFICE_LABEL.get, horizontal=True,
                      key='res_margin_office')
    d = g[(g['office'] == office) & g['actual'].notna() & g['fc_margin'].notna()]
    if d.empty:
        st.info("No vote shares entered for this office yet. Add dem_side_pct and rep_pct to the results file.")
        return
    err = d['error']
    c = st.columns(4)
    c[0].metric("Races with vote shares", len(d))
    c[1].metric("Average miss", fmt_margin(err.mean()), help="Result minus forecast margin, averaged. D+2 means "
                "Democrats ran 2 points ahead of our forecast on average (a systematic lean, not random noise).")
    c[2].metric("Typical miss (RMSE)", f"{np.sqrt((err ** 2).mean()):.1f} pts",
                help="Root mean squared error of the margin. Our 2000–2024 backtest at 28 days out: 6.7 pts.")
    if 'race_sd' in d and d['race_sd'].notna().any():
        # The forecast's full spread is the race's own error plus the shared national error
        # (the simulation adds both), so the 90% range uses both.
        shared = d['shared_sd'].fillna(0) if 'shared_sd' in d else 0
        inside = (err.abs() <= 1.645 * np.sqrt(d['race_sd'] ** 2 + shared ** 2)).mean()
        c[3].metric("Results inside our 90% range", f"{inside:.0%}", help="Should be near 90% if the forecast's "
                    "uncertainty was right. Much lower: overconfident. Much higher: too cautious. The range "
                    "includes both the race's own error and the shared national error.")
    else:
        c[3].metric("Median miss", f"{err.abs().median():.1f} pts")

    cols = st.columns([3, 2])
    with cols[0]:
        st.plotly_chart(margin_scatter(d, 'fc_margin', 'Our forecast'), width='stretch', key='res_scatter')
        st.caption("Each dot is a race. Above the dotted line, Democrats did better than we forecast; below it, "
                   "Republicans did. An × marks a race where our favorite lost.")
    with cols[1]:
        st.plotly_chart(error_hist(err, "Result minus forecast (pts, + = Democrats did better)"), width='stretch',
                        key='res_hist')
        b = d.assign(abs_err=err.abs()).sort_values('abs_err', ascending=False).head(10)
        st.markdown("**Biggest margin surprises**")
        st.dataframe(pd.DataFrame({'Race': b['label'], 'Forecast': b['fc_margin'].map(fmt_margin),
                                   'Result': b['actual'].map(fmt_margin), 'Miss': b['error'].map(fmt_margin)}),
                     hide_index=True, width='stretch')

    if office != 'house':
        render_polling_error(d, office)


def render_polling_error(d, office):
    st.markdown("##### How the polls did")
    p = d[d['poll_avg'].notna()].copy()
    if p.empty:
        st.info("None of these races had polls.")
        return
    p['raw_miss'] = p['actual'] - p['poll_avg']
    p['corr_miss'] = p['actual'] - p['poll_avg_corrected'] if 'poll_avg_corrected' in p else np.nan
    c = st.columns(4)
    c[0].metric("Races with polls", len(p))
    c[1].metric("Raw polling average miss", fmt_margin(p['raw_miss'].mean()),
                help="Result minus the plain weighted polling average. D+2 means polls underestimated Democrats.")
    c[2].metric("After our corrections", fmt_margin(p['corr_miss'].mean()),
                help="Same, after the house-effect and lean corrections (state polls pulled 1.8 pts toward R).")
    c[3].metric("Polls picked the winner", f"{(np.sign(p['poll_avg']) == np.sign(p['actual'])).mean():.0%}")
    cols = st.columns([3, 2])
    with cols[0]:
        st.plotly_chart(error_map(p, 'raw_miss', 'Result vs. polls'), width='stretch', key='res_poll_map')
    with cols[1]:
        st.plotly_chart(error_bars(p.assign(label=p['state'].map(_title_case)), 'raw_miss', 'Result vs. polls'),
                        width='stretch', key='res_poll_bars')
    st.caption(f"{OFFICE_LABEL[office]} races with at least one poll. Blue: Democrats beat their polls; red: "
               "Republicans did.")


def render_pollsters(data, mtime):
    st.subheader("Pollster scorecard")
    m = pollster_errors(mtime, data['final_date'])
    if m.empty:
        st.info("No final-weeks polls match races with vote shares yet.")
        return
    t = m.groupby('pollster').agg(polls=('error', 'size'), races=('label', 'nunique'),
                                  avg_abs=('error', lambda e: e.abs().mean()), bias=('error', 'mean'),
                                  right=('right_winner', 'mean'), sponsor=('sponsor', 'first')).reset_index()
    t = t[t['polls'] >= MIN_POLLSTER_POLLS].sort_values('avg_abs')
    c = st.columns(4)
    c[0].metric("Polls graded", len(m), help=f"Polls ending in the final {POLL_WINDOW_DAYS} days, in races with "
                                            "vote shares entered.")
    c[1].metric("Average poll miss", f"{m['error'].abs().mean():.1f} pts")
    c[2].metric("Average lean", fmt_margin(m['error'].mean()), help="Poll margin minus result. D+2 means polls "
                                                                     "overstated Democrats by 2 on average.")
    c[3].metric("Picked the winner", f"{m['right_winner'].mean():.0%}")
    st.dataframe(pd.DataFrame({
        'Pollster': t['pollster'], 'Polls': t['polls'], 'Races': t['races'],
        'Average miss (pts)': t['avg_abs'].round(1), 'Lean': t['bias'].map(fmt_margin),
        'Picked the winner': t['right'] * 100,
        'Partisan sponsor': t['sponsor'].fillna('—').astype(str).str.title(),
    }), hide_index=True, width='stretch', height=min(35 * (len(t) + 1) + 3, 560), column_config={
        'Picked the winner': st.column_config.ProgressColumn(color=DEM, format='%.0f%%', min_value=0, max_value=100)})
    st.caption(f"Pollsters with at least {MIN_POLLSTER_POLLS} polls in the final {POLL_WINDOW_DAYS} days. Lean is the "
               "poll's margin minus the result: D+3 means the pollster overstated Democrats by 3 on average. Raw "
               "toplines, before any of our corrections.")
    with st.expander("Every graded poll"):
        mm = m.sort_values('error', key=abs, ascending=False)
        st.dataframe(pd.DataFrame({'Race': mm['label'], 'Pollster': mm['pollster'],
                                   'Ended': mm['end_date'].dt.strftime('%b %d'), 'Poll': mm['margin'].map(fmt_margin),
                                   'Result': mm['actual'].map(fmt_margin), 'Miss': mm['error'].map(fmt_margin)}),
                     hide_index=True, width='stretch', height=420)


def render_misses(g):
    st.subheader("Where we got it wrong")
    m = g[(g['model_p'] >= 0.5) != g['won']].copy()
    if m.empty:
        st.success("Every favorite won.")
        return
    m['conf'] = np.maximum(m['model_p'], 1 - m['model_p'])
    m = m.sort_values('conf', ascending=False)
    st.dataframe(pd.DataFrame({
        'Race': m['label'],
        'We favored': np.where(m['model_p'] >= 0.5, np.where(m['dem_side_party'] == 'INDEPENDENT', 'Independent',
                                                            'Democrat'), 'Republican'),
        'Our chance for the favorite': m['conf'] * 100,
        'Won by': m['winner_party'].str.title(),
        'Forecast': m['fc_margin'].map(fmt_margin),
        'Result': m['actual'].map(fmt_margin),
        'Polls': m['poll_avg'].map(fmt_margin) if 'poll_avg' in m else '—',
    }), hide_index=True, width='stretch', height=min(35 * (len(m) + 1) + 3, 420), column_config={
        'Our chance for the favorite': st.column_config.ProgressColumn(color=NEUTRAL, format='%.0f%%', min_value=50,
                                                                       max_value=100)})
    st.caption(f"{len(m)} races where our final favorite lost, most confident first. A forecast that's working should "
               "miss some 60–70% races; misses above 90% are the ones worth explaining.")


def render_all_races(g):
    st.subheader("Every race")
    c = st.columns([2, 2, 3])
    office = c[0].selectbox("Office", ['all', 'senate', 'governor', 'house'], key='res_all_office',
                            format_func=lambda o: 'All' if o == 'all' else OFFICE_LABEL[o])
    show = c[1].selectbox("Show", ['All', 'Competitive (10–90%)', 'Flips', 'Misses'], key='res_all_show')
    q = c[2].text_input("Search state, district or candidate", key='res_all_q')
    d = g if office == 'all' else g[g['office'] == office]
    if show.startswith('Competitive'):
        d = d[d['model_p'].between(0.1, 0.9)]
    elif show == 'Flips':
        d = d[d['flip']]
    elif show == 'Misses':
        d = d[(d['model_p'] >= 0.5) != d['won']]
    if q:
        ql = q.lower()
        d = d[d['label'].str.lower().str.contains(ql, regex=False) | d['state'].str.lower().str.contains(ql, regex=False)
              | d['dem_name'].fillna('').str.lower().str.contains(ql, regex=False)
              | d['rep_name'].fillna('').str.lower().str.contains(ql, regex=False)]
    d = d.sort_values(['office', 'model_p'])
    st.dataframe(pd.DataFrame({
        'Race': d['label'],
        'Dem side': d['dem_name'].fillna('—'),
        'Republican': d['rep_name'].fillna('—'),
        'Winner': d['winner_party'].str.title(),
        'Dem %': d['dem_side_pct'], 'Rep %': d['rep_pct'],
        'Result': d['actual'].map(fmt_margin),
        'Forecast': d['fc_margin'].map(fmt_margin),
        'Miss': d['error'].map(fmt_margin),
        'Dem-side win chance': d['model_p'] * 100,
        'Favorite won': np.where((d['model_p'] >= 0.5) == d['won'], '✓', '✗'),
        'Flip': np.where(d['flip'], 'flip', ''),
    }), hide_index=True, width='stretch', height=520, column_config={
        'Dem %': st.column_config.NumberColumn(format='%.1f'), 'Rep %': st.column_config.NumberColumn(format='%.1f'),
        'Dem-side win chance': st.column_config.ProgressColumn(color=DEM, format='%.0f%%', min_value=0, max_value=100)})
    st.caption(f"{len(d)} races. Miss is the result minus our forecast margin (D+ means Democrats did better). "
               "Uncontested races and races without two-party vote shares show no margin.")


def render_data_checks(data):
    a = data['all'].copy()
    a['total'] = a['dem_side_pct'] + a['rep_pct']
    a['label'] = [race_label(o, s, d) for o, s, d in zip(a['office'], a['state'], a['district'])]
    called = a[a['winner_party'].isin(PARTIES)]
    issues = []
    for _, r in called[called['total'] > 100.5].iterrows():
        issues.append((r['label'], f"Vote shares add to {r['total']:.1f}%"))
    both = called.dropna(subset=['dem_side_pct', 'rep_pct'])
    lead_d = both['dem_side_pct'] > both['rep_pct']
    winner_d = both['winner_party'] != 'REPUBLICAN'
    for _, r in both[(lead_d != winner_d) & (both['dem_side_pct'] != both['rep_pct'])].iterrows():
        issues.append((r['label'], f"Winner is {r['winner_party'].title()} but the shares are "
                                   f"{r['dem_side_pct']:.1f}–{r['rep_pct']:.1f}"))
    no_share = called[called['dem_side_pct'].isna() & called['rep_pct'].isna()]
    for _, r in no_share.iterrows():
        issues.append((r['label'], "Winner entered without vote shares"))
    with st.expander(f"🧹 Results file checks ({len(issues)})", expanded=False):
        if issues:
            st.dataframe(pd.DataFrame(issues, columns=['Race', 'Check']), hide_index=True, width='stretch')
            st.caption("Worth a look in forecasts/2026/results_2026.csv. This page only reads that file.")
        else:
            st.success("No problems found.")


def render_results(model_version):
    st.title("🏁 2026 Results")
    st.caption("How the Ledger Forecast did. Every race is scored against the final forecast published before polls "
               "closed, and against simple alternatives (polls alone, fundamentals alone, the experts' ratings). "
               "Results come from forecasts/2026/results_2026.csv.")
    if not os.path.exists(GR.RESULTS_CSV):
        st.info("No results file yet. Set it up on the Election Night tab.")
        return
    mtime = os.path.getmtime(GR.RESULTS_CSV)
    data = build(mtime)
    if data['res'].empty:
        st.info("No winners entered yet.")
        return
    g = data['final']
    render_status(data)
    render_data_checks(data)
    st.divider()
    outcomes = render_chambers(data)
    if os.path.exists(TOPLINE_CSV):
        st.markdown("##### Our chamber odds over the campaign")
        st.plotly_chart(odds_over_time(pd.read_csv(TOPLINE_CSV), outcomes), width='stretch', key='res_odds')
    st.divider()
    render_flips(g)
    st.divider()
    render_scorecard(data)
    st.divider()
    render_margins(g)
    st.divider()
    render_pollsters(data, mtime)
    st.divider()
    render_misses(g)
    st.divider()
    render_all_races(g)
    st.caption(f"Final forecast: {data['final_date']} snapshot (model {data['summary'].get('model_version', model_version)}).")


# ── Post builder: results posts ───────────────────────────────────────────────
POST_KINDS = {'report': 'Report card', 'races': 'Forecast vs. result', 'misses': 'Misses',
              'polls': 'Polling error', 'pollsters': 'Most accurate pollsters', 'alts': 'Us vs. the alternatives'}


def render_result_posts(model_version):
    import result_posts as RP
    from brand import MODEL_NAME
    st.caption("Posts about how the forecast did, from the same numbers as the 🏁 Results tab. Copy the text, or "
               "download the image if emoji won't survive the platform.")
    if not os.path.exists(GR.RESULTS_CSV):
        st.info("No results file yet.")
        return
    mtime = os.path.getmtime(GR.RESULTS_CSV)
    data = build(mtime)
    g = data['final']
    if g.empty:
        st.info("No winners entered yet.")
        return
    kind = st.radio("Post", list(POST_KINDS), format_func=POST_KINDS.get, horizontal=True, key='rpost_kind')
    as_of = pd.Timestamp.fromtimestamp(mtime)
    final = pd.Timestamp(data['final_date'])
    footer = (f"{MODEL_NAME} · final forecast {final:%b} {final.day} vs. results as of {as_of:%b} {as_of.day} · "
              f"{len(g)} races")
    header, lines, tiles, subtitle = None, None, None, None

    if kind == 'report':
        title = f"How the {MODEL_NAME[4:]} did" if MODEL_NAME.startswith('The ') else f"How {MODEL_NAME} did"
        tiles = RP.report_card(data)
        subtitle = "2026 midterms · graded against the final forecast, published before polls closed"
    elif kind in ('races', 'misses', 'polls'):
        c = st.columns(2)
        offices = ['senate', 'governor'] if kind == 'polls' else ['senate', 'governor', 'house']
        if kind == 'misses':
            offices = ['all'] + offices
        office = c[0].radio("Office", offices, horizontal=True, key=f'rpost_office_{kind}',
                            format_func=lambda o: 'All offices' if o == 'all' else OFFICE_LABEL[o])
        label = 'All races' if office == 'all' else OFFICE_LABEL[office]
        if kind == 'races':
            which = c[1].radio("Races", ['competitive', 'flips', 'misses', 'all'], horizontal=True,
                               key='rpost_which', format_func={'competitive': 'Competitive (10–90%)', 'flips': 'Flips',
                                                               'misses': 'Misses', 'all': 'All'}.get)
            lines = RP.forecast_vs_result(g, office, which)
            title = f"{label}: Forecast vs. Result"
            header = "Winner and margin · our final chance for the winner"
        elif kind == 'misses':
            lines = RP.misses(g, None if office == 'all' else office)
            n = len(g) if office == 'all' else int((g['office'] == office).sum())
            title = f"Where we missed: {label}"
            header = f"{len(lines)} of {n} races went against our favorite"
        else:
            lines, avg, n = RP.polling_error(g, office)
            title = f"{label} Polls vs. Results"
            header = (f"On average, {'Democrats' if avg > 0 else 'Republicans'} beat their polls by {abs(avg):.1f} "
                      f"across {n} races")
    elif kind == 'pollsters':
        m = pollster_errors(mtime, data['final_date'])
        c = st.columns(2)
        top = c[0].slider("How many", 5, 20, 10, key='rpost_top')
        min_polls = c[1].slider("Minimum polls", 1, 8, 3, key='rpost_min')
        lines = RP.pollster_lines(m, top, min_polls)
        title = "Most Accurate Pollsters of 2026"
        header = f"Average miss on the margin, polls from the final {POLL_WINDOW_DAYS} days"
    else:
        office = st.radio("Races", ['all', 'senate', 'governor', 'house'], horizontal=True, key='rpost_alt_office',
                          format_func=lambda o: 'All races' if o == 'all' else OFFICE_LABEL[o])
        lines = RP.alternatives(data['scores'], data['final_date'], office)
        title = "Our forecast vs. the alternatives"
        header = "Brier score, final forecast (lower is better)"

    if lines and len(lines) > 1 and kind != 'alts':
        bodies = [ln['body'] for ln in lines]
        keep = st.multiselect("Lines", bodies, default=bodies, key=f'rpost_lines_{kind}_{len(bodies)}_{hash(tuple(bodies))}',
                              help="Every line is included; remove any you don't want.")
        lines = [ln for ln in lines if ln['body'] in keep]
    c = st.columns([3, 1])
    title = c[0].text_input("Title", title, key=f'rpost_title_{kind}')
    dark = c[1].checkbox("Dark image", True, key='rpost_dark')
    if header is not None:
        header = st.text_input("Subtitle", header, key=f'rpost_header_{kind}')
    if subtitle is not None:
        subtitle = st.text_input("Subtitle", subtitle, key=f'rpost_sub_{kind}')
    footer = st.text_input("Footer", footer, key=f'rpost_footer_{kind}',
                           help="Keep the results date in it, so a post made before every race was in can't pass "
                                "for the final tally.")
    if lines is not None and not lines:
        st.info("Nothing to show for this selection.")
        return

    l, r = st.columns(2)
    with l:
        st.markdown("**Text** (copy button at top right)")
        text = (RP.report_text(title, tiles, footer or None, [subtitle] if subtitle else None) if tiles is not None
                else RP.lines_text(title, lines, footer or None, header or None))
        st.code(text, language=None)
    with r:
        png = (RP.report_png(title, tiles, footer or None, dark, subtitle or None) if tiles is not None
               else RP.lines_png(title, lines, footer or None, dark, header or None))
        st.markdown("**Image**")
        st.image(png, width='stretch')
        st.download_button("Download image (PNG)", png, key='rpost_png', mime='image/png',
                           file_name=f"results_{kind}_{as_of:%Y-%m-%d}.png")
    st.caption("✓ / ✗: whether our final favorite won. \"We had\" is our final chance for the candidate who won. "
               "Margins come from the vote shares in results_2026.csv; races without shares just say \"won\".")
