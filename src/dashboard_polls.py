"""
Streamlit views for the polls themselves: the latest-polls feed (every 2026 poll we use,
with its corrections and weight) and the pollster ratings behind those corrections.
Render-only, like dashboard_2026.
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_2026 import DEM, REP, IND, NEUTRAL, INK_2, INK_3, STATE_ABBREV, _layout, _title_case, fmt_margin
from polls_average import poll_shares, CORRECTION_STRENGTH, SPONSORED_WEIGHT, UNRATED_WEIGHT, _quality_weight

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RATINGS_CSV = os.path.join(BASE_DIR, 'data', 'polling_data', 'pollster_ratings.csv')
DROPPED_CSV = os.path.join(BASE_DIR, 'data', 'polls_2026', 'dropped_polls.csv')
OFFICE_LABEL = {'senate': 'Senate', 'governor': 'Governor', 'house': 'House'}


def _race_label(office, state, district=None):
    if office == 'house':
        d = int(district) if pd.notna(district) else 0
        return f"House · {STATE_ABBREV.get(state, state)}-{'AL' if d == 0 else d}"
    return f"{OFFICE_LABEL[office]} · {_title_case(state)}"


def poll_feed(annotated, house_annotated, as_of):
    """Every 2026 poll in one table, newest first, with its share of its race's average today."""
    frames = [a for a in (annotated, house_annotated) if a is not None and len(a)]
    p = pd.concat(frames, ignore_index=True)
    keys = ['office', 'state'] + (['district'] if 'district' in p else [])
    p['district'] = p['district'] if 'district' in p else np.nan
    shares = pd.Series(0.0, index=p.index)
    for _, g in p.groupby(['office', 'state', p['district'].fillna(-1)]):
        shares.loc[g.index] = poll_shares(g, as_of)
    p['share'] = shares
    p['race'] = [_race_label(o, s, d) for o, s, d in zip(p['office'], p['state'], p['district'])]
    p['kind'] = np.where(p['sponsor'] == 'DEMOCRAT', 'Dem-sponsored',
                np.where(p['sponsor'] == 'REPUBLICAN', 'Rep-sponsored',
                np.where(p['sponsor'] == 'INDEPENDENT', 'Ind-sponsored',
                np.where(p['matched_pollster'].notna(), 'Rated', 'Unrated'))))
    return p.sort_values(['end_date', 'race'], ascending=[False, True])


def render_polls(f, annotated, house_annotated):
    as_of = pd.Timestamp(f['as_of'])
    st.title("2026 Polls")
    st.caption("Every general-election poll the forecast uses, newest first. Polls come from Wikipedia's race "
               "pages (which cite each pollster's release) plus any we add by hand from a pollster's own release; "
               "nothing from prediction-market-affiliated aggregators.")
    tab_feed, tab_ratings = st.tabs(["📋 Latest polls", "⭐ Pollster ratings"])
    p = poll_feed(annotated, house_annotated, as_of)
    with tab_feed:
        render_feed(p, as_of)
    with tab_ratings:
        render_ratings(p, f)


def render_feed(p, as_of):
    c = st.columns(4)
    recent = lambda d: p[p['end_date'] > as_of - pd.Timedelta(days=d)]
    c[0].metric("Polls in the last 7 days", len(recent(7)))
    c[1].metric("Last 30 days", len(recent(30)))
    c[2].metric("Races polled", p['race'].nunique())
    c[3].metric("Pollsters", p['pollster'].nunique())

    fc = st.columns([2, 2, 2, 2])
    offices = fc[0].multiselect("Office", ['senate', 'governor', 'house'], default=['senate', 'governor', 'house'],
                                format_func=OFFICE_LABEL.get, key='feed_office')
    states = sorted(p['state'].unique())
    state = fc[1].selectbox("State", ['All states'] + states, format_func=lambda s: s if s == 'All states' else _title_case(s),
                            key='feed_state')
    days = fc[2].select_slider("Released in the last", [7, 14, 30, 60, 120, 365, 730], value=30,
                               format_func=lambda d: f"{d} days", key='feed_days')
    kinds = fc[3].multiselect("Poll type", ['Rated', 'Unrated', 'Dem-sponsored', 'Rep-sponsored', 'Ind-sponsored'],
                              default=['Rated', 'Unrated', 'Dem-sponsored', 'Rep-sponsored', 'Ind-sponsored'],
                              key='feed_kinds')
    q = st.text_input("Search pollster or candidate", key='feed_q')
    t = p[p['office'].isin(offices) & p['kind'].isin(kinds) & (p['end_date'] > as_of - pd.Timedelta(days=days))]
    if state != 'All states':
        t = t[t['state'] == state]
    if q:
        hay = (t['pollster'] + ' ' + t['dem_candidate'].fillna('') + ' ' + t['rep_candidate'].fillna('')).str.lower()
        t = t[hay.str.contains(q.lower(), regex=False)]
    if t.empty:
        st.info("No polls match these filters.")
        return
    st.dataframe(pd.DataFrame({
        'Ends': t['end_date'].dt.strftime('%b %d'),
        'Race': t['race'],
        'Pollster': t['pollster'],
        'Type': t['kind'],
        'Sample': t['sample_size'].fillna(0).astype(int).astype(str).str.replace('^0$', '—', regex=True)
                  + ' ' + t['population'].fillna(''),
        'Matchup': (t['dem_candidate'].fillna('Dem').map(lambda n: str(n).split()[-1]) + ' ' + t['dem_pct'].map(lambda v: f"{v:.0f}")
                    + ' – ' + t['rep_candidate'].fillna('Rep').map(lambda n: str(n).split()[-1]) + ' ' + t['rep_pct'].map(lambda v: f"{v:.0f}")),
        'Reported': t['margin'].map(fmt_margin),
        'After lean correction': t['adjusted_margin'].map(fmt_margin),
        'Share of race average': t['share'] * 100,
        'Source': t['source_url'],
    }), hide_index=True, width='stretch', height=min(35 * (len(t) + 1) + 3, 640),
        column_config={
            'Share of race average': st.column_config.ProgressColumn(
                color=DEM, format='%.0f%%', min_value=0, max_value=100,
                help="How much this poll counts in its race's polling average today. Newer, larger polls from more "
                     "accurate pollsters count more; a pollster's many polls in one race share their weight; polls "
                     "older than 120 days count for nothing."),
            'After lean correction': st.column_config.TextColumn(
                help=f"The reported margin minus {CORRECTION_STRENGTH:.0%} of the pollster's historical partisan "
                     "lean (or, for campaign- and party-sponsored polls, the typical lean of sponsored polls)."),
            'Source': st.column_config.LinkColumn(display_text=r'https?://(?:www\.)?([^/]+).*')})
    st.caption(f"{len(t)} polls shown. Margins are the Democratic side minus the Republican, in points. "
               "Sponsored polls (paid for by a campaign or party, marked (D)/(R) by Wikipedia) count half after "
               "their lean is corrected.")

    if os.path.exists(DROPPED_CSV):
        d = pd.read_csv(DROPPED_CSV)
        if d.empty:
            return
        d['end_date'] = pd.to_datetime(d['end_date'], errors='coerce')
        with st.expander(f"Polls we left out ({len(d)}) and why"):
            st.dataframe(pd.DataFrame({
                'Ends': d['end_date'].dt.strftime('%b %d, %Y'),
                'Race': [_race_label(o, s, dd) for o, s, dd in zip(d['office'], d['state'], d.get('district', pd.Series(np.nan, index=d.index)))],
                'Pollster': d['pollster'],
                'Matchup': d['dem_candidate'].fillna('') + ' vs. ' + d['rep_candidate'].fillna(''),
                'Reason': d['reason'],
            }), hide_index=True, width='stretch', height=320)


# ── Pollster ratings ──────────────────────────────────────────────────────────
def load_ratings():
    return pd.read_csv(RATINGS_CSV)


def pollster_2026(p, f):
    """Each 2026 pollster: polls, how we treat it, and how far its raw numbers sat from the
    race averages this cycle (Senate and governor races with at least 3 other polls)."""
    avg = pd.concat([f['senate'].assign(office='senate'), f['governor'].assign(office='governor')])
    avg = avg.set_index(['office', 'state'])['poll_avg']
    q = p[p['office'] != 'house'].copy()
    q['race_avg'] = [avg.get((o, s), np.nan) for o, s in zip(q['office'], q['state'])]
    q['vs_avg'] = q['margin'] - q['race_avg']
    g = q.groupby('pollster').agg(polls_2026=('pollster', 'size'), vs_avg=('vs_avg', 'mean'),
                                  last=('end_date', 'max'))
    allp = p.groupby('pollster').agg(n_all=('pollster', 'size'), matched=('matched_pollster', 'first'),
                                     sponsor=('sponsor', 'first'), correction=('house_correction', 'mean'),
                                     weight=('quality_weight', 'mean'), last_all=('end_date', 'max'))
    out = allp.join(g, how='left').reset_index()
    out.loc[out['polls_2026'].fillna(0) < 3, 'vs_avg'] = np.nan
    return out.sort_values('n_all', ascending=False)


def ratings_scatter(r, active):
    r = r[r['n_polls'] >= 10].copy()
    r['active'] = r['pollster'].isin(active)
    fig = go.Figure()
    for is_active, color, name in ((False, NEUTRAL, 'Not polling 2026 races'), (True, DEM, 'Polling 2026 races')):
        d = r[r['active'] == is_active]
        fig.add_trace(go.Scatter(
            x=d['house_effect'], y=d['plus_minus'], mode='markers', name=name,
            marker=dict(size=np.clip(np.sqrt(d['n_polls']) * 1.6, 8, 30), color=color,
                        line=dict(color='white', width=1.5), opacity=0.9 if is_active else 0.6),
            text=d['pollster'],
            customdata=np.stack([d['n_polls'], d['first_cycle'], d['last_cycle'], d['mean_abs_error']], axis=1),
            hovertemplate="<b>%{text}</b><br>Lean %{x:+.1f} (+ = Dem) · accuracy vs peers %{y:+.2f}"
                          "<br>%{customdata[0]} polls, %{customdata[1]}–%{customdata[2]} · avg miss %{customdata[3]:.1f} pts"
                          "<extra></extra>"))
    for _, d in r[r['active']].nlargest(12, 'n_polls').iterrows():
        fig.add_annotation(x=d['house_effect'], y=d['plus_minus'], text=d['pollster'], showarrow=False,
                           yshift=14, font=dict(size=10, color=INK_2))
    fig.add_vline(x=0, line_color=INK_3, line_width=1)
    fig.add_hline(y=0, line_color=INK_3, line_width=1)
    fig.update_xaxes(title="Partisan lean vs. other pollsters in the same races (← R · D →)")
    fig.update_yaxes(title="Error vs. peers (lower is better)", autorange='reversed')
    return _layout(fig, height=480, legend=dict(orientation='h', y=1.08, x=0))


def render_ratings(p, f):
    r = load_ratings()
    st.caption("Our ratings come from about 18,000 polls with known results (FiveThirtyEight's public archive of "
               "final-three-week polls, 1998–2024). Each pollster is compared with **other pollsters in the same "
               "race**, so a year when everyone missed doesn't count against anyone. Small track records are shrunk "
               "toward average, so a pollster with three lucky polls can't top the list.")
    st.plotly_chart(ratings_scatter(r, set(p['matched_pollster'].dropna())), width='stretch', key='ratings_scatter')
    st.caption("Each dot is a pollster with at least 10 rated polls; bigger dots have more. Up is more accurate. "
               "Left/right is the pollster's typical lean relative to its peers, which we correct by half "
               f"(the backtest showed full and half corrections perform about the same; half is safer).")

    st.subheader("Pollsters in 2026 races")
    t = pollster_2026(p, f)
    st.dataframe(pd.DataFrame({
        'Pollster': t['pollster'],
        '2026 polls': t['n_all'],
        'Latest': t['last_all'].dt.strftime('%b %d'),
        'Rated as': np.where(t['sponsor'].notna(), 'Sponsored (' + t['sponsor'].fillna('').str.title() + ')',
                             t['matched'].fillna('No track record')),
        'Lean correction': t['correction'].map(lambda v: '—' if not v else fmt_margin(-v)),
        'Weight': t['weight'] * 100,
        '2026 vs. race averages': t['vs_avg'].map(fmt_margin),
    }), hide_index=True, width='stretch', height=520, column_config={
        'Weight': st.column_config.ProgressColumn(
            color=DEM, format='%.0f%%', min_value=0, max_value=100,
            help=f"Quality weight. Accurate pollsters count fully; less accurate ones less; pollsters with no "
                 f"record {UNRATED_WEIGHT:.0%}; sponsored polls {SPONSORED_WEIGHT:.0%}."),
        'Lean correction': st.column_config.TextColumn(
            help="Points added to each of its polls' margins before averaging (D+ moves them toward Democrats)."),
        '2026 vs. race averages': st.column_config.TextColumn(
            help="How far this pollster's reported margins sat from each race's polling average this year, on "
                 "average (pollsters with 3+ Senate/governor polls). A quick check, not a rating: it says nothing "
                 "about who's right until the results are in."),
    })

    st.subheader("Full ratings table")
    q = st.text_input("Find a pollster", key='ratings_q')
    rr = r[r['pollster'].str.contains(q, case=False, regex=False)] if q else r
    rr = rr.sort_values('plus_minus')
    st.dataframe(pd.DataFrame({
        'Pollster': rr['pollster'],
        'Rated polls': rr['n_polls'],
        'Races': rr['n_races'],
        'Years': rr['first_cycle'].astype(str) + '–' + rr['last_cycle'].astype(str),
        'Avg miss (pts)': rr['mean_abs_error'].round(1),
        'Error vs. peers': rr['plus_minus'].round(2),
        'Lean': rr['house_effect'].map(fmt_margin),
        'Weight': rr['plus_minus'].map(_quality_weight) * 100,
        'Transparency (AAPOR/Roper)': rr['aapor_roper'].map({True: '✓', False: ''}),
    }), hide_index=True, width='stretch', height=420, column_config={
        'Weight': st.column_config.ProgressColumn(color=DEM, format='%.0f%%', min_value=0, max_value=100),
        'Error vs. peers': st.column_config.NumberColumn(
            help="Negative = smaller misses than other pollsters in the same races. After shrinkage toward 0."),
    })
