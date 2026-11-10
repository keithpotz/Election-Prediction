"""
Streamlit views for the 2026 forecast: overview, Senate, House, governors, and track
record. app.py owns data loading and caching; every function here only renders.
Inputs are treated as read-only (copied before any change), because the forecast dict
comes from a shared cache.
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECASTS_DIR = os.path.join(BASE_DIR, '..', 'forecasts')

# Party colors (validated reference palette): blue / red, violet for independents,
# neutral gray midpoint for toss-ups. Chart text and rules use mid-gray tones that read
# on both Streamlit's light and dark themes; body text is left to the theme.
DEM, REP, IND, NEUTRAL = '#2a78d6', '#e34948', '#7b6fd6', '#c9c8c2'
INK, INK_2, INK_3, GRID = '#0b0b0b', '#8a8984', '#8a8984', 'rgba(128,128,128,0.22)'
PROB_SCALE = [[0.0, REP], [0.5, NEUTRAL], [1.0, DEM]]

from states import STATE_ABBREV  # noqa: E402  (re-exported for app.py)
from polls_2026 import POLL_SOURCE_NOTE  # noqa: E402
from brand import MODEL_NAME, TAGLINE  # noqa: E402

def _layout(fig, height=320, **kw):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12), hoverlabel=dict(bgcolor='white', font_color=INK),
        **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig

def fmt_margin(m):
    if m is None or pd.isna(m):
        return '—'
    return f"D+{m:.1f}" if m > 0 else (f"R+{-m:.1f}" if m < 0 else "Even")

def _title_case(state):
    return state.title()


# ── Overview ──────────────────────────────────────────────────────────────────
def seat_distribution_chart(dist, threshold, label, total):
    """Bars = share of simulations ending at each Democratic seat count, colored by who
    controls the chamber in that outcome."""
    dist = dist[dist > 0.0005]
    colors = [DEM if k >= threshold else REP for k in dist.index]
    fig = go.Figure(go.Bar(
        x=dist.index, y=dist.values * 100, marker_color=colors, marker_line_width=0,
        hovertemplate=f"%{{x}} Democratic {label} seats (%{{customdata}} R)<br>%{{y:.1f}}% of simulations<extra></extra>",
        customdata=[total - k for k in dist.index]))
    fig.add_vline(x=threshold - 0.5, line_dash='dash', line_color=INK_3, line_width=1)
    fig.add_annotation(x=threshold - 0.5, y=1.0, yref='paper', yanchor='bottom', xanchor='center',
                       text=f"{threshold} for Democratic control", showarrow=False, font=dict(color=INK_2, size=11))
    fig.update_xaxes(title=f"Democratic {label} seats")
    fig.update_yaxes(title="% of simulations")
    _layout(fig, height=290, bargap=0.15, showlegend=False)
    fig.update_layout(margin=dict(t=36))
    return fig

def forecast_history_chart(log):
    log = log.copy()
    log['published_at'] = pd.to_datetime(log['published_at'])
    fig = go.Figure()
    for col, name, color in (('senate_dem_control_prob', 'Senate', DEM), ('house_dem_control_prob', 'House', IND)):
        fig.add_trace(go.Scatter(
            x=log['published_at'], y=log[col] * 100, name=name, mode='lines+markers+text',
            line=dict(color=color, width=2), marker=dict(size=8),
            text=[''] * (len(log) - 1) + [f"{name} {log[col].iloc[-1] * 100:.0f}%"],
            textposition='middle right', textfont=dict(color=INK_2), cliponaxis=False,
            customdata=log['model_version'],
            hovertemplate=f"{name}: %{{y:.0f}}% Dem control<br>%{{x|%b %d %H:%M}} · model %{{customdata}}<extra></extra>"))
    fig.add_hline(y=50, line_dash='dot', line_color=INK_3, line_width=1)
    fig.update_yaxes(title="Democratic control chance (%)", range=[0, 100])
    _layout(fig, height=280, legend=dict(orientation='h', y=1.12, x=0))
    fig.update_layout(margin=dict(r=90))   # room for the end-of-line labels
    return fig

def render_overview(f, model_version, log):
    s = f['summary']
    as_of = pd.Timestamp(f['as_of'])
    st.title(f"{MODEL_NAME}: 2026 Midterms")
    st.markdown(f"*{TAGLINE}*")
    st.caption(
        f"As of {as_of:%B} {as_of.day}, {as_of.year} · {f['days_to_election']} days to Election Day (Nov 3) · "
        f"model {model_version} · generic-ballot polls D{f['generic_ballot_poll']:+.1f}, read as an expected House "
        f"vote of D{f['national_environment']:+.1f} after removing the polls' historical {f['generic_ballot_bias_removed']:.1f}-pt "
        "Democratic lean")

    c1, c2, c3 = st.columns(3)
    sen, house, gov = s['senate'], s['house'], s['governor']
    with c1:
        st.metric("Senate: Democratic control", f"{sen['dem_control_prob']:.0%}",
                  help="Share of 20,000 simulations where Democrats (incl. independents who caucus with them) "
                       "reach 51 seats. VP Vance breaks 50-50 ties for Republicans.")
        st.caption(f"Republican control {sen['rep_control_prob']:.0%} · median {sen['dem_seats_median']} D seats "
                   f"(90%: {sen['dem_seats_90pct'][0]}–{sen['dem_seats_90pct'][1]})")
    with c2:
        st.metric("House: Democratic control", f"{house['dem_control_prob']:.0%}")
        st.caption(f"Median {house['dem_seats_median']} D seats (90%: {house['dem_seats_90pct'][0]}–"
                   f"{house['dem_seats_90pct'][1]}) · 218 for a majority")
    with c3:
        st.metric("Governorships won by Democrats", f"{gov['dem_governorships_median']} of {gov['races']}")
        st.caption(f"90% range: {gov['dem_governorships_90pct'][0]}–{gov['dem_governorships_90pct'][1]} of the 36 races on the ballot")

    st.divider()
    l, r = st.columns(2)
    with l:
        st.subheader("Senate: how many seats?")
        st.plotly_chart(seat_distribution_chart(sen['dem_seats_dist'], 51, 'Senate', 100), width='stretch', key='ov_senate_dist')
    with r:
        st.subheader("House: how many seats?")
        st.plotly_chart(seat_distribution_chart(house['dem_seats_dist'], 218, 'House', 435), width='stretch', key='ov_house_dist')

    l, r = st.columns(2)
    with l:
        st.subheader("Closest races")
        close = pd.concat([
            f['senate'].assign(Office='Senate'), f['governor'].assign(Office='Governor')], ignore_index=True)
        close['p'] = close['dem_side_win_prob']
        close = close.loc[(close['p'] - 0.5).abs().sort_values().index].head(10)
        st.dataframe(pd.DataFrame({
            'Race': close['Office'] + ' · ' + close['state'].map(_title_case),
            'Matchup': close['dem_candidate'].fillna('Dem') + ' vs. ' + close['rep_candidate'].fillna('Rep'),
            'Forecast': close['forecast_margin'].map(fmt_margin),
            'Dem-side win %': close['p'] * 100,
        }), hide_index=True, width='stretch', column_config={
            'Dem-side win %': st.column_config.ProgressColumn(color=DEM, format='%.0f%%', min_value=0, max_value=100)})
    with r:
        st.subheader("Forecast over time")
        if log is not None and len(log):
            st.plotly_chart(forecast_history_chart(log), width='stretch', key='ov_history')
            st.caption("Each point is a published snapshot in forecasts/2026/topline_log.csv.")
        else:
            st.info("No published snapshots yet. Run `python src/publish_forecast.py`.")

    env = s.get('environment_check')
    if env and env.get('implied_by_race_polls') is not None:
        st.subheader("Cross-check: what our own polls say about the national environment")
        c = st.columns(len(env['by_office']) + 2)
        c[0].metric("Generic-ballot polls", f"D{env['generic_ballot_poll']:+.1f}",
                    help="The non-market-tied generic-ballot average, before the model removes its historical lean "
                         f"(the forecast uses D{env['environment_used']:+.1f}).")
        c[1].metric("Implied by Senate + House polls", f"D{env['implied_by_race_polls']:+.1f}",
                    delta=f"{env['gap']:+.1f} vs generic", delta_color='off')
        for i, (label, v) in enumerate(env['by_office'].items()):
            c[i + 2].metric(label, f"D{v['implied']:+.1f}",
                            help=f"From {v['races']} polled races" + (" (noisy: governors run far from their "
                                 "state's lean; not in the combined read)" if label == 'Governor' else ''))
        if env['flag']:
            st.warning("Race polls and generic-ballot polls disagree by more than 3 points. One of them may be off.")
        else:
            st.caption("Both are poll readings, compared before any lean correction: race-by-race polls tell the same "
                       "story as the generic ballot, which is a good sign the national input isn't off.")

    with st.expander("How this forecast works"):
        st.markdown(f"""
- **Polls first.** Every Senate, governor, and House district poll (from Wikipedia's race pages, which compile
  pollsters' own releases; no prediction-market-affiliated sources) is averaged with weights for recency,
  sample size, and pollster track record, after correcting each pollster's partisan lean (rated from 18,000
  historical polls with known results).
- **Fundamentals as a backstop.** A structural model (state partisanship, incumbency, national environment)
  fills in races with few or no polls. The two are blended by how accurate each has historically been, so
  polls take over as they accumulate and election day nears.
- **House.** All 435 districts use their 2026 partisan lean on this year's maps, incumbency, and the national
  environment. District polls aren't included yet.
- **Everything moves together.** Each of 20,000 simulations draws one national polling error that hits every
  race at once (like 2016 or 2020), which is why chamber odds are less certain than individual races suggest.
- **Graded.** Backtested on every polled race 2000–2022; see the Track Record tab.
""")


# ── Senate / Governor ─────────────────────────────────────────────────────────
def _race_hover(r, office):
    d_party = 'I' if r['dem_side_party'] == 'INDEPENDENT' else 'D'
    match = (f"{r['dem_candidate']} ({d_party}) vs. {r['rep_candidate']} (R)" if isinstance(r['dem_candidate'], str)
             else "Candidates not yet in poll tables")
    polls = f"{r['n_polls']} polls, avg {fmt_margin(r['poll_avg'])}" if r['n_polls'] else "No polls"
    cons = r.get('consensus_rating')
    return (f"<b>{_title_case(r['state'])} {office}</b>{' (special)' if r.get('special') else ''}<br>{match}<br>"
            f"Forecast: {fmt_margin(r['forecast_margin'])} · {d_party} win {r['dem_side_win_prob']:.0%}<br>"
            f"{polls} · fundamentals {fmt_margin(r['fund_pred'])}<br>"
            f"Experts' consensus: {fmt_margin(cons) if pd.notna(cons) else '—'}")

def race_map(df, office):
    d = df.copy()
    d['po'] = d['state'].map(STATE_ABBREV)
    # Independents winning (Osborn) show in violet-free form here: the scale encodes
    # Dem-side probability, and the hover names the candidate's party.
    fig = go.Figure(go.Choropleth(
        locations=d['po'], z=d['dem_side_win_prob'], locationmode='USA-states',
        colorscale=PROB_SCALE, zmin=0, zmax=1, marker_line_color='white', marker_line_width=1,
        hovertext=[_race_hover(r, office) for _, r in d.iterrows()], hoverinfo='text',
        colorbar=dict(title='Dem-side<br>win %', tickvals=[0, 0.25, 0.5, 0.75, 1],
                      ticktext=['R 100', '75', '50', '75', 'D 100'], len=0.7, thickness=12)))
    fig.update_layout(geo=dict(scope='usa', bgcolor='rgba(0,0,0,0)', lakecolor='rgba(0,0,0,0)',
                               landcolor='rgba(128,128,128,0.12)', showland=True,
                               subunitcolor='rgba(128,128,128,0.4)'))
    return _layout(fig, height=420)

def race_table(df):
    d = df.sort_values('forecast_margin')
    out = pd.DataFrame({
        'State': d['state'].map(_title_case) + np.where(d.get('special', False), ' (special)', ''),
        'Democratic side': d['dem_candidate'].fillna('—') + np.where(d['dem_side_party'] == 'INDEPENDENT', ' (I)', ''),
        'Republican': d['rep_candidate'].fillna('—'),
        'Held by': d['incumbent_party'].str[0] + np.where(d['open_seat'], ' (open)', ''),
        'Polls': d['n_polls'],
        'Poll avg': d['poll_avg'].map(fmt_margin),
        'Fundamentals': d['fund_pred'].map(fmt_margin),
        'Forecast': d['forecast_margin'].map(fmt_margin),
        'Dem-side win %': d['dem_side_win_prob'] * 100,
        'Experts': d['consensus_rating'].map(fmt_margin),
    })
    return out

def poll_chart(polls, avg_margin, d_name, r_name, series=None):
    p = polls.sort_values('end_date')
    fig = go.Figure()
    if series is not None and series['poll_avg'].notna().any():
        fig.add_trace(go.Scatter(
            x=series['date'], y=series['poll_avg'], mode='lines', name='Polling average',
            line=dict(color=INK_2, width=2), connectgaps=False,
            hovertemplate="Average %{x|%b %d}: %{y:+.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=p['end_date'], y=p['adjusted_margin'], mode='markers', name='Poll (after lean correction)',
        marker=dict(size=9, color=np.where(p['adjusted_margin'] >= 0, DEM, REP), line=dict(color='white', width=2)),
        customdata=np.stack([p['pollster'], p['margin'].map(fmt_margin), p['adjusted_margin'].map(fmt_margin),
                             p['sample_size'].fillna(0).astype(int).astype(str) + ' ' + p['population'].fillna('')], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{x|%b %d}<br>Reported %{customdata[1]} → corrected %{customdata[2]}"
                      "<br>n = %{customdata[3]}<extra></extra>"))
    fig.add_hline(y=0, line_color=INK_3, line_width=1)
    if series is None and avg_margin is not None and not pd.isna(avg_margin):
        fig.add_hline(y=avg_margin, line_dash='dash', line_color=INK_2, line_width=1.5,
                      annotation_text=f"Weighted average {fmt_margin(avg_margin)}", annotation_font_color=INK_2)
    last = lambda n: str(n).split()[-1] if isinstance(n, str) and n else n
    fig.update_yaxes(title=f"Margin (+ {last(d_name)}, − {last(r_name)})")
    return _layout(fig, height=320, showlegend=False)

def share_card_block(f, race, office, key, model_version):
    """Card preview + PNG download for one race."""
    from share_cards import make_card, card_filename
    png = make_card(race, office, pd.Timestamp(f['as_of']), model_version, f['days_to_election'])
    with st.expander("📣 Share card for this race"):
        st.image(png, width='stretch')
        st.download_button("Download card (PNG)", png, file_name=card_filename(race, office), mime='image/png', key=key)
        st.caption("Sized for social media (1200×675). It shows a probability, not a call, and carries the date and "
                   "model version, so a saved copy can't pass for a later forecast.")

def render_race_detail(f, annotated, office, label, model_version='', history=None):
    df = f[office]
    states = df.sort_values('state')['state'].tolist()
    default = (df['dem_side_win_prob'] - 0.5).abs().idxmin()
    state = st.selectbox(f"Look inside a {label} race", states, index=states.index(df.loc[default, 'state']),
                         format_func=_title_case, key=f"detail_{office}")
    r = df[df['state'] == state].iloc[0]
    party = 'Independent' if r['dem_side_party'] == 'INDEPENDENT' else 'Democrat'
    c = st.columns(4)
    c[0].metric("Forecast margin", fmt_margin(r['forecast_margin']),
                help="Blend of the polling average and the fundamentals model, weighted by how accurate each has been.")
    c[1].metric(f"{party} win chance", f"{r['dem_side_win_prob']:.0%}")
    c[2].metric("Weight on polls", f"{r['poll_weight']:.0%}",
                help="Rises with more polls and fewer days to go. Unpolled races rely on fundamentals alone.")
    cons = r.get('consensus_rating')
    c[3].metric("Experts' consensus", fmt_margin(cons) if pd.notna(cons) else '—',
                help="Average of ~10 professional forecasters' ratings (Safe = 30, Likely = 15, Lean = 7, Tilt = 3), "
                     "scraped from Wikipedia. Shown for comparison; not used by the model.")
    share_card_block(f, r, office, f"card_{office}_{state}", model_version)
    st.caption(f"Polling average {fmt_margin(r['poll_avg'])} from {r['n_polls']} polls "
               f"(worth {r['effective_polls']:.1f} independent polls) · fundamentals {fmt_margin(r['fund_pred'])} · "
               f"partisan lean {fmt_margin(r['lean'])} · held by {r['incumbent_party'].title()}"
               f"{' (open seat)' if r['open_seat'] else ''}")

    polls = annotated[(annotated['office'] == office) & (annotated['state'] == state)]
    hist = history[(history['office'] == office) & (history['state'] == state)] if history is not None and len(history) else None
    l, rcol = st.columns(2)
    with l:
        st.markdown("**Polls and the polling average**")
        if polls.empty:
            st.info("No general-election polls found for this race. The forecast uses fundamentals alone.")
        else:
            series = race_average_series(polls, f['as_of'])
            st.plotly_chart(poll_chart(polls, r['poll_avg'], r['dem_candidate'], r['rep_candidate'], series),
                            width='stretch', key=f'polls_{office}')
    with rcol:
        st.markdown(f"**Our {party.lower()} win chance over time**")
        if hist is not None and len(hist):
            st.plotly_chart(win_history_chart(hist, r['dem_side_win_prob'], f['as_of'], party), width='stretch',
                            key=f'winhist_{office}')
            st.caption("One point per published daily forecast.")
        else:
            st.caption("No published forecasts yet.")
    if polls.empty:
        return
    st.caption("Dots are individual polls after their lean correction; the line is the weighted polling average "
               "on each day (before the forecast's statewide lean correction and its blend with fundamentals).")
    shown = polls.sort_values('end_date', ascending=False)
    st.dataframe(pd.DataFrame({
        'Pollster': shown['pollster'],
        'Ends': shown['end_date'].dt.strftime('%b %d, %Y'),
        'Sample': shown['sample_size'].fillna(0).astype(int).astype(str).str.replace('^0$', '—', regex=True)
                  + ' ' + shown['population'].fillna(''),
        r['dem_candidate'] or 'Dem': shown['dem_pct'].map(lambda v: f"{v:.0f}%"),
        r['rep_candidate'] or 'Rep': shown['rep_pct'].map(lambda v: f"{v:.0f}%"),
        'Reported': shown['margin'].map(fmt_margin),
        'Lean correction': shown['house_correction'].map(lambda v: f"{-v:+.1f}" if v else '—'),
        'Rated as': np.where(shown['sponsor'].notna(), 'Sponsored (' + shown['sponsor'].fillna('').str.title() + ')',
                             shown['matched_pollster'].fillna('Unrated')),
        'Source': shown['source_url'],
    }), hide_index=True, width='stretch', height=min(35 * (len(shown) + 1) + 3, 420),
        column_config={'Source': st.column_config.LinkColumn(
            display_text=r'https?://(?:www\.)?([^/]+).*',
            help="Links to the Wikipedia page the poll was taken from, which cites the pollster's release.")})

# ── Seat bars, House state map, and the share image ──────────────────────────
# Explicit shades rather than opacity, which darkens on dark backgrounds: lighter tints
# for less certain seats, deeper shades for Senate seats not up (already decided).
DEM_TINT, REP_TINT, DEM_DEEP, REP_DEEP = '#86b6ef', '#f09a99', '#1c5cab', '#a8302f'
HOUSE_BANDS = [   # (low, high, label, color, opacity) on the Democratic win probability
    (0.90, 1.01, 'Safe or likely Democratic', DEM, 1.0),
    (0.65, 0.90, 'Lean Democratic', DEM_TINT, 1.0),
    (0.35, 0.65, 'Toss-up', NEUTRAL, 1.0),
    (0.10, 0.35, 'Lean Republican', REP_TINT, 1.0),
    (-0.01, 0.10, 'Safe or likely Republican', REP, 1.0),
]

def senate_bar_segments(s):
    """Senate seats left to right: Democratic seats not up, projected Democratic wins,
    any seat neither median claims (an independent or a coin flip), projected Republican
    wins, Republican seats not up. Medians, so the bar shows the most likely split."""
    hold_d, hold_r = s['holdovers']['DEMOCRAT'], s['holdovers']['REPUBLICAN']
    d, r = s['dem_seats_median'], s['rep_seats_median']
    return [
        (hold_d, DEM_DEEP, 1.0, f"{hold_d} Democratic seats not up in 2026"),
        (d - hold_d, DEM, 1.0, f"{d - hold_d} projected Democratic wins in 2026"),
        (100 - d - r, NEUTRAL, 1.0, f"{100 - d - r} seat(s) too close to assign, or won by an independent"),
        (r - hold_r, REP, 1.0, f"{r - hold_r} projected Republican wins in 2026"),
        (hold_r, REP_DEEP, 1.0, f"{hold_r} Republican seats not up in 2026"),
    ]

def house_bar_segments(h):
    """All 435 districts left to right by Democratic win chance, grouped into rating bands."""
    p = h['dem_win_prob']
    return [(int(((p >= lo) & (p < hi)).sum()), color, opacity, f"{int(((p >= lo) & (p < hi)).sum())} {label}")
            for lo, hi, label, color, opacity in HOUSE_BANDS]

def seat_bar_traces(segments):
    return [go.Bar(x=[n], y=[''], orientation='h', marker=dict(color=color, opacity=opacity, line_width=0),
                   hovertemplate=f"{text}<extra></extra>", showlegend=False)
            for n, color, opacity, text in segments if n > 0]

def seat_bar(segments, total, line_at, line_text):
    fig = go.Figure(seat_bar_traces(segments))
    fig.add_vline(x=line_at, line_color=INK_2, line_width=2, line_dash='dot')
    fig.add_annotation(x=line_at, y=1, yref='paper', yanchor='bottom', showarrow=False, font=dict(color=INK_2, size=11),
                       text=line_text)
    fig.update_xaxes(range=[0, total], visible=False)
    fig.update_yaxes(visible=False)
    _layout(fig, height=90, barmode='stack', bargap=0)
    fig.update_layout(margin=dict(l=0, r=0, t=26, b=0))
    return fig

def senate_seat_bar(s):
    return seat_bar(senate_bar_segments(s), 100, 50.5,
                    "51 for Democratic control · Republicans need 50 (VP Vance breaks ties)")

def house_seat_bar(h):
    return seat_bar(house_bar_segments(h), 435, 217.5, "218 for a majority")

def _seat_header(dem, rep, dem_note, rep_note):
    l, _, r = st.columns([2, 3, 2])
    with l:
        st.markdown(f"<div style='color:{DEM};font-weight:700;font-size:1.1rem'>Democrats</div>"
                    f"<div style='font-size:2.4rem;font-weight:700;line-height:1.1'>{dem}</div>",
                    unsafe_allow_html=True)
        st.caption(dem_note)
    with r:
        st.markdown(f"<div style='text-align:right'><div style='color:{REP};font-weight:700;font-size:1.1rem'>Republicans</div>"
                    f"<div style='font-size:2.4rem;font-weight:700;line-height:1.1'>{rep}</div></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:right;color:{INK_2};font-size:0.875rem'>{rep_note}</div>", unsafe_allow_html=True)

def senate_with_rep_fields(s):
    """Fills the Republican seat fields for a forecast cached before they existed,
    approximated from the Democratic numbers (independents rarely win more than one seat)."""
    s = dict(s)
    if 'rep_seats_median' not in s:
        ind = round(s.get('independent_seats_expected', 0))
        s['rep_seats_median'] = 100 - s['dem_seats_median'] - ind
        s['rep_seats_90pct'] = [100 - s['dem_seats_90pct'][1] - ind, 100 - s['dem_seats_90pct'][0] - ind]
    return s

def house_numbers(s):
    """Republican House numbers mirror the Democratic ones (every seat goes to one party)."""
    return {'rep_seats_median': 435 - s['dem_seats_median'],
            'rep_seats_90pct': [435 - s['dem_seats_90pct'][1], 435 - s['dem_seats_90pct'][0]],
            # from the rounded Democratic percentage, so the two always add to 100%
            'rep_control_prob': (100 - round(s['dem_control_prob'] * 100)) / 100}

def render_senate_seat_header(s):
    """Projected seats for each party, chance of control, and the seat bar."""
    _seat_header(s['dem_seats_median'], s['rep_seats_median'],
                 f"{s['dem_control_prob']:.0%} chance of control · 90% range {s['dem_seats_90pct'][0]}–{s['dem_seats_90pct'][1]}",
                 f"{s['rep_control_prob']:.0%} chance of control · 90% range {s['rep_seats_90pct'][0]}–{s['rep_seats_90pct'][1]}")
    st.plotly_chart(senate_seat_bar(s), width='stretch', key='senate_seat_bar', config={'displayModeBar': False})
    st.caption("Projected seats are the most likely totals across 20,000 simulations, counting independents who caucus "
               "with Democrats. The deeper-colored ends of the bar are seats not up this year.")

def render_house_seat_header(s, h):
    r = house_numbers(s)
    _seat_header(s['dem_seats_median'], r['rep_seats_median'],
                 f"{s['dem_control_prob']:.0%} chance of control · 90% range {s['dem_seats_90pct'][0]}–{s['dem_seats_90pct'][1]}",
                 f"{r['rep_control_prob']:.0%} chance of control · 90% range {r['rep_seats_90pct'][0]}–{r['rep_seats_90pct'][1]}")
    st.plotly_chart(house_seat_bar(h), width='stretch', key='house_seat_bar', config={'displayModeBar': False})
    st.caption("The numbers above are the most likely seat totals across 20,000 simulations. The bar sorts all 435 "
               "districts by how likely each party is to win them (Lean: 65–90%; Toss-up: 35–65%), so its split can "
               "differ from the totals by a few seats.")

def house_state_table(h):
    """Per state: seats, expected Democratic seats, and the competitive districts."""
    rows = []
    for state, g in h.groupby('state'):
        comp = g[(g['dem_win_prob'] > 0.1) & (g['dem_win_prob'] < 0.9)].sort_values('dem_win_prob')
        rows.append({'state': state, 'seats': len(g), 'exp_dem': float(g['dem_win_prob'].sum()),
                     'fav_dem': int((g['dem_win_prob'] >= 0.5).sum()),
                     'competitive': ', '.join(f"{STATE_ABBREV[state]}-{'AL' if d == 0 else d} ({p:.0%} D)"
                                              for d, p in zip(comp['district'], comp['dem_win_prob']))})
    t = pd.DataFrame(rows)
    t['dem_share'] = t['exp_dem'] / t['seats']
    return t

def house_map_trace(h, colorbar=True):
    t = house_state_table(h)
    hover = [f"<b>{_title_case(r['state'])}</b><br>{r['seats']} seat{'s' if r['seats'] > 1 else ''} · Democrats favored in "
             f"{r['fav_dem']}, expected {r['exp_dem']:.1f}"
             + (f"<br>Competitive: {r['competitive']}" if r['competitive'] else '<br>No competitive districts')
             for _, r in t.iterrows()]
    return go.Choropleth(
        locations=t['state'].map(STATE_ABBREV), z=t['dem_share'], locationmode='USA-states',
        colorscale=PROB_SCALE, zmin=0, zmax=1, marker_line_color='white', marker_line_width=1,
        hovertext=hover, hoverinfo='text', showscale=colorbar,
        colorbar=dict(title='Expected<br>Dem share<br>of seats', tickvals=[0, 0.5, 1], ticktext=['All R', 'Even', 'All D'],
                      len=0.7, thickness=12))

def house_map(h):
    fig = go.Figure(house_map_trace(h))
    fig.update_layout(geo=dict(scope='usa', bgcolor='rgba(0,0,0,0)', lakecolor='rgba(0,0,0,0)',
                               landcolor='rgba(128,128,128,0.12)', showland=True, subunitcolor='rgba(128,128,128,0.4)'))
    return _layout(fig, height=420)

def share_image(office, f, model_version):
    """Headline numbers, seat bar and map as one 1200x900 PNG (dark, like the post builder)."""
    from plotly.subplots import make_subplots
    bg, ink, ink_2 = '#15202b', '#ffffff', '#8b98a5'
    as_of = pd.Timestamp(f['as_of'])
    if office == 'senate':
        s = senate_with_rep_fields(f['summary']['senate'])
        segments, total, line_at, line_text = senate_bar_segments(s), 100, 50.5, "51 for Democratic control"
        dem, rep = s['dem_seats_median'], s['rep_seats_median']
        dem_p, rep_p = s['dem_control_prob'], s['rep_control_prob']
        d = f['senate'].copy()
        d['po'] = d['state'].map(STATE_ABBREV)
        geo_trace = go.Choropleth(locations=d['po'], z=d['dem_side_win_prob'], locationmode='USA-states',
                                  colorscale=PROB_SCALE, zmin=0, zmax=1, marker_line_color=bg, marker_line_width=1,
                                  showscale=False)
        legend = "Map: each state's Senate race, shaded by win chance (blue Democratic, red Republican, gray toss-up)"
    else:
        s = f['summary']['house']
        r = house_numbers(s)
        segments, total, line_at, line_text = house_bar_segments(f['house']), 435, 217.5, "218 for a majority"
        dem, rep, dem_p, rep_p = s['dem_seats_median'], r['rep_seats_median'], s['dem_control_prob'], r['rep_control_prob']
        geo_trace = house_map_trace(f['house'], colorbar=False)
        geo_trace.marker.line.color = bg
        legend = "Map: each state shaded by the share of its House seats Democrats are expected to win"
    fig = make_subplots(rows=2, cols=1, row_heights=[0.09, 0.91], vertical_spacing=0.03,
                        specs=[[{'type': 'xy'}], [{'type': 'geo'}]])
    for t in seat_bar_traces(segments):
        fig.add_trace(t, row=1, col=1)
    fig.update_xaxes(range=[0, total], visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.add_shape(type='line', x0=line_at, x1=line_at, y0=-0.6, y1=0.6, line=dict(color=ink, width=2, dash='dot'), row=1, col=1)
    fig.add_trace(geo_trace, row=2, col=1)
    fig.update_geos(scope='usa', bgcolor=bg, lakecolor=bg, landcolor='#22303c', showland=True, subunitcolor=bg)
    a = lambda **k: fig.add_annotation(xref='paper', yref='paper', showarrow=False, **k)
    a(x=0.5, y=1.235, text=f"<b>2026 {office.title()} Forecast</b>", font=dict(size=34, color=ink), xanchor='center')
    a(x=0.5, y=1.30, text=f"<b>{MODEL_NAME.upper()}</b>", font=dict(size=15, color=ink_2), xanchor='center')
    a(x=0, y=1.13, text=f"<b>Democrats {dem}</b>", font=dict(size=30, color='#5598e7'), xanchor='left')
    a(x=0, y=1.075, text=f"{dem_p:.0%} chance of control", font=dict(size=16, color=ink_2), xanchor='left')
    a(x=1, y=1.13, text=f"<b>{rep} Republicans</b>", font=dict(size=30, color='#e66767'), xanchor='right')
    a(x=1, y=1.075, text=f"{rep_p:.0%} chance of control", font=dict(size=16, color=ink_2), xanchor='right')
    a(x=line_at / total, y=1.075, text=line_text, font=dict(size=13, color=ink_2), xanchor='center')
    a(x=0.5, y=-0.02, text=legend, font=dict(size=13, color=ink_2), xanchor='center')
    a(x=0.5, y=-0.065, text=f"{MODEL_NAME} · as of {as_of:%b} {as_of.day}, {as_of.year} · model {model_version} · "
                            "a forecast, not a call · no prediction-market data", font=dict(size=13, color=ink_2), xanchor='center')
    fig.update_layout(width=1200, height=900, paper_bgcolor=bg, plot_bgcolor=bg, barmode='stack', bargap=0,
                      margin=dict(l=40, r=40, t=210, b=80), showlegend=False)
    return fig.to_image(format='png', width=1200, height=900)

def share_image_block(office, f, model_version):
    with st.expander("📣 Share image: headline, seat bar and map"):
        if st.button("Build image", key=f'share_build_{office}'):
            try:
                st.session_state[f'share_png_{office}'] = share_image(office, f, model_version)
            except Exception as e:
                st.error(f"Couldn't build the image: {e}")
        png = st.session_state.get(f'share_png_{office}')
        if png:
            st.image(png, width='stretch')
            as_of = pd.Timestamp(f['as_of'])
            st.download_button("Download image (PNG)", png, mime='image/png', key=f'share_dl_{office}',
                               file_name=f"{office}_forecast_{as_of:%Y-%m-%d}.png")
        st.caption("1200×900, dark background. Dated and versioned, so a saved copy can't pass for a later forecast.")

def render_statewide(f, annotated, office, label, chamber_note, model_version='', history=None):
    df = f[office]
    st.title(f"2026 {label} Forecast")
    if office == 'senate':
        render_senate_seat_header(senate_with_rep_fields(f['summary']['senate']))
    st.caption(chamber_note)
    st.plotly_chart(race_map(df, label), width='stretch', key=f'map_{office}')
    if office == 'senate':
        share_image_block('senate', f, model_version)
        st.divider()
        render_senate_paths(f)
        st.divider()
    st.subheader("Every race")
    st.dataframe(race_table(df), hide_index=True, width='stretch', height=min(38 * (len(df) + 1), 640),
                 column_config={'Dem-side win %': st.column_config.ProgressColumn(color=DEM, 
                     format='%.0f%%', min_value=0, max_value=100,
                     help="Chance the Democratic (or, in Nebraska's Senate race, independent) candidate wins.")})
    st.caption(f"{POLL_SOURCE_NOTE} "
               "Margins are Democratic-side minus Republican, in points. 'Experts' is the professional "
               "forecasters' consensus, on a rating scale rather than a vote margin.")
    st.divider()
    render_race_detail(f, annotated, office, label, model_version, history)


# ── House ─────────────────────────────────────────────────────────────────────
def render_house(f, model_version=''):
    s = f['summary']['house']
    h = f['house'].copy()
    st.title("2026 House Forecast")
    render_house_seat_header(s, h)
    cal = s['calibration']
    st.caption(f"All 435 districts on this year's maps (including the 9 states that redrew for 2026). District "
               f"margin ≈ {cal['coefficients']['pvi']:.2f} × Cook PVI + {cal['coefficients']['incumbency']:.1f} for an "
               f"incumbent, shifted by the national environment. Tested on 2022: "
               f"{cal['out_of_year']['winner_accuracy']:.1%} of winners right, typical miss ±{cal['district_sd']:.1f} pts. "
               f"{int((h['n_polls'] > 0).sum())} districts also have polls, blended in by how reliable they are "
               "(most have only one or two, often campaign-sponsored, so they get modest weight).")
    st.plotly_chart(house_map(h), width='stretch', key='house_map')
    st.caption(f"States are shaded by the share of their seats Democrats are expected to win; hover for "
               f"each state's competitive districts. {int(((h['dem_win_prob'] > 0.1) & (h['dem_win_prob'] < 0.9)).sum())} "
               "districts are in play (neither party above 90%). District-level shapes aren't shown: nine "
               "states redrew their maps for 2026.")
    share_image_block('house', f, model_version)
    st.subheader("How many seats?")
    st.plotly_chart(seat_distribution_chart(s['dem_seats_dist'], 218, 'House', 435), width='stretch', key='house_dist')
    render_house_tipping(h)

    st.subheader("Competitive districts")
    comp = h[(h['dem_win_prob'] > 0.1) & (h['dem_win_prob'] < 0.9)].copy()
    comp['label'] = comp['state'].map(STATE_ABBREV) + '-' + comp['district'].map(lambda d: 'AL' if d == 0 else str(d))
    comp = comp.sort_values('dem_win_prob')
    fig = go.Figure(go.Scatter(
        x=comp['dem_win_prob'] * 100, y=comp['label'], mode='markers',
        marker=dict(size=10, color=comp['dem_win_prob'], colorscale=PROB_SCALE, cmin=0, cmax=1,
                    line=dict(color='white', width=2)),
        customdata=np.stack([comp['incumbent'].fillna('Open seat'), comp['incumbent_party'].fillna('—'),
                             comp['pvi'].map(fmt_margin), comp['predicted_margin'].map(fmt_margin)], axis=1),
        hovertemplate="<b>%{y}</b><br>Dem win %{x:.0f}%<br>Incumbent: %{customdata[0]} (%{customdata[1]})"
                      "<br>Cook PVI %{customdata[2]} · forecast %{customdata[3]}<extra></extra>"))
    fig.add_vline(x=50, line_dash='dot', line_color=INK_3)
    fig.update_xaxes(title="Democratic win chance (%)", range=[0, 100])
    st.plotly_chart(_layout(fig, height=max(320, 18 * len(comp))), width='stretch', key='house_competitive')

    if 'consensus_rating' in h and h['consensus_rating'].notna().any():
        st.subheader("Where we disagree most with the experts")
        st.caption("Our win probability versus the professional forecasters' consensus (Cook, Sabato, Inside Elections "
                   "and others without prediction-market ties), converted to a probability on the same scale used in "
                   "grading. These are the districts most worth a second look. Experts often know about candidate "
                   "quality or private polling our model can't see.")
        from grade_2026 import expert_prob
        dd = h[h['consensus_rating'].notna() & ~h['uncontested'].astype(bool)].copy()
        dd['experts_p'] = dd['consensus_rating'].map(expert_prob)
        dd['gap'] = dd['dem_win_prob'] - dd['experts_p']
        dd = dd.loc[dd['gap'].abs().sort_values(ascending=False).index].head(12)
        st.dataframe(pd.DataFrame({
            'District': dd['state'].map(STATE_ABBREV) + '-' + dd['district'].map(lambda d: 'AL' if d == 0 else str(d)).astype(str),
            'Incumbent': np.where(dd['incumbent_running'].astype(bool), dd['incumbent'].fillna(''), 'Open seat'),
            'Our Dem win %': (dd['dem_win_prob'] * 100).round(0),
            'Experts Dem win %': (dd['experts_p'] * 100).round(0),
            'Experts rating': dd['consensus_rating'].map(fmt_margin),
            'District polls': dd['n_polls'],
        }), hide_index=True, width='stretch')

    st.subheader("Share card for a district")
    labels = (h['state'].map(STATE_ABBREV) + '-' + h['district'].map(lambda d: 'AL' if d == 0 else str(d)).astype(str))
    order = (h['dem_win_prob'] - 0.5).abs().sort_values().index
    pick = st.selectbox("District (closest races first)", list(labels.loc[order]), key='house_card_pick')
    share_card_block(f, h.loc[labels[labels == pick].index[0]], 'house', f"card_house_{pick}", model_version)

    st.subheader("All districts")
    fc = st.columns([2, 2, 3])
    state = fc[0].selectbox("State", ['All states'] + sorted(h['state'].unique()), format_func=_title_case)
    view = fc[1].selectbox("Show", ['All', 'Competitive (10–90%)', 'Toss-ups (35–65%)', 'Open seats', 'Projected flips',
                                    'Polled districts'])
    q = fc[2].text_input("Search candidate or incumbent")
    t = h if state == 'All states' else h[h['state'] == state]
    if view.startswith('Competitive'):
        t = t[(t['dem_win_prob'] > 0.1) & (t['dem_win_prob'] < 0.9)]
    elif view.startswith('Toss'):
        t = t[(t['dem_win_prob'] >= 0.35) & (t['dem_win_prob'] <= 0.65)]
    elif view == 'Open seats':
        t = t[~t['incumbent_running'].astype(bool)]
    elif view == 'Polled districts':
        t = t[t['n_polls'] > 0]
    elif view == 'Projected flips':
        t = t[t['held_by'].notna() & (t['held_by'] != t['predicted_winner'])]
    if q:
        hay = (t['incumbent'].fillna('') + ' ' + t['dem_candidate'].fillna('') + ' ' + t['rep_candidate'].fillna('')).str.lower()
        t = t[hay.str.contains(q.lower(), regex=False)]
    if t.empty:
        st.info("No districts match these filters.")
        return
    st.dataframe(pd.DataFrame({
        'District': t['state'].map(STATE_ABBREV) + '-' + t['district'].map(lambda d: 'AL' if d == 0 else str(d)).astype(str),
        'Cook PVI': t['pvi'].map(fmt_margin),
        'Held by': t['held_by'].fillna('New seat').str.title(),
        'Incumbent': np.where(t['incumbent_running'].astype(bool), t['incumbent'].fillna(''), 'Open seat'),
        'Democrat': t['dem_candidate'].fillna('—'),
        'Republican': t['rep_candidate'].fillna('—'),
        'District polls': np.where(t['n_polls'] > 0, t['poll_avg'].map(fmt_margin) + ' (' + t['n_polls'].astype(str) + ')', '—'),
        'Forecast': np.where(t['uncontested'], 'Uncontested', t['predicted_margin'].map(fmt_margin)),
        'Dem win %': t['dem_win_prob'] * 100,
        'Experts': t['consensus_rating'].map(fmt_margin) if 'consensus_rating' in t else '—',
    }).sort_values('Dem win %'), hide_index=True, width='stretch', height=420,
        column_config={'Dem win %': st.column_config.ProgressColumn(color=DEM, format='%.0f%%', min_value=0, max_value=100)})
    st.caption(f"{len(t)} of 435 districts shown.")


# ── Track record ──────────────────────────────────────────────────────────────
def render_track_record(model_version):
    st.title("Track Record & Methodology")
    st.caption("How well does this forecast work? Every polled Senate and governor race from 2000 to 2022 was "
               "re-forecast using only information from other election years, then scored against what happened.")
    report_path = os.path.join(FORECASTS_DIR, 'calibration', 'calibration_report.json')
    if not os.path.exists(report_path):
        st.info("Run `python src/calibration.py` to generate the backtest.")
        return
    with open(report_path) as fh:
        report = json.load(fh)

    chart = os.path.join(FORECASTS_DIR, 'calibration', 'calibration_28d.png')
    if os.path.exists(chart):
        st.image(chart, width='stretch')
        st.caption("Left: when the model gave a favorite a given chance, how often they won. Dots inside the gray "
                   "bars mean the probabilities are honest. Right: probability quality versus simpler methods "
                   "(Brier score, lower is better).")

    for key, label in (('gb_bias_corrected_28d', '28 days before the election'), ('gb_bias_corrected_0d', 'Election eve')):
        st.subheader(f"Scores: {label}")
        sc = pd.DataFrame(report['scores'][key])
        st.dataframe(pd.DataFrame({
            'Method': sc['method'], 'Typical miss (pts)': sc['rmse'].round(1),
            'Winners right': (sc['winner_accuracy'] * 100).round(1).astype(str) + '%',
            'Brier (lower = better)': sc['brier'].round(3), 'Log loss': sc['log_loss'].round(3),
        }), hide_index=True, width='stretch')

    st.subheader("Calibration detail (28 days out)")
    cal = pd.DataFrame(report['calibration']['gb_bias_corrected_28d'])
    st.dataframe(pd.DataFrame({
        'Forecast chance for favorite': cal['bucket'], 'Races': cal['races'],
        'Average forecast': (cal['predicted'] * 100).round(0).astype(int).astype(str) + '%',
        'Favorite actually won': (cal['actual'] * 100).round(0).astype(int).astype(str) + '%',
        'Expected if calibrated': (cal['expected_low'] * 100).round(0).astype(int).astype(str) + '–'
                                  + (cal['expected_high'] * 100).round(0).astype(int).astype(str) + '%',
    }), hide_index=True, width='stretch')

    st.subheader("Generic-ballot correction")
    gb = pd.DataFrame(report['generic_ballot_bias'])
    st.markdown(f"Generic-ballot polls overstated Democrats in **{int(gb['cycles_overstating_dem'].max())} of "
                f"{int(gb['cycles'].max())}** cycles (1998–2022), by about **{gb['mean_dem_overstatement'].mean():.1f} pts**. "
                "Removing that lean improved the backtest, so the forecast applies it.")

    st.subheader("Data quality")
    try:
        from poll_checks import load_checks
        checks = load_checks()
    except Exception:
        checks = None
    if checks is None:
        st.info("No data checks yet. They run on every poll refresh (`python src/poll_checks.py`).")
    else:
        warn = checks[checks['severity'] == 'warning']
        st.markdown(f"Automatic checks on the current poll file: **{len(warn)} warnings**, "
                    f"{len(checks) - len(warn)} informational notes. Warnings are worth a look before publishing; "
                    "flagged polls stay in the average unless they're clearly wrong.")
        st.dataframe(checks.rename(columns=str.title), hide_index=True, width='stretch',
                     height=min(35 * (len(checks) + 1) + 3, 420))

    changelog = os.path.join(FORECASTS_DIR, 'CHANGELOG.md')
    if os.path.exists(changelog):
        with st.expander(f"Methodology changelog (current model: {model_version})"):
            with open(changelog, encoding='utf-8') as fh:
                st.markdown(fh.read())


# ── Comparisons ───────────────────────────────────────────────────────────────
def render_comparisons(f, inputs):
    import comparisons as CMP
    st.title("How we compare")
    st.caption("Our forecast next to other forecasters' published ratings, and each pollster next to our averages. "
               "Forecasters with prediction-market ties (Decision Desk HQ, RealClearPolitics, Silver Bulletin) are "
               "left out, consistent with the rest of the forecast.")

    totals = CMP.forecaster_totals('senate')
    if len(totals):
        st.subheader("Senate: projected Democratic seats")
        ours = f['summary']['senate']
        t = totals.copy()
        fig = go.Figure()
        fig.add_trace(go.Bar(y=t['forecaster'], x=t['dem'], orientation='h', marker_color=NEUTRAL, name='Leaning or safe D',
                             hovertemplate="%{y}: %{x} D seats + %{customdata} toss-ups<extra></extra>", customdata=t['tossups']))
        fig.add_trace(go.Bar(y=t['forecaster'], x=t['tossups'], base=t['dem'], orientation='h', marker_color=NEUTRAL,
                             marker_pattern_shape='/', opacity=0.5, name='Toss-ups', hoverinfo='skip'))
        fig.add_trace(go.Scatter(y=['Our model'], x=[ours['dem_seats_median']], mode='markers', marker=dict(size=12, color=DEM),
                                 error_x=dict(type='data', symmetric=False, array=[ours['dem_seats_90pct'][1] - ours['dem_seats_median']],
                                              arrayminus=[ours['dem_seats_median'] - ours['dem_seats_90pct'][0]], color=DEM),
                                 name='Our median (90% range)', hovertemplate="Our median %{x} D seats<extra></extra>"))
        fig.add_vline(x=51, line_dash='dash', line_color=INK_3, annotation_text='51 for control', annotation_font_color=INK_2)
        fig.update_xaxes(title='Democratic Senate seats (incl. independents who caucus with them)', range=[40, 60])
        st.plotly_chart(_layout(fig, height=330, barmode='overlay', legend=dict(orientation='h', y=-0.25)),
                        width='stretch', key='cmp_senate_totals')
        st.caption("Forecasters' totals count seats rated at least Tilt toward a party; the hatched part is their toss-ups.")

    for office, label in (('senate', 'Senate'), ('governor', 'Governor')):
        t = CMP.forecaster_table(f, office)
        if t.empty:
            continue
        st.subheader(f"{label}: race-by-race ratings")
        cols = [c for c in ('Cook', 'Sabato', 'IE', 'Econ', 'FPO', 'ST', 'Fox') if c in t]
        show = t[['state', 'ours', 'our_prob'] + cols + ['gap']].copy()
        show['state'] = show['state'].map(_title_case)
        show['our_prob'] = (show['our_prob'] * 100).round(0)
        show['gap'] = show['gap'].round(1)
        st.dataframe(show.rename(columns={'state': 'State', 'ours': 'Our rating', 'our_prob': 'Our Dem-side win %',
                                          'IE': 'Inside Elections', 'Econ': 'Economist', 'FPO': 'FiftyPlusOne',
                                          'ST': 'Split Ticket', 'gap': 'Gap (+ = we lean D)'}),
                     hide_index=True, width='stretch', height=min(35 * (len(show) + 1) + 3, 420))
        more_d = int((t['gap'] > 3).sum()); more_r = int((t['gap'] < -3).sum())
        st.caption(f"Sorted by how far we are from the forecasters' average. We're notably more Democratic in "
                   f"{more_d} races and more Republican in {more_r}. Our probabilities are converted to rating words "
                   "(Tossup under 55%, Tilt 55–65%, Lean 65–78%, Likely 78–91%, Safe above 91%).")

    st.subheader("Pollsters: who leans which way this cycle")
    leans = CMP.pollster_leans(inputs['annotated_polls'], inputs['poll_avgs'])
    if len(leans):
        l = leans.copy()
        fig = go.Figure(go.Bar(
            y=l['name'], x=l['lean'], orientation='h',
            marker_color=[DEM if v > 0 else REP for v in l['lean']],
            customdata=np.stack([l['polls'], l['races'], l['sponsor'].fillna('none')], axis=1),
            hovertemplate="%{y}: %{x:+.1f} pts vs our averages<br>%{customdata[0]} polls in %{customdata[1]} races"
                          "<br>sponsor: %{customdata[2]}<extra></extra>"))
        fig.add_vline(x=0, line_color=INK_3)
        fig.update_xaxes(title='Average distance from our race averages, last 60 days (+ = more Democratic)')
        st.plotly_chart(_layout(fig, height=max(300, 22 * len(l))), width='stretch', key='cmp_pollster_leans')
        st.caption("Pollsters with at least 3 Senate or governor polls in the last 60 days, compared with our averages "
                   "of the same races before any lean correction. A lean here is not an error: the field average can "
                   "itself be off, as it was in 2024.")


# ── Trends ────────────────────────────────────────────────────────────────────
# Approve / disapprove are not parties, so they take neutral categorical slots
# (aqua, orange) rather than blue/red. Validated pair; low contrast on light
# backgrounds, so the lines are direct-labeled.
APPROVE, DISAPPROVE = '#1baf7a', '#eb6834'
OFFICE_COLORS = {'senate': '#2a78d6', 'governor': '#eb6834', 'house': '#1baf7a'}
SENATE_REP_SEATS_2025 = 53      # Republican seats entering 2026 (53–47)

def _last_valid(s, col):
    v = s.dropna(subset=[col])
    return v.iloc[-1] if len(v) else None

def approval_chart(polls, series, start=None, height=360, latest=None):
    """Individual approval polls as faint dots, the smoothed trend as lines. start: show
    only from this date on (the trend is still computed from the full history). latest:
    the published 30-day average {'approve', 'disapprove'} for the end-of-line labels, so
    they match the headline numbers; defaults to the line's own last values."""
    if start is not None:
        polls = polls[polls['end_date'] >= pd.Timestamp(start)]
        series = series[series['date'] >= pd.Timestamp(start)]
    fig = go.Figure()
    for col, name, color in (('approve', 'Approve', APPROVE), ('disapprove', 'Disapprove', DISAPPROVE)):
        fig.add_trace(go.Scatter(
            x=polls['end_date'], y=polls[col], mode='markers', name=f"{name} (single poll)",
            marker=dict(size=8, color=color, opacity=0.3), showlegend=False,
            customdata=np.stack([polls['pollster'], polls['sample_size'].fillna(0)], axis=1),
            hovertemplate=f"%{{customdata[0]}}: %{{y:.0f}}% {name.lower()}<br>%{{x|%b %d, %Y}} · n=%{{customdata[1]:,.0f}}<extra></extra>"))
        last = _last_valid(series, col)
        fig.add_trace(go.Scatter(
            x=series['date'], y=series[col], mode='lines', name=name, connectgaps=False,
            line=dict(color=color, width=2.5, shape='spline', smoothing=0.6),
            hovertemplate=f"{name} trend %{{y:.1f}}%<br>%{{x|%b %d, %Y}}<extra></extra>"))
        if last is not None:
            value = latest[col] if latest is not None else last[col]
            fig.add_annotation(x=last['date'], y=last[col], text=f"{name} {value:.1f}%", showarrow=False,
                               xanchor='left', xshift=6, font=dict(color=INK_2, size=12))
    fig.add_hline(y=50, line_dash='dot', line_color=INK_3, line_width=1)
    fig.update_yaxes(title='% of respondents', range=[20, 80])
    # Fixed range: the end-of-line labels would otherwise stretch the axis months past the data
    fig.update_xaxes(range=[polls['end_date'].min() - pd.Timedelta(days=7), polls['end_date'].max() + pd.Timedelta(days=21)])
    _layout(fig, height=height, legend=dict(orientation='h', y=1.1, x=0), hovermode='closest')
    fig.update_yaxes(ticksuffix='%', title=None)
    fig.update_layout(margin=dict(l=50, r=120))
    return fig

def midterm_history_chart(history, fit, current_approval, ours, chamber):
    """Past midterms: approval vs the president's party's seat change, with 2026 marked."""
    slope, intercept, sd = fit
    col = f'{chamber}_change'
    fig = go.Figure()
    for party, color, label in (('D', DEM, 'Democratic president'), ('R', REP, 'Republican president')):
        h = history[history['party'] == party]
        fig.add_trace(go.Scatter(
            x=h['approval'], y=h[col], mode='markers+text', name=label,
            marker=dict(size=10, color=color, line=dict(width=1, color='white')),
            text=h['year'].astype(str), textposition='top center', textfont=dict(size=10, color=INK_2),
            customdata=np.stack([h['year'], h['president']], axis=1),
            hovertemplate="%{customdata[0]} · %{customdata[1]}<br>approval %{x}%<br>seat change %{y:+d}<extra></extra>"))
    xs = np.array([min(history['approval'].min(), current_approval) - 3, history['approval'].max() + 3])
    fig.add_trace(go.Scatter(x=xs, y=intercept + slope * xs, mode='lines', name='Historical trend',
                             line=dict(color=INK_3, width=2, dash='dash'), hoverinfo='skip'))
    implied = intercept + slope * current_approval
    fig.add_trace(go.Scatter(
        x=[current_approval - 0.6], y=[implied], mode='markers', name='2026: history alone',
        marker=dict(size=12, color=NEUTRAL, symbol='diamond', line=dict(width=1, color=INK_2)),
        error_y=dict(type='data', array=[sd], color=INK_3),
        hovertemplate=f"History alone: {implied:+.0f} seats (±{sd:.0f}, one std. dev.)<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[current_approval + 0.6], y=[ours['median']], mode='markers', name='2026: our forecast',
        marker=dict(size=12, color=REP, symbol='star', line=dict(width=1, color='white')),
        error_y=dict(type='data', symmetric=False, array=[ours['hi'] - ours['median']],
                     arrayminus=[ours['median'] - ours['lo']], color=REP),
        hovertemplate=f"Our forecast: {ours['median']:+d} Republican seats (90%: {ours['lo']:+d} to {ours['hi']:+d})<extra></extra>"))
    fig.add_vline(x=current_approval, line_dash='dot', line_color=INK_3, line_width=1,
                  annotation_text=f"Trump today {current_approval:.0f}%", annotation_font_color=INK_2,
                  annotation_position='bottom right')
    fig.add_hline(y=0, line_color=INK_3, line_width=1)
    fig.update_xaxes(title="President's approval before the midterm (%)")
    fig.update_yaxes(title=f"President's party {'House' if chamber == 'house' else 'Senate'} seat change")
    return _layout(fig, height=440, legend=dict(orientation='h', y=-0.18, x=0))

def polls_per_week_chart(weekly):
    fig = go.Figure()
    for office in weekly.columns:
        fig.add_trace(go.Bar(x=weekly.index, y=weekly[office], name=office.title(), marker_color=OFFICE_COLORS[office],
                             marker_line_width=0,
                             hovertemplate=f"Week ending %{{x|%b %d}}: %{{y}} {office} polls<extra></extra>"))
    fig.update_yaxes(title='Polls released')
    return _layout(fig, height=300, barmode='stack', bargap=0.2,
                   legend=dict(orientation='h', y=1.12, x=0, traceorder='normal'))

def _seat_change(f, chamber, rep_house_2024):
    """Our forecast as the Republicans' seat change: median and 90% range."""
    if chamber == 'house':
        s = f['summary']['house']
        rep = lambda d: (435 - d) - rep_house_2024
    else:
        s = f['summary']['senate']
        ind = round(s.get('independent_seats_expected', 0))
        rep = lambda d: (100 - d - ind) - SENATE_REP_SEATS_2025
    return {'median': rep(s['dem_seats_median']), 'lo': rep(s['dem_seats_90pct'][1]), 'hi': rep(s['dem_seats_90pct'][0])}

RATING_TICKS = {-30: 'Safe R', -15: 'Likely R', -7: 'Lean R', -3: 'Tilt R', 0: 'Toss-up',
                3: 'Tilt D', 7: 'Lean D', 15: 'Likely D', 30: 'Safe D'}
LINE = '#1baf7a'    # single-series lines that aren't parties (economy panel)

def _race_label(office, state):
    return f"{'Sen' if office == 'senate' else 'Gov'} · {_title_case(state)}"

def _favored_party(r):
    return r['dem_side_party'] if r['forecast_margin'] > 0 else 'REPUBLICAN'

def generic_ballot_chart(h):
    fig = go.Figure()
    styles = {'Our average': (INK_2, 3), 'FiftyPlusOne': (APPROVE, 2), 'NBC News': (DISAPPROVE, 2)}
    for name, g in h.groupby('aggregator', sort=False):
        color, width = styles.get(name, (NEUTRAL, 2))
        fig.add_trace(go.Scatter(x=g['date'], y=g['margin'], name=name, mode='lines+markers',
                                 line=dict(color=color, width=width), marker=dict(size=8),
                                 hovertemplate=f"{name}: D%{{y:+.1f}}<br>%{{x|%b %d}}<extra></extra>"))
    fig.update_yaxes(title='Democratic lead (pts)', range=[0, max(12, h['margin'].max() + 2)])
    return _layout(fig, height=300, legend=dict(orientation='h', y=1.15, x=0))

def economy_chart(series):
    from plotly.subplots import make_subplots
    from trends import ECONOMY_SERIES
    have = [s for s in ECONOMY_SERIES if s[0] in series]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[label for _, label, _, _ in have], vertical_spacing=0.2)
    for i, (sid, label, unit, _) in enumerate(have):
        d = series[sid]
        fmt = '$%{y:.2f}' if unit == '$/gal' else ('%{y:.1f}%' if unit == '%' else '%{y:.1f}')
        fig.add_trace(go.Scatter(x=d['date'], y=d['value'], mode='lines', line=dict(color=LINE, width=2), showlegend=False,
                                 hovertemplate=f"{label}: {fmt}<br>%{{x|%b %Y}}<extra></extra>"),
                      row=i // 2 + 1, col=i % 2 + 1)
        last = d.iloc[-1]
        fig.add_annotation(x=last['date'], y=last['value'], showarrow=False, xanchor='left', xshift=5,
                           text=(f"${last['value']:.2f}" if unit == '$/gal' else f"{last['value']:.1f}{'%' if unit == '%' else ''}"),
                           font=dict(color=INK_2, size=11), row=i // 2 + 1, col=i % 2 + 1)
    fig.update_annotations(font_size=12)
    _layout(fig, height=460)
    fig.update_layout(margin=dict(r=50, t=40))
    return fig

def polls_vs_fundamentals_chart(df, office):
    d = df.copy()
    d['party'] = d.apply(_favored_party, axis=1)
    d['label'] = d['state'].map(_title_case)
    labeled = set(d.loc[d['gap'].abs().sort_values(ascending=False).index[:8], 'state'])
    lim = max(d['fund_pred'].abs().max(), d['poll_avg'].abs().max()) + 5
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode='lines', line=dict(color=INK_3, dash='dash', width=1),
                             name='Polls match fundamentals', hoverinfo='skip'))
    for party, color, name in (('DEMOCRAT', DEM, 'Democrat favored'), ('INDEPENDENT', IND, 'Independent favored'),
                               ('REPUBLICAN', REP, 'Republican favored')):
        g = d[d['party'] == party]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=g['fund_pred'], y=g['poll_avg'], mode='markers+text', name=name,
            marker=dict(size=10, color=color, line=dict(width=1, color='white')),
            text=[l if s in labeled else '' for l, s in zip(g['label'], g['state'])],
            textposition='top center', textfont=dict(size=10, color=INK_2),
            customdata=np.stack([g['label'], g['dem_candidate'].fillna('Dem'), g['rep_candidate'].fillna('Rep'),
                                 g['n_polls'], g['gap']], axis=1),
            hovertemplate="%{customdata[0]}: %{customdata[1]} vs. %{customdata[2]}<br>Fundamentals %{x:+.1f} · "
                          "polls %{y:+.1f} (%{customdata[3]} polls)<br>Polls run %{customdata[4]:+.1f} vs. fundamentals"
                          "<extra></extra>"))
    fig.add_hline(y=0, line_color=INK_3, line_width=1)
    fig.add_vline(x=0, line_color=INK_3, line_width=1)
    fig.update_xaxes(title='Fundamentals: state lean + incumbency + national mood (D − R)', range=[-lim, lim])
    fig.update_yaxes(title='Polling average (D − R)', range=[-lim, lim])
    return _layout(fig, height=480, legend=dict(orientation='h', y=-0.18, x=0))

def experts_chart(t):
    from grade_2026 import EXPERT_RATING_TO_PROB
    xs = np.linspace(-30, 30, 121)
    pts = np.array(EXPERT_RATING_TO_PROB)
    implied = np.where(xs >= 0, np.interp(np.abs(xs), pts[:, 0], pts[:, 1]),
                       1 - np.interp(np.abs(xs), pts[:, 0], pts[:, 1])) * 100
    d = t.copy()
    d['label'] = d['state'].map(_title_case)
    d['off'] = d['gap'].abs() >= 5
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=implied, mode='lines', name='What the rating usually implies',
                             line=dict(color=INK_3, dash='dash', width=1), hoverinfo='skip'))
    for off, color, name in ((False, NEUTRAL, 'We roughly agree'), (True, IND, 'We differ by a rating step or more')):
        g = d[d['off'] == off]
        fig.add_trace(go.Scatter(
            x=g['forecasters_avg'], y=g['our_prob'] * 100, mode='markers+text' if off else 'markers', name=name,
            marker=dict(size=10, color=color, line=dict(width=1, color=INK_2)),
            text=g['label'] if off else None, textposition='middle right', textfont=dict(size=10, color=INK_2),
            customdata=np.stack([g['label'], g['ours']], axis=1),
            hovertemplate="%{customdata[0]}<br>Experts average %{x:+.1f} on the rating scale<br>"
                          "Our Dem-side chance %{y:.0f}% (%{customdata[1]})<extra></extra>"))
    fig.update_xaxes(title="Experts' average rating (Cook, Sabato, Inside Elections and others)",
                     tickvals=list(RATING_TICKS), ticktext=list(RATING_TICKS.values()), range=[-33, 36])
    fig.update_yaxes(title='Our Democratic-side win chance (%)', range=[-3, 103])
    return _layout(fig, height=460, legend=dict(orientation='h', y=-0.2, x=0))

def partisan_split_chart(g):
    order = (g[g['group'] == 'Nonpartisan'].set_index('state')['mean']
             .reindex(g['state'].unique()).fillna(g.groupby('state')['mean'].mean()).sort_values().index)
    fig = go.Figure()
    for state in order:
        s = g[g['state'] == state]
        fig.add_trace(go.Scatter(x=[s['mean'].min(), s['mean'].max()], y=[_title_case(state)] * 2, mode='lines',
                                 line=dict(color=GRID, width=6), showlegend=False, hoverinfo='skip'))
    for group, color in (('Democratic-sponsored', DEM), ('Nonpartisan', NEUTRAL), ('Republican-sponsored', REP)):
        s = g[g['group'] == group]
        fig.add_trace(go.Scatter(
            x=s['mean'], y=s['state'].map(_title_case), mode='markers', name=group,
            marker=dict(size=12, color=color, line=dict(width=1, color=INK_2)),
            customdata=s['size'], hovertemplate=f"{group}: D%{{x:+.1f}} (%{{customdata}} polls)<extra></extra>"))
    fig.add_vline(x=0, line_color=INK_3, line_width=1)
    fig.update_xaxes(title='Average margin, last 90 days (D − R)')
    return _layout(fig, height=max(300, 34 * len(order) + 100), legend=dict(orientation='h', y=1.08, x=0))

def movers_chart(m):
    fig = go.Figure()
    labels = [_race_label(o, s) for o, s in zip(m['office'], m['state'])][::-1]
    rows = m.iloc[::-1]
    for (_, r), label in zip(rows.iterrows(), labels):
        color = DEM if r['change'] > 0 else REP
        fig.add_trace(go.Scatter(x=[r['before'] * 100, r['after'] * 100], y=[label, label], mode='lines',
                                 line=dict(color=color, width=3), showlegend=False, hoverinfo='skip'))
        fig.add_annotation(x=r['after'] * 100, y=label, ax=r['before'] * 100, ay=label, axref='x', ayref='y',
                           showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor=color, text='')
    fig.add_trace(go.Scatter(x=rows['before'] * 100, y=labels, mode='markers', name='Before',
                             marker=dict(size=9, color=NEUTRAL, line=dict(width=1, color=INK_2)),
                             hovertemplate="Before: %{x:.0f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=rows['after'] * 100, y=labels, mode='markers+text', name='Now', marker=dict(size=1, opacity=0),
                             text=[f"{a * 100:.0f}% ({c * 100:+.0f})" for a, c in zip(rows['after'], rows['change'])],
                             textposition=['middle right' if c > 0 else 'middle left' for c in rows['change']],
                             textfont=dict(size=11, color=INK_2), hovertemplate="Now: %{x:.0f}%<extra></extra>"))
    fig.add_vline(x=50, line_dash='dot', line_color=INK_3, line_width=1)
    fig.update_xaxes(title='Democratic-side win chance (%)', range=[0, 100])
    return _layout(fig, height=max(280, 34 * len(m) + 80), showlegend=False)

def close_races_chart(c, f):
    from plotly.subplots import make_subplots
    races = c[['office', 'state']].drop_duplicates()
    races = races.assign(cur=[f[o].set_index('state').loc[s, 'forecast_margin'] for o, s in zip(races['office'], races['state'])])
    races = races.sort_values('cur')
    n, cols = len(races), 3
    rows = -(-n // cols)
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.06,
                        horizontal_spacing=0.03, subplot_titles=[_race_label(o, s) for o, s in zip(races['office'], races['state'])])
    lim = max(6, np.nanmax(np.abs(c['poll_avg'])) + 1)
    for i, (_, r) in enumerate(races.iterrows()):
        d = c[(c['office'] == r['office']) & (c['state'] == r['state'])]
        last = d['poll_avg'].dropna()
        color = DEM if len(last) and last.iloc[-1] > 0 else REP
        fig.add_trace(go.Scatter(x=d['date'], y=d['poll_avg'], mode='lines', line=dict(color=color, width=2), showlegend=False,
                                 hovertemplate=f"{_race_label(r['office'], r['state'])}: D%{{y:+.1f}}<br>%{{x|%b %d}}<extra></extra>"),
                      row=i // cols + 1, col=i % cols + 1)
        fig.add_hline(y=0, line_color=INK_3, line_width=1, row=i // cols + 1, col=i % cols + 1)
    fig.update_yaxes(range=[-lim, lim])
    fig.update_annotations(font_size=12)
    _layout(fig, height=170 * rows + 60)
    fig.update_layout(margin=dict(t=40))
    return fig

def seats_votes_chart(t, f):
    s = f['summary']['house']
    fig = go.Figure()
    t = t.copy()
    t['era'] = np.where(t['year'] >= 1996, 'since', 'before')
    for maj, color, name in (('D', DEM, 'Democratic majority'), ('R', REP, 'Republican majority')):
        g = t[(t['DEMOCRAT'] >= 218) == (maj == 'D')]
        fig.add_trace(go.Scatter(
            x=g['margin'], y=g['DEMOCRAT'], mode='markers+text', name=name,
            marker=dict(size=10, color=color, opacity=[1.0 if e == 'since' else 0.45 for e in g['era']],
                        line=dict(width=1, color='white')),
            text=g['year'].astype(str), textposition='top center', textfont=dict(size=10, color=INK_2),
            hovertemplate="%{text}: national vote D%{x:+.1f} → %{y} Democratic seats<extra></extra>"))
    recent = t[t['era'] == 'since']
    slope, intercept = np.polyfit(recent['margin'], recent['DEMOCRAT'], 1)
    xs = np.array([t['margin'].min() - 2, max(t['margin'].max(), f['national_environment']) + 2])
    fig.add_trace(go.Scatter(x=xs, y=intercept + slope * xs, mode='lines', name='Trend since 1996',
                             line=dict(color=INK_3, dash='dash', width=2), hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=[f['national_environment']], y=[s['dem_seats_median']], mode='markers', name='2026 forecast',
        marker=dict(size=16, color=DEM, symbol='star', line=dict(width=1, color='white')),
        error_y=dict(type='data', symmetric=False, array=[s['dem_seats_90pct'][1] - s['dem_seats_median']],
                     arrayminus=[s['dem_seats_median'] - s['dem_seats_90pct'][0]], color=DEM),
        hovertemplate=f"2026 forecast: expected House vote D{f['national_environment']:+.1f} → {s['dem_seats_median']} "
                      f"Democratic seats (90%: {s['dem_seats_90pct'][0]}–{s['dem_seats_90pct'][1]})<extra></extra>"))
    fig.add_hline(y=218, line_dash='dot', line_color=INK_3, line_width=1, annotation_text='218 for a majority',
                  annotation_font_color=INK_2, annotation_position='bottom right')
    fig.add_vline(x=0, line_color=INK_3, line_width=1)
    fig.update_xaxes(title='National House popular vote (D − R, pts)')
    fig.update_yaxes(title='Democratic House seats won')
    return _layout(fig, height=460, legend=dict(orientation='h', y=-0.18, x=0)), slope

def render_trends(f, d):
    """d: the dict built by app.load_trends()."""
    st.title("Trends")
    st.caption("The national mood behind the forecast, how the races are moving, and the history behind it.")
    mood, races, hist, volume = st.tabs(["🌡️ National mood", "🗳️ Races", "📜 History", "📬 Polling volume"])

    series = d['approval_series']
    current = series.dropna(subset=['approve']) if series is not None and len(series) else pd.DataFrame()
    with mood:
        st.subheader("Presidential approval")
        if current.empty:
            st.info("No approval polls loaded. Use Refresh in the sidebar.")
        else:
            cur = current.iloc[-1]
            month_ago = current[current['date'] <= cur['date'] - pd.Timedelta(days=30)]
            prev = month_ago.iloc[-1] if len(month_ago) else None
            delta = lambda col: None if prev is None else f"{cur[col] - prev[col]:+.1f} in 30 days"
            c1, c2, c3 = st.columns(3)
            c1.metric("Approve", f"{cur['approve']:.1f}%", delta('approve'))
            c2.metric("Disapprove", f"{cur['disapprove']:.1f}%", delta('disapprove'), delta_color='inverse')
            c3.metric("Net approval", f"{cur['net']:+.1f}", delta('net'))
            st.plotly_chart(approval_chart(d['approval_polls'], d.get('approval_trend', series),
                                           latest={'approve': cur['approve'], 'disapprove': cur['disapprove']}),
                            width='stretch', key='tr_approval')
            st.caption(f"Dots are single polls; lines are a smoothed trend through them. The numbers above and at the "
                       f"line ends are our 30-day average as of {cur['date']:%b %d, %Y} (recent and larger polls count "
                       "more; no pollster counts more than once). Polls from Wikipedia and the UC Santa Barbara "
                       "Presidency Project; late 2025 has only about one poll a month.")

        st.divider()
        st.subheader("Generic congressional ballot")
        gb = d['generic_ballot']
        if gb is not None and len(gb):
            st.plotly_chart(generic_ballot_chart(gb), width='stretch', key='tr_gb')
            st.caption("Our average (the forecast's input) and the averages it's built from. Wikipedia only shows today's "
                       "averages, so we log them on every daily update; the history starts October 2026 and fills in "
                       "from here. Aggregators with prediction-market ties are left out.")
        else:
            st.info("No generic-ballot history yet. It's logged on each daily update.")

        st.divider()
        st.subheader("The economy")
        if d['economy']:
            st.plotly_chart(economy_chart(d['economy']), width='stretch', key='tr_economy')
            st.caption("Federal Reserve Economic Data (FRED), since 2021. Inflation is the 12-month change in the "
                       "consumer price index; gas is the weekly national average for regular. Context only: the forecast "
                       "reads the economy through polls and the national environment, not these numbers directly.")
        else:
            st.info("Economic data unavailable (FRED didn't respond, or no API key is set).")

    with races:
        office = st.radio("Office", ['senate', 'governor'], horizontal=True, key='tr_race_office',
                          format_func=lambda o: o.title())
        st.subheader("Polls vs. fundamentals")
        pf = d['polls_vs_fund'][office]
        if len(pf):
            st.plotly_chart(polls_vs_fundamentals_chart(pf, office), width='stretch', key=f'tr_pvf_{office}')
            above = pf.loc[pf['gap'].idxmax()]
            st.caption("Each dot is a polled race. Fundamentals is what the state's lean, incumbency and the national "
                       "mood alone predict; the polling average is what voters are telling pollsters. Dots above the "
                       "dashed line are Democratic-side candidates running ahead of their state's usual result; below it, "
                       f"Republicans are. The biggest gap: {_title_case(above['state'])}, where polls run "
                       f"{above['gap']:+.1f} points from fundamentals. Labels mark the eight largest gaps. Expect lopsided "
                       "races to sit closer to zero than the dashed line: polls report undecided voters, who mostly "
                       "break toward the state's usual party, so a poll showing R+8 in a deep-red state isn't a Democratic "
                       "surge. That's one reason the forecast blends polls with fundamentals instead of using polls alone.")
        st.divider()
        st.subheader("Our forecast vs. the experts")
        ex = d['vs_experts'][office]
        if len(ex):
            st.plotly_chart(experts_chart(ex), width='stretch', key=f'tr_exp_{office}')
            st.caption("Each dot is a race: across, the average rating from professional forecasters (Cook, Sabato, "
                       "Inside Elections, The Economist, FiftyPlusOne, Split Ticket, Fox); up, our Democratic-side win "
                       "chance. The dashed curve is the chance each rating usually implies. Labeled races are where we "
                       "differ by a rating step or more. Most of the gaps are lightly polled races where our model "
                       "stays cautious and the experts call them safe.")
        st.divider()
        st.subheader("Partisan vs. nonpartisan polls")
        ps = d['partisan_split'][office]
        if len(ps):
            st.plotly_chart(partisan_split_chart(ps), width='stretch', key=f'tr_partisan_{office}')
            st.caption("Average margin from each kind of poll in races within 10 points, last 90 days, before any "
                       "correction. Sponsored polls usually lean toward their sponsor; our averages count them at reduced "
                       "weight and correct each pollster's lean. Races without any sponsored polls aren't shown.")
        else:
            st.info("No sponsored polls in competitive races in the last 90 days.")

        st.divider()
        st.subheader("Biggest movers this week")
        m = d['movers']
        if m is not None and len(m):
            st.plotly_chart(movers_chart(m), width='stretch', key='tr_movers')
            st.caption(f"Change in the Democratic-side win chance between the forecasts of {m['start'].iloc[0]:%b %d} and "
                       f"{m['end'].iloc[0]:%b %d}, Senate and governor races together. Moves come from new polls and "
                       "from model updates; the October 6–8 model changes account for much of this first week.")
        else:
            st.info("Movers appear once there are at least two days of published forecasts.")

        st.divider()
        st.subheader("Close races: polling average over time")
        cr = d['close_races']
        if cr is not None and len(cr):
            st.plotly_chart(close_races_chart(cr, f), width='stretch', key='tr_close')
            st.caption("Every race our forecast has within 5 points, Senate and governor, sorted from most Republican to "
                       "most Democratic. Each line is the polling average (house-effect corrected, before the forecast's "
                       "lean correction) as it stood on each day since July. Line color shows today's leader.")

    with hist:
        st.subheader("Approval and midterm losses, 1946–2022")
        if current.empty:
            st.info("Needs the current approval average, which isn't loaded.")
        else:
            chamber = st.radio("Chamber", ['House', 'Senate'], horizontal=True, key='tr_chamber').lower()
            approve_now = float(current['approve'].iloc[-1])
            ours = _seat_change(f, chamber, d['rep_house_2024'])
            slope, intercept, sd = d['fits'][chamber]
            implied = intercept + slope * approve_now
            st.plotly_chart(midterm_history_chart(d['history'], d['fits'][chamber], approve_now, ours, chamber),
                            width='stretch', key=f'tr_history_{chamber}')
            st.caption(
                f"Each dot is a midterm: Gallup approval at its last reading before the election, and the seats the "
                f"president's party gained or lost (UCSB American Presidency Project). The dashed line is the straight-line "
                f"fit. At today's {approve_now:.0f}%, history alone points to about {implied:+.0f} {chamber.title()} seats "
                f"for Republicans (give or take {sd:.0f}); our forecast says {ours['median']:+d}. "
                + ("The gap is expected. Far fewer House seats are competitive now than before 2000, so the same national "
                   "swing moves fewer seats, and our model works race by race from current polls rather than from this one "
                   "national relationship. Today's approval is also lower than at any midterm on the chart except 1946, "
                   "so the line is stretched past most of its data."
                   if chamber == 'house' else
                   "Senate losses track approval only loosely: a third of seats are up, and which ones matters more than "
                   "the national mood."))

        st.divider()
        st.subheader("House seats vs. the national vote, 1976–2024")
        sv = d['seats_votes']
        if sv is not None and len(sv):
            fig, slope = seats_votes_chart(sv, f)
            st.plotly_chart(fig, width='stretch', key='tr_seats_votes')
            st.caption(f"Each dot is a House election: the national popular-vote margin and the seats Democrats won. Since "
                       f"1996 (solid dots), each point of national margin has been worth about {slope:.0f} seats; before "
                       "the 1994 realignment (faded) Democrats won far more seats per vote. The star is our 2026 forecast "
                       "at its expected national vote, with the 90% range. Seat counts come from MIT Election Lab district "
                       "results and can differ from official totals by a seat or two in some years.")

    with volume:
        st.subheader("Polls coming in")
        if d['weekly_polls'] is not None and len(d['weekly_polls']):
            st.plotly_chart(polls_per_week_chart(d['weekly_polls']), width='stretch', key='tr_weekly')
            st.caption("Senate, governor, and House district polls by the week they finished fielding, since January "
                       "2026. The latest week is still in progress.")


# ── Post builder ──────────────────────────────────────────────────────────────
# House lists start with only these districts selected (|margin| under this); 400+ lines is no post
HOUSE_POST_MARGIN = 3.0

def render_posts(f, annotated, model_version, house_annotated=None):
    import posts as P
    st.title("Post builder")
    kind = st.segmented_control("Post", ['Race list', 'What moved', 'Results'], default='Race list', key='post_kind')
    if kind == 'What moved':
        render_what_moved(annotated)
        return
    if kind == 'Results':
        import dashboard_results
        dashboard_results.render_result_posts(model_version)
        return
    st.caption("Race lists for social media: a colored dot for whoever leads, the lead, and (Flip) when that party "
               "doesn't hold the seat now. Copy the text, or download the image if emoji won't survive the platform.")
    as_of = pd.Timestamp(f['as_of'])
    c1, c2 = st.columns(2)
    office = c1.radio("Office", ['senate', 'governor', 'house'], format_func=P.OFFICE_LABEL.get, horizontal=True,
                      key='post_office')
    if office == 'house':
        annotated = house_annotated if house_annotated is not None else annotated.iloc[:0]
    source = c2.radio("Numbers from", ['pollster', 'averages', 'forecast'], horizontal=True, key='post_source',
                      format_func={'pollster': 'One pollster', 'averages': 'Our polling averages',
                                   'forecast': 'Our forecast'}.get)
    label = P.OFFICE_LABEL[office]
    if source == 'pollster':
        choices = P.pollster_choices(annotated, office)
        if not choices:
            st.info("No pollster has polled two or more of these races in the last 60 days.")
            return
        c1, c2 = st.columns([2, 1])
        pollster = c1.selectbox("Pollster", choices, key=f'post_pollster_{office}',
                                help="Pollsters with recent polls in at least two races, busiest first.")
        days = c2.slider("Polls from the last N days", 14, 120, 60, step=7, key='post_days')
        rows, (d0, d1) = P.rows_from_pollster(annotated, f, office, pollster, days)
        title = f"{label} Polling — {pollster}"
        footer = (f"Latest {pollster} poll in each race, fielded {d0:%b} {d0.day} – {d1:%b} {d1.day} · "
                  f"via {MODEL_NAME}") if rows else ''
        default = [r['state'] for r in rows]
        extra_label = None
    elif source == 'averages':
        rows = P.rows_from_averages(f, office)
        title = f"{label} Polling Averages"
        footer = f"{MODEL_NAME} polling averages as of {as_of:%b} {as_of.day}, {as_of.year}"
        default = [r['state'] for r in rows]
        extra_label = "Show number of polls"
    else:
        rows = P.rows_from_forecast(f, office)
        title = f"{label} Forecast"
        footer = f"{MODEL_NAME} as of {as_of:%b} {as_of.day}, {as_of.year} · model {model_version} · a forecast, not a call"
        default = [r['state'] for r in rows]
        extra_label = "Show win chance"

    race_help = "Every race is included; remove any you don't want in the post."
    if office == 'house' and source != 'pollster':
        default = [r['state'] for r in rows if abs(r['dem_margin']) < HOUSE_POST_MARGIN]
        race_help = (f"Starts with districts within {HOUSE_POST_MARGIN:g} points. Add any other contested "
                     "district from the list.")
    races = st.multiselect("Races", [r['state'] for r in rows], default=default, key=f'post_races_{office}_{source}',
                           help=race_help)
    rows = [r for r in rows if r['state'] in races]
    c1, c2, c3 = st.columns(3)
    show_flip = c1.checkbox("Mark flips", True, key='post_flip')
    show_extra = c2.checkbox(extra_label, True, key='post_extra') if extra_label else False
    dark = c3.checkbox("Dark image", True, key='post_dark')
    title = st.text_input("Title", title, key=f'post_title_{office}_{source}')
    footer = st.text_input("Footer", footer, key=f'post_footer_{office}_{source}',
                           help="Keep the date in it, so an old post can't pass for a current one.")
    if not rows:
        st.info("No races selected.")
        return

    l, r = st.columns(2)
    with l:
        st.markdown("**Text** (copy button at top right)")
        st.code(P.post_text(title, rows, footer or None, show_flip, show_extra), language=None)
    with r:
        png = P.post_png(title, rows, footer or None, show_flip, show_extra, dark)
        st.markdown("**Image**")
        st.image(png, width='stretch')
        st.download_button("Download image (PNG)", png, key='post_png', mime='image/png',
                           file_name=f"{office}_{source}_{as_of:%Y-%m-%d}.png")
    st.caption("Flip means the leader's party differs from the party that holds the seat now. Pollster numbers are "
               "the pollster's own, not our averages. Averages and the forecast show one decimal so close races "
               "don't round to a tie.")


def render_what_moved(annotated):
    """A weekly 'what moved' post from two published snapshots, with the reason for each move."""
    import weekly as W
    dates = W.snapshot_dates()
    if len(dates) < 2:
        st.info("Needs at least two published forecasts.")
        return
    st.caption("Compares two published daily forecasts and explains each race's move: new polls, the national "
               "environment, or a model change logged in the changelog. Markdown pastes straight into Substack.")
    d0_default, d1_default = W.default_pair()
    c1, c2 = st.columns(2)
    d1 = c2.selectbox("To", dates[::-1], index=dates[::-1].index(d1_default), key='wm_to')
    earlier = [d for d in dates if d < d1]
    if not earlier:
        st.info("Pick a later end date.")
        return
    d0 = c1.selectbox("From", earlier[::-1], index=earlier[::-1].index(d0_default) if d0_default in earlier else 0,
                      key='wm_from')
    w = W.what_moved(d0, d1, annotated)
    t = w['topline']
    m1, m2, m3 = st.columns(3)
    m1.metric("Senate control (D)", f"{t['senate'][1]:.0%}", f"{(t['senate'][1] - t['senate'][0]) * 100:+.0f} pts")
    m2.metric("House control (D)", f"{t['house'][1]:.0%}", f"{(t['house'][1] - t['house'][0]) * 100:+.0f} pts")
    m3.metric("Governorships (D, median)", t['governors'][1], f"{t['governors'][1] - t['governors'][0]:+d}")
    if w['method_changes']:
        st.warning("The model changed between these dates (" + ", ".join(v for v, _ in w['method_changes']) +
                   "). Moves marked \"Mostly the model update\" come from that, not from new information.")
    l, r = st.columns(2)
    with l:
        st.markdown("**Markdown** (copy button at top right)")
        st.code(W.post_markdown(w), language='markdown')
    with r:
        st.markdown("**Image**")
        if len(w['movers']):
            png = W.movers_png(w, dark=st.checkbox("Dark image", True, key='wm_dark'))
            st.image(png, width='stretch')
            st.download_button("Download image (PNG)", png, key='wm_png', mime='image/png',
                               file_name=f"what_moved_{d1}.png")
        else:
            st.info(f"No Senate or governor race moved {W.MIN_MOVE:.0%} or more.")
    with st.expander("Every race, with the breakdown"):
        a = w['all'].copy()
        a['race'] = [f"{W.OFFICE_LABEL[o]} · {_title_case(s_)}" for o, s_ in zip(a['office'], a['state'])]
        a = a.loc[a['change'].abs().sort_values(ascending=False).index]
        st.dataframe(a[['race', 'p_before', 'p_after', 'margin_before', 'margin_after', 'from_polls',
                        'from_environment', 'from_method', 'reason']], hide_index=True, width='stretch',
                     column_config={
                         'race': 'Race', 'p_before': st.column_config.NumberColumn('Before', format='percent'),
                         'p_after': st.column_config.NumberColumn('After', format='percent'),
                         'margin_before': st.column_config.NumberColumn('Margin before', format='%+.1f'),
                         'margin_after': st.column_config.NumberColumn('Margin after', format='%+.1f'),
                         'from_polls': st.column_config.NumberColumn('From polls', format='%+.1f'),
                         'from_environment': st.column_config.NumberColumn('From environment', format='%+.1f'),
                         'from_method': st.column_config.NumberColumn('From model change', format='%+.1f'),
                         'reason': 'Why'})
        st.caption("Margins are Democratic-side minus Republican, in points. The three parts add up to the change "
                   "in margin. From model change also picks up rounding, so it's rarely exactly zero.")


# ── Path to control: snake chart, tipping point, paths (simulations.py) ───────
def _prob_color(p):
    from plotly.colors import sample_colorscale
    return sample_colorscale(PROB_SCALE, [float(p)])[0]

def senate_needs(f):
    """Democratic wins needed among this year's races, and the net gain that means."""
    s = f['summary']['senate']
    sen = f['senate']
    need = 51 - s['holdovers']['DEMOCRAT']
    held_now = s['holdovers']['DEMOCRAT'] + int((sen['incumbent_party'] == 'DEMOCRAT').sum())
    return need, 51 - held_now

def snake_chart(f):
    """Every Senate race in a row, most Democratic on the left, shaded by win chance. The
    line falls where Democrats reach 51."""
    sen = f['senate'].sort_values('dem_side_win_prob', ascending=False).reset_index(drop=True)
    need, _ = senate_needs(f)
    # Independents' wins don't count toward 51, so the line sits after the need-th Democratic race
    dem_rows = sen.index[sen['dem_side_party'] == 'DEMOCRAT']
    line_at = int(dem_rows[need - 1]) + 1 if 0 < need <= len(dem_rows) else None
    tp = sen['tipping_point_prob'] if 'tipping_point_prob' in sen else pd.Series(0.0, index=sen.index)
    fig = go.Figure()
    for i, r in sen.iterrows():
        p = r['dem_side_win_prob']
        ind = r['dem_side_party'] == 'INDEPENDENT'
        fig.add_trace(go.Bar(
            x=[1], y=[''], orientation='h',
            marker=dict(color=_prob_color(p), line=dict(color='white', width=2),
                        pattern=dict(shape='/' if ind else '', fgcolor='rgba(255,255,255,0.45)', size=8, solidity=0.15)),
            text=STATE_ABBREV[r['state']] + ('*' if ind else ''), textposition='inside', insidetextanchor='middle',
            textfont=dict(color='#0b0b0b' if 0.25 < p < 0.75 else 'white', size=12),
            hovertemplate=(f"<b>{_title_case(r['state'])}</b><br>{'Independent' if ind else 'Democrat'} win {p:.0%}"
                           f"<br>Forecast {fmt_margin(r['forecast_margin'])}<br>Tipping point in {tp.iloc[i]:.0%} of "
                           "simulations<extra></extra>"),
            showlegend=False))
    if line_at:
        fig.add_vline(x=line_at, line_color=INK_2, line_width=2, line_dash='dot')
        fig.add_annotation(x=line_at, y=1, yref='paper', yanchor='bottom', showarrow=False,
                           font=dict(color=INK_2, size=11), text="51st Democratic seat")
    fig.update_xaxes(range=[0, len(sen)], visible=False)
    fig.update_yaxes(visible=False)
    _layout(fig, height=110, barmode='stack', bargap=0)
    fig.update_layout(margin=dict(l=0, r=0, t=26, b=0))
    return fig

def tipping_point_chart(df, label_fn, n=10):
    t = df.nlargest(n, 'tipping_point_prob').iloc[::-1]
    pcol = 'dem_side_win_prob' if 'dem_side_win_prob' in t else 'dem_win_prob'
    fig = go.Figure(go.Bar(
        x=t['tipping_point_prob'] * 100, y=[label_fn(r) for _, r in t.iterrows()], orientation='h',
        marker=dict(color=[_prob_color(p) for p in t[pcol]], line=dict(color='white', width=1)),
        text=[f"{v:.0%}" for v in t['tipping_point_prob']], textposition='outside', cliponaxis=False,
        textfont=dict(color=INK_2),
        customdata=t[pcol] * 100,
        hovertemplate="%{y}: deciding seat in %{x:.1f}% of simulations<br>Dem-side win %{customdata:.0f}%<extra></extra>"))
    fig.update_xaxes(title="Chance this race decides control (%)", range=[0, t['tipping_point_prob'].max() * 125])
    return _layout(fig, height=max(240, 28 * len(t)), showlegend=False)

def _path_text(c):
    parts = []
    if c['gains']:
        parts.append("flip " + ', '.join(c['gains']))
    held_won = [x for x in c['wins'] if x not in c['gains']]
    if held_won:
        parts.append("hold " + ', '.join(held_won))
    if c['losses']:
        parts.append("lose " + ', '.join(c['losses']))
    return '; '.join(parts) or "lose every toss-up and still win"

def render_senate_paths(f):
    s = f['summary']['senate']
    sen = f['senate']
    if 'tipping_point_prob' not in sen or 'paths' not in s:
        return
    need, net = senate_needs(f)
    st.subheader("🐍 The path to 51")
    st.caption(f"Every Senate race on the ballot, ordered from most to least Democratic and shaded by win chance. "
               f"Democrats have {s['holdovers']['DEMOCRAT']} seats not up this year, so they need {need} of these "
               f"{len(sen)} races: a net gain of {net} seat{'s' if net != 1 else ''}. Striped races (*) have an "
               "independent instead of a Democrat; their wins don't count toward 51.")
    st.plotly_chart(snake_chart(f), width='stretch', key='senate_snake', config={'displayModeBar': False})
    top = sen.nlargest(1, 'tipping_point_prob').iloc[0]
    l, r = st.columns(2)
    with l:
        st.markdown(f"**Most likely tipping point: {_title_case(top['state'])}** "
                    f"({top['tipping_point_prob']:.0%} of simulations)")
        st.plotly_chart(tipping_point_chart(sen, lambda r: _title_case(r['state'])), width='stretch', key='senate_tp')
        st.caption("In each simulation, line the races up from most to least Democratic: the tipping point is the one "
                   "that delivers the 51st seat. Whoever wins it wins the Senate. When the chances are spread across "
                   "many states, no single race decides it.")
    with r:
        for party, name, color in (('DEMOCRAT', 'Democrats', DEM), ('REPUBLICAN', 'Republicans', REP)):
            p = s['paths'][party]
            st.markdown(f"<span style='color:{color};font-weight:700'>How {name} win the Senate</span> "
                        f"({p['control']:.0%} of simulations)", unsafe_allow_html=True)
            if not p['combos']:
                st.caption("No winning simulations.")
                continue
            st.markdown('\n'.join(f"- **{c['share_of_wins']:.0%}** of their wins: {_path_text(c)}"
                                  for c in p['combos'][:4]))
        st.caption(f"The most common outcomes of the {len(s['paths']['DEMOCRAT'].get('tossups', []))} races neither "
                   "party wins in more than 80% of simulations. There are many ways to a majority, so even the most "
                   "common one is a small share.")
    with st.expander("Must-win races: how often each side wins a race when it takes the Senate"):
        dd = {x['race']: x for x in s['paths']['DEMOCRAT']['races']}
        rr = {x['race']: x for x in s['paths']['REPUBLICAN']['races']}
        rows = [{'Race': k, 'Dem side wins (all simulations)': dd[k]['p'] * 100,
                 'Dem side wins, when Democrats take the Senate': dd[k]['p_when_control'] * 100,
                 'Republican wins, when Republicans take the Senate': rr[k]['p_when_control'] * 100 if k in rr else np.nan}
                for k in dd]
        t = pd.DataFrame(rows).sort_values('Dem side wins (all simulations)', ascending=False)
        bar = lambda c: st.column_config.ProgressColumn(color=c, format='%.0f%%', min_value=0, max_value=100)
        st.dataframe(t, hide_index=True, width='stretch', column_config={
            'Dem side wins (all simulations)': bar(DEM), 'Dem side wins, when Democrats take the Senate': bar(DEM),
            'Republican wins, when Republicans take the Senate': bar(REP)})
        st.caption("A race near 100% in a party's winning simulations is close to a must-win for that party.")

def render_house_tipping(h):
    if 'tipping_point_prob' not in h:
        return
    st.subheader("Tipping-point districts")
    lab = lambda r: f"{STATE_ABBREV[r['state']]}-{'AL' if r['district'] == 0 else r['district']}"
    st.plotly_chart(tipping_point_chart(h, lab, n=12), width='stretch', key='house_tp')
    st.caption("Sort the 435 districts in each simulation from most to least Democratic: the 218th is the tipping "
               "point. With so many seats no district decides it often; these sit at the majority line most.")


# ── Race pages: average and win chance over time ─────────────────────────────
def race_average_series(polls, as_of, start='2026-03-01'):
    """The race's polling average (house-effect corrected) every other day."""
    from polls_average import race_average
    if polls.empty:
        return pd.DataFrame(columns=['date', 'poll_avg'])
    start = max(pd.Timestamp(start), polls['end_date'].min())
    days = pd.date_range(start, pd.Timestamp(as_of), freq='2D')
    rows = [(d, (race_average(polls, d) or {}).get('poll_avg', np.nan)) for d in days]
    return pd.DataFrame(rows, columns=['date', 'poll_avg'])

def win_history_chart(hist, today_p, as_of, label):
    h = hist[['date', 'dem_side_win_prob']].sort_values('date')
    if pd.Timestamp(as_of) not in set(h['date']):
        h = pd.concat([h, pd.DataFrame({'date': [pd.Timestamp(as_of)], 'dem_side_win_prob': [today_p]})])
    color = IND if label == 'Independent' else DEM
    fig = go.Figure(go.Scatter(
        x=h['date'], y=h['dem_side_win_prob'] * 100, mode='lines+markers', line=dict(color=color, width=2),
        marker=dict(size=8), hovertemplate="%{x|%b %d}: %{y:.0f}%<extra></extra>"))
    fig.add_hline(y=50, line_dash='dot', line_color=INK_3, line_width=1)
    fig.update_yaxes(title=f"{label} win chance (%)", range=[0, 100])
    return _layout(fig, height=260, showlegend=False)


# ── What if the polls are off? ───────────────────────────────────────────────
def miss_label(m):
    return "0 (as corrected)" if m == 0 else (f"R+{m:g}: polls overstate Democrats" if m > 0
                                             else f"D+{-m:g}: polls understate Democrats")

def sensitivity_chart(sweep, current):
    fig = go.Figure()
    for col, name, color in (('senate_dem', 'Senate', DEM), ('house_dem', 'House', IND)):
        fig.add_trace(go.Scatter(
            x=sweep['miss'], y=sweep[col] * 100, name=name, mode='lines+markers+text',
            line=dict(color=color, width=2), marker=dict(size=8),
            text=[''] * (len(sweep) - 1) + [name], textposition='middle right', textfont=dict(color=INK_2),
            cliponaxis=False,
            hovertemplate=f"{name}: %{{y:.0f}}% Democratic control<extra></extra>"))
    fig.add_vline(x=0, line_color=INK_3, line_width=1)
    if current:
        fig.add_vline(x=current, line_color=INK_2, line_width=2, line_dash='dot',
                      annotation_text="Your scenario", annotation_font_color=INK_2)
    fig.add_hline(y=50, line_dash='dot', line_color=INK_3, line_width=1)
    fig.update_xaxes(title="Extra polling miss (points)  ← polls understate Democrats · overstate Democrats →",
                     tickvals=sweep['miss'],
                     ticktext=[f"D+{-m:g}" if m < 0 else (f"R+{m:g}" if m > 0 else '0') for m in sweep['miss']])
    fig.update_yaxes(title="Democratic control chance (%)", range=[0, 100])
    _layout(fig, height=360, legend=dict(orientation='h', y=1.1, x=0), hovermode='x unified')
    fig.update_layout(margin=dict(r=70))
    return fig

def render_scenarios(base, f, sweep, miss):
    st.title("What if the polls are wrong?")
    st.caption("The forecast already allows for a polling miss: each of the 20,000 simulations draws one shared error "
               "that hits every race at once. This page asks a sharper question: what if we knew the polls were off "
               "by a set amount, beyond the corrections the model already makes? Use the **polling miss** slider in "
               "the sidebar. Every tab then shows that scenario.")
    st.plotly_chart(sensitivity_chart(sweep, miss), width='stretch', key='scen_sweep')
    st.caption("For scale: in 2016 and 2020 state polls overstated Democrats by roughly 3–6 points (R+3 to R+6 here); "
               "in 2022 they were close; in 2024 Senate polls overstated Democrats by about 2.5. Each point on the "
               "chart uses 4,000 simulations, so it can differ from the main forecast by a point or two.")
    c = st.columns(4)
    sb, s = base['summary'], f['summary']
    pts = lambda a, b: f"{(a - b) * 100:+.0f} pts" if miss else None
    c[0].metric("Senate: Democratic control", f"{s['senate']['dem_control_prob']:.0%}",
                delta=pts(s['senate']['dem_control_prob'], sb['senate']['dem_control_prob']))
    c[1].metric("Senate seats (D median)", s['senate']['dem_seats_median'],
                delta=(s['senate']['dem_seats_median'] - sb['senate']['dem_seats_median']) if miss else None)
    c[2].metric("House: Democratic control", f"{s['house']['dem_control_prob']:.0%}",
                delta=pts(s['house']['dem_control_prob'], sb['house']['dem_control_prob']))
    c[3].metric("House seats (D median)", s['house']['dem_seats_median'],
                delta=(s['house']['dem_seats_median'] - sb['house']['dem_seats_median']) if miss else None)
    if not miss:
        st.info("Move the **polling miss** slider in the sidebar to see which races change hands.")
        return
    st.subheader("Races that change favorite in this scenario")
    rows = []
    for office, label in (('senate', 'Senate'), ('governor', 'Governor')):
        a, b = base[office].set_index('state'), f[office].set_index('state')
        for state in a.index:
            pa, pb = a.loc[state, 'dem_side_win_prob'], b.loc[state, 'dem_side_win_prob']
            if (pa >= 0.5) != (pb >= 0.5):
                rows.append({'Race': f"{label} · {_title_case(state)}", 'Forecast': f"{pa:.0%}",
                             'This scenario': f"{pb:.0%}",
                             'Margin': f"{fmt_margin(a.loc[state, 'forecast_margin'])} → "
                                       f"{fmt_margin(b.loc[state, 'forecast_margin'])}"})
    ha, hb = base['house'], f['house']
    for i in ha.index[(ha['dem_win_prob'] >= 0.5) != (hb.loc[ha.index, 'dem_win_prob'] >= 0.5)]:
        r = ha.loc[i]
        rows.append({'Race': f"House · {STATE_ABBREV[r['state']]}-{'AL' if r['district'] == 0 else r['district']}",
                     'Forecast': f"{r['dem_win_prob']:.0%}", 'This scenario': f"{hb.loc[i, 'dem_win_prob']:.0%}",
                     'Margin': f"{fmt_margin(r['predicted_margin'])} → {fmt_margin(hb.loc[i, 'predicted_margin'])}"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch', height=min(35 * (len(rows) + 1) + 3, 520))
        st.caption(f"{len(rows)} races. Percentages are the Democratic side's win chance.")
    else:
        st.caption("No race changes favorite.")


# ── 2028: saved predictions ───────────────────────────────────────────────────
def ev_history_chart(history):
    """One bar per saved 2028 prediction, newest on top: Democratic and Republican
    electoral votes out of 538, with the 270 line. Hover shows the inputs behind it."""
    h = history.copy()
    h['saved_at'] = pd.to_datetime(h['saved_at'])
    h = h.sort_values('saved_at')
    labels = [f"{t:%b} {t.day}, {t:%Y} {t:%H:%M}" if (h['saved_at'].dt.date == t.date()).sum() > 1
              else f"{t:%b} {t.day}, {t:%Y}" for t in h['saved_at']]
    inputs = [f"GDP {r['gdp_growth']:.1f}% · unemployment {r['unemployment']:.1f}% · approval {r['approval']:.0f}%"
              f"<br>Incumbent party: {str(r['incumbent']).title()}" for _, r in h.iterrows()]
    fig = go.Figure()
    for col, name, color, pos in (('dem_ev', 'Democrat', DEM, 'inside'), ('rep_ev', 'Republican', REP, 'inside')):
        fig.add_trace(go.Bar(
            y=labels, x=h[col], orientation='h', name=name, marker=dict(color=color, line=dict(color='white', width=2)),
            text=[f"{name[0]} {v}" for v in h[col]], textposition=pos, insidetextanchor='middle',
            textfont=dict(color='white', size=13), customdata=inputs,
            hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{x}} electoral votes<br>%{{customdata}}<extra></extra>"))
    fig.add_vline(x=270, line_color=INK_2, line_width=2, line_dash='dot')
    fig.add_annotation(x=270, y=1, yref='paper', yanchor='bottom', showarrow=False, text="270 to win",
                       font=dict(color=INK_2, size=12))
    fig.update_xaxes(range=[0, 538], tickvals=[0, 100, 200, 270, 338, 438, 538], title=None)
    fig.update_yaxes(title=None)
    _layout(fig, height=max(180, 46 * len(h) + 90), barmode='stack', bargap=0.35,
            legend=dict(orientation='h', y=1.12, x=0, traceorder='normal'))
    fig.update_layout(margin=dict(t=50, l=130))
    return fig
