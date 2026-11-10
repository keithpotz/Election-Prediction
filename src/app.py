import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
from datetime import datetime, date
from electoral_votes import calculate_ec_result, calculate_predicted_ec
from data_sources import fetch_all, fetch_approval_rating, fetch_generic_ballot
from forecast_2026 import load_inputs, build_forecast
from publish_forecast import MODEL_VERSION
import dashboard_2026 as dash
from dashboard_2026 import STATE_ABBREV
from fundamentals import fundamentals_adjustment
from config import FRED_API_KEY

from brand import MODEL_NAME, PUBLICATION, TAGLINE
st.set_page_config(page_title=MODEL_NAME, layout="wide", page_icon="🗳️")
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(BASE_DIR, "data/prediction_history.csv")

PARTY_COLORS = {"DEMOCRAT": "#1f77b4", "REPUBLICAN": "#d62728", "OTHER": "#7f7f7f"}
MAX_TOTAL_ADJUSTMENT = 20.0  # combined cap on fundamentals_adjustment() + the approval adjustment

def _or_default(value, default):
    """Like `value or default`, but a real fetched 0/0.0 counts as present, not missing."""
    return value if value is not None else default

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_historical():
    return pd.read_csv(os.path.join(BASE_DIR, "data/polling_data/cleaned_polls.csv"))

@st.cache_data
def load_featured():
    return pd.read_csv(os.path.join(BASE_DIR, "data/polling_data/featured_data.csv"))

# ── 2026 forecast loaders ─────────────────────────────────────────────────────
# Models, scraped race data, and poll averages are built once (a few seconds);
# rerunning the forecast for a different generic-ballot value then takes ~0.3s.
@st.cache_resource(show_spinner="Loading 2026 forecast models and polls...")
def forecast_inputs():
    return load_inputs()

# Each cached forecast keeps its simulations (~15 MB) for election-night mode
@st.cache_resource(max_entries=16, show_spinner="Simulating every race...")
def run_forecast(generic_ballot_poll, extra_poll_miss=0.0):
    return build_forecast(generic_ballot_poll, inputs=forecast_inputs(), extra_poll_miss=extra_poll_miss)

POLL_MISS_SWEEP = list(range(-5, 6))

@st.cache_data(max_entries=8, show_spinner="Running the forecast under 11 polling misses (about 15 seconds)...")
def poll_miss_sweep(generic_ballot_poll):
    rows = []
    for m in POLL_MISS_SWEEP:
        f = build_forecast(generic_ballot_poll, inputs=forecast_inputs(), extra_poll_miss=float(m),
                           n_sims=4000, keep_sims=False)
        s = f['summary']
        rows.append({'miss': m, 'senate_dem': s['senate']['dem_control_prob'],
                     'senate_rep': s['senate']['rep_control_prob'], 'house_dem': s['house']['dem_control_prob'],
                     'senate_seats': s['senate']['dem_seats_median'], 'house_seats': s['house']['dem_seats_median']})
    return pd.DataFrame(rows)

@st.cache_data(ttl=600)
def load_race_history():
    import trends as T
    return T.race_history()

@st.cache_data(ttl=600)
def load_topline_log():
    path = os.path.join(BASE_DIR, '..', 'forecasts', '2026', 'topline_log.csv')
    return pd.read_csv(path) if os.path.exists(path) else None

def refresh_2026_data():
    """Re-scrape polls, race status, and House districts, then drop cached forecasts."""
    from polls_2026 import refresh_polls
    from race_status_2026 import load_or_scrape_race_status
    from house_districts import load_house_districts
    from national_polls import refresh as refresh_national
    refresh_national()
    for office in ('senate', 'governor'):
        load_or_scrape_race_status(refresh=True, office=office)
    load_house_districts(refresh=True)
    refresh_polls()
    from poll_checks import run_checks
    run_checks()
    forecast_inputs.clear()
    run_forecast.clear()
    load_trends_national.clear()
    load_close_race_averages.clear()

@st.cache_data(ttl=86400)
def load_counties_geojson():
    try:
        r = requests.get(
            "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
            timeout=15
        )
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=3600)
def load_approval_trend():
    """Individual approval polls from Wikipedia (columns renamed for the 2028 tab's chart)."""
    try:
        from national_polls import load, load_approval_polls
        load()
        df = load_approval_polls().rename(columns={
            'end_date': 'enddate', 'approve': 'adjusted_approve', 'disapprove': 'adjusted_disapprove'})
        return df.sort_values('enddate')
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="Building approval trend...")
def load_trends_national():
    """Approval polls and daily average, generic-ballot history, FRED economy series,
    midterm history fits, 2024 House baseline, House seats vs votes, weekly poll counts."""
    import trends as T
    from national_polls import load, load_approval_polls, load_generic_ballot_log
    from polls_2026 import load_polls
    try:
        load()
        approval = load_approval_polls()
    except Exception:
        approval = pd.DataFrame()
    try:
        weekly = T.polls_per_week(load_polls(include_house=True))
    except Exception:
        weekly = pd.DataFrame()
    return {
        'approval_polls': approval,
        'approval_series': T.approval_series(approval) if len(approval) else pd.DataFrame(),
        'approval_trend': T.approval_trend_line(approval) if len(approval) else pd.DataFrame(),
        'generic_ballot': T.generic_ballot_history(load_generic_ballot_log(), load_topline_log()),
        'economy': T.economy(FRED_API_KEY) if FRED_API_KEY else {},
        'history': T.MIDTERM_HISTORY,
        'fits': {c: T.approval_seat_fit(chamber=c) for c in ('house', 'senate')},
        'rep_house_2024': T.rep_house_seats_2024(),
        'seats_votes': T.house_seats_votes(),
        'weekly_polls': weekly,
    }

@st.cache_data(ttl=3600, show_spinner="Rebuilding race averages over time...")
def load_close_race_averages(_annotated, _forecast, key):
    """Slow (a few seconds): every close race's polling average for each day since July.
    `key` (the forecast's as-of date and generic ballot) decides when to recompute."""
    import trends as T
    return T.close_race_averages(_annotated, _forecast)

def load_trends(forecast, annotated):
    import trends as T
    d = dict(load_trends_national())
    offices = ('senate', 'governor')
    d['polls_vs_fund'] = {o: T.polls_vs_fundamentals(forecast, o) for o in offices}
    d['vs_experts'] = {o: T.ours_vs_experts(forecast, o) for o in offices}
    d['partisan_split'] = {o: T.partisan_poll_split(annotated, forecast, o) for o in offices}
    d['movers'] = T.movers(T.race_history())
    d['close_races'] = load_close_race_averages(annotated, forecast, (forecast['as_of'], forecast['generic_ballot_poll']))
    return d

@st.cache_data(ttl=3600)
def load_generic_ballot_aggregators():
    try:
        from national_polls import load
        return pd.DataFrame(load()['generic_ballot']['aggregators'])
    except Exception:
        return pd.DataFrame()

def load_history():
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame(columns=[
        'saved_at','pred_year','incumbent','gdp_growth',
        'unemployment','approval','dem_ev','rep_ev','winner'
    ])

def save_prediction(year, incumbent, gdp_growth, unemployment, approval, dem_ev, rep_ev, winner):
    record = pd.DataFrame([{
        'saved_at':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pred_year':    year,
        'incumbent':    incumbent,
        'gdp_growth':   gdp_growth,
        'unemployment': unemployment,
        'approval':     approval,
        'dem_ev':       dem_ev,
        'rep_ev':       rep_ev,
        'winner':       winner,
    }])
    if os.path.exists(HISTORY_PATH):
        record.to_csv(HISTORY_PATH, mode='a', header=False, index=False)
    else:
        record.to_csv(HISTORY_PATH, index=False)

# ── Prediction engine ─────────────────────────────────────────────────────────
# 2028 presidential projection = the last election's county results + one national swing.
# A leave-one-election-out test (2004-2024) found the county XGBoost model no better than
# this once it's told the national swing, and worse in 2020 and 2024 (modeling.py,
# evaluate_honest; CHANGELOG 2026.10.08). Uncertainty, in pts of state margin, is the
# historical state-level error of uniform swing (5.2) plus an unknown national swing.
STATE_SWING_SD    = 5.2
NATIONAL_SWING_SD = 4.0

def predict_with_ci(year, scenario, approval, featured_df, models=None, feature_cols=None,
                    polling_overrides=None):
    """
    scenario: dict with gdp_growth, unemployment, inflation, consumer_confidence,
    incumbent_party, incumbent_on_ballot, consecutive_terms, prior_midterm_house_margin.
    The national swing (fundamentals + approval) moves the incumbent party's margin from
    the last election; zero swing means a repeat of it. models/feature_cols are unused.
    """
    incumbent_party = scenario['incumbent_party']
    base_year = int(featured_df['year'].max())
    last = featured_df[(featured_df['year'] == base_year) &
                       featured_df['party'].isin(['DEMOCRAT', 'REPUBLICAN'])]
    meta = last[['state', 'state_po', 'county_fips', 'county_name', 'party', 'totalvotes', 'pct']].copy()
    meta['county_fips'] = meta['county_fips'].astype(int).astype(str).str.zfill(5)

    fund_adj = fundamentals_adjustment(
        scenario['gdp_growth'], scenario['unemployment'], scenario['inflation'],
        scenario['consumer_confidence'], scenario['incumbent_on_ballot'],
        scenario['consecutive_terms'], scenario['prior_midterm_house_margin'],
        incumbent_party,
    )
    approval_adj = (approval - 50) * 0.15 if approval != 50 else 0.0
    # Bound the combined national swing (fundamentals + approval) as one quantity.
    total_adj = max(-MAX_TOTAL_ADJUSTMENT, min(MAX_TOTAL_ADJUSTMENT, fund_adj + approval_adj))
    # total_adj is a swing in margin: half moves to the incumbent, half from the challenger
    sign = np.where(meta['party'] == incumbent_party, 1.0, -1.0)
    meta['predicted_pct'] = (meta['pct'] + sign * total_adj / 2).clip(0, 100)

    county_results = meta.rename(columns={'totalvotes': 'avg_votes'})[
        ['state', 'state_po', 'county_fips', 'county_name', 'party', 'avg_votes', 'predicted_pct']].copy()

    w = meta['totalvotes']
    state_results = (meta.assign(wp=meta['predicted_pct'] * w)
                         .groupby(['state', 'state_po', 'party'])[['wp', 'totalvotes']].sum())
    state_results['predicted_pct'] = state_results['wp'] / state_results['totalvotes']
    state_results = state_results[['predicted_pct']].reset_index()
    half_width = 1.96 * float(np.hypot(STATE_SWING_SD, NATIONAL_SWING_SD)) / 2   # 95%, in vote share
    state_results['ci_lower'] = (state_results['predicted_pct'] - half_width).clip(0, 100)
    state_results['ci_upper'] = (state_results['predicted_pct'] + half_width).clip(0, 100)
    state_results = state_results.round(2)

    if polling_overrides:
        for spo, override in polling_overrides.items():
            for party, val in override.items():
                mask = (state_results['state_po']==spo) & (state_results['party']==party)
                state_results.loc[mask,['predicted_pct','ci_lower','ci_upper']] = val

    return state_results, county_results

# ── Map builders ──────────────────────────────────────────────────────────────
def build_historical_map(year_df):
    agg     = year_df.groupby(['state_po','party'])['votes'].sum().reset_index()
    winners = agg.loc[agg.groupby('state_po')['votes'].idxmax()].copy()
    dem     = agg[agg['party']=='DEMOCRAT'].set_index('state_po')['votes']
    rep     = agg[agg['party']=='REPUBLICAN'].set_index('state_po')['votes']
    total   = agg.groupby('state_po')['votes'].sum()
    spo     = winners['state_po'].values
    winners['dem_pct']   = (dem.reindex(spo).values / total.reindex(spo).values * 100).round(1)
    winners['rep_pct']   = (rep.reindex(spo).values / total.reindex(spo).values * 100).round(1)
    winners['color_val'] = winners['party'].map({'DEMOCRAT':0,'REPUBLICAN':1}).fillna(0.5)
    winners['hover']     = winners.apply(
        lambda r: f"<b>{r['state_po']}</b><br>Winner: {r['party']}<br>Dem: {r['dem_pct']:.1f}%<br>Rep: {r['rep_pct']:.1f}%",
        axis=1
    )
    fig = go.Figure(go.Choropleth(
        locations=winners['state_po'], z=winners['color_val'],
        locationmode='USA-states',
        colorscale=[[0,'#1f77b4'],[0.5,'#999999'],[1,'#d62728']],
        zmin=0, zmax=1, showscale=False,
        hovertext=winners['hover'], hoverinfo='text',
    ))
    fig.update_layout(geo_scope='usa', margin=dict(l=0,r=0,t=0,b=0), height=380)
    return fig

def _build_binary_state_choropleth(df, color_col, hover_col, height=380):
    """Shared D-vs-R state choropleth builder used by build_prediction_map."""
    fig = go.Figure(go.Choropleth(
        locations=df['state_po'], z=df[color_col],
        locationmode='USA-states',
        colorscale=[[0,'#1f77b4'],[1,'#d62728']],
        zmin=0, zmax=1, showscale=False,
        hovertext=df[hover_col], hoverinfo='text',
    ))
    fig.update_layout(geo_scope='usa', margin=dict(l=0,r=0,t=0,b=0), height=height)
    return fig

def build_prediction_map(states_dict):
    rows = []
    for state, info in states_dict.items():
        rows.append({
            'state':     state,
            'winner':    info['winner'],
            'dem_pct':   info['dem_pct'],
            'rep_pct':   info['rep_pct'],
            'ev':        info['ev'],
            'color_val': 0 if info['winner']=='DEMOCRAT' else 1,
        })
    df          = pd.DataFrame(rows)
    df['state_po'] = df['state'].map(STATE_ABBREV)
    df          = df.dropna(subset=['state_po'])
    df['hover'] = df.apply(
        lambda r: f"<b>{r['state_po']}</b><br>Projected: {r['winner']}<br>Dem: {r['dem_pct']:.1f}%<br>Rep: {r['rep_pct']:.1f}%<br>EV: {r['ev']}",
        axis=1
    )
    return _build_binary_state_choropleth(df, 'color_val', 'hover')

def build_county_map(county_results, selected_state, geojson):
    df    = county_results[county_results['state']==selected_state].copy()
    pivot = df.pivot_table(
        index=['county_fips','county_name','avg_votes'],
        columns='party', values='predicted_pct'
    ).reset_index()
    pivot.columns.name = None
    if 'DEMOCRAT' not in pivot.columns or 'REPUBLICAN' not in pivot.columns:
        return None
    pivot['winner']    = pivot.apply(lambda r: 'DEMOCRAT' if r['DEMOCRAT']>r['REPUBLICAN'] else 'REPUBLICAN', axis=1)
    pivot['color_val'] = pivot['winner'].map({'DEMOCRAT':0,'REPUBLICAN':1})
    pivot['hover']     = pivot.apply(
        lambda r: f"<b>{r['county_name']}</b><br>Dem: {r['DEMOCRAT']:.1f}%<br>Rep: {r['REPUBLICAN']:.1f}%",
        axis=1
    )
    fig = go.Figure(go.Choropleth(
        # Only ship this state's county shapes: the full US file is ~3 MB and was
        # re-sent to the browser on every rerun, freezing the page.
        geojson={'type': 'FeatureCollection',
                 'features': [f for f in geojson['features'] if f['id'] in set(pivot['county_fips'])]},
        locations=pivot['county_fips'], z=pivot['color_val'],
        colorscale=[[0,'#1f77b4'],[1,'#d62728']],
        zmin=0, zmax=1, showscale=False,
        hovertext=pivot['hover'], hoverinfo='text',
        marker_line_width=0.3,
    ))
    fig.update_layout(
        geo=dict(scope='usa', showlakes=False, fitbounds='locations'),
        margin=dict(l=0,r=0,t=30,b=0), height=420,
        title_text=f"{selected_state} — County Projections",
    )
    return fig

# ── Load data ─────────────────────────────────────────────────────────────────
historical   = load_historical()
featured     = load_featured()
models, feature_cols = None, None   # the 2028 projection is uniform swing (see predict_with_ci)
approval_trend = load_approval_trend()
generic_ballot_aggs = load_generic_ballot_aggregators()

# ── Session state for live data ───────────────────────────────────────────────
if 'live_data' not in st.session_state:
    approval, approval_source             = fetch_approval_rating()
    generic_ballot, generic_ballot_source = fetch_generic_ballot()
    st.session_state.live_data = {
        'gdp_growth':               None,
        'gdp_date':                 None,
        'unemployment':             None,
        'unemp_date':               None,
        'inflation':                None,
        'inflation_date':           None,
        'consumer_confidence':      None,
        'consumer_confidence_date': None,
        'approval':                 approval,
        'approval_source':          approval_source,
        'generic_ballot':           generic_ballot,
        'generic_ballot_source':    generic_ballot_source,
        'fetched_at':               datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

live = st.session_state.live_data

# ── Sidebar — global data refresh ────────────────────────────────────────────
st.sidebar.title(f"🗳️ {MODEL_NAME}")
st.sidebar.caption(f"{TAGLINE} · from {PUBLICATION}")
st.sidebar.divider()
st.sidebar.subheader("Live Data")
if FRED_API_KEY:
    if st.sidebar.button("🔄 Refresh All Data"):
        with st.spinner("Fetching latest data..."):
            st.session_state.live_data = fetch_all(FRED_API_KEY)
            live = st.session_state.live_data
            st.sidebar.success("Data refreshed!")

if live['fetched_at']:
    st.sidebar.caption(f"Last updated: {live['fetched_at']}")
if live['gdp_growth'] is not None:
    st.sidebar.caption(f"GDP Growth: **{live['gdp_growth']}%** ({live['gdp_date']})")
if live['unemployment'] is not None:
    st.sidebar.caption(f"Unemployment: **{live['unemployment']}%** ({live['unemp_date']})")
if live['inflation'] is not None:
    st.sidebar.caption(f"Inflation (CPI YoY): **{live['inflation']}%** ({live['inflation_date']})")
if live['consumer_confidence'] is not None:
    st.sidebar.caption(f"Consumer Confidence: **{live['consumer_confidence']}** ({live['consumer_confidence_date']})")
if live['approval'] is not None:
    st.sidebar.caption(f"Approval: **{live['approval']}%**")
    st.sidebar.caption(f"Source: {live['approval_source']}")
else:
    st.sidebar.caption("Add FRED API key to config.py to enable auto-refresh.")
if live['generic_ballot'] is not None:
    st.sidebar.caption(f"Generic Ballot (D−R): **{live['generic_ballot']:+.1f}**")
    st.sidebar.caption(f"Source: {live['generic_ballot_source']}")

# ── Sidebar: 2026 forecast input ─────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader("2026 Forecast")
generic_ballot_poll = st.sidebar.slider(
    "Generic ballot polling average (D−R)", -15.0, 20.0,
    float(round(_or_default(live['generic_ballot'], 0.0), 1)), 0.1,
    help="Defaults to the average of generic-ballot aggregators with no prediction-market ties (currently "
         "FiftyPlusOne and NBC News). Move it to see how every race shifts with the national "
         "environment. The model removes the generic ballot's historical ~3-pt Democratic lean before using it.")
poll_miss = st.sidebar.select_slider(
    "What if the polls are off? (polling miss)", options=[float(x) / 2 for x in range(-12, 13)], value=0.0,
    format_func=lambda m: "None: the forecast" if m == 0 else (f"R+{m:g}" if m > 0 else f"D+{-m:g}"),
    help="Shifts every poll (the generic ballot and each race's polls) by this many points, beyond the corrections "
         "the forecast already makes. R+3 means the polls overstate Democrats by 3 (about the 2024 Senate miss); "
         "R+6 is roughly 2020. D+ means they understate Democrats. The published forecast is 0. See the What if? tab.")
if st.sidebar.button("🔄 Refresh polls & race data",
                     help="Re-scrapes all 71 Senate/governor poll pages, race ratings, and House districts "
                          "from Wikipedia. Takes 1–2 minutes."):
    with st.spinner("Re-scraping polls and race data (1–2 minutes)..."):
        refresh_2026_data()
    st.rerun()
try:
    from polls_2026 import load_polls
    _p = load_polls()
    st.sidebar.caption(f"Polls: {len(_p)} · scraped {pd.Timestamp(_p['scraped_at'].iloc[0]):%b %d %H:%M} · "
                       f"latest poll {_p['end_date'].max():%b %d}")
except Exception:
    st.sidebar.caption("No 2026 poll file yet. Click Refresh.")
try:
    from poll_checks import load_checks
    _c = load_checks()
    if _c is not None and (_c['severity'] == 'warning').any():
        st.sidebar.warning(f"{int((_c['severity'] == 'warning').sum())} poll data warnings. See Track Record → Data quality.")
except Exception:
    pass

# ── Tabs ──────────────────────────────────────────────────────────────────────
forecast = run_forecast(generic_ballot_poll, poll_miss)
base_forecast = run_forecast(generic_ballot_poll, 0.0) if poll_miss else forecast
if poll_miss:
    st.warning(f"Scenario view: polls miss by {'R' if poll_miss > 0 else 'D'}+{abs(poll_miss):g} beyond the usual "
               "corrections. This is a what-if, not the forecast. Set the sidebar slider back to 0 for the forecast.")
inputs = forecast_inputs()

(tab_overview, tab_senate, tab_house, tab_governors, tab_polls, tab_whatif, tab_night, tab_results, tab_compare,
 tab_trends, tab_posts, tab_track, tab_home, tab_historical, tab_prediction, tab_history) = st.tabs([
    "📊 2026 Overview", "🏛️ Senate", "🏘️ House", "🏢 Governors", "📋 Polls", "🎚️ What if?", "🗳️ Election Night",
    "🏁 Results",
    "🔎 Comparisons", "📈 Trends", "📣 Post Builder", "📏 Track Record",
    "🗓️ 2028 Outlook", "📜 Historical Results", "🔮 2028 Prediction", "📈 Prediction History",
])

# ══════════════════════════════════════════════════════════════════════════════
# 2026 MIDTERMS
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    dash.render_overview(forecast, MODEL_VERSION, load_topline_log())
    st.divider()
    st.subheader("Generic ballot: where the national input comes from")
    if not generic_ballot_aggs.empty:
        g = generic_ballot_aggs.sort_values('margin')
        fig = go.Figure(go.Bar(
            x=g['margin'], y=g['aggregator'], orientation='h',
            marker_color=[dash.NEUTRAL if t else dash.DEM for t in g['market_tied']],
            text=[f"D{m:+.1f}" + ("  (excluded: prediction-market ties)" if t else "") for m, t in zip(g['margin'], g['market_tied'])],
            textposition='outside', cliponaxis=False,
            hovertemplate="%{y}: D%{x:+.1f}<extra></extra>"))
        fig.add_vline(x=generic_ballot_poll, line_dash='dash', line_color=dash.INK_2,
                      annotation_text=f"Forecast input D{generic_ballot_poll:+.1f}", annotation_font_color=dash.INK_2)
        fig.update_xaxes(title="Dem − Rep (pts)", range=[0, max(g['margin'].max() * 1.6, generic_ballot_poll + 3)])
        st.plotly_chart(dash._layout(fig, height=260), width='stretch', key='gb_aggs')
        st.caption("Generic-ballot averages listed on Wikipedia. Blue bars are used; gray bars belong to organizations "
                   "with business ties to prediction markets (Decision Desk HQ, RealClearPolitics, Silver Bulletin) "
                   "and are excluded.")
    else:
        st.info("Generic ballot aggregators unavailable.")

with tab_senate:
    _s = forecast['summary']['senate']
    dash.render_statewide(
        forecast, inputs['annotated_polls'], 'senate', 'Senate',
        f"35 races: 33 regular seats plus the Florida and Ohio specials. Seats not up this year: "
        f"{_s['holdovers']['DEMOCRAT']} D, {_s['holdovers']['REPUBLICAN']} R. Democrats need 51 for control "
        f"({_s['dem_control_prob']:.0%} chance); Republicans need 50 with VP Vance ({_s['rep_control_prob']:.0%}). "
        "In Nebraska the main challenger is independent Dan Osborn, who isn't counted toward either caucus.",
        MODEL_VERSION, load_race_history())

with tab_house:
    dash.render_house(forecast, MODEL_VERSION)

with tab_governors:
    _g = forecast['summary']['governor']
    dash.render_statewide(
        forecast, inputs['annotated_polls'], 'governor', 'Governor',
        f"36 governor races. Democrats are projected to win a median of {_g['dem_governorships_median']} "
        f"(90% range {_g['dem_governorships_90pct'][0]}–{_g['dem_governorships_90pct'][1]}). Governor fundamentals are "
        "weaker than the Senate's (governors often run far from their state's lean), so polls carry most of the weight.",
        MODEL_VERSION, load_race_history())

with tab_polls:
    import dashboard_polls
    dashboard_polls.render_polls(forecast, inputs['annotated_polls'], inputs.get('house_annotated_polls'))

with tab_whatif:
    dash.render_scenarios(base_forecast, forecast, poll_miss_sweep(generic_ballot_poll), poll_miss)

with tab_night:
    import dashboard_night
    if poll_miss:
        st.info("Election-night mode uses the forecast, not a what-if. Set the polling-miss slider back to 0.")
    else:
        dashboard_night.render_election_night(forecast, MODEL_VERSION)

with tab_results:
    import dashboard_results
    dashboard_results.render_results(MODEL_VERSION)

with tab_compare:
    dash.render_comparisons(forecast, inputs)

with tab_trends:
    dash.render_trends(forecast, load_trends(forecast, inputs['annotated_polls']))

with tab_posts:
    dash.render_posts(forecast, inputs['annotated_polls'], MODEL_VERSION, inputs.get('house_annotated_polls'))

with tab_track:
    dash.render_track_record(MODEL_VERSION)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    st.title("2028 Presidential Outlook")
    st.caption("Real-time economic conditions, presidential approval, and model-based forecasts")
    st.divider()

    # Days to next election
    next_election  = date(2028, 11, 7)
    days_remaining = (next_election - date.today()).days

    # Current approval — explicit None-check: a live-fetched value could legitimately
    # be 0 (or any falsy-looking number), which `or` would wrongly treat as "missing"
    current_approval = _or_default(live['approval'], 50)
    approval_source  = live['approval_source'] or "Wikipedia approval polls"

    # Latest FRED values
    current_gdp   = live['gdp_growth']
    current_unemp = live['unemployment']

    # ── Top metrics ───────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🏛️ Current President",    "Donald Trump",   "Republican")
    c2.metric("👍 Approval Rating",       f"{current_approval}%",
              delta=f"{current_approval-50:+.0f}% vs 50%",
              delta_color="normal")
    c3.metric("📈 GDP Growth",
              f"{current_gdp:.1f}%" if current_gdp is not None else "—",
              "Annual" if current_gdp is not None else "Click Refresh")
    c4.metric("👷 Unemployment",
              f"{current_unemp:.1f}%" if current_unemp is not None else "—",
              "Monthly" if current_unemp is not None else "Click Refresh")
    c5.metric("🗓️ Days to 2028 Election", f"{days_remaining:,}", "Nov 7, 2028")

    st.divider()

    col_left, col_right = st.columns(2)

    # ── Approval trend chart ──────────────────────────────────────────────────
    with col_left:
        st.subheader("Presidential Approval Trend")
        _nat = load_trends_national()
        if len(_nat['approval_polls']) and len(_nat['approval_series']):
            _span = st.segmented_control("Show", ['6 months', '1 year', 'Full term'], default='1 year',
                                         key='outlook_approval_span') or '1 year'
            _days = {'6 months': 182, '1 year': 365}.get(_span)
            _start = _nat['approval_polls']['end_date'].max() - pd.Timedelta(days=_days) if _days else None
            _cur = _nat['approval_series'].dropna(subset=['approve']).iloc[-1]
            st.plotly_chart(dash.approval_chart(_nat['approval_polls'], _nat['approval_trend'], _start, height=340,
                                                latest={'approve': _cur['approve'], 'disapprove': _cur['disapprove']}),
                            width='stretch', key='outlook_approval')
            st.caption(f"Dots are single polls; lines are a smoothed trend. Line-end numbers are our 30-day average. "
                       f"Source: {approval_source}. More in the 📈 Trends tab.")
        else:
            st.info("No approval polls loaded. Click Refresh polls & race data in the sidebar.")

    # ── Economic indicators ───────────────────────────────────────────────────
    with col_right:
        st.subheader("Economic Indicators")
        if current_gdp is not None or current_unemp is not None:
            fig2, axes = plt.subplots(2, 1, figsize=(6, 5))

            # GDP gauge
            ax1 = axes[0]
            gdp_display = _or_default(current_gdp, 0)
            gdp_color = '#2ecc71' if gdp_display > 0 else '#e74c3c'
            ax1.barh(['GDP Growth'], [gdp_display], color=gdp_color)
            ax1.axvline(0, color='black', linewidth=0.8)
            ax1.set_xlim(-5, 8)
            ax1.set_xlabel("% Annual Growth")
            ax1.set_title(f"Real GDP Growth: {current_gdp:.1f}%" if current_gdp is not None else "GDP: N/A")

            # Unemployment gauge
            ax2 = axes[1]
            unemp_display = _or_default(current_unemp, 0)
            unemp_color = '#e74c3c' if unemp_display > 6 else '#2ecc71'
            ax2.barh(['Unemployment'], [unemp_display], color=unemp_color)
            ax2.axvline(4, color='gray', linestyle='--', linewidth=0.8, label='4% baseline')
            ax2.set_xlim(0, 15)
            ax2.set_xlabel("% Unemployment Rate")
            ax2.set_title(f"Unemployment: {current_unemp:.1f}%" if current_unemp is not None else "Unemployment: N/A")

            plt.tight_layout()
            st.pyplot(fig2); plt.close()
            st.caption("Source: Federal Reserve (FRED)")
        else:
            st.info("Click **Refresh All Data** in the sidebar to load live FRED data.")

    st.divider()

    # ── Quick EC snapshot ─────────────────────────────────────────────────────
    st.subheader("Current Model Snapshot — 2028 Projection")
    st.caption("The 2024 result in every county, shifted by one national swing from current economic "
               "conditions and approval (Republican incumbent party, open seat).")
    if st.button("Run Quick Projection"):
        with st.spinner("Running projection..."):
            snap_scenario = {
                'gdp_growth':          _or_default(current_gdp, 2.5),
                'unemployment':        _or_default(current_unemp, 4.0),
                'inflation':           _or_default(live['inflation'], 3.0),
                'consumer_confidence': _or_default(live['consumer_confidence'], 70.0),
                'incumbent_party':     'REPUBLICAN',
                'incumbent_on_ballot': False,  # Trump is term-limited entering 2028
                'consecutive_terms':   1,      # GOP has held the White House one term (since 2025)
                'prior_midterm_house_margin': _or_default(live['generic_ballot'], 0.0),
            }
            sr, _ = predict_with_ci(
                2028, snap_scenario, current_approval,
                featured, models, feature_cols
            )
            ec = calculate_predicted_ec(sr, 2028)
        d_ev = ec['DEMOCRAT']
        r_ev = ec['REPUBLICAN']
        w    = ec['winner']
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Democrat EV",   d_ev, delta=f"{d_ev-270:+d} vs 270")
        cc2.metric("Republican EV", r_ev, delta=f"{r_ev-270:+d} vs 270")
        cc3.metric("Projected Winner", w)
        st.plotly_chart(build_prediction_map(ec['states']), width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORICAL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_historical:
    st.title("Historical Election Results")

    # Inline filters
    f1,f2,f3 = st.columns(3)
    selected_year  = f1.selectbox("Election Year", sorted(historical["year"].unique(), reverse=True))
    selected_state = f2.selectbox("State", ["ALL STATES"] + sorted(historical["state"].unique()))
    selected_party = f3.selectbox("Party", ["ALL","DEMOCRAT","REPUBLICAN","OTHER"])

    df = historical[historical["year"]==selected_year].copy()
    if selected_state != "ALL STATES":
        df = df[df["state"]==selected_state]
    if selected_party != "ALL":
        df = df[df["party"]==selected_party]

    if not df.empty:
        dem   = df[df["party"]=="DEMOCRAT"]
        rep   = df[df["party"]=="REPUBLICAN"]
        dem_v = int(dem["votes"].sum()) if not dem.empty else 0
        rep_v = int(rep["votes"].sum()) if not rep.empty else 0
        total = int(df["votes"].sum())
        denom = (dem_v+rep_v) or 1
        ec    = calculate_ec_result(historical[historical["year"]==selected_year], selected_year)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Democrat Vote Share",      f"{dem_v/denom*100:.1f}%")
        c2.metric("Republican Vote Share",    f"{rep_v/denom*100:.1f}%")
        c3.metric("Total Votes Cast",         f"{total:,}")
        c4.metric("Electoral Votes (D / R)",  f"{ec['DEMOCRAT']} / {ec['REPUBLICAN']}")
        c5.metric("Electoral College Winner", ec["winner"])

    st.divider()

    if df.empty:
        st.warning("No data for this selection.")
    else:
        st.subheader("Results Map")
        year_map_df = historical[historical["year"]==selected_year]
        if 'state_po' in year_map_df.columns:
            st.plotly_chart(build_historical_map(year_map_df), width='stretch')

        col_left, col_right = st.columns(2)
        total_votes_cast = df["votes"].sum()
        agg = df.groupby(["candidate_name","party"])["votes"].sum().reset_index()
        agg["pct"] = (agg["votes"]/total_votes_cast*100).round(2)
        agg = agg.sort_values("pct", ascending=False).head(10)

        with col_left:
            st.subheader("Vote Share by Candidate")
            fig, ax = plt.subplots(figsize=(7,4))
            colors = [PARTY_COLORS.get(p,"#aec7e8") for p in agg["party"]]
            ax.barh(agg["candidate_name"], agg["pct"], color=colors)
            ax.invert_yaxis()
            ax.set_xlabel("Vote Share (%)")
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
            st.pyplot(fig); plt.close()

        with col_right:
            st.subheader("Vote Share by Party")
            party_df = df.groupby("party")["votes"].sum().reset_index()
            party_df["pct"] = party_df["votes"]/total_votes_cast*100
            party_df = party_df[party_df["pct"]>0]
            pie_colors = [PARTY_COLORS.get(p,"#aec7e8") for p in party_df["party"]]
            fig2, ax2 = plt.subplots(figsize=(5,4))
            ax2.pie(party_df["pct"], labels=party_df["party"],
                    autopct="%1.1f%%", colors=pie_colors, startangle=140)
            st.pyplot(fig2); plt.close()

        st.subheader("Full Results Table")
        display = df[["year","state","candidate_name","party","votes","pct","sample_size"]].copy()
        display.columns = ["Year","State","Candidate","Party","Votes","Vote %","Total Votes"]
        st.dataframe(display.sort_values("Vote %",ascending=False).reset_index(drop=True), width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MAKE A PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_prediction:
    st.title("Make a Prediction")

    # Inline inputs
    i1,i2,i3,i4,i5 = st.columns(5)
    pred_year    = i1.number_input("Election Year",           min_value=2024, max_value=2100, value=2028, step=4)
    incumbent    = i2.selectbox("Incumbent Party",            ["REPUBLICAN","DEMOCRAT"])
    gdp_growth   = i3.slider("GDP Growth (%)",                -5.0, 8.0,  float(_or_default(live['gdp_growth'],   2.5)), 0.1)
    unemployment = i4.slider("Unemployment (%)",              2.0,  15.0, float(_or_default(live['unemployment'], 4.0)), 0.1)
    approval     = i5.slider("Presidential Approval (%)",     20,   80,   int(_or_default(live['approval'],       50)),  1)

    j1,j2,j3,j4 = st.columns(4)
    inflation           = j1.slider("CPI Inflation (YoY %)",         -2.0, 12.0,  float(_or_default(live['inflation'], 3.0)), 0.1)
    consumer_confidence = j2.slider("Consumer Confidence (UMCSENT)", 40.0, 110.0, float(_or_default(live['consumer_confidence'], 70.0)), 0.5)
    incumbent_on_ballot = j3.checkbox("Incumbent president on the ballot?", value=(pred_year != 2028),
                                      help="Trump is term-limited, so no incumbent runs in 2028.")
    consecutive_terms   = j4.number_input("Consecutive terms incumbent party has held the WH",
                                           min_value=1, max_value=5, value=1, step=1)

    k1, _, _, _ = st.columns(4)
    prior_midterm_house_margin = k1.slider(
        "Prior Midterm House Margin (D−R pts)", -20.0, 20.0,
        float(_or_default(live['generic_ballot'], 0.0)), 0.5,
        help="National House popular-vote margin from the midterm two years before this election. "
             "Defaults to the live generic congressional ballot as a nowcast for the upcoming midterm."
    )

    st.caption(f"Approval source: {live['approval_source'] or 'Manual'}")
    st.divider()

    scenario = {
        'gdp_growth':          gdp_growth,
        'unemployment':        unemployment,
        'inflation':           inflation,
        'consumer_confidence': consumer_confidence,
        'incumbent_party':     incumbent,
        'incumbent_on_ballot': incumbent_on_ballot,
        'consecutive_terms':   consecutive_terms,
        'prior_midterm_house_margin': prior_midterm_house_margin,
    }

    with st.spinner("Running predictions..."):
        state_results, county_results = predict_with_ci(
            pred_year, scenario, approval,
            featured, models, feature_cols
        )
        ec = calculate_predicted_ec(state_results, pred_year)

    dem_ev = ec["DEMOCRAT"]
    rep_ev = ec["REPUBLICAN"]
    winner = ec["winner"]

    # ── Polling overrides ─────────────────────────────────────────────────────
    with st.expander("➕ Add State Polling Data (overrides model for edited states)"):
        st.caption("Edit Dem % or Rep % for any state to override the model.")
        override_base = state_results.pivot_table(
            index=['state_po'], columns='party', values='predicted_pct'
        ).reset_index()
        override_base.columns.name = None
        if 'DEMOCRAT'   not in override_base.columns: override_base['DEMOCRAT']   = 0.0
        if 'REPUBLICAN' not in override_base.columns: override_base['REPUBLICAN'] = 0.0
        override_base = override_base[['state_po','DEMOCRAT','REPUBLICAN']].rename(
            columns={'DEMOCRAT':'Dem %','REPUBLICAN':'Rep %'}
        )
        edited = st.data_editor(override_base, width='stretch', hide_index=True, key="polling_overrides")

        overrides = {}
        for _, row in edited.iterrows():
            orig = override_base[override_base['state_po']==row['state_po']]
            if orig.empty: continue
            if abs(row['Dem %']-float(orig['Dem %'].values[0])) > 0.01 or \
               abs(row['Rep %']-float(orig['Rep %'].values[0])) > 0.01:
                overrides[row['state_po']] = {'DEMOCRAT':row['Dem %'],'REPUBLICAN':row['Rep %']}

        if overrides:
            with st.spinner("Re-running with polling overrides..."):
                state_results, county_results = predict_with_ci(
                    pred_year, scenario, approval,
                    featured, models, feature_cols, polling_overrides=overrides
                )
                ec     = calculate_predicted_ec(state_results, pred_year)
                dem_ev = ec["DEMOCRAT"]
                rep_ev = ec["REPUBLICAN"]
                winner = ec["winner"]

    # ── EC metrics ────────────────────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Democrat Electoral Votes",   dem_ev, delta=f"{dem_ev-270:+d} vs 270")
    c2.metric("Republican Electoral Votes", rep_ev, delta=f"{rep_ev-270:+d} vs 270")
    c3.metric("Projected Winner",           winner)
    c4.metric("Presidential Approval",      f"{approval}%", delta=f"{approval-50:+d} vs 50%")

    st.divider()

    # ── EC bar ────────────────────────────────────────────────────────────────
    st.subheader("Electoral Vote Breakdown")
    fig, ax = plt.subplots(figsize=(10,1.2))
    ax.barh([""], [dem_ev], color="#1f77b4", label=f"Democrat ({dem_ev})")
    ax.barh([""], [rep_ev], left=[dem_ev], color="#d62728", label=f"Republican ({rep_ev})")
    remaining = 538-dem_ev-rep_ev
    if remaining > 0:
        ax.barh([""], [remaining], left=[dem_ev+rep_ev], color="#cccccc", label="Other")
    ax.axvline(270, color="black", linestyle="--", linewidth=1.5, label="270 to win")
    ax.set_xlim(0,538); ax.set_xlabel("Electoral Votes")
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig); plt.close()

    st.divider()

    st.subheader("Projected Results Map")
    st.plotly_chart(build_prediction_map(ec["states"]), width='stretch')

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("State-by-State Projections")
        rows = []
        for state, info in sorted(ec["states"].items()):
            sr      = state_results[state_results['state']==state]
            dem_row = sr[sr['party']=='DEMOCRAT']
            rep_row = sr[sr['party']=='REPUBLICAN']
            dem_ci  = f"±{(dem_row['ci_upper'].values[0]-dem_row['ci_lower'].values[0])/2:.1f}" if not dem_row.empty else ""
            rep_ci  = f"±{(rep_row['ci_upper'].values[0]-rep_row['ci_lower'].values[0])/2:.1f}" if not rep_row.empty else ""
            rows.append({
                "State":  state,
                "Winner": info['winner'],
                "Dem %":  f"{info['dem_pct']} {dem_ci}",
                "Rep %":  f"{info['rep_pct']} {rep_ci}",
                "Margin": round(info['dem_pct']-info['rep_pct'],2),
                "EV":     info['ev'],
            })
        table_df = pd.DataFrame(rows).sort_values("Margin",ascending=False).reset_index(drop=True)
        st.dataframe(table_df, width='stretch', height=500)

    with col_right:
        st.subheader("Closest States (Battlegrounds)")
        battle    = table_df.copy()
        battle['abs_margin'] = battle['Margin'].abs()
        battle    = battle.nsmallest(15,'abs_margin').sort_values('Margin')
        err_vals  = []
        for _, row in battle.iterrows():
            sr      = state_results[state_results['state']==row['State']]
            dem_row = sr[sr['party']=='DEMOCRAT']
            ci_half = (dem_row['ci_upper'].values[0]-dem_row['ci_lower'].values[0])/2 if not dem_row.empty else 0
            err_vals.append(ci_half)
        fig3, ax3 = plt.subplots(figsize=(6,6))
        bar_colors = [PARTY_COLORS["DEMOCRAT"] if m>0 else PARTY_COLORS["REPUBLICAN"] for m in battle['Margin']]
        ax3.barh(battle['State'], battle['Margin'], color=bar_colors,
                 xerr=err_vals, error_kw=dict(ecolor='gray',capsize=3,elinewidth=1))
        ax3.axvline(0,color='black',linewidth=0.8)
        ax3.set_xlabel("Margin (positive = Democrat lead)")
        ax3.set_title("15 Closest States")
        st.pyplot(fig3); plt.close()

    st.divider()

    st.subheader("County-Level Drill-Down")
    all_states            = sorted(county_results['state'].unique())
    selected_county_state = st.selectbox("Select a state", all_states)
    geojson               = load_counties_geojson()
    if geojson is None:
        st.warning("County map unavailable — could not load county boundaries.")
    else:
        county_fig = build_county_map(county_results, selected_county_state, geojson)
        if county_fig:
            st.plotly_chart(county_fig, width='stretch')

    st.divider()

    if st.button("💾 Save This Prediction", type="primary"):
        save_prediction(pred_year, incumbent, gdp_growth, unemployment,
                        approval, dem_ev, rep_ev, winner)
        st.success("Prediction saved! View it in the Prediction History tab.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.title("Prediction History")
    st.caption("Track how projections shift as economic conditions change")

    history = load_history()

    if history.empty:
        st.info("No predictions saved yet. Go to Make a Prediction and click Save This Prediction.")
    else:
        c1,c2,c3 = st.columns(3)
        last = history.iloc[-1]
        c1.metric("Total Saved",          len(history))
        c2.metric("Latest Projected Winner", last['winner'])
        c3.metric("Latest EC (D / R)",    f"{int(last['dem_ev'])} / {int(last['rep_ev'])}")

        st.divider()

        if len(history) > 1:
            st.subheader("Electoral Vote Trend")
            st.plotly_chart(dash.ev_history_chart(history), width='stretch', key='ev_history')
            st.caption("Each bar is one saved 2028 prediction (electoral votes out of 538), newest on top. Hover for the "
                       "inputs behind it.")

        st.divider()
        st.subheader("All Saved Predictions")
        display = history.copy()
        display.columns = ['Saved At','Year','Incumbent','GDP Growth',
                           'Unemployment','Approval','Dem EV','Rep EV','Winner']
        st.dataframe(display, width='stretch')

        if st.button("🗑️ Clear History", type="secondary"):
            os.remove(HISTORY_PATH)
            st.rerun()
