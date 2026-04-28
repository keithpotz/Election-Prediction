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
from data_sources import fetch_all, fetch_approval_rating
from config import FRED_API_KEY

st.set_page_config(page_title="Election Prediction", layout="wide", page_icon="🗳️")
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(BASE_DIR, "data/prediction_history.csv")
APPROVAL_CSV = os.path.join(BASE_DIR, "data/polling_data/Silver Bulletin Trump approval polls - Sheet1.csv")

PARTY_COLORS = {"DEMOCRAT": "#1f77b4", "REPUBLICAN": "#d62728", "OTHER": "#7f7f7f"}
STATE_ABBREV = {
    'ALABAMA':'AL','ALASKA':'AK','ARIZONA':'AZ','ARKANSAS':'AR','CALIFORNIA':'CA',
    'COLORADO':'CO','CONNECTICUT':'CT','DELAWARE':'DE','DISTRICT OF COLUMBIA':'DC',
    'FLORIDA':'FL','GEORGIA':'GA','HAWAII':'HI','IDAHO':'ID','ILLINOIS':'IL',
    'INDIANA':'IN','IOWA':'IA','KANSAS':'KS','KENTUCKY':'KY','LOUISIANA':'LA',
    'MAINE':'ME','MARYLAND':'MD','MASSACHUSETTS':'MA','MICHIGAN':'MI','MINNESOTA':'MN',
    'MISSISSIPPI':'MS','MISSOURI':'MO','MONTANA':'MT','NEBRASKA':'NE','NEVADA':'NV',
    'NEW HAMPSHIRE':'NH','NEW JERSEY':'NJ','NEW MEXICO':'NM','NEW YORK':'NY',
    'NORTH CAROLINA':'NC','NORTH DAKOTA':'ND','OHIO':'OH','OKLAHOMA':'OK',
    'OREGON':'OR','PENNSYLVANIA':'PA','RHODE ISLAND':'RI','SOUTH CAROLINA':'SC',
    'SOUTH DAKOTA':'SD','TENNESSEE':'TN','TEXAS':'TX','UTAH':'UT','VERMONT':'VT',
    'VIRGINIA':'VA','WASHINGTON':'WA','WEST VIRGINIA':'WV','WISCONSIN':'WI','WYOMING':'WY',
}

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_historical():
    return pd.read_csv(os.path.join(BASE_DIR, "data/polling_data/cleaned_polls.csv"))

@st.cache_data
def load_featured():
    return pd.read_csv(os.path.join(BASE_DIR, "data/polling_data/featured_data.csv"))

@st.cache_resource
def load_model():
    model    = joblib.load(os.path.join(BASE_DIR, "models/random_forest_model.pkl"))
    features = joblib.load(os.path.join(BASE_DIR, "models/training_features.pkl"))
    return model, features

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

@st.cache_data
def load_approval_trend():
    try:
        df = pd.read_csv(APPROVAL_CSV)
        df = df[df['subgroup'] == 'All polls'].copy()
        df['enddate']          = pd.to_datetime(df['enddate'], errors='coerce')
        df['adjusted_approve'] = pd.to_numeric(df['adjusted_approve'], errors='coerce')
        df['adjusted_disapprove'] = pd.to_numeric(df['adjusted_disapprove'], errors='coerce')
        df = df.dropna(subset=['enddate','adjusted_approve'])
        return df.sort_values('enddate')
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
def predict_with_ci(year, gdp_growth, unemployment, incumbent_party, approval,
                    featured_df, model, feature_cols, polling_overrides=None):
    county_profiles = featured_df.groupby(
        ['state','state_po','county_fips','county_name']
    ).agg(
        county_lean    = ('county_lean', 'first'),
        state_lean     = ('state_lean',  'first'),
        avg_totalvotes = ('totalvotes',   'mean'),
    ).reset_index()

    rows = []
    for _, c in county_profiles.iterrows():
        for party in ['DEMOCRAT','REPUBLICAN']:
            rows.append({
                'year':         year,
                'totalvotes':   c['avg_totalvotes'],
                'gdp_growth':   gdp_growth,
                'unemployment': unemployment,
                'is_incumbent': 1 if party == incumbent_party else 0,
                'state_lean':   c['state_lean'],
                'county_lean':  c['county_lean'],
                'state':        c['state'],
                'party':        party,
                '__state_po':   c['state_po'],
                '__fips':       str(int(c['county_fips'])).zfill(5),
                '__county':     c['county_name'],
                '__avg_votes':  c['avg_totalvotes'],
            })

    pred_df   = pd.DataFrame(rows)
    meta      = pred_df[['state','__state_po','__fips','__county','party','__avg_votes']].copy()
    encode_df = pred_df.drop(columns=['__state_po','__fips','__county','__avg_votes'])
    encode_df = pd.get_dummies(encode_df, columns=['state','party'], drop_first=True)
    encode_df.columns = encode_df.columns.str.replace('__','_')
    encode_df = encode_df.reindex(columns=feature_cols, fill_value=0)

    tree_preds            = np.array([t.predict(encode_df) for t in model.estimators_])
    meta['predicted_pct'] = tree_preds.mean(axis=0).clip(0,100)
    meta['pred_std']      = tree_preds.std(axis=0)

    if approval != 50:
        adj  = (approval - 50) * 0.15
        mask = meta['party'] == incumbent_party
        meta.loc[mask,'predicted_pct'] = (meta.loc[mask,'predicted_pct'] + adj).clip(0,100)

    county_results = meta.rename(columns={
        '__state_po':'state_po','__fips':'county_fips',
        '__county':'county_name','__avg_votes':'avg_votes',
    })[['state','state_po','county_fips','county_name','party','avg_votes','predicted_pct']].copy()

    def state_agg(x):
        w     = x['__avg_votes']
        w_sum = w.sum() or 1
        mp    = (x['predicted_pct'] * w).sum() / w_sum
        sp    = (x['pred_std']      * w).sum() / w_sum
        return pd.Series({
            'predicted_pct': mp,
            'ci_lower':      max(0,   mp - 1.96*sp),
            'ci_upper':      min(100, mp + 1.96*sp),
        })

    state_results = meta.groupby(['state','__state_po','party']).apply(state_agg).reset_index()
    state_results.columns = ['state','state_po','party','predicted_pct','ci_lower','ci_upper']
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
    fig = go.Figure(go.Choropleth(
        locations=df['state_po'], z=df['color_val'],
        locationmode='USA-states',
        colorscale=[[0,'#1f77b4'],[1,'#d62728']],
        zmin=0, zmax=1, showscale=False,
        hovertext=df['hover'], hoverinfo='text',
    ))
    fig.update_layout(geo_scope='usa', margin=dict(l=0,r=0,t=0,b=0), height=380)
    return fig

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
        geojson=geojson, locations=pivot['county_fips'], z=pivot['color_val'],
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
model, feature_cols = load_model()
approval_trend = load_approval_trend()

# ── Session state for live data ───────────────────────────────────────────────
if 'live_data' not in st.session_state:
    approval, approval_source = fetch_approval_rating()
    st.session_state.live_data = {
        'gdp_growth':      None,
        'gdp_date':        None,
        'unemployment':    None,
        'unemp_date':      None,
        'approval':        approval,
        'approval_source': approval_source,
        'fetched_at':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

live = st.session_state.live_data

# ── Sidebar — global data refresh ────────────────────────────────────────────
st.sidebar.title("🗳️ Election Prediction")
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
if live['approval'] is not None:
    st.sidebar.caption(f"Approval: **{live['approval']}%**")
    st.sidebar.caption(f"Source: {live['approval_source']}")
else:
    st.sidebar.caption("Add FRED API key to config.py to enable auto-refresh.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_home, tab_historical, tab_prediction, tab_history = st.tabs([
    "🏠 Home", "📊 Historical Results", "🔮 Make a Prediction", "📈 Prediction History"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    st.title("🗳️ Election Prediction Dashboard")
    st.caption("Real-time economic conditions, presidential approval, and model-based forecasts")
    st.divider()

    # Days to next election
    next_election  = date(2028, 11, 7)
    days_remaining = (next_election - date.today()).days

    # Current approval
    current_approval = live['approval'] or 50
    approval_source  = live['approval_source'] or "Silver Bulletin"

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
              f"{current_gdp:.1f}%" if current_gdp else "—",
              "Annual" if current_gdp else "Click Refresh")
    c4.metric("👷 Unemployment",
              f"{current_unemp:.1f}%" if current_unemp else "—",
              "Monthly" if current_unemp else "Click Refresh")
    c5.metric("🗓️ Days to 2028 Election", f"{days_remaining:,}", "Nov 7, 2028")

    st.divider()

    col_left, col_right = st.columns(2)

    # ── Approval trend chart ──────────────────────────────────────────────────
    with col_left:
        st.subheader("Presidential Approval Trend")
        if not approval_trend.empty:
            # Last 180 days
            cutoff = approval_trend['enddate'].max() - pd.Timedelta(days=180)
            trend  = approval_trend[approval_trend['enddate'] >= cutoff]
            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(trend['enddate'], trend['adjusted_approve'],
                    color='#2ecc71', linewidth=2, label='Approve')
            if 'adjusted_disapprove' in trend.columns:
                ax.plot(trend['enddate'], trend['adjusted_disapprove'],
                        color='#e74c3c', linewidth=2, label='Disapprove')
            ax.axhline(50, color='gray', linestyle='--', linewidth=0.8)
            ax.set_ylabel("Approval (%)")
            ax.set_ylim(20, 70)
            ax.legend()
            ax.set_title("Last 180 Days · Silver Bulletin (adjusted)")
            plt.xticks(rotation=30, ha='right')
            st.pyplot(fig); plt.close()
            st.caption(f"Source: {approval_source}")
        else:
            st.info("No approval data loaded. Drop the Silver Bulletin CSV in src/data/polling_data/")

    # ── Economic indicators ───────────────────────────────────────────────────
    with col_right:
        st.subheader("Economic Indicators")
        if current_gdp is not None or current_unemp is not None:
            fig2, axes = plt.subplots(2, 1, figsize=(6, 5))

            # GDP gauge
            ax1 = axes[0]
            gdp_color = '#2ecc71' if (current_gdp or 0) > 0 else '#e74c3c'
            ax1.barh(['GDP Growth'], [current_gdp or 0], color=gdp_color)
            ax1.axvline(0, color='black', linewidth=0.8)
            ax1.set_xlim(-5, 8)
            ax1.set_xlabel("% Annual Growth")
            ax1.set_title(f"Real GDP Growth: {current_gdp:.1f}%" if current_gdp else "GDP: N/A")

            # Unemployment gauge
            ax2 = axes[1]
            unemp_color = '#e74c3c' if (current_unemp or 0) > 6 else '#2ecc71'
            ax2.barh(['Unemployment'], [current_unemp or 0], color=unemp_color)
            ax2.axvline(4, color='gray', linestyle='--', linewidth=0.8, label='4% baseline')
            ax2.set_xlim(0, 15)
            ax2.set_xlabel("% Unemployment Rate")
            ax2.set_title(f"Unemployment: {current_unemp:.1f}%" if current_unemp else "Unemployment: N/A")

            plt.tight_layout()
            st.pyplot(fig2); plt.close()
            st.caption("Source: Federal Reserve (FRED)")
        else:
            st.info("Click **Refresh All Data** in the sidebar to load live FRED data.")

    st.divider()

    # ── Quick EC snapshot ─────────────────────────────────────────────────────
    st.subheader("Current Model Snapshot — 2028 Projection")
    st.caption("Based on current economic conditions and approval rating assuming Republican incumbent")
    if st.button("Run Quick Projection"):
        with st.spinner("Running projection..."):
            snap_gdp   = current_gdp   or 2.5
            snap_unemp = current_unemp or 4.0
            snap_appr  = current_approval
            sr, _      = predict_with_ci(
                2028, snap_gdp, snap_unemp, 'REPUBLICAN', snap_appr,
                featured, model, feature_cols
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
    gdp_growth   = i3.slider("GDP Growth (%)",                -5.0, 8.0,  float(live['gdp_growth']   or 2.5), 0.1)
    unemployment = i4.slider("Unemployment (%)",              2.0,  15.0, float(live['unemployment'] or 4.0), 0.1)
    approval     = i5.slider("Presidential Approval (%)",     20,   80,   int(live['approval']       or 50),  1)

    st.caption(f"Approval source: {live['approval_source'] or 'Manual'}")
    st.divider()

    with st.spinner("Running predictions..."):
        state_results, county_results = predict_with_ci(
            pred_year, gdp_growth, unemployment, incumbent, approval,
            featured, model, feature_cols
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
                    pred_year, gdp_growth, unemployment, incumbent, approval,
                    featured, model, feature_cols, polling_overrides=overrides
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
            fig, ax = plt.subplots(figsize=(10,4))
            x = range(len(history))
            ax.plot(x, history['dem_ev'], color="#1f77b4", marker='o', label="Democrat EV")
            ax.plot(x, history['rep_ev'], color="#d62728", marker='o', label="Republican EV")
            ax.axhline(270, color='black', linestyle='--', linewidth=1, label="270 to win")
            ax.set_xticks(list(x))
            ax.set_xticklabels(history['saved_at'], rotation=30, ha='right', fontsize=7)
            ax.set_ylabel("Electoral Votes")
            ax.legend()
            st.pyplot(fig); plt.close()

        st.divider()
        st.subheader("All Saved Predictions")
        display = history.copy()
        display.columns = ['Saved At','Year','Incumbent','GDP Growth',
                           'Unemployment','Approval','Dem EV','Rep EV','Winner']
        st.dataframe(display, width='stretch')

        if st.button("🗑️ Clear History", type="secondary"):
            os.remove(HISTORY_PATH)
            st.rerun()
