#!/usr/bin/env python3
"""
Geospatial Crime Analysis Dashboard
Interactive visualization for FBI Field Specialist Application
Rebuilt with corrected column names and 11-page navigation.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# ---------------------------------------------------------------------------
# Page config -- NO emojis
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FBI Crime Pattern Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")

# ---------------------------------------------------------------------------
# Zone definitions (all 6)
# ---------------------------------------------------------------------------
ZONES = {
    'US-Mexico Border': {
        'states': ['CA', 'AZ', 'NM', 'TX'],
        'counties': [
            'San Diego', 'Imperial', 'Yuma', 'Pima', 'Santa Cruz', 'Cochise',
            'Hidalgo', 'Luna', 'Dona Ana', 'El Paso', 'Hudspeth', 'Culberson',
            'Jeff Davis', 'Presidio', 'Brewster', 'Terrell', 'Val Verde',
            'Kinney', 'Maverick', 'Dimmit', 'Webb', 'Zapata', 'Starr',
            'Hidalgo', 'Cameron', 'Willacy', 'Brooks',
        ],
    },
    'I-35 Corridor': {
        'states': ['TX'],
        'counties': [
            'Denton', 'Collin', 'Dallas', 'Ellis', 'Hill', 'McLennan',
            'Bell', 'Williamson', 'Travis', 'Hays', 'Comal', 'Bexar',
            'Atascosa', 'Frio', 'Medina', 'Webb',
        ],
    },
    'Pacific Northwest': {
        'states': ['WA', 'OR'],
        'counties': [
            'King', 'Pierce', 'Snohomish', 'Thurston', 'Multnomah',
            'Clackamas', 'Washington', 'Marion',
        ],
    },
    'Midwest Metro': {
        'states': ['IL', 'WI', 'IN', 'OH', 'MI'],
        'counties': [
            'Cook', 'Milwaukee', 'Wayne', 'Marion', 'Cuyahoga', 'Franklin',
            'Hamilton',
        ],
    },
    'Northeast Corridor': {
        'states': ['NY', 'NJ', 'PA', 'MA', 'MD'],
        'counties': [
            'New York', 'Kings', 'Queens', 'Bronx', 'Richmond',
            'Philadelphia', 'Baltimore', 'Suffolk', 'Essex',
        ],
    },
    'Southern California': {
        'states': ['CA'],
        'counties': [
            'Los Angeles', 'Orange', 'San Diego', 'Riverside',
            'San Bernardino', 'Ventura',
        ],
    },
}

# ---------------------------------------------------------------------------
# State coordinates (abbreviations only)
# ---------------------------------------------------------------------------
STATE_COORDS = {
    'AL': (32.806671, -86.791130), 'AK': (61.370716, -152.404419),
    'AZ': (33.729759, -111.431221), 'AR': (34.969704, -92.373123),
    'CA': (36.116203, -119.681564), 'CO': (39.059811, -105.311104),
    'CT': (41.597782, -72.755371), 'DE': (39.318523, -75.507141),
    'FL': (27.766279, -81.686783), 'GA': (33.040619, -83.643074),
    'HI': (21.094318, -157.498337), 'ID': (44.240459, -114.478828),
    'IL': (40.349457, -88.986137), 'IN': (39.849426, -86.258278),
    'IA': (42.011539, -93.210526), 'KS': (38.526600, -96.726486),
    'KY': (37.668140, -84.670067), 'LA': (31.169546, -91.867805),
    'ME': (44.693947, -69.381927), 'MD': (39.063946, -76.802101),
    'MA': (42.230171, -71.530106), 'MI': (43.326618, -84.536095),
    'MN': (45.694454, -93.900192), 'MS': (32.741646, -89.678696),
    'MO': (38.456085, -92.288368), 'MT': (46.921925, -110.454353),
    'NE': (41.125370, -98.268082), 'NV': (38.313515, -117.055374),
    'NH': (43.452492, -71.563896), 'NJ': (40.298904, -74.521011),
    'NM': (34.840515, -106.248482), 'NY': (42.165726, -74.948051),
    'NC': (35.630066, -79.806419), 'ND': (47.528912, -99.784012),
    'OH': (40.388783, -82.764915), 'OK': (35.565342, -96.928917),
    'OR': (44.572021, -122.070938), 'PA': (40.590752, -77.209755),
    'RI': (41.680893, -71.511780), 'SC': (33.856892, -80.945007),
    'SD': (44.299782, -99.438828), 'TN': (35.747845, -86.692345),
    'TX': (31.054487, -97.563461), 'UT': (40.150032, -111.862434),
    'VT': (44.045876, -72.710686), 'VA': (37.769337, -78.169968),
    'WA': (47.400902, -121.490494), 'WV': (38.491226, -80.954453),
    'WI': (44.268543, -89.616508), 'WY': (42.755966, -107.302490),
    'DC': (38.897438, -77.026817), 'PR': (18.220833, -66.590149),
    'GU': (13.444304, 144.793731), 'VI': (18.335765, -64.896335),
}

# Full-name -> abbreviation mapping used when loading raw bodies data
STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT',
    'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
    'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
    'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI',
    'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC', 'Puerto Rico': 'PR',
    'Guam': 'GU', 'Virgin Islands': 'VI',
}

# ---------------------------------------------------------------------------
# Custom dark-theme CSS -- NO emojis
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1e2127; padding: 10px; border-radius: 5px;}
    h1 {color: #00d4ff;}
    h2 {color: #00a8cc;}
    .alert-red {
        background-color: #ff4444; padding: 10px; border-radius: 5px;
        color: white; font-weight: bold;
    }
    .alert-orange {
        background-color: #ff8800; padding: 10px; border-radius: 5px;
        color: white; font-weight: bold;
    }
    .alert-yellow {
        background-color: #ffcc00; padding: 10px; border-radius: 5px;
        color: black; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ===================================================================
# DATA LOADERS -- all wrapped in try/except for graceful fallback
# ===================================================================

@st.cache_data
def load_raw_data():
    """Load raw missing-persons and unidentified-bodies CSVs."""
    try:
        mp_frames = []
        bodies_frames = []
        for fname in os.listdir(RAW_DIR):
            fpath = os.path.join(RAW_DIR, fname)
            if fname.endswith('_missing_persons.csv'):
                df = pd.read_csv(fpath)
                df['data_type'] = 'Missing Persons'
                mp_frames.append(df)
            elif fname.endswith('_unidentified_bodies.csv'):
                df = pd.read_csv(fpath)
                df['data_type'] = 'Unidentified Bodies'
                bodies_frames.append(df)
        df_mp = pd.concat(mp_frames, ignore_index=True) if mp_frames else pd.DataFrame()
        df_bodies = pd.concat(bodies_frames, ignore_index=True) if bodies_frames else pd.DataFrame()

        # Parse dates -> year
        if 'DLC' in df_mp.columns:
            df_mp['year'] = pd.to_datetime(df_mp['DLC'], errors='coerce').dt.year
        if 'DBF' in df_bodies.columns:
            df_bodies['year'] = pd.to_datetime(df_bodies['DBF'], errors='coerce').dt.year

        # Normalise State to abbreviation
        if 'State' in df_bodies.columns:
            df_bodies['State'] = df_bodies['State'].map(STATE_ABBREV).fillna(df_bodies['State'])
        return df_mp, df_bodies
    except Exception as exc:
        st.warning(f"Could not load raw data: {exc}")
        return pd.DataFrame(), pd.DataFrame()


def _load_analysis_csv(filename):
    """Helper: load a CSV from the analysis directory or return None."""
    fpath = os.path.join(ANALYSIS_DIR, filename)
    try:
        if os.path.exists(fpath):
            return pd.read_csv(fpath)
    except Exception:
        pass
    return None


@st.cache_data
def load_outlier_data():
    return _load_analysis_csv("county_decade_outliers.csv")

@st.cache_data
def load_ml_anomaly_scores():
    return _load_analysis_csv("ml_anomaly_scores.csv")

@st.cache_data
def load_spatial_autocorrelation():
    return _load_analysis_csv("spatial_autocorrelation.csv")

@st.cache_data
def load_global_morans():
    return _load_analysis_csv("global_morans_i.csv")

@st.cache_data
def load_county_clusters():
    return _load_analysis_csv("county_clusters.csv")

@st.cache_data
def load_zone_forecasts():
    return _load_analysis_csv("zone_forecasts.csv")

@st.cache_data
def load_zone_trends():
    return _load_analysis_csv("zone_trends.csv")

@st.cache_data
def load_arima_forecasts():
    return _load_analysis_csv("arima_forecasts.csv")

@st.cache_data
def load_forecast_backtest():
    return _load_analysis_csv("forecast_backtest.csv")

@st.cache_data
def load_temporal_trends():
    return _load_analysis_csv("temporal_trends.csv")

@st.cache_data
def load_zone_rate_comparisons():
    return _load_analysis_csv("zone_rate_comparisons.csv")

@st.cache_data
def load_zone_overall_tests():
    return _load_analysis_csv("zone_overall_tests.csv")

@st.cache_data
def load_covariate_adjusted():
    return _load_analysis_csv("covariate_adjusted_outliers.csv")


# ===================================================================
# MAIN
# ===================================================================

def main():
    st.title("Geospatial Crime Pattern Analysis System")
    st.markdown("**Multi-Tier Anomaly Detection for Serial Crime & Trafficking Networks**")
    st.markdown("---")

    # Sidebar navigation -- 11 pages
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        [
            "Overview",
            "Raw Count Map",
            "Sigma Heat Map",
            "Temporal Trends",
            "Outlier Detection",
            "Zone Forecasting",
            "Validation",
            "Robust Statistics",
            "Spatial Analysis",
            "Advanced Forecasting",
            "ML Anomaly Detection",
        ],
    )

    # Dispatch
    if page == "Overview":
        page_overview()
    elif page == "Raw Count Map":
        page_raw_count_map()
    elif page == "Sigma Heat Map":
        page_sigma_heat_map()
    elif page == "Temporal Trends":
        page_temporal_trends()
    elif page == "Outlier Detection":
        page_outlier_detection()
    elif page == "Zone Forecasting":
        page_zone_forecasting()
    elif page == "Validation":
        page_validation()
    elif page == "Robust Statistics":
        page_robust_statistics()
    elif page == "Spatial Analysis":
        page_spatial_analysis()
    elif page == "Advanced Forecasting":
        page_advanced_forecasting()
    elif page == "ML Anomaly Detection":
        page_ml_anomaly()


# ===================================================================
# PAGE 1 -- Overview
# ===================================================================

def page_overview():
    st.header("System Overview")

    df_outliers = load_outlier_data()
    df_mp, df_bodies = load_raw_data()

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    total_mp = len(df_mp) if not df_mp.empty else 0
    total_bodies = len(df_bodies) if not df_bodies.empty else 0

    with col1:
        st.metric("Missing Persons Records", f"{total_mp:,}")
    with col2:
        st.metric("Unidentified Bodies Records", f"{total_bodies:,}")
    with col3:
        st.metric("Total Cases", f"{total_mp + total_bodies:,}")
    with col4:
        if df_outliers is not None and 'alert_level' in df_outliers.columns:
            alert_counts = df_outliers['alert_level'].value_counts()
            red_count = alert_counts.get('RED', 0)
            orange_count = alert_counts.get('ORANGE', 0)
            st.metric("RED + ORANGE Alerts", f"{red_count + orange_count:,}")
        else:
            st.metric("Alerts", "N/A")

    st.markdown("---")

    # Alert breakdown
    if df_outliers is not None and 'alert_level' in df_outliers.columns:
        st.subheader("Alert Level Distribution")
        alert_counts = df_outliers['alert_level'].value_counts()
        total = len(df_outliers)
        acol1, acol2, acol3, acol4 = st.columns(4)
        with acol1:
            v = alert_counts.get('RED', 0)
            st.metric("RED Alerts", v, f"{v / total * 100:.1f}%")
        with acol2:
            v = alert_counts.get('ORANGE', 0)
            st.metric("ORANGE Alerts", v, f"{v / total * 100:.1f}%")
        with acol3:
            v = alert_counts.get('YELLOW', 0)
            st.metric("YELLOW Alerts", v, f"{v / total * 100:.1f}%")
        with acol4:
            v = alert_counts.get('GREEN', 0)
            st.metric("GREEN (Normal)", v, f"{v / total * 100:.1f}%")

    st.markdown("---")

    # Critical findings
    st.subheader("Critical Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="alert-red">I-35 CORRIDOR CRISIS</div>', unsafe_allow_html=True)
        st.write("**Accelerating trend** in missing persons along I-35.")
        st.write("**Status:** Human trafficking superhighway")
        st.markdown('<div class="alert-orange">TEXAS BORDER SURGE</div>', unsafe_allow_html=True)
        st.write("Significant increase in missing persons along the southern border in recent decades.")
    with col2:
        st.markdown('<div class="alert-orange">PIMA COUNTY, AZ</div>', unsafe_allow_html=True)
        st.write("Extreme statistical outlier for unidentified bodies.")
        st.write("**Pattern:** Border / cartel dumping ground")
        if df_outliers is not None and 'composite_z_score' in df_outliers.columns:
            top5 = df_outliers.sort_values('composite_z_score', ascending=False).head(5)
            st.write("**Top 5 composite z-scores:**")
            st.dataframe(
                top5[['State', 'County', 'decade', 'composite_z_score', 'alert_level']].reset_index(drop=True),
                use_container_width=True,
            )

    st.markdown("---")

    # Top 20 states bar chart
    st.subheader("Top 20 States by Total Cases")
    if not df_mp.empty or not df_bodies.empty:
        mp_by_state = df_mp.groupby('State').size().reset_index(name='mp_count') if not df_mp.empty else pd.DataFrame(columns=['State', 'mp_count'])
        bodies_by_state = df_bodies.groupby('State').size().reset_index(name='bodies_count') if not df_bodies.empty else pd.DataFrame(columns=['State', 'bodies_count'])
        state_summary = pd.merge(mp_by_state, bodies_by_state, on='State', how='outer').fillna(0)
        state_summary['total'] = state_summary['mp_count'] + state_summary['bodies_count']

        fig = px.bar(
            state_summary.sort_values('total', ascending=False).head(20),
            x='State',
            y=['mp_count', 'bodies_count'],
            title="Top 20 States by Total Cases",
            labels={'value': 'Count', 'variable': 'Type'},
            color_discrete_map={'mp_count': '#00d4ff', 'bodies_count': '#ff4444'},
        )
        fig.update_layout(template='plotly_dark', height=450, xaxis_title="State", yaxis_title="Number of Cases")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Raw data not available for state-level chart.")


# ===================================================================
# PAGE 2 -- Raw Count Map
# ===================================================================

def page_raw_count_map():
    st.header("Geographic Distribution - Raw Counts")
    st.markdown(
        "**County-level map.** Dot size = number of cases. "
        "Blue = Missing Persons, Red = Unidentified Bodies."
    )

    df_mp, df_bodies = load_raw_data()
    if df_mp.empty and df_bodies.empty:
        st.warning("Raw data files not found.")
        return

    # Year range slider
    all_years = []
    if 'year' in df_mp.columns:
        all_years.extend(df_mp['year'].dropna().astype(int).tolist())
    if 'year' in df_bodies.columns:
        all_years.extend(df_bodies['year'].dropna().astype(int).tolist())
    if not all_years:
        st.warning("No year data available.")
        return

    min_yr, max_yr = int(min(all_years)), int(max(all_years))
    year_range = st.slider("Select Year Range", min_value=min_yr, max_value=max_yr, value=(2000, max_yr))

    # Filter
    mp_f = df_mp[(df_mp['year'] >= year_range[0]) & (df_mp['year'] <= year_range[1])] if not df_mp.empty else pd.DataFrame()
    bodies_f = df_bodies[(df_bodies['year'] >= year_range[0]) & (df_bodies['year'] <= year_range[1])] if not df_bodies.empty else pd.DataFrame()

    def _agg_and_map(df, count_col, color_scale, title):
        if df.empty or 'State' not in df.columns or 'County' not in df.columns:
            return
        agg = df.groupby(['State', 'County']).size().reset_index(name=count_col)
        agg['lat'] = agg['State'].apply(lambda s: STATE_COORDS.get(s, (None, None))[0])
        agg['lon'] = agg['State'].apply(lambda s: STATE_COORDS.get(s, (None, None))[1])
        agg = agg.dropna(subset=['lat', 'lon'])
        agg = agg[agg[count_col] > 0].sort_values(count_col, ascending=False).head(500)
        if agg.empty:
            return
        fig = px.scatter_geo(
            agg, lat='lat', lon='lon', size=count_col, color=count_col,
            hover_name='County',
            hover_data={'State': True, count_col: True, 'lat': False, 'lon': False},
            color_continuous_scale=color_scale, size_max=50,
            title=f"{title} ({year_range[0]}-{year_range[1]})", scope='usa',
        )
        fig.update_layout(
            template='plotly_dark', height=650,
            geo=dict(bgcolor='#0e1117', lakecolor='#0e1117', landcolor='#1e2127'),
        )
        st.plotly_chart(fig, use_container_width=True)

    _agg_and_map(mp_f, 'mp_count', 'Blues', 'Missing Persons by County')
    _agg_and_map(bodies_f, 'bodies_count', 'Reds', 'Unidentified Bodies by County')

    # Summary
    st.subheader("Summary Statistics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Missing Persons (filtered)", f"{len(mp_f):,}")
    with c2:
        st.metric("Unidentified Bodies (filtered)", f"{len(bodies_f):,}")
    with c3:
        st.metric("Total", f"{len(mp_f) + len(bodies_f):,}")


# ===================================================================
# PAGE 3 -- Sigma Heat Map
# ===================================================================

def page_sigma_heat_map():
    st.header("Statistical Outlier Heat Map (Sigma)")

    st.markdown(
        "Color intensity = z-score magnitude. "
        "Red = extreme (>3 sigma), Orange = significant (2-3 sigma), "
        "Yellow = moderate (1-2 sigma), Green = normal (<1 sigma)."
    )

    df_outliers = load_outlier_data()
    if df_outliers is None:
        st.warning("county_decade_outliers.csv not found.")
        return

    # Decade filter
    available_decades = sorted(df_outliers['decade'].unique())
    selected_decades = st.multiselect("Filter by Decade (leave empty for all)", options=available_decades, default=[])
    df_f = df_outliers[df_outliers['decade'].isin(selected_decades)] if selected_decades else df_outliers.copy()

    view_level = st.radio("View Level", ["State", "County"], horizontal=True)
    metric = st.radio("Metric", ["MP z-score", "Bodies z-score", "Combined (max)"], horizontal=True)

    if metric == "MP z-score":
        zcol = 'mp_z_score'
        cscale = 'Blues'
    elif metric == "Bodies z-score":
        zcol = 'bodies_z_score'
        cscale = 'Reds'
    else:
        df_f = df_f.copy()
        df_f['max_z'] = df_f[['mp_z_score', 'bodies_z_score']].max(axis=1)
        zcol = 'max_z'
        cscale = 'YlOrRd'

    decade_label = f" ({', '.join(map(str, selected_decades))})" if selected_decades else " (All Decades)"

    if view_level == "State":
        state_agg = df_f.groupby('State')[zcol].max().reset_index()
        state_agg['lat'] = state_agg['State'].apply(lambda s: STATE_COORDS.get(s, (None, None))[0])
        state_agg['lon'] = state_agg['State'].apply(lambda s: STATE_COORDS.get(s, (None, None))[1])
        state_agg = state_agg.dropna(subset=['lat', 'lon'])
        state_agg['size_val'] = state_agg[zcol].apply(lambda x: max(abs(x), 0.1))

        fig = px.scatter_geo(
            state_agg, lat='lat', lon='lon', size='size_val', color=zcol,
            hover_name='State',
            hover_data={zcol: ':.2f', 'size_val': False, 'lat': False, 'lon': False},
            color_continuous_scale=cscale, size_max=80,
            title=f"State-Level {metric}{decade_label}", scope='usa',
        )
    else:
        county_df = df_f.copy()
        county_df['lat'] = county_df['State'].apply(lambda s: STATE_COORDS.get(s, (None, None))[0])
        county_df['lon'] = county_df['State'].apply(lambda s: STATE_COORDS.get(s, (None, None))[1])
        county_df = county_df.dropna(subset=['lat', 'lon'])
        county_df = county_df[county_df[zcol] > 1]
        county_df = county_df.sort_values(zcol, ascending=False).head(500)
        county_df['size_val'] = county_df[zcol]

        fig = px.scatter_geo(
            county_df, lat='lat', lon='lon', size='size_val', color=zcol,
            hover_name='County',
            hover_data={
                'State': True, zcol: ':.2f',
                'missing_count': True, 'bodies_count': True,
                'decade': True, 'alert_level': True,
                'size_val': False, 'lat': False, 'lon': False,
            },
            color_continuous_scale=cscale, size_max=60,
            title=f"County-Level {metric} (Outliers, z>1){decade_label}", scope='usa',
        )

    fig.update_layout(
        template='plotly_dark', height=700,
        geo=dict(bgcolor='#0e1117', lakecolor='#0e1117', landcolor='#1e2127',
                 showland=True, showcountries=True, countrycolor='#444444'),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown(
        "**Color Scale:** "
        "Green/Low (<1 sigma) = Normal | "
        "Yellow (1-2 sigma) = Moderate | "
        "Orange (2-3 sigma) = Significant | "
        "Red/High (>3 sigma) = Extreme"
    )

    # Top outliers table
    st.subheader(f"Top 20 Outliers by {metric}")
    top = df_f.sort_values(zcol, ascending=False).head(20)
    display_cols = [c for c in ['State', 'County', 'decade', 'missing_count', 'bodies_count',
                                'mp_z_score', 'bodies_z_score', 'alert_level'] if c in top.columns]
    st.dataframe(top[display_cols].reset_index(drop=True), use_container_width=True, height=400)


# ===================================================================
# PAGE 4 -- Temporal Trends
# ===================================================================

def page_temporal_trends():
    st.header("Temporal Trends Analysis")

    df_mp, df_bodies = load_raw_data()
    if df_mp.empty and df_bodies.empty:
        st.warning("Raw data not available.")
        return

    # Year slider
    all_years = []
    if 'year' in df_mp.columns:
        all_years.extend(df_mp['year'].dropna().astype(int).tolist())
    if 'year' in df_bodies.columns:
        all_years.extend(df_bodies['year'].dropna().astype(int).tolist())
    if not all_years:
        st.warning("No year data.")
        return

    min_yr, max_yr = int(min(all_years)), int(max(all_years))
    year_range = st.slider("Year Range", min_value=min_yr, max_value=max_yr, value=(1980, max_yr))

    mp_f = df_mp[(df_mp['year'] >= year_range[0]) & (df_mp['year'] <= year_range[1])] if not df_mp.empty else pd.DataFrame()
    bodies_f = df_bodies[(df_bodies['year'] >= year_range[0]) & (df_bodies['year'] <= year_range[1])] if not df_bodies.empty else pd.DataFrame()

    mp_by_year = mp_f.groupby('year').size().reset_index(name='mp_count') if not mp_f.empty else pd.DataFrame(columns=['year', 'mp_count'])
    bodies_by_year = bodies_f.groupby('year').size().reset_index(name='bodies_count') if not bodies_f.empty else pd.DataFrame(columns=['year', 'bodies_count'])
    trends = pd.merge(mp_by_year, bodies_by_year, on='year', how='outer').fillna(0).sort_values('year')

    # National trend lines (dual axis)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=trends['year'], y=trends['mp_count'], name="Missing Persons",
                   line=dict(color='#00d4ff', width=3), mode='lines+markers'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=trends['year'], y=trends['bodies_count'], name="Unidentified Bodies",
                   line=dict(color='#ff4444', width=3), mode='lines+markers'),
        secondary_y=True,
    )
    fig.update_layout(title="National Trends Over Time", template='plotly_dark', height=500, hovermode='x unified')
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Missing Persons", secondary_y=False, color='#00d4ff')
    fig.update_yaxes(title_text="Unidentified Bodies", secondary_y=True, color='#ff4444')
    st.plotly_chart(fig, use_container_width=True)

    # Decade comparison
    st.subheader("Decade Comparison")
    if not mp_f.empty:
        mp_f = mp_f.copy()
        mp_f['decade'] = (mp_f['year'] // 10) * 10
    if not bodies_f.empty:
        bodies_f = bodies_f.copy()
        bodies_f['decade'] = (bodies_f['year'] // 10) * 10

    mp_dec = mp_f.groupby('decade').size().reset_index(name='mp_count') if not mp_f.empty else pd.DataFrame(columns=['decade', 'mp_count'])
    bod_dec = bodies_f.groupby('decade').size().reset_index(name='bodies_count') if not bodies_f.empty else pd.DataFrame(columns=['decade', 'bodies_count'])
    decades = pd.merge(mp_dec, bod_dec, on='decade', how='outer').fillna(0)
    decades = decades[decades['decade'] >= 1980]

    fig2 = go.Figure(data=[
        go.Bar(name='Missing Persons', x=decades['decade'], y=decades['mp_count'], marker_color='#00d4ff'),
        go.Bar(name='Unidentified Bodies', x=decades['decade'], y=decades['bodies_count'], marker_color='#ff4444'),
    ])
    fig2.update_layout(title="Cases by Decade", template='plotly_dark', barmode='group', height=400)
    st.plotly_chart(fig2, use_container_width=True)


# ===================================================================
# PAGE 5 -- Outlier Detection
# ===================================================================

def page_outlier_detection():
    st.header("Statistical Outlier Detection")

    df_outliers = load_outlier_data()
    if df_outliers is None:
        st.warning("county_decade_outliers.csv not found. Run the analysis pipeline first.")
        return

    st.markdown(
        "**Methodology:** Z-score based classification\n\n"
        "- **RED**: >3 sigma (99.7%+ confidence)\n"
        "- **ORANGE**: >2 sigma (95%+ confidence)\n"
        "- **YELLOW**: >1 sigma (68%+ confidence)\n"
        "- **GREEN**: <1 sigma (normal variation)"
    )

    # Alert distribution
    alert_counts = df_outliers['alert_level'].value_counts()
    total = len(df_outliers)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        v = alert_counts.get('RED', 0)
        st.metric("RED Alerts", v, f"{v / total * 100:.1f}%")
    with c2:
        v = alert_counts.get('ORANGE', 0)
        st.metric("ORANGE Alerts", v, f"{v / total * 100:.1f}%")
    with c3:
        v = alert_counts.get('YELLOW', 0)
        st.metric("YELLOW Alerts", v, f"{v / total * 100:.1f}%")
    with c4:
        v = alert_counts.get('GREEN', 0)
        st.metric("GREEN", v, f"{v / total * 100:.1f}%")

    st.markdown("---")

    # Top outliers by composite_z_score
    st.subheader("Top 20 Outliers by Composite Z-Score")
    top = df_outliers.sort_values('composite_z_score', ascending=False).head(20)
    display_cols = [c for c in ['State', 'County', 'decade', 'missing_count', 'bodies_count',
                                'mp_z_score', 'bodies_z_score', 'composite_z_score',
                                'alert_level', 'alert_reason'] if c in top.columns]
    st.dataframe(top[display_cols].reset_index(drop=True), use_container_width=True, height=500)

    # Scatter
    fig = px.scatter(
        top, x='mp_z_score', y='bodies_z_score', size='composite_z_score',
        color='alert_level', hover_data=['County', 'State', 'decade'],
        title="Outlier Distribution (Top 20)",
        color_discrete_map={'RED': '#ff0000', 'ORANGE': '#ff8800', 'YELLOW': '#ffcc00', 'GREEN': '#00ff00'},
    )
    fig.update_layout(template='plotly_dark', height=500,
                      xaxis_title="Missing Persons Z-Score", yaxis_title="Bodies Z-Score")
    st.plotly_chart(fig, use_container_width=True)

    # FDR correction results
    st.markdown("---")
    st.subheader("FDR Correction Results")
    if 'fdr_significant' in df_outliers.columns:
        fdr_true = df_outliers['fdr_significant'].sum()
        fdr_false = total - fdr_true
        st.write(f"**FDR Significant:** {fdr_true} counties ({fdr_true / total * 100:.1f}%)")
        st.write(f"**Not FDR Significant:** {fdr_false} counties ({fdr_false / total * 100:.1f}%)")

        # Show top FDR-significant counties
        fdr_sig = df_outliers[df_outliers['fdr_significant'] == True].sort_values('composite_z_score', ascending=False).head(20)
        if not fdr_sig.empty:
            fdr_cols = [c for c in ['State', 'County', 'decade', 'mp_adjusted_p', 'bodies_adjusted_p',
                                    'composite_z_score', 'alert_level'] if c in fdr_sig.columns]
            st.dataframe(fdr_sig[fdr_cols].reset_index(drop=True), use_container_width=True)
    else:
        st.info("FDR columns not found in data.")


# ===================================================================
# PAGE 6 -- Zone Forecasting
# ===================================================================

def page_zone_forecasting():
    st.header("Geographic Zone Forecasting")

    df_forecasts = load_zone_forecasts()
    df_trends = load_zone_trends()

    if df_forecasts is None and df_trends is None:
        st.warning("zone_forecasts.csv and zone_trends.csv not found.")
        return

    available_zones = list(ZONES.keys())
    selected_zone = st.selectbox("Select Zone", available_zones)

    # Historical data
    st.subheader(f"{selected_zone} - Historical Trends + Forecast")

    hist = df_trends[df_trends['zone'] == selected_zone] if df_trends is not None else pd.DataFrame()
    fcast = df_forecasts[df_forecasts['zone'] == selected_zone] if df_forecasts is not None else pd.DataFrame()

    for metric_label, hist_col, fc_col, fc_lo, fc_hi, color in [
        ("Missing Persons", "mp_count", "mp_forecast", "mp_lower_95", "mp_upper_95", "#00d4ff"),
        ("Unidentified Bodies", "bodies_count", "bodies_forecast", "bodies_lower_95", "bodies_upper_95", "#ff4444"),
    ]:
        fig = go.Figure()
        # Historical
        if not hist.empty and hist_col in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist['year'], y=hist[hist_col], name=f'Historical {metric_label}',
                mode='lines+markers', line=dict(color=color, width=3),
            ))
        # Forecast + CI
        if not fcast.empty and fc_col in fcast.columns:
            fig.add_trace(go.Scatter(
                x=fcast['year'], y=fcast[fc_col], name=f'Forecast {metric_label}',
                mode='lines+markers', line=dict(color=color, width=3, dash='dot'),
            ))
            if fc_lo in fcast.columns and fc_hi in fcast.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([fcast['year'], fcast['year'][::-1]]),
                    y=pd.concat([fcast[fc_hi], fcast[fc_lo][::-1]]),
                    fill='toself', fillcolor=f'rgba({",".join(str(int(color.lstrip("#")[i:i+2], 16)) for i in (0,2,4))},0.15)',
                    line=dict(color='rgba(0,0,0,0)'), name='95% CI', showlegend=True,
                ))

        fig.update_layout(
            title=f"{selected_zone} - {metric_label}", template='plotly_dark',
            height=450, xaxis_title="Year", yaxis_title="Count", hovermode='x unified',
        )
        st.plotly_chart(fig, use_container_width=True)

    # Slope / R2 info
    if not fcast.empty:
        st.subheader("Linear Trend Statistics")
        slope_cols = [c for c in ['mp_slope', 'bodies_slope', 'mp_r2', 'bodies_r2'] if c in fcast.columns]
        if slope_cols:
            row = fcast.iloc[0]
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("MP Slope", f"{row.get('mp_slope', 0):+.2f}/yr")
            with sc2:
                st.metric("MP R-squared", f"{row.get('mp_r2', 0):.3f}")
            with sc3:
                st.metric("Bodies Slope", f"{row.get('bodies_slope', 0):+.2f}/yr")
            with sc4:
                st.metric("Bodies R-squared", f"{row.get('bodies_r2', 0):.3f}")

    # Forecast table
    if not fcast.empty:
        st.subheader("Forecast Data")
        st.dataframe(fcast.reset_index(drop=True), use_container_width=True)


# ===================================================================
# PAGE 7 -- Validation
# ===================================================================

def page_validation():
    st.header("System Validation - Known Serial Killers")

    st.markdown("Testing the anomaly detection system against **known serial killers** to validate accuracy.")

    df_outliers = load_outlier_data()
    if df_outliers is None:
        st.warning("Outlier data not available.")
        return

    tests = [
        {
            'name': 'Gary Ridgway (Green River Killer)',
            'state': 'WA', 'county': 'King', 'decade': 1980,
            'expected': 'Should detect - dumped bodies, active 1982-1998',
        },
        {
            'name': 'John Wayne Gacy',
            'state': 'IL', 'county': 'Cook', 'decade': 1970,
            'expected': 'Should detect - 33 victims buried on property',
        },
        {
            'name': 'Jeffrey Dahmer',
            'state': 'WI', 'county': 'Milwaukee', 'decade': 1980,
            'expected': 'May not detect - destroyed bodies (validates multi-tier design)',
        },
        {
            'name': 'Ted Bundy (Washington period)',
            'state': 'WA', 'county': 'King', 'decade': 1970,
            'expected': 'Should detect - multiple victims in King County area',
        },
        {
            'name': 'Samuel Little',
            'state': 'CA', 'county': 'Los Angeles', 'decade': 1980,
            'expected': 'Should detect - prolific serial killer, 93 confirmed victims',
        },
    ]

    for test in tests:
        st.subheader(test['name'])
        match = df_outliers[
            (df_outliers['State'] == test['state']) &
            (df_outliers['County'] == test['county']) &
            (df_outliers['decade'] == test['decade'])
        ]

        c1, c2, c3 = st.columns(3)
        if len(match) > 0:
            row = match.iloc[0]
            with c1:
                st.metric("Missing Persons", int(row.get('missing_count', 0)),
                          f"z={row.get('mp_z_score', 0):.2f}")
            with c2:
                st.metric("Unidentified Bodies", int(row.get('bodies_count', 0)),
                          f"z={row.get('bodies_z_score', 0):.2f}")
            with c3:
                st.metric("Alert Level", str(row.get('alert_level', 'N/A')))

            z_max = max(abs(row.get('mp_z_score', 0)), abs(row.get('bodies_z_score', 0)))
            if z_max > 1:
                st.success(f"**DETECTED** as statistical outlier (max |z| = {z_max:.2f})")
            else:
                st.info(f"Not flagged -- {test['expected']}")
        else:
            st.error("No data found for this location / time period")
        st.markdown("---")


# ===================================================================
# PAGE 8 -- Robust Statistics (NEW)
# ===================================================================

def page_robust_statistics():
    st.header("Robust Statistics & FDR Correction")

    df_outliers = load_outlier_data()
    if df_outliers is None:
        st.warning("county_decade_outliers.csv not found.")
        return

    # --- Standard vs Robust Z-scores ---
    st.subheader("Standard vs Robust Z-Scores")
    st.markdown(
        "Robust z-scores use median / MAD instead of mean / std, "
        "making them resistant to extreme outliers pulling the distribution."
    )

    for label, std_col, robust_col, color in [
        ("Missing Persons", "mp_z_score", "mp_robust_z", "#00d4ff"),
        ("Unidentified Bodies", "bodies_z_score", "bodies_robust_z", "#ff4444"),
    ]:
        if std_col in df_outliers.columns and robust_col in df_outliers.columns:
            sample = df_outliers.dropna(subset=[std_col, robust_col])
            if len(sample) > 2000:
                sample = sample.sample(2000, random_state=42)
            fig = px.scatter(
                sample, x=std_col, y=robust_col, color='alert_level',
                color_discrete_map={'RED': '#ff0000', 'ORANGE': '#ff8800', 'YELLOW': '#ffcc00', 'GREEN': '#00ff00'},
                title=f"{label}: Standard vs Robust Z-Score",
                hover_data=['State', 'County', 'decade'],
            )
            fig.add_shape(type='line', x0=sample[std_col].min(), x1=sample[std_col].max(),
                          y0=sample[std_col].min(), y1=sample[std_col].max(),
                          line=dict(color='white', dash='dash'))
            fig.update_layout(template='plotly_dark', height=450,
                              xaxis_title=f"Standard Z ({std_col})",
                              yaxis_title=f"Robust Z ({robust_col})")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- FDR Correction Results ---
    st.subheader("FDR Correction Summary")
    if 'fdr_significant' in df_outliers.columns:
        total = len(df_outliers)
        fdr_true = int(df_outliers['fdr_significant'].sum())
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total County-Decades", total)
        with c2:
            st.metric("FDR Significant", fdr_true)
        with c3:
            st.metric("FDR Rate", f"{fdr_true / total * 100:.1f}%")

        # Cross-tab: alert_level vs fdr_significant
        st.write("**Alert Level vs FDR Significance:**")
        xtab = pd.crosstab(df_outliers['alert_level'], df_outliers['fdr_significant'], margins=True)
        xtab.columns = [str(c) for c in xtab.columns]
        st.dataframe(xtab, use_container_width=True)

    st.markdown("---")

    # --- Empirical Bayes Shrinkage ---
    st.subheader("Empirical Bayes Shrinkage Effect")
    st.markdown(
        "Small counties get shrunk toward the global mean. "
        "The shrinkage_weight column (0=full shrinkage, 1=no shrinkage) controls the degree."
    )
    if 'shrinkage_weight' in df_outliers.columns and 'missing_per_100k' in df_outliers.columns and 'mp_rate_shrunk' in df_outliers.columns:
        sample = df_outliers.dropna(subset=['missing_per_100k', 'mp_rate_shrunk', 'shrinkage_weight'])
        if len(sample) > 2000:
            sample = sample.sample(2000, random_state=42)
        fig = px.scatter(
            sample, x='missing_per_100k', y='mp_rate_shrunk',
            color='shrinkage_weight', color_continuous_scale='Viridis',
            title="Raw MP Rate vs Shrunk MP Rate (color = shrinkage weight)",
            hover_data=['State', 'County', 'decade', 'population'],
        )
        fig.add_shape(type='line',
                      x0=0, x1=sample['missing_per_100k'].quantile(0.99),
                      y0=0, y1=sample['missing_per_100k'].quantile(0.99),
                      line=dict(color='white', dash='dash'))
        fig.update_layout(template='plotly_dark', height=450,
                          xaxis_title="Raw Missing per 100k",
                          yaxis_title="Shrunk Missing per 100k")
        st.plotly_chart(fig, use_container_width=True)

        # Small county flag
        if 'small_county_flag' in df_outliers.columns:
            small_count = int(df_outliers['small_county_flag'].sum())
            st.write(f"**Small county flag count:** {small_count} out of {len(df_outliers)} "
                     f"({small_count / len(df_outliers) * 100:.1f}%)")

    st.markdown("---")

    # --- Distribution Diagnostics ---
    st.subheader("Distribution Diagnostics")
    for col_name, label in [('mp_z_score', 'MP Z-Score'), ('bodies_z_score', 'Bodies Z-Score'),
                            ('mp_robust_z', 'MP Robust Z'), ('bodies_robust_z', 'Bodies Robust Z')]:
        if col_name in df_outliers.columns:
            vals = df_outliers[col_name].dropna()
            fig = px.histogram(vals, nbins=80, title=f"Distribution: {label}",
                               labels={'value': label, 'count': 'Frequency'})
            fig.update_layout(template='plotly_dark', height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# PAGE 9 -- Spatial Analysis (NEW)
# ===================================================================

def page_spatial_analysis():
    st.header("Spatial Autocorrelation Analysis")

    # --- Global Moran's I ---
    st.subheader("Global Moran's I")
    df_gm = load_global_morans()
    if df_gm is not None:
        st.markdown("Global spatial autocorrelation indicates whether similar values cluster geographically.")
        st.dataframe(df_gm.reset_index(drop=True), use_container_width=True)

        # Highlight significant
        if 'p_value' in df_gm.columns:
            sig = df_gm[df_gm['p_value'] < 0.05]
            st.write(f"**Significant (p < 0.05):** {len(sig)} out of {len(df_gm)} tests")
    else:
        st.info("global_morans_i.csv not found.")

    st.markdown("---")

    # --- LISA Clusters ---
    st.subheader("Local Spatial Autocorrelation (LISA)")
    df_lisa = load_spatial_autocorrelation()
    if df_lisa is not None:
        # Decade filter
        decades = sorted(df_lisa['decade'].unique())
        sel_decade = st.selectbox("Decade", decades, index=len(decades) - 1 if decades else 0)
        df_dec = df_lisa[df_lisa['decade'] == sel_decade]

        metric_choice = st.radio("Metric", ["Missing Persons", "Bodies"], horizontal=True)
        cluster_col = 'mp_lisa_cluster' if metric_choice == "Missing Persons" else 'bodies_lisa_cluster'
        p_col = 'mp_lisa_p' if metric_choice == "Missing Persons" else 'bodies_lisa_p'

        # Filter to significant
        sig_df = df_dec[df_dec[p_col] < 0.05].copy() if p_col in df_dec.columns else df_dec.copy()

        # Cluster distribution
        if cluster_col in sig_df.columns:
            cluster_counts = sig_df[cluster_col].value_counts()
            st.write("**Significant LISA Cluster Distribution:**")
            for ctype in ['HH', 'HL', 'LH', 'LL', 'NS']:
                cnt = cluster_counts.get(ctype, 0)
                st.write(f"  - {ctype}: {cnt}")

            # Bar chart
            fig = px.bar(
                x=cluster_counts.index, y=cluster_counts.values,
                title=f"LISA Cluster Types ({metric_choice}, {sel_decade}, p<0.05)",
                labels={'x': 'Cluster Type', 'y': 'Count'},
                color=cluster_counts.index,
                color_discrete_map={'HH': '#ff0000', 'HL': '#ff8800', 'LH': '#4488ff', 'LL': '#00cc00', 'NS': '#888888'},
            )
            fig.update_layout(template='plotly_dark', height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Top HH clusters
            hh = sig_df[sig_df[cluster_col] == 'HH'].copy()
            if not hh.empty:
                st.subheader("Top HH (High-High) Clusters")
                st.markdown("Counties with high values surrounded by other high-value neighbors (hotspots).")
                i_col = 'mp_lisa_i' if metric_choice == "Missing Persons" else 'bodies_lisa_i'
                hh_sorted = hh.sort_values(i_col, ascending=False).head(20)
                show_cols = [c for c in ['State', 'County', 'decade', i_col, p_col] if c in hh_sorted.columns]
                st.dataframe(hh_sorted[show_cols].reset_index(drop=True), use_container_width=True)
            else:
                st.info("No HH clusters found for this selection.")
    else:
        st.info("spatial_autocorrelation.csv not found.")


# ===================================================================
# PAGE 10 -- Advanced Forecasting (NEW)
# ===================================================================

def page_advanced_forecasting():
    st.header("Advanced Forecasting (ARIMA & Diagnostics)")

    # --- ARIMA vs Linear comparison ---
    st.subheader("ARIMA vs Linear Model Comparison")
    df_bt = load_forecast_backtest()
    if df_bt is not None:
        st.dataframe(df_bt.reset_index(drop=True), use_container_width=True)
        if 'better_model' in df_bt.columns:
            model_counts = df_bt['better_model'].value_counts()
            st.write("**Better Model Counts:**")
            for m, c in model_counts.items():
                st.write(f"  - {m}: {c}")
    else:
        st.info("forecast_backtest.csv not found.")

    st.markdown("---")

    # --- Mann-Kendall & Structural Breaks ---
    st.subheader("Temporal Trend Tests (Mann-Kendall & Structural Breaks)")
    df_tt = load_temporal_trends()
    if df_tt is not None:
        st.dataframe(df_tt.reset_index(drop=True), use_container_width=True)

        # Highlight significant trends
        if 'mk_trend' in df_tt.columns:
            st.write("**Mann-Kendall Trend Summary:**")
            trend_counts = df_tt['mk_trend'].value_counts()
            for t, c in trend_counts.items():
                st.write(f"  - {t}: {c}")

        # Structural breaks
        if 'break_year' in df_tt.columns and 'break_significance' in df_tt.columns:
            breaks = df_tt[df_tt['break_significance'] == True]
            if not breaks.empty:
                st.write("**Significant Structural Breaks:**")
                for _, r in breaks.iterrows():
                    st.write(f"  - {r.get('zone', '?')} / {r.get('metric', '?')}: break at year {r.get('break_year', '?')}")
    else:
        st.info("temporal_trends.csv not found.")

    st.markdown("---")

    # --- ARIMA Forecasts with CIs ---
    st.subheader("ARIMA Forecasts with Confidence Intervals")
    df_arima = load_arima_forecasts()
    if df_arima is not None:
        zones_in_data = df_arima['zone'].unique().tolist()
        sel_zone = st.selectbox("Zone", zones_in_data, key="arima_zone")
        zone_data = df_arima[df_arima['zone'] == sel_zone]

        metrics_in_data = zone_data['metric'].unique().tolist()
        sel_metric = st.selectbox("Metric", metrics_in_data, key="arima_metric")
        metric_data = zone_data[zone_data['metric'] == sel_metric].sort_values('year')

        if not metric_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=metric_data['year'], y=metric_data['arima_forecast'],
                name='ARIMA Forecast', mode='lines+markers',
                line=dict(color='#00d4ff', width=3),
            ))
            if 'arima_lower_95' in metric_data.columns and 'arima_upper_95' in metric_data.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([metric_data['year'], metric_data['year'][::-1]]),
                    y=pd.concat([metric_data['arima_upper_95'], metric_data['arima_lower_95'][::-1]]),
                    fill='toself', fillcolor='rgba(0,212,255,0.15)',
                    line=dict(color='rgba(0,0,0,0)'), name='95% CI',
                ))
            fig.update_layout(
                title=f"ARIMA Forecast: {sel_zone} - {sel_metric}",
                template='plotly_dark', height=450,
                xaxis_title="Year", yaxis_title="Forecast Value", hovermode='x unified',
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show order and AIC
            if 'arima_order' in metric_data.columns and 'arima_aic' in metric_data.columns:
                row = metric_data.iloc[0]
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("ARIMA Order", str(row.get('arima_order', 'N/A')))
                with c2:
                    st.metric("AIC", f"{row.get('arima_aic', 0):.2f}")

            st.dataframe(metric_data.reset_index(drop=True), use_container_width=True)
    else:
        st.info("arima_forecasts.csv not found.")


# ===================================================================
# PAGE 11 -- ML Anomaly Detection (NEW)
# ===================================================================

def page_ml_anomaly():
    st.header("ML-Based Anomaly Detection")

    # ---- ML Anomaly Scores ----
    st.subheader("Ensemble Anomaly Scores")
    df_ml = load_ml_anomaly_scores()
    if df_ml is not None:
        # Classification distribution
        if 'ml_classification' in df_ml.columns:
            class_counts = df_ml['ml_classification'].value_counts()
            fig = px.bar(
                x=class_counts.index, y=class_counts.values,
                title="ML Classification Distribution",
                labels={'x': 'Classification', 'y': 'Count'},
                color=class_counts.index,
                color_discrete_map={'Anomalous': '#ff4444', 'Normal': '#00cc00', 'Borderline': '#ffcc00'},
            )
            fig.update_layout(template='plotly_dark', height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Top anomalies
        st.write("**Top 20 ML Anomalies (by ensemble_score):**")
        if 'ensemble_score' in df_ml.columns:
            top_anom = df_ml.sort_values('ensemble_score', ascending=False).head(20)
            show_cols = [c for c in ['State', 'County', 'state_name', 'decade',
                                     'missing_per_100k', 'bodies_per_100k',
                                     'if_score', 'lof_score', 'ensemble_score',
                                     'ml_classification'] if c in top_anom.columns]
            st.dataframe(top_anom[show_cols].reset_index(drop=True), use_container_width=True)

        # Concordance table
        if 'is_concordant' in df_ml.columns and 'statistical_alert' in df_ml.columns and 'ml_classification' in df_ml.columns:
            st.markdown("---")
            st.subheader("Statistical vs ML Concordance")
            conc_count = int(df_ml['is_concordant'].sum())
            total = len(df_ml)
            st.write(f"**Concordant:** {conc_count} / {total} ({conc_count / total * 100:.1f}%)")

            xtab = pd.crosstab(df_ml['statistical_alert'], df_ml['ml_classification'], margins=True)
            xtab.columns = [str(c) for c in xtab.columns]
            st.dataframe(xtab, use_container_width=True)
    else:
        st.info("ml_anomaly_scores.csv not found.")

    st.markdown("---")

    # ---- County Clusters ----
    st.subheader("County Cluster Profiles")
    df_clust = load_county_clusters()
    if df_clust is not None:
        # Cluster profiles
        for clust_col in ['dbscan_cluster', 'hierarchical_cluster']:
            if clust_col in df_clust.columns:
                st.write(f"**{clust_col} distribution:**")
                cc = df_clust[clust_col].value_counts().sort_index()
                for k, v in cc.items():
                    label = f"Cluster {k}" if k != -1 else "Noise (DBSCAN -1)"
                    st.write(f"  - {label}: {v} counties")

        # Cluster scatter
        if 'mean_mp_rate' in df_clust.columns and 'mean_bodies_rate' in df_clust.columns and 'dbscan_cluster' in df_clust.columns:
            plot_df = df_clust.copy()
            plot_df['dbscan_cluster'] = plot_df['dbscan_cluster'].astype(str)
            fig = px.scatter(
                plot_df, x='mean_mp_rate', y='mean_bodies_rate',
                color='dbscan_cluster',
                hover_data=['State', 'County', 'state_name'],
                title="County Clusters (DBSCAN) - MP Rate vs Bodies Rate",
            )
            fig.update_layout(template='plotly_dark', height=500,
                              xaxis_title="Mean MP Rate per 100k",
                              yaxis_title="Mean Bodies Rate per 100k")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("county_clusters.csv not found.")

    st.markdown("---")

    # ---- Covariate-Adjusted (RF Overperformance) ----
    st.subheader("Covariate-Adjusted Outliers (Random Forest)")
    df_cov = load_covariate_adjusted()
    if df_cov is not None:
        st.markdown(
            "Counties that have **higher** actual rates than predicted by socioeconomic covariates "
            "(poverty, income, unemployment, etc.) may indicate non-random causes."
        )

        # Top RF overperformance
        if 'rf_overperformance_rank' in df_cov.columns:
            top_rf = df_cov.sort_values('rf_overperformance_rank').head(20)
            show_cols = [c for c in ['State', 'County', 'state_name', 'decade',
                                     'missing_per_100k', 'bodies_per_100k',
                                     'rf_residual_mp_z', 'rf_residual_bodies_z',
                                     'adj_anomaly', 'rf_overperformance_rank'] if c in top_rf.columns]
            st.write("**Top 20 Counties by RF Overperformance Rank:**")
            st.dataframe(top_rf[show_cols].reset_index(drop=True), use_container_width=True)

        # Feature importance proxy: show distributions of residuals
        st.markdown("---")
        st.write("**Residual Distributions (RF model):**")
        for col, label in [('rf_residual_mp_z', 'RF Residual MP Z'),
                           ('rf_residual_bodies_z', 'RF Residual Bodies Z')]:
            if col in df_cov.columns:
                vals = df_cov[col].dropna()
                fig = px.histogram(vals, nbins=60, title=f"Distribution: {label}",
                                   labels={'value': label, 'count': 'Frequency'})
                fig.update_layout(template='plotly_dark', height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Adjusted anomaly breakdown
        if 'adj_anomaly' in df_cov.columns:
            adj_counts = df_cov['adj_anomaly'].value_counts()
            st.write("**Covariate-Adjusted Anomaly Classification:**")
            for k, v in adj_counts.items():
                st.write(f"  - {k}: {v}")
    else:
        st.info("covariate_adjusted_outliers.csv not found.")


# ===================================================================
# ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    main()
