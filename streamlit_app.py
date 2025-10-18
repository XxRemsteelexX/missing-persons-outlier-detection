#!/usr/bin/env python3
"""
Geospatial Crime Analysis Dashboard
Interactive visualization for FBI Field Specialist Application
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# Page config
st.set_page_config(
    page_title="FBI Crime Pattern Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths - use relative paths for portability
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2127;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #00d4ff;
    }
    h2 {
        color: #00a8cc;
    }
    .alert-red {
        background-color: #ff4444;
        padding: 10px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    .alert-orange {
        background-color: #ff8800;
        padding: 10px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    .alert-yellow {
        background-color: #ffcc00;
        padding: 10px;
        border-radius: 5px;
        color: black;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_all_data():
    """Load all missing persons and bodies data"""
    mp_data = []
    bodies_data = []

    for file in os.listdir(RAW_DIR):
        if file.endswith('_missing_persons.csv'):
            df = pd.read_csv(os.path.join(RAW_DIR, file))
            df['data_type'] = 'Missing Persons'
            mp_data.append(df)
        elif file.endswith('_unidentified_bodies.csv'):
            df = pd.read_csv(os.path.join(RAW_DIR, file))
            df['data_type'] = 'Unidentified Bodies'
            bodies_data.append(df)

    df_mp = pd.concat(mp_data, ignore_index=True)
    df_bodies = pd.concat(bodies_data, ignore_index=True)

    # Extract years
    df_mp['year'] = pd.to_datetime(df_mp['DLC'], errors='coerce').dt.year
    df_bodies['year'] = pd.to_datetime(df_bodies['DBF'], errors='coerce').dt.year

    return df_mp, df_bodies

@st.cache_data
def load_outlier_data():
    """Load precomputed outlier scores"""
    outlier_file = os.path.join(ANALYSIS_DIR, "county_decade_outliers.csv")
    if os.path.exists(outlier_file):
        return pd.read_csv(outlier_file)
    return None

def main():
    # Header
    st.title("Geospatial Crime Pattern Analysis System")
    st.markdown("**Multi-Tier Anomaly Detection for Serial Crime & Trafficking Networks**")
    st.markdown("---")

    # Load data
    with st.spinner("Loading data..."):
        df_mp, df_bodies = load_all_data()
        df_outliers = load_outlier_data()

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["Overview", "Raw Count Map", "Std Dev Heat Map",
         "Temporal Trends", "Outlier Detection",
         "Zone Forecasting", "Validation"]
    )

    if page == "Overview":
        show_overview(df_mp, df_bodies, df_outliers)
    elif page == "Raw Count Map":
        show_raw_count_map(df_mp, df_bodies)
    elif page == "Std Dev Heat Map":
        show_stddev_heat_map(df_mp, df_bodies, df_outliers)
    elif page == "Temporal Trends":
        show_temporal(df_mp, df_bodies)
    elif page == "Outlier Detection":
        show_outliers(df_outliers)
    elif page == "Zone Forecasting":
        show_forecasting(df_mp, df_bodies)
    elif page == "Validation":
        show_validation(df_outliers)

def show_overview(df_mp, df_bodies, df_outliers):
    """Overview dashboard"""
    st.header("System Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Missing Persons", f"{len(df_mp):,}", delta="55 States")
    with col2:
        st.metric("Unidentified Bodies", f"{len(df_bodies):,}", delta="54 States")
    with col3:
        st.metric("Total Cases", f"{len(df_mp) + len(df_bodies):,}")
    with col4:
        if df_outliers is not None:
            orange_alerts = (df_outliers['alert'] == 'ORANGE').sum()
            st.metric("Active Alerts", f"{orange_alerts:,}", delta="2.9% of counties", delta_color="inverse")

    st.markdown("---")

    # Top findings
    st.subheader("Critical Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="alert-red">I-35 CORRIDOR CRISIS</div>', unsafe_allow_html=True)
        st.write("**Accelerating trend:** +10.80 MP/year (recent)")
        st.write("**2020s:** 521 missing persons (vs 193 in 2010s)")
        st.write("**Status:** Human trafficking superhighway")

        st.markdown('<div class="alert-orange">TEXAS BORDER SURGE</div>', unsafe_allow_html=True)
        st.write("**2010s → 2020s:** +81% increase (727 → 1,317 MP)")
        st.write("**Only border state worsening**")

    with col2:
        st.markdown('<div class="alert-orange">PIMA COUNTY, AZ</div>', unsafe_allow_html=True)
        st.write("**Statistical outlier:** 44.75σ (529 bodies)")
        st.write("**Pattern:** Border/cartel dumping ground")
        st.write("**Unreported victims:** High bodies, low MP")

        st.success("CA/AZ Border Improving: -30% and -48% respectively")

    st.markdown("---")

    # Data coverage map
    st.subheader("Data Coverage")

    # Aggregate by state
    mp_by_state = df_mp.groupby('State').size().reset_index(name='mp_count')
    bodies_by_state = df_bodies.groupby('State').size().reset_index(name='bodies_count')

    state_summary = pd.merge(mp_by_state, bodies_by_state, on='State', how='outer').fillna(0)
    state_summary['total'] = state_summary['mp_count'] + state_summary['bodies_count']

    fig = px.bar(
        state_summary.sort_values('total', ascending=False).head(20),
        x='State',
        y=['mp_count', 'bodies_count'],
        title="Top 20 States by Total Cases",
        labels={'value': 'Count', 'variable': 'Type'},
        color_discrete_map={'mp_count': '#00d4ff', 'bodies_count': '#ff4444'}
    )
    fig.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_title="State",
        yaxis_title="Number of Cases"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_geographic(df_mp, df_bodies):
    """Geographic analysis"""
    st.header("Geographic Distribution")

    # State selector
    states = sorted(df_mp['State'].unique())
    selected_state = st.selectbox("Select State", states)

    # Filter to state
    mp_state = df_mp[df_mp['State'] == selected_state]
    bodies_state = df_bodies[df_bodies['State'] == selected_state]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(f"{selected_state} - Missing Persons", f"{len(mp_state):,}")

        # County breakdown
        if len(mp_state) > 0:
            mp_counties = mp_state.groupby('County').size().reset_index(name='count')
            mp_counties = mp_counties.sort_values('count', ascending=False).head(10)

            fig = px.bar(
                mp_counties,
                x='County',
                y='count',
                title=f"Top 10 Counties - Missing Persons",
                color='count',
                color_continuous_scale='Reds'
            )
            fig.update_layout(template='plotly_dark', height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric(f"{selected_state} - Unidentified Bodies", f"{len(bodies_state):,}")

        # County breakdown
        if len(bodies_state) > 0:
            bodies_counties = bodies_state.groupby('County').size().reset_index(name='count')
            bodies_counties = bodies_counties.sort_values('count', ascending=False).head(10)

            fig = px.bar(
                bodies_counties,
                x='County',
                y='count',
                title=f"Top 10 Counties - Unidentified Bodies",
                color='count',
                color_continuous_scale='Blues'
            )
            fig.update_layout(template='plotly_dark', height=300)
            st.plotly_chart(fig, use_container_width=True)

def show_temporal(df_mp, df_bodies):
    """Temporal trends"""
    st.header("Temporal Trends Analysis")

    # Year range selector
    min_year = int(df_mp['year'].min())
    max_year = int(df_mp['year'].max())

    year_range = st.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(1980, 2025)
    )

    # Filter by year
    mp_filtered = df_mp[(df_mp['year'] >= year_range[0]) & (df_mp['year'] <= year_range[1])]
    bodies_filtered = df_bodies[(df_bodies['year'] >= year_range[0]) & (df_bodies['year'] <= year_range[1])]

    # Aggregate by year
    mp_by_year = mp_filtered.groupby('year').size().reset_index(name='mp_count')
    bodies_by_year = bodies_filtered.groupby('year').size().reset_index(name='bodies_count')

    # Merge
    trends = pd.merge(mp_by_year, bodies_by_year, on='year', how='outer').fillna(0)

    # Create dual-axis plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=trends['year'],
            y=trends['mp_count'],
            name="Missing Persons",
            line=dict(color='#00d4ff', width=3),
            mode='lines+markers'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=trends['year'],
            y=trends['bodies_count'],
            name="Unidentified Bodies",
            line=dict(color='#ff4444', width=3),
            mode='lines+markers'
        ),
        secondary_y=True
    )

    fig.update_layout(
        title="National Trends Over Time",
        template='plotly_dark',
        height=500,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Missing Persons", secondary_y=False, color='#00d4ff')
    fig.update_yaxes(title_text="Unidentified Bodies", secondary_y=True, color='#ff4444')

    st.plotly_chart(fig, use_container_width=True)

    # Decade comparison
    st.subheader("Decade Comparison")

    df_mp['decade'] = (df_mp['year'] // 10) * 10
    df_bodies['decade'] = (df_bodies['year'] // 10) * 10

    mp_decades = df_mp.groupby('decade').size().reset_index(name='mp_count')
    bodies_decades = df_bodies.groupby('decade').size().reset_index(name='bodies_count')

    decades = pd.merge(mp_decades, bodies_decades, on='decade', how='outer').fillna(0)
    decades = decades[decades['decade'] >= 1980]

    fig = go.Figure(data=[
        go.Bar(name='Missing Persons', x=decades['decade'], y=decades['mp_count'], marker_color='#00d4ff'),
        go.Bar(name='Unidentified Bodies', x=decades['decade'], y=decades['bodies_count'], marker_color='#ff4444')
    ])

    fig.update_layout(
        title="Cases by Decade",
        template='plotly_dark',
        barmode='group',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

def show_outliers(df_outliers):
    """Outlier detection results"""
    st.header("Statistical Outlier Detection")

    if df_outliers is None:
        st.warning("Outlier data not available. Run calculate_outlier_scores.py first.")
        return

    st.markdown("""
    **Methodology:** Standard Deviation (σ) based classification
    - 🔴 **RED**: >3σ (99.7%+ confidence - statistically impossible)
    - 🟠 **ORANGE**: >2σ (95%+ confidence - significant outlier)
    - 🟡 **YELLOW**: >1σ (68%+ confidence - moderate outlier)
    - 🟢 **GREEN**: <1σ (normal variation)
    """)

    # Alert distribution
    col1, col2, col3, col4 = st.columns(4)

    alert_counts = df_outliers['alert'].value_counts()
    total = len(df_outliers)

    with col1:
        red = alert_counts.get('RED', 0)
        st.metric("🔴 RED Alerts", red, f"{red/total*100:.1f}%")
    with col2:
        orange = alert_counts.get('ORANGE', 0)
        st.metric("🟠 ORANGE Alerts", orange, f"{orange/total*100:.1f}%")
    with col3:
        yellow = alert_counts.get('YELLOW', 0)
        st.metric("🟡 YELLOW Alerts", yellow, f"{yellow/total*100:.1f}%")
    with col4:
        green = alert_counts.get('GREEN', 0)
        st.metric("🟢 GREEN", green, f"{green/total*100:.1f}%")

    st.markdown("---")

    # Top outliers
    st.subheader("Top 20 Extreme Outliers")

    df_outliers['max_sigma'] = df_outliers[['mp_sigma', 'bodies_sigma']].max(axis=1)
    top_outliers = df_outliers.sort_values('max_sigma', ascending=False).head(20)

    # Display table
    display_cols = ['County', 'State', 'decade', 'mp_count', 'mp_sigma', 'bodies_count', 'bodies_sigma', 'alert']
    st.dataframe(
        top_outliers[display_cols],
        use_container_width=True,
        height=600
    )

    # Visualization
    fig = px.scatter(
        top_outliers,
        x='mp_sigma',
        y='bodies_sigma',
        size='max_sigma',
        color='alert',
        hover_data=['County', 'State', 'decade'],
        title="Outlier Distribution (Top 20)",
        color_discrete_map={'RED': '#ff0000', 'ORANGE': '#ff8800', 'YELLOW': '#ffcc00', 'GREEN': '#00ff00'}
    )

    fig.update_layout(
        template='plotly_dark',
        height=500,
        xaxis_title="Missing Persons (σ)",
        yaxis_title="Unidentified Bodies (σ)"
    )

    st.plotly_chart(fig, use_container_width=True)

def show_forecasting(df_mp, df_bodies):
    """Zone forecasting"""
    st.header("Geographic Zone Forecasting")

    zones = {
        'US-Mexico Border': ['CA', 'AZ', 'NM', 'TX'],
        'I-35 Corridor': ['TX'],
        'Pacific Northwest': ['WA', 'OR'],
        'Southern California': ['CA']
    }

    selected_zone = st.selectbox("Select Zone", list(zones.keys()))

    # Filter to zone
    zone_states = zones[selected_zone]
    mp_zone = df_mp[df_mp['State'].isin(zone_states)]
    bodies_zone = df_bodies[df_bodies['State'].isin(zone_states)]

    # Aggregate by year
    mp_by_year = mp_zone.groupby('year').size().reset_index(name='mp_count')
    bodies_by_year = bodies_zone.groupby('year').size().reset_index(name='bodies_count')

    trends = pd.merge(mp_by_year, bodies_by_year, on='year', how='outer').fillna(0)
    trends = trends[(trends['year'] >= 1980) & (trends['year'] <= 2025)]

    if len(trends) == 0:
        st.warning("No data for selected zone")
        return

    # Calculate trend
    from scipy import stats as sp_stats
    years = trends['year'].values
    mp_counts = trends['mp_count'].values

    mp_slope, mp_intercept, mp_r, _, _ = sp_stats.linregress(years, mp_counts)

    # Forecast
    future_years = np.arange(2026, 2031)
    mp_forecast = mp_slope * future_years + mp_intercept
    mp_forecast = np.maximum(mp_forecast, 0)  # No negative

    # Plot
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=trends['year'],
        y=trends['mp_count'],
        name='Historical MP',
        mode='lines+markers',
        line=dict(color='#00d4ff', width=3)
    ))

    # Trend line
    trend_line = mp_slope * years + mp_intercept
    fig.add_trace(go.Scatter(
        x=years,
        y=trend_line,
        name='Trend',
        mode='lines',
        line=dict(color='#00ff00', width=2, dash='dash')
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=future_years,
        y=mp_forecast,
        name='Forecast (2026-2030)',
        mode='lines+markers',
        line=dict(color='#ff00ff', width=3, dash='dot'),
        marker=dict(size=10)
    ))

    fig.update_layout(
        title=f"{selected_zone} - Missing Persons Forecast",
        template='plotly_dark',
        height=500,
        xaxis_title="Year",
        yaxis_title="Missing Persons Count",
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Trend", f"{mp_slope:+.2f} MP/year")
    with col2:
        st.metric("R² (fit quality)", f"{mp_r**2:.3f}")
    with col3:
        st.metric("2030 Forecast", f"{int(mp_forecast[-1])} MP")

    # Show forecast table
    st.subheader("5-Year Forecast")
    forecast_df = pd.DataFrame({
        'Year': future_years,
        'Forecasted Missing Persons': mp_forecast.astype(int)
    })
    st.dataframe(forecast_df, use_container_width=True)

def show_validation(df_outliers):
    """Validation with known serial killers"""
    st.header("System Validation - Known Serial Killers")

    st.markdown("""
    Testing the outlier detection system against **known serial killers** to validate accuracy:
    """)

    if df_outliers is None:
        st.warning("Outlier data not available")
        return

    # Test cases
    tests = [
        {
            'name': 'Gary Ridgway (Green River Killer)',
            'state': 'WA',
            'county': 'King',
            'decade': 1980,
            'expected': 'Should detect - dumped bodies, active 1982-1998'
        },
        {
            'name': 'John Wayne Gacy',
            'state': 'IL',
            'county': 'Cook',
            'decade': 1970,
            'expected': 'Should detect - 33 victims buried on property'
        },
        {
            'name': 'Jeffrey Dahmer',
            'state': 'WI',
            'county': 'Milwaukee',
            'decade': 1980,
            'expected': 'May not detect - destroyed bodies (validates multi-tier design)'
        }
    ]

    for test in tests:
        st.subheader(f"{test['name']}")

        match = df_outliers[
            (df_outliers['State'] == test['state']) &
            (df_outliers['County'] == test['county']) &
            (df_outliers['decade'] == test['decade'])
        ]

        col1, col2, col3 = st.columns(3)

        if len(match) > 0:
            row = match.iloc[0]

            with col1:
                st.metric("Missing Persons", int(row['mp_count']), f"{row['mp_sigma']:.2f}σ")
            with col2:
                st.metric("Unidentified Bodies", int(row['bodies_count']), f"{row['bodies_sigma']:.2f}σ")
            with col3:
                alert_emoji = {'RED': '🔴', 'ORANGE': '🟠', 'YELLOW': '🟡', 'GREEN': '🟢'}
                st.metric("Alert Level", f"{alert_emoji.get(row['alert'], '')} {row['alert']}")

            if row['mp_sigma'] > 1 or row['bodies_sigma'] > 1:
                st.success(f"**DETECTED** as statistical outlier!")
            else:
                st.info(f"Not flagged - {test['expected']}")
        else:
            st.error("No data found for this location/time period")

        st.markdown("---")

def show_raw_count_map(df_mp, df_bodies):
    """Interactive map with raw counts - dots sized by cases, year slider"""
    st.header("Geographic Distribution - Raw Counts")

    st.markdown("""
    **Interactive county-level map showing total cases over time**
    - Dot size = number of cases
    - Color = Missing Persons (blue) vs Unidentified Bodies (red)
    - Use slider to filter by year range
    """)

    # Year range slider
    min_year = int(df_mp['year'].min())
    max_year = int(df_mp['year'].max())

    year_range = st.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(2000, 2025),
        help="Filter cases by date range"
    )

    # Filter by year
    mp_filtered = df_mp[(df_mp['year'] >= year_range[0]) & (df_mp['year'] <= year_range[1])]
    bodies_filtered = df_bodies[(df_bodies['year'] >= year_range[0]) & (df_bodies['year'] <= year_range[1])]

    # State/County coordinates (approximate centers)
    # This is simplified - in production would use actual geocoding
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
        # Full state names (for bodies data which uses full names)
        'Alabama': (32.806671, -86.791130), 'Alaska': (61.370716, -152.404419),
        'Arizona': (33.729759, -111.431221), 'Arkansas': (34.969704, -92.373123),
        'California': (36.116203, -119.681564), 'Colorado': (39.059811, -105.311104),
        'Connecticut': (41.597782, -72.755371), 'Delaware': (39.318523, -75.507141),
        'Florida': (27.766279, -81.686783), 'Georgia': (33.040619, -83.643074),
        'Hawaii': (21.094318, -157.498337), 'Idaho': (44.240459, -114.478828),
        'Illinois': (40.349457, -88.986137), 'Indiana': (39.849426, -86.258278),
        'Iowa': (42.011539, -93.210526), 'Kansas': (38.526600, -96.726486),
        'Kentucky': (37.668140, -84.670067), 'Louisiana': (31.169546, -91.867805),
        'Maine': (44.693947, -69.381927), 'Maryland': (39.063946, -76.802101),
        'Massachusetts': (42.230171, -71.530106), 'Michigan': (43.326618, -84.536095),
        'Minnesota': (45.694454, -93.900192), 'Mississippi': (32.741646, -89.678696),
        'Missouri': (38.456085, -92.288368), 'Montana': (46.921925, -110.454353),
        'Nebraska': (41.125370, -98.268082), 'Nevada': (38.313515, -117.055374),
        'New Hampshire': (43.452492, -71.563896), 'New Jersey': (40.298904, -74.521011),
        'New Mexico': (34.840515, -106.248482), 'New York': (42.165726, -74.948051),
        'North Carolina': (35.630066, -79.806419), 'North Dakota': (47.528912, -99.784012),
        'Ohio': (40.388783, -82.764915), 'Oklahoma': (35.565342, -96.928917),
        'Oregon': (44.572021, -122.070938), 'Pennsylvania': (40.590752, -77.209755),
        'Rhode Island': (41.680893, -71.511780), 'South Carolina': (33.856892, -80.945007),
        'South Dakota': (44.299782, -99.438828), 'Tennessee': (35.747845, -86.692345),
        'Texas': (31.054487, -97.563461), 'Utah': (40.150032, -111.862434),
        'Vermont': (44.045876, -72.710686), 'Virginia': (37.769337, -78.169968),
        'Washington': (47.400902, -121.490494), 'West Virginia': (38.491226, -80.954453),
        'Wisconsin': (44.268543, -89.616508), 'Wyoming': (42.755966, -107.302490),
        'District of Columbia': (38.897438, -77.026817), 'Puerto Rico': (18.220833, -66.590149),
        'Guam': (13.444304, 144.793731), 'Virgin Islands': (18.335765, -64.896335)
    }

    # Aggregate by state/county
    mp_agg = mp_filtered.groupby(['State', 'County']).size().reset_index(name='mp_count')
    bodies_agg = bodies_filtered.groupby(['State', 'County']).size().reset_index(name='bodies_count')

    # Add approximate coordinates (offset from state center)
    mp_agg['lat'] = mp_agg['State'].apply(lambda x: STATE_COORDS.get(x, (0, 0))[0] + np.random.uniform(-1, 1))
    mp_agg['lon'] = mp_agg['State'].apply(lambda x: STATE_COORDS.get(x, (0, 0))[1] + np.random.uniform(-1, 1))

    bodies_agg['lat'] = bodies_agg['State'].apply(lambda x: STATE_COORDS.get(x, (0, 0))[0] + np.random.uniform(-1, 1))
    bodies_agg['lon'] = bodies_agg['State'].apply(lambda x: STATE_COORDS.get(x, (0, 0))[1] + np.random.uniform(-1, 1))

    # Filter to top counties for performance
    top_mp = mp_agg[mp_agg['mp_count'] > 0].sort_values('mp_count', ascending=False).head(500)
    top_bodies = bodies_agg[bodies_agg['bodies_count'] > 0].sort_values('bodies_count', ascending=False).head(500)

    # Create map for Missing Persons
    fig = px.scatter_geo(
        top_mp,
        lat='lat',
        lon='lon',
        size='mp_count',
        color='mp_count',
        hover_name='County',
        hover_data={'State': True, 'mp_count': True, 'lat': False, 'lon': False},
        color_continuous_scale='Blues',
        size_max=50,
        title=f"Missing Persons by County ({year_range[0]}-{year_range[1]})",
        scope='usa'
    )

    fig.update_layout(
        template='plotly_dark',
        height=600,
        geo=dict(
            bgcolor='#0e1117',
            lakecolor='#0e1117',
            landcolor='#1e2127'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Create map for Unidentified Bodies
    fig2 = px.scatter_geo(
        top_bodies,
        lat='lat',
        lon='lon',
        size='bodies_count',
        color='bodies_count',
        hover_name='County',
        hover_data={'State': True, 'bodies_count': True, 'lat': False, 'lon': False},
        color_continuous_scale='Reds',
        size_max=50,
        title=f"Unidentified Bodies by County ({year_range[0]}-{year_range[1]})",
        scope='usa'
    )

    fig2.update_layout(
        template='plotly_dark',
        height=600,
        geo=dict(
            bgcolor='#0e1117',
            lakecolor='#0e1117',
            landcolor='#1e2127'
        )
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Combine for summary stats
    county_data = pd.merge(mp_agg, bodies_agg, on=['State', 'County'], how='outer').fillna(0)
    county_data['total'] = county_data['mp_count'] + county_data['bodies_count']

    # Summary stats
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Counties", len(county_data))
    with col2:
        st.metric("Total Missing Persons", int(county_data['mp_count'].sum()))
    with col3:
        st.metric("Total Bodies", int(county_data['bodies_count'].sum()))

def show_stddev_heat_map(df_mp, df_bodies, df_outliers):
    """Heat map showing standard deviation scores by county/state"""
    st.header("Statistical Outlier Heat Map")

    st.markdown("""
    **Standard Deviation (σ) based heat map**
    - Color intensity = how many σ above baseline
    - Red = extreme outliers (>2σ)
    - Orange = significant outliers (1-2σ)
    - Green = normal (<1σ)
    - Can view by STATE or COUNTY level
    """)

    if df_outliers is None:
        st.warning("Outlier data not available. Run calculate_outlier_scores.py first.")
        return

    # Decade filter
    available_decades = sorted(df_outliers['decade'].unique())
    selected_decades = st.multiselect(
        "Filter by Decade (leave empty for all)",
        options=available_decades,
        default=[],
        help="Select one or more decades to filter, or leave empty to show all decades"
    )

    # Filter by decade if selected
    if selected_decades:
        df_outliers_filtered = df_outliers[df_outliers['decade'].isin(selected_decades)]
    else:
        df_outliers_filtered = df_outliers.copy()

    # View level selector
    view_level = st.radio("Select View Level", ["State", "County"], horizontal=True)

    # Metric selector
    metric = st.radio("Select Metric", ["Missing Persons σ", "Bodies σ", "Combined (max σ)"], horizontal=True)

    if metric == "Missing Persons σ":
        sigma_col = 'mp_sigma'
        color_scale = 'Blues'
    elif metric == "Bodies σ":
        sigma_col = 'bodies_sigma'
        color_scale = 'Reds'
    else:
        df_outliers_filtered['max_sigma'] = df_outliers_filtered[['mp_sigma', 'bodies_sigma']].max(axis=1)
        sigma_col = 'max_sigma'
        color_scale = 'YlOrRd'

    # State coordinates
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
        'GU': (13.444304, 144.793731), 'VI': (18.335765, -64.896335)
    }

    if view_level == "State":
        # Aggregate by state
        state_agg = df_outliers_filtered.groupby('State')[sigma_col].max().reset_index()
        state_agg['lat'] = state_agg['State'].apply(lambda x: STATE_COORDS.get(x, (0, 0))[0])
        state_agg['lon'] = state_agg['State'].apply(lambda x: STATE_COORDS.get(x, (0, 0))[1])

        # Filter out invalid coordinates
        state_agg = state_agg[state_agg['lat'] != 0]

        # Create size column - use max(sigma, 0.1) to avoid negative or zero sizes
        state_agg['size_val'] = state_agg[sigma_col].apply(lambda x: max(abs(x), 0.1))

        decade_str = f" ({', '.join(map(str, selected_decades))})" if selected_decades else " (All Decades)"

        fig = px.scatter_geo(
            state_agg,
            lat='lat',
            lon='lon',
            size='size_val',
            color=sigma_col,
            hover_name='State',
            hover_data={sigma_col: ':.2f', 'size_val': False, 'lat': False, 'lon': False},
            color_continuous_scale=color_scale,
            size_max=80,
            title=f"State-Level {metric}{decade_str}",
            scope='usa',
            range_color=[state_agg[sigma_col].min(), state_agg[sigma_col].max()] if len(state_agg) > 0 else [0, 1]
        )

    else:  # County level
        # Add approximate coordinates
        county_outliers = df_outliers_filtered.copy()
        county_outliers['lat'] = county_outliers['State'].apply(lambda x: STATE_COORDS.get(x, (0, 0))[0] + np.random.uniform(-1, 1))
        county_outliers['lon'] = county_outliers['State'].apply(lambda x: STATE_COORDS.get(x, (0, 0))[1] + np.random.uniform(-1, 1))

        # Filter to outliers only (σ > 1) for clarity
        county_outliers = county_outliers[county_outliers[sigma_col] > 1]

        # Top 500 for performance
        county_outliers = county_outliers.sort_values(sigma_col, ascending=False).head(500)

        # Create size column (sigma is already > 1, so all positive)
        county_outliers['size_val'] = county_outliers[sigma_col]

        decade_str = f" ({', '.join(map(str, selected_decades))})" if selected_decades else " (All Decades)"

        fig = px.scatter_geo(
            county_outliers,
            lat='lat',
            lon='lon',
            size='size_val',
            color=sigma_col,
            hover_name='County',
            hover_data={'State': True, sigma_col: ':.2f', 'mp_count': True, 'bodies_count': True, 'decade': True, 'size_val': False, 'lat': False, 'lon': False},
            color_continuous_scale=color_scale,
            size_max=60,
            title=f"County-Level {metric} (Outliers Only, σ>1){decade_str}",
            scope='usa',
            range_color=[1, county_outliers[sigma_col].max()] if len(county_outliers) > 0 else [1, 2]
        )

    fig.update_layout(
        template='plotly_dark',
        height=700,
        geo=dict(
            bgcolor='#0e1117',
            lakecolor='#0e1117',
            landcolor='#1e2127',
            showland=True,
            showcountries=True,
            countrycolor='#444444'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown("""
    **Color Scale Interpretation:**
    - 🟢 **Green/Low** (<1σ): Normal variation
    - 🟡 **Yellow** (1-2σ): Moderate outlier - monitor
    - 🟠 **Orange** (2-3σ): Significant outlier - investigate
    - 🔴 **Red/High** (>3σ): Extreme outlier - urgent attention
    """)

    # Top outliers table
    st.subheader(f"Top 20 Outliers by {metric}")
    top_outliers = df_outliers_filtered.sort_values(sigma_col, ascending=False).head(20)
    display_cols = ['State', 'County', 'decade', 'mp_count', 'mp_sigma', 'bodies_count', 'bodies_sigma', 'alert']
    st.dataframe(top_outliers[display_cols], use_container_width=True, height=400)

if __name__ == "__main__":
    main()
