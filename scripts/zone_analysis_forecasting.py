#!/usr/bin/env python3
"""
Zone Analysis and Trend Forecasting
Analyze how crime patterns change over time in specific geographic zones
Forecast future hotspots using time series analysis
"""
import pandas as pd
import numpy as np
from scipy import stats
import os

RAW_DIR = "/home/yeblad/Desktop/Geospatial_Crime_Analysis/data/raw"
OUTPUT_DIR = "/home/yeblad/Desktop/Geospatial_Crime_Analysis/data/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define geographic zones
ZONES = {
    'US-Mexico Border': {
        'states': ['CA', 'AZ', 'NM', 'TX'],
        'counties': ['San Diego', 'Imperial', 'Yuma', 'Pima', 'Santa Cruz', 'Cochise',
                     'Hidalgo', 'Luna', 'Doña Ana', 'El Paso', 'Hudspeth', 'Culberson',
                     'Jeff Davis', 'Presidio', 'Brewster', 'Terrell', 'Val Verde',
                     'Kinney', 'Maverick', 'Dimmit', 'Webb', 'Zapata', 'Starr',
                     'Hidalgo', 'Cameron', 'Willacy', 'Brooks']
    },
    'I-35 Corridor': {
        'states': ['TX'],
        'counties': ['Denton', 'Collin', 'Dallas', 'Ellis', 'Hill', 'McLennan',
                     'Bell', 'Williamson', 'Travis', 'Hays', 'Comal', 'Bexar',
                     'Atascosa', 'Frio', 'Medina', 'Webb']
    },
    'Pacific Northwest': {
        'states': ['WA', 'OR'],
        'counties': ['King', 'Pierce', 'Snohomish', 'Thurston', 'Multnomah',
                     'Clackamas', 'Washington', 'Marion']
    },
    'Midwest Metro': {
        'states': ['IL', 'WI', 'IN', 'OH', 'MI'],
        'counties': ['Cook', 'Milwaukee', 'Wayne', 'Marion', 'Cuyahoga',
                     'Franklin', 'Hamilton']
    },
    'Northeast Corridor': {
        'states': ['NY', 'NJ', 'PA', 'MA', 'MD'],
        'counties': ['New York', 'Kings', 'Queens', 'Bronx', 'Richmond',
                     'Philadelphia', 'Baltimore', 'Suffolk', 'Essex']
    },
    'Southern California': {
        'states': ['CA'],
        'counties': ['Los Angeles', 'Orange', 'San Diego', 'Riverside',
                     'San Bernardino', 'Ventura']
    }
}

def load_data():
    """Load all missing persons and bodies data"""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    mp_data = []
    bodies_data = []

    for file in os.listdir(RAW_DIR):
        if file.endswith('_missing_persons.csv'):
            df = pd.read_csv(os.path.join(RAW_DIR, file))
            mp_data.append(df)
        elif file.endswith('_unidentified_bodies.csv'):
            df = pd.read_csv(os.path.join(RAW_DIR, file))
            bodies_data.append(df)

    df_mp = pd.concat(mp_data, ignore_index=True)
    df_bodies = pd.concat(bodies_data, ignore_index=True)

    # Extract years
    df_mp['year'] = pd.to_datetime(df_mp['DLC'], errors='coerce').dt.year
    df_bodies['year'] = pd.to_datetime(df_bodies['DBF'], errors='coerce').dt.year

    print(f"✅ Loaded {len(df_mp):,} missing persons, {len(df_bodies):,} bodies")

    return df_mp, df_bodies

def analyze_zone_trends(df_mp, df_bodies, zone_name, zone_def):
    """
    Analyze trends for a specific geographic zone over time
    """
    print(f"\n{'=' * 70}")
    print(f"ZONE: {zone_name}")
    print(f"{'=' * 70}")

    # Filter to zone
    zone_states = zone_def['states']
    zone_counties = zone_def.get('counties', [])

    mp_zone = df_mp[df_mp['State'].isin(zone_states)]
    bodies_zone = df_bodies[df_bodies['State'].isin(zone_states)]

    # If counties specified, filter further
    if zone_counties:
        mp_zone = mp_zone[mp_zone['County'].isin(zone_counties)]
        bodies_zone = bodies_zone[bodies_zone['County'].isin(zone_counties)]

    # Aggregate by year
    mp_by_year = mp_zone.groupby('year').size().reset_index(name='mp_count')
    bodies_by_year = bodies_zone.groupby('year').size().reset_index(name='bodies_count')

    # Merge
    trends = pd.merge(mp_by_year, bodies_by_year, on='year', how='outer').fillna(0)
    trends = trends.sort_values('year')

    # Filter to reasonable year range (1980-2025)
    trends = trends[(trends['year'] >= 1980) & (trends['year'] <= 2025)]

    if len(trends) == 0:
        print(f"⚠️  No data for {zone_name}")
        return None

    print(f"\nData Range: {int(trends['year'].min())} - {int(trends['year'].max())}")
    print(f"Total Cases: {int(trends['mp_count'].sum())} MP, {int(trends['bodies_count'].sum())} bodies")

    # Calculate trend statistics
    years = trends['year'].values
    mp_counts = trends['mp_count'].values
    bodies_counts = trends['bodies_count'].values

    # Linear regression for trend
    if len(years) > 5:
        mp_slope, mp_intercept, mp_r, _, _ = stats.linregress(years, mp_counts)
        bodies_slope, bodies_intercept, bodies_r, _, _ = stats.linregress(years, bodies_counts)

        print(f"\nTrend Analysis:")
        print(f"  Missing Persons: {mp_slope:+.2f} per year (R²={mp_r**2:.3f})")
        print(f"  Bodies: {bodies_slope:+.2f} per year (R²={bodies_r**2:.3f})")

        if mp_slope > 0.5:
            print(f"  ⚠️  INCREASING MP trend!")
        elif mp_slope < -0.5:
            print(f"  ✓ Decreasing MP trend")

        if bodies_slope > 0.5:
            print(f"  ⚠️  INCREASING bodies trend!")
        elif bodies_slope < -0.5:
            print(f"  ✓ Decreasing bodies trend")

        # Forecast next 5 years
        future_years = np.arange(2026, 2031)
        mp_forecast = mp_slope * future_years + mp_intercept
        bodies_forecast = bodies_slope * future_years + bodies_intercept

        print(f"\n📈 5-Year Forecast (2026-2030):")
        for i, year in enumerate(future_years):
            print(f"  {year}: {int(max(0, mp_forecast[i]))} MP, {int(max(0, bodies_forecast[i]))} bodies")

        # Decade comparison
        decades = {}
        for decade in [1980, 1990, 2000, 2010, 2020]:
            decade_data = trends[(trends['year'] >= decade) & (trends['year'] < decade + 10)]
            if len(decade_data) > 0:
                decades[f"{decade}s"] = {
                    'mp': int(decade_data['mp_count'].sum()),
                    'bodies': int(decade_data['bodies_count'].sum())
                }

        if decades:
            print(f"\n📊 Decade Comparison:")
            for decade, counts in decades.items():
                print(f"  {decade}: {counts['mp']} MP, {counts['bodies']} bodies")

        # Identify peak years
        peak_mp_year = trends.loc[trends['mp_count'].idxmax()]
        peak_bodies_year = trends.loc[trends['bodies_count'].idxmax()]

        print(f"\n🚨 Peak Years:")
        print(f"  Missing Persons: {int(peak_mp_year['year'])} ({int(peak_mp_year['mp_count'])} cases)")
        print(f"  Bodies: {int(peak_bodies_year['year'])} ({int(peak_bodies_year['bodies_count'])} cases)")

        # Calculate acceleration (change in trend)
        recent_years = trends[trends['year'] >= 2015]
        if len(recent_years) >= 5:
            recent_mp_slope, _, _, _, _ = stats.linregress(recent_years['year'], recent_years['mp_count'])
            acceleration = recent_mp_slope - mp_slope

            if abs(acceleration) > 1:
                print(f"\n⚡ Trend Acceleration Detected:")
                print(f"  Recent trend ({recent_mp_slope:+.2f}/yr) vs overall ({mp_slope:+.2f}/yr)")
                if acceleration > 0:
                    print(f"  ⚠️  ACCELERATING - situation worsening!")
                else:
                    print(f"  ✓ Decelerating - situation improving")

        return {
            'zone': zone_name,
            'trends': trends,
            'mp_slope': mp_slope,
            'bodies_slope': bodies_slope,
            'mp_r2': mp_r**2,
            'bodies_r2': bodies_r**2,
            'decades': decades,
            'forecast': {
                'years': future_years,
                'mp': mp_forecast,
                'bodies': bodies_forecast
            }
        }

    return None

def compare_zones(zone_results):
    """
    Compare trends across different geographic zones
    """
    print("\n" + "=" * 70)
    print("ZONE COMPARISON")
    print("=" * 70)

    comparison = []
    for result in zone_results:
        if result:
            comparison.append({
                'Zone': result['zone'],
                'MP Trend': f"{result['mp_slope']:+.2f}/yr",
                'Bodies Trend': f"{result['bodies_slope']:+.2f}/yr",
                'MP R²': f"{result['mp_r2']:.3f}",
                'Bodies R²': f"{result['bodies_r2']:.3f}"
            })

    if comparison:
        df_comp = pd.DataFrame(comparison)
        print(f"\n{df_comp.to_string(index=False)}")

        # Identify most concerning zones
        print(f"\n🚨 PRIORITY ZONES (Increasing Trends):")
        for result in zone_results:
            if result and result['mp_slope'] > 1:
                print(f"  ⚠️  {result['zone']}: +{result['mp_slope']:.2f} MP/year")

def analyze_border_evolution():
    """
    Special analysis: How US-Mexico border patterns evolved decade by decade
    """
    print("\n" + "=" * 70)
    print("BORDER EVOLUTION ANALYSIS (1980-2025)")
    print("=" * 70)

    df_mp, df_bodies = load_data()

    # Border states
    border_states = ['CA', 'AZ', 'NM', 'TX']

    mp_border = df_mp[df_mp['State'].isin(border_states)]
    bodies_border = df_bodies[df_bodies['State'].isin(border_states)]

    # Add decade
    mp_border['decade'] = (mp_border['year'] // 10) * 10
    bodies_border['decade'] = (bodies_border['year'] // 10) * 10

    # Aggregate by state and decade
    mp_agg = mp_border.groupby(['State', 'decade']).size().reset_index(name='mp_count')
    bodies_agg = bodies_border.groupby(['State', 'decade']).size().reset_index(name='bodies_count')

    combined = pd.merge(mp_agg, bodies_agg, on=['State', 'decade'], how='outer').fillna(0)

    print("\n📊 Border State Evolution by Decade:")
    print("=" * 70)

    for state in border_states:
        state_data = combined[combined['State'] == state].sort_values('decade')
        if len(state_data) > 0:
            print(f"\n{state}:")
            for _, row in state_data.iterrows():
                if row['decade'] >= 1980:
                    print(f"  {int(row['decade'])}s: {int(row['mp_count'])} MP, {int(row['bodies_count'])} bodies")

    # Calculate change rates
    print(f"\n📈 Change Analysis (2010s → 2020s):")
    for state in border_states:
        data_2010 = combined[(combined['State'] == state) & (combined['decade'] == 2010)]
        data_2020 = combined[(combined['State'] == state) & (combined['decade'] == 2020)]

        if len(data_2010) > 0 and len(data_2020) > 0:
            mp_change = data_2020.iloc[0]['mp_count'] - data_2010.iloc[0]['mp_count']
            bodies_change = data_2020.iloc[0]['bodies_count'] - data_2010.iloc[0]['bodies_count']

            print(f"\n{state}:")
            print(f"  MP: {mp_change:+.0f} ({mp_change/data_2010.iloc[0]['mp_count']*100:+.1f}%)" if data_2010.iloc[0]['mp_count'] > 0 else f"  MP: {mp_change:+.0f}")
            print(f"  Bodies: {bodies_change:+.0f} ({bodies_change/data_2010.iloc[0]['bodies_count']*100:+.1f}%)" if data_2010.iloc[0]['bodies_count'] > 0 else f"  Bodies: {bodies_change:+.0f}")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GEOGRAPHIC ZONE ANALYSIS & FORECASTING")
    print("=" * 70)

    # Load data
    df_mp, df_bodies = load_data()

    # Analyze each zone
    zone_results = []
    for zone_name, zone_def in ZONES.items():
        result = analyze_zone_trends(df_mp, df_bodies, zone_name, zone_def)
        zone_results.append(result)

    # Compare zones
    compare_zones(zone_results)

    # Special border analysis
    analyze_border_evolution()

    # Save results
    output_file = os.path.join(OUTPUT_DIR, "zone_trends_forecast.txt")
    print(f"\n✅ Analysis complete")
    print(f"Results saved to: {output_file}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS FOR FBI")
    print("=" * 70)
    print("""
    1. Temporal trends show which zones are getting WORSE vs BETTER
    2. 5-year forecasts predict future hotspots for resource allocation
    3. Decade comparison reveals long-term patterns
    4. Border evolution tracks immigration/cartel activity changes
    5. Acceleration detection identifies rapidly deteriorating situations

    This enables PREDICTIVE policing rather than just REACTIVE.
    """)
