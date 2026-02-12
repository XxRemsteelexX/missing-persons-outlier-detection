#!/usr/bin/env python3
"""
Zone Analysis and Trend Forecasting
Analyze how crime patterns change over time in specific geographic zones.
Forecast future hotspots using time series analysis with prediction intervals.

Phase 1 fixes:
  - Zone definitions imported from utils.py (single source of truth, all 6 zones)
  - 95% prediction intervals on linear regression forecasts
  - Forecasts saved to CSV (was broken before)
  - State matching uses abbreviations consistently
"""
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import RAW_DIR, ANALYSIS_DIR, ZONES, normalize_state

os.makedirs(ANALYSIS_DIR, exist_ok=True)


def load_data():
    """Load all missing persons and bodies data with normalized state codes."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    mp_data = []
    bodies_data = []

    for file in sorted(os.listdir(RAW_DIR)):
        if file.endswith('_missing_persons.csv'):
            df = pd.read_csv(os.path.join(RAW_DIR, file))
            mp_data.append(df)
        elif file.endswith('_unidentified_bodies.csv'):
            df = pd.read_csv(os.path.join(RAW_DIR, file))
            bodies_data.append(df)

    df_mp = pd.concat(mp_data, ignore_index=True)
    df_bodies = pd.concat(bodies_data, ignore_index=True)

    # Normalize state codes
    df_mp['State'] = df_mp['State'].apply(normalize_state)
    df_bodies['State'] = df_bodies['State'].apply(normalize_state)

    # Extract years
    df_mp['year'] = pd.to_datetime(df_mp['DLC'], errors='coerce').dt.year
    df_bodies['year'] = pd.to_datetime(df_bodies['DBF'], errors='coerce').dt.year

    print(f"Loaded {len(df_mp):,} missing persons, {len(df_bodies):,} bodies")

    return df_mp, df_bodies


def _prediction_interval(years, counts, future_years, alpha=0.05):
    """
    Compute linear regression with prediction intervals.

    Returns:
        forecast: predicted values for future_years
        lower: lower bound of prediction interval
        upper: upper bound of prediction interval
        slope, intercept, r_squared: regression stats
    """
    n = len(years)
    if n < 5:
        return None

    slope, intercept, r_value, p_value, std_err = stats.linregress(years, counts)
    r_squared = r_value ** 2

    # Predicted values
    y_hat = slope * years + intercept
    residuals = counts - y_hat

    # Residual standard error
    s_e = np.sqrt(np.sum(residuals ** 2) / (n - 2))

    # Mean of x
    x_bar = np.mean(years)
    ss_x = np.sum((years - x_bar) ** 2)

    # t critical value
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 2)

    # Forecast
    forecast = slope * future_years + intercept

    # Prediction interval for each future point
    lower = np.zeros_like(future_years, dtype=float)
    upper = np.zeros_like(future_years, dtype=float)

    for i, x_new in enumerate(future_years):
        se_pred = s_e * np.sqrt(1 + 1/n + (x_new - x_bar)**2 / ss_x)
        lower[i] = forecast[i] - t_crit * se_pred
        upper[i] = forecast[i] + t_crit * se_pred

    # Floor at 0 (can't have negative counts)
    forecast = np.maximum(forecast, 0)
    lower = np.maximum(lower, 0)
    upper = np.maximum(upper, 0)

    return {
        'forecast': forecast,
        'lower_95': lower,
        'upper_95': upper,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'p_value': p_value,
        'residual_se': s_e,
    }


def analyze_zone_trends(df_mp, df_bodies, zone_name, zone_def):
    """
    Analyze trends for a specific geographic zone over time.
    Returns zone statistics with prediction intervals.
    """
    print(f"\n{'=' * 70}")
    print(f"ZONE: {zone_name}")
    print(f"{'=' * 70}")

    zone_states = zone_def['states']
    zone_counties = zone_def.get('counties', [])

    mp_zone = df_mp[df_mp['State'].isin(zone_states)]
    bodies_zone = df_bodies[df_bodies['State'].isin(zone_states)]

    if zone_counties:
        mp_zone = mp_zone[mp_zone['County'].isin(zone_counties)]
        bodies_zone = bodies_zone[bodies_zone['County'].isin(zone_counties)]

    # Aggregate by year
    mp_by_year = mp_zone.groupby('year').size().reset_index(name='mp_count')
    bodies_by_year = bodies_zone.groupby('year').size().reset_index(name='bodies_count')

    trends = pd.merge(mp_by_year, bodies_by_year, on='year', how='outer').fillna(0)
    trends = trends.sort_values('year')
    trends = trends[(trends['year'] >= 1980) & (trends['year'] <= 2025)]

    if len(trends) == 0:
        print(f"  No data for {zone_name}")
        return None

    print(f"\nData Range: {int(trends['year'].min())} - {int(trends['year'].max())}")
    print(f"Total Cases: {int(trends['mp_count'].sum())} MP, {int(trends['bodies_count'].sum())} bodies")

    years = trends['year'].values.astype(float)
    mp_counts = trends['mp_count'].values.astype(float)
    bodies_counts = trends['bodies_count'].values.astype(float)

    future_years = np.arange(2026, 2031, dtype=float)

    # Regression with prediction intervals
    mp_result = _prediction_interval(years, mp_counts, future_years)
    bodies_result = _prediction_interval(years, bodies_counts, future_years)

    if mp_result is None:
        print(f"  Insufficient data for regression")
        return None

    print(f"\nTrend Analysis:")
    print(f"  Missing Persons: {mp_result['slope']:+.2f} per year (R2={mp_result['r_squared']:.3f})")
    print(f"  Bodies: {bodies_result['slope']:+.2f} per year (R2={bodies_result['r_squared']:.3f})")

    if mp_result['slope'] > 0.5:
        print(f"  WARNING: INCREASING MP trend")
    elif mp_result['slope'] < -0.5:
        print(f"  Decreasing MP trend")

    # Print forecasts with intervals
    print(f"\n5-Year Forecast with 95% Prediction Intervals (2026-2030):")
    for i, yr in enumerate(future_years.astype(int)):
        mp_f = mp_result['forecast'][i]
        mp_lo = mp_result['lower_95'][i]
        mp_hi = mp_result['upper_95'][i]
        bd_f = bodies_result['forecast'][i]
        bd_lo = bodies_result['lower_95'][i]
        bd_hi = bodies_result['upper_95'][i]
        print(f"  {yr}: MP {mp_f:.0f} [{mp_lo:.0f}, {mp_hi:.0f}] | Bodies {bd_f:.0f} [{bd_lo:.0f}, {bd_hi:.0f}]")

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
        print(f"\nDecade Comparison:")
        for decade, counts in decades.items():
            print(f"  {decade}: {counts['mp']} MP, {counts['bodies']} bodies")

    # Peak years
    peak_mp_year = trends.loc[trends['mp_count'].idxmax()]
    peak_bodies_year = trends.loc[trends['bodies_count'].idxmax()]

    print(f"\nPeak Years:")
    print(f"  Missing Persons: {int(peak_mp_year['year'])} ({int(peak_mp_year['mp_count'])} cases)")
    print(f"  Bodies: {int(peak_bodies_year['year'])} ({int(peak_bodies_year['bodies_count'])} cases)")

    # Acceleration detection
    recent_years = trends[trends['year'] >= 2015]
    recent_mp_slope = None
    if len(recent_years) >= 5:
        recent_mp_slope, _, _, _, _ = stats.linregress(
            recent_years['year'].values, recent_years['mp_count'].values
        )
        acceleration = recent_mp_slope - mp_result['slope']
        if abs(acceleration) > 1:
            print(f"\nTrend Acceleration Detected:")
            print(f"  Recent trend ({recent_mp_slope:+.2f}/yr) vs overall ({mp_result['slope']:+.2f}/yr)")
            if acceleration > 0:
                print(f"  ACCELERATING - situation worsening")
            else:
                print(f"  Decelerating - situation improving")

    return {
        'zone': zone_name,
        'trends': trends,
        'mp_slope': mp_result['slope'],
        'bodies_slope': bodies_result['slope'],
        'mp_r2': mp_result['r_squared'],
        'bodies_r2': bodies_result['r_squared'],
        'mp_residual_se': mp_result['residual_se'],
        'bodies_residual_se': bodies_result['residual_se'],
        'decades': decades,
        'forecast': {
            'years': future_years.astype(int),
            'mp_forecast': mp_result['forecast'],
            'mp_lower_95': mp_result['lower_95'],
            'mp_upper_95': mp_result['upper_95'],
            'bodies_forecast': bodies_result['forecast'],
            'bodies_lower_95': bodies_result['lower_95'],
            'bodies_upper_95': bodies_result['upper_95'],
        },
        'recent_mp_slope': recent_mp_slope,
    }


def compare_zones(zone_results):
    """Compare trends across different geographic zones."""
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
                'MP R2': f"{result['mp_r2']:.3f}",
                'Bodies R2': f"{result['bodies_r2']:.3f}",
            })

    if comparison:
        df_comp = pd.DataFrame(comparison)
        print(f"\n{df_comp.to_string(index=False)}")

        print(f"\nPRIORITY ZONES (Increasing Trends):")
        for result in zone_results:
            if result and result['mp_slope'] > 1:
                print(f"  {result['zone']}: +{result['mp_slope']:.2f} MP/year")


def save_zone_forecasts(zone_results):
    """Save all zone forecasts to CSV with prediction intervals."""
    print("\n" + "=" * 70)
    print("SAVING ZONE FORECASTS")
    print("=" * 70)

    rows = []
    for result in zone_results:
        if result is None:
            continue

        fc = result['forecast']
        for i, yr in enumerate(fc['years']):
            rows.append({
                'zone': result['zone'],
                'year': int(yr),
                'mp_forecast': fc['mp_forecast'][i],
                'mp_lower_95': fc['mp_lower_95'][i],
                'mp_upper_95': fc['mp_upper_95'][i],
                'bodies_forecast': fc['bodies_forecast'][i],
                'bodies_lower_95': fc['bodies_lower_95'][i],
                'bodies_upper_95': fc['bodies_upper_95'][i],
                'mp_slope': result['mp_slope'],
                'bodies_slope': result['bodies_slope'],
                'mp_r2': result['mp_r2'],
                'bodies_r2': result['bodies_r2'],
            })

    if rows:
        df_fc = pd.DataFrame(rows)
        output_file = os.path.join(ANALYSIS_DIR, "zone_forecasts.csv")
        df_fc.to_csv(output_file, index=False)
        print(f"Saved {len(df_fc)} forecast records to {output_file}")

    # Also save zone trend data (historical)
    trend_rows = []
    for result in zone_results:
        if result is None:
            continue
        for _, row in result['trends'].iterrows():
            trend_rows.append({
                'zone': result['zone'],
                'year': int(row['year']),
                'mp_count': int(row['mp_count']),
                'bodies_count': int(row['bodies_count']),
            })

    if trend_rows:
        df_trends = pd.DataFrame(trend_rows)
        output_file = os.path.join(ANALYSIS_DIR, "zone_trends.csv")
        df_trends.to_csv(output_file, index=False)
        print(f"Saved {len(df_trends)} trend records to {output_file}")


def analyze_border_evolution():
    """
    Special analysis: How US-Mexico border patterns evolved decade by decade.
    """
    print("\n" + "=" * 70)
    print("BORDER EVOLUTION ANALYSIS (1980-2025)")
    print("=" * 70)

    df_mp, df_bodies = load_data()

    border_states = ['CA', 'AZ', 'NM', 'TX']

    mp_border = df_mp[df_mp['State'].isin(border_states)]
    bodies_border = df_bodies[df_bodies['State'].isin(border_states)]

    mp_border = mp_border.copy()
    bodies_border = bodies_border.copy()
    mp_border['decade'] = (mp_border['year'] // 10) * 10
    bodies_border['decade'] = (bodies_border['year'] // 10) * 10

    mp_agg = mp_border.groupby(['State', 'decade']).size().reset_index(name='mp_count')
    bodies_agg = bodies_border.groupby(['State', 'decade']).size().reset_index(name='bodies_count')

    combined = pd.merge(mp_agg, bodies_agg, on=['State', 'decade'], how='outer').fillna(0)

    print("\nBorder State Evolution by Decade:")
    print("=" * 70)

    for state in border_states:
        state_data = combined[combined['State'] == state].sort_values('decade')
        if len(state_data) > 0:
            print(f"\n{state}:")
            for _, row in state_data.iterrows():
                if row['decade'] >= 1980:
                    print(f"  {int(row['decade'])}s: {int(row['mp_count'])} MP, {int(row['bodies_count'])} bodies")

    # Change rates
    print(f"\nChange Analysis (2010s -> 2020s):")
    for state in border_states:
        data_2010 = combined[(combined['State'] == state) & (combined['decade'] == 2010)]
        data_2020 = combined[(combined['State'] == state) & (combined['decade'] == 2020)]

        if len(data_2010) > 0 and len(data_2020) > 0:
            mp_change = data_2020.iloc[0]['mp_count'] - data_2010.iloc[0]['mp_count']
            mp_base = data_2010.iloc[0]['mp_count']
            bodies_change = data_2020.iloc[0]['bodies_count'] - data_2010.iloc[0]['bodies_count']
            bodies_base = data_2010.iloc[0]['bodies_count']

            print(f"\n{state}:")
            if mp_base > 0:
                print(f"  MP: {mp_change:+.0f} ({mp_change/mp_base*100:+.1f}%)")
            else:
                print(f"  MP: {mp_change:+.0f}")
            if bodies_base > 0:
                print(f"  Bodies: {bodies_change:+.0f} ({bodies_change/bodies_base*100:+.1f}%)")
            else:
                print(f"  Bodies: {bodies_change:+.0f}")


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

    # Save forecasts with prediction intervals
    save_zone_forecasts(zone_results)

    # Special border analysis
    analyze_border_evolution()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
