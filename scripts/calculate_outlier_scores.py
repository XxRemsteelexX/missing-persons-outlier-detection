#!/usr/bin/env python3
"""
Calculate Outlier Scores for Missing Persons and Unidentified Bodies
Using Standard Deviation Method with Known Serial Killer Validation
"""
import pandas as pd
import numpy as np
from scipy import stats
import os

DATA_DIR = "/home/yeblad/Desktop/Geospatial_Crime_Analysis/data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
POP_DIR = os.path.join(DATA_DIR, "population")
OUTPUT_DIR = os.path.join(DATA_DIR, "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_aggregate_data():
    """
    Load all missing persons and unidentified bodies data
    Aggregate by county/state/year
    """
    print("=" * 70)
    print("LOADING AND AGGREGATING DATA")
    print("=" * 70)

    # Load all state files
    mp_data = []
    bodies_data = []

    for file in os.listdir(RAW_DIR):
        if file.endswith('_missing_persons.csv'):
            df = pd.read_csv(os.path.join(RAW_DIR, file))
            # Parse state from filename
            state = file.replace('_missing_persons.csv', '').replace('_', ' ').title()
            df['state_name'] = state
            mp_data.append(df)
            print(f"  ✓ Loaded {state}: {len(df)} missing persons")

        elif file.endswith('_unidentified_bodies.csv'):
            df = pd.read_csv(os.path.join(RAW_DIR, file))
            state = file.replace('_unidentified_bodies.csv', '').replace('_', ' ').title()
            df['state_name'] = state
            bodies_data.append(df)

    df_mp = pd.concat(mp_data, ignore_index=True)
    df_bodies = pd.concat(bodies_data, ignore_index=True)

    print(f"\n✅ Total: {len(df_mp):,} missing persons, {len(df_bodies):,} unidentified bodies")

    return df_mp, df_bodies

def extract_year_from_date(df, date_col):
    """Extract year from date column"""
    df['year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
    return df

def aggregate_by_county_year(df_mp, df_bodies):
    """
    Aggregate counts by county and year
    """
    print("\n" + "=" * 70)
    print("AGGREGATING BY COUNTY AND YEAR")
    print("=" * 70)

    # Extract years
    df_mp = extract_year_from_date(df_mp, 'DLC')
    df_bodies = extract_year_from_date(df_bodies, 'DBF')

    # Aggregate missing persons
    mp_agg = df_mp.groupby(['State', 'County', 'year']).size().reset_index(name='missing_count')

    # Aggregate bodies
    bodies_agg = df_bodies.groupby(['State', 'County', 'year']).size().reset_index(name='bodies_count')

    # Merge
    df_combined = pd.merge(
        mp_agg,
        bodies_agg,
        on=['State', 'County', 'year'],
        how='outer'
    ).fillna(0)

    print(f"\n✅ Aggregated to {len(df_combined):,} county-year combinations")

    return df_combined

def merge_with_population(df_county_year):
    """
    Merge with population data to calculate rates per 100K
    """
    print("\n" + "=" * 70)
    print("MERGING WITH POPULATION DATA")
    print("=" * 70)

    # Load county population
    pop_file = os.path.join(POP_DIR, "county_population_2020_2024.csv")

    if not os.path.exists(pop_file):
        print("⚠️  No county population data found")
        print("   Using state-level population as fallback...")

        state_pop_file = os.path.join(POP_DIR, "state_population_1980_2024.csv")
        if os.path.exists(state_pop_file):
            df_pop = pd.read_csv(state_pop_file)
            df_merged = pd.merge(
                df_county_year,
                df_pop,
                left_on=['State', 'year'],
                right_on=['state', 'year'],
                how='left'
            )
        else:
            print("❌ No population data available")
            return df_county_year
    else:
        df_pop = pd.read_csv(pop_file)
        df_merged = pd.merge(
            df_county_year,
            df_pop,
            left_on=['State', 'County', 'year'],
            right_on=['state', 'county', 'year'],
            how='left'
        )

    # Calculate rates per 100K
    df_merged['missing_per_100k'] = (df_merged['missing_count'] / df_merged['population']) * 100000
    df_merged['bodies_per_100k'] = (df_merged['bodies_count'] / df_merged['population']) * 100000

    # Fill NaN with 0
    df_merged = df_merged.fillna(0)

    print(f"✅ Merged {len(df_merged):,} records with population data")

    return df_merged

def calculate_standard_deviation_scores(df):
    """
    Calculate Z-scores (standard deviations from mean)
    For both missing persons and unidentified bodies
    """
    print("\n" + "=" * 70)
    print("CALCULATING STANDARD DEVIATION SCORES")
    print("=" * 70)

    # Calculate national baseline (mean + std dev)
    mp_mean = df['missing_per_100k'].mean()
    mp_std = df['missing_per_100k'].std()

    bodies_mean = df['bodies_per_100k'].mean()
    bodies_std = df['bodies_per_100k'].std()

    print(f"\nNational Baselines:")
    print(f"  Missing Persons: {mp_mean:.2f} ± {mp_std:.2f} per 100K")
    print(f"  Unidentified Bodies: {bodies_mean:.2f} ± {bodies_std:.2f} per 100K")

    # Calculate Z-scores
    df['mp_z_score'] = (df['missing_per_100k'] - mp_mean) / mp_std
    df['bodies_z_score'] = (df['bodies_per_100k'] - bodies_mean) / bodies_std

    # Composite score (combine both)
    df['composite_z_score'] = (df['mp_z_score'] + df['bodies_z_score']) / 2

    print(f"\n✅ Calculated Z-scores for {len(df):,} county-year combinations")

    return df

def classify_alert_level(row):
    """
    Classify alert level based on Z-scores
    Green: < 1σ (normal)
    Yellow: 1-2σ (moderate outlier)
    Orange: 2-3σ (significant outlier)
    Red: > 3σ (extreme outlier - URGENT)
    """
    mp_z = row['mp_z_score']
    bodies_z = row['bodies_z_score']
    composite_z = row['composite_z_score']

    # Check if BOTH are elevated (classic serial killer pattern)
    if mp_z > 1 and bodies_z > 1:
        if composite_z > 3:
            return 'RED', 'Both elevated >3σ - URGENT'
        elif composite_z > 2:
            return 'ORANGE', 'Both elevated >2σ - High Priority'
        elif composite_z > 1:
            return 'YELLOW', 'Both elevated >1σ - Monitor'
    # High MP, low bodies (destroyer pattern)
    elif mp_z > 2 and bodies_z < 1:
        return 'ORANGE', 'High MP / Low Bodies - Destroyer Pattern'
    # High bodies, low MP (transient/unreported pattern)
    elif bodies_z > 2 and mp_z < 1:
        return 'ORANGE', 'High Bodies / Low MP - Transient Pattern'
    # Single elevated
    elif composite_z > 3:
        return 'RED', 'Extreme outlier >3σ'
    elif composite_z > 2:
        return 'ORANGE', 'Significant outlier >2σ'
    elif composite_z > 1:
        return 'YELLOW', 'Moderate outlier >1σ'
    else:
        return 'GREEN', 'Normal'

def apply_alert_classification(df):
    """
    Apply alert classification to all records
    """
    print("\n" + "=" * 70)
    print("APPLYING ALERT CLASSIFICATION")
    print("=" * 70)

    df[['alert_level', 'alert_reason']] = df.apply(
        classify_alert_level,
        axis=1,
        result_type='expand'
    )

    # Count by alert level
    alert_counts = df['alert_level'].value_counts()
    print("\nAlert Level Distribution:")
    for level in ['RED', 'ORANGE', 'YELLOW', 'GREEN']:
        count = alert_counts.get(level, 0)
        print(f"  {level}: {count:,} counties")

    return df

def validate_with_known_killers(df):
    """
    Validate system using known serial killers
    """
    print("\n" + "=" * 70)
    print("VALIDATING WITH KNOWN SERIAL KILLERS")
    print("=" * 70)

    # Known killer patterns to validate
    test_cases = [
        {'name': 'Green River Killer (Ridgway)', 'state': 'Washington', 'county': 'King', 'years': range(1982, 1999)},
        {'name': 'John Wayne Gacy', 'state': 'Illinois', 'county': 'Cook', 'years': [1978]},
        {'name': 'Jeffrey Dahmer', 'state': 'Wisconsin', 'county': 'Milwaukee', 'years': range(1978, 1992)}
    ]

    for test in test_cases:
        print(f"\n{test['name']}:")
        subset = df[
            (df['State'].str.contains(test['state'], case=False, na=False)) &
            (df['County'].str.contains(test['county'], case=False, na=False)) &
            (df['year'].isin(test['years']))
        ]

        if len(subset) > 0:
            avg_mp_z = subset['mp_z_score'].mean()
            avg_bodies_z = subset['bodies_z_score'].mean()
            alerts = subset['alert_level'].value_counts()

            print(f"  Years analyzed: {len(subset)}")
            print(f"  Avg MP Z-score: {avg_mp_z:.2f}σ")
            print(f"  Avg Bodies Z-score: {avg_bodies_z:.2f}σ")
            print(f"  Alerts: {dict(alerts)}")

            if avg_mp_z > 1 or avg_bodies_z > 1:
                print(f"  ✅ DETECTED as outlier")
            else:
                print(f"  ⚠️  NOT detected (may validate multi-tier need)")
        else:
            print(f"  ❌ No data found")

    return df

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("OUTLIER DETECTION SYSTEM")
    print("Standard Deviation Method with Multi-Tier Alerts")
    print("=" * 70)

    # Load data
    df_mp, df_bodies = load_and_aggregate_data()

    # Aggregate by county/year
    df_county_year = aggregate_by_county_year(df_mp, df_bodies)

    # Merge with population
    df_merged = merge_with_population(df_county_year)

    # Calculate Z-scores
    df_scores = calculate_standard_deviation_scores(df_merged)

    # Apply alert classification
    df_alerts = apply_alert_classification(df_scores)

    # Validate with known killers
    df_validated = validate_with_known_killers(df_alerts)

    # Save results
    output_file = os.path.join(OUTPUT_DIR, "county_outlier_scores.csv")
    df_validated.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n✅ Saved outlier scores to: {output_file}")

    # Show top 20 RED alerts
    red_alerts = df_validated[df_validated['alert_level'] == 'RED'].sort_values('composite_z_score', ascending=False).head(20)

    if len(red_alerts) > 0:
        print(f"\n🚨 TOP 20 RED ALERTS (Extreme Outliers):")
        print("=" * 70)
        for idx, row in red_alerts.iterrows():
            print(f"{row['County']}, {row['State']} ({int(row['year'])})")
            print(f"  MP: {row['missing_count']:.0f} ({row['missing_per_100k']:.1f}/100K, {row['mp_z_score']:.2f}σ)")
            print(f"  Bodies: {row['bodies_count']:.0f} ({row['bodies_per_100k']:.1f}/100K, {row['bodies_z_score']:.2f}σ)")
            print(f"  Alert: {row['alert_reason']}")
            print()
