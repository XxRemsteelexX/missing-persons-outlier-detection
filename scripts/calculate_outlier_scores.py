#!/usr/bin/env python3
"""
Calculate Outlier Scores for Missing Persons and Unidentified Bodies
Using Standard Deviation Method with Known Serial Killer Validation

Phase 0 fixes:
  - State names normalized to consistent abbreviations via utils.normalize_state()
  - Population merge fixed: uses state_abbrev + county for matching
  - Falls back to state-level population when county-level unavailable
  - Per-100K rates verified to be non-zero for populated counties
  - Also produces decade-level aggregation for downstream analysis
"""
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    DATA_DIR, RAW_DIR, POP_DIR, ANALYSIS_DIR,
    normalize_state, normalize_state_to_full, STATE_FULL,
)

os.makedirs(ANALYSIS_DIR, exist_ok=True)


def load_and_aggregate_data():
    """
    Load all missing persons and unidentified bodies data.
    Normalize state names to abbreviations on load.
    """
    print("=" * 70)
    print("LOADING AND AGGREGATING DATA")
    print("=" * 70)

    mp_data = []
    bodies_data = []

    for file in sorted(os.listdir(RAW_DIR)):
        if file.endswith('_missing_persons.csv'):
            df = pd.read_csv(os.path.join(RAW_DIR, file))
            mp_data.append(df)
            print(f"  Loaded {file}: {len(df)} missing persons")

        elif file.endswith('_unidentified_bodies.csv'):
            df = pd.read_csv(os.path.join(RAW_DIR, file))
            bodies_data.append(df)

    df_mp = pd.concat(mp_data, ignore_index=True)
    df_bodies = pd.concat(bodies_data, ignore_index=True)

    # Normalize state columns to abbreviations
    df_mp['State'] = df_mp['State'].apply(normalize_state)
    df_bodies['State'] = df_bodies['State'].apply(normalize_state)

    print(f"\nTotal: {len(df_mp):,} missing persons, {len(df_bodies):,} unidentified bodies")
    print(f"Unique states (MP): {df_mp['State'].nunique()}")
    print(f"Unique states (Bodies): {df_bodies['State'].nunique()}")

    return df_mp, df_bodies


def extract_year_from_date(df, date_col):
    """Extract year from date column."""
    df = df.copy()
    df['year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
    return df


def aggregate_by_county_year(df_mp, df_bodies):
    """
    Aggregate counts by county, state, and year.
    """
    print("\n" + "=" * 70)
    print("AGGREGATING BY COUNTY AND YEAR")
    print("=" * 70)

    df_mp = extract_year_from_date(df_mp, 'DLC')
    df_bodies = extract_year_from_date(df_bodies, 'DBF')

    # Aggregate missing persons by State (abbrev), County, year
    mp_agg = df_mp.groupby(['State', 'County', 'year']).size().reset_index(
        name='missing_count'
    )

    # Aggregate bodies
    bodies_agg = df_bodies.groupby(['State', 'County', 'year']).size().reset_index(
        name='bodies_count'
    )

    # Merge
    df_combined = pd.merge(
        mp_agg, bodies_agg,
        on=['State', 'County', 'year'],
        how='outer'
    ).fillna(0)

    # Ensure integer counts
    df_combined['missing_count'] = df_combined['missing_count'].astype(int)
    df_combined['bodies_count'] = df_combined['bodies_count'].astype(int)

    # Add full state name for display
    df_combined['state_name'] = df_combined['State'].map(
        lambda x: STATE_FULL.get(x, x)
    )

    print(f"\nAggregated to {len(df_combined):,} county-year combinations")
    print(f"Year range: {df_combined['year'].min():.0f} - {df_combined['year'].max():.0f}")

    return df_combined


def load_population_data():
    """
    Load population data and prepare for merging.
    Returns both county-level and state-level population DataFrames.
    """
    county_pop = None
    state_pop = None

    # Try county-level first
    county_file = os.path.join(POP_DIR, "county_population_2000_2024.csv")
    if os.path.exists(county_file):
        county_pop = pd.read_csv(county_file)
        # Normalize state abbreviations
        if 'state_abbrev' in county_pop.columns:
            county_pop['state_abbrev'] = county_pop['state_abbrev'].apply(normalize_state)
        elif 'state' in county_pop.columns:
            county_pop['state_abbrev'] = county_pop['state'].apply(normalize_state)
        print(f"  Loaded county population: {len(county_pop):,} records")

    # State-level fallback
    state_file = os.path.join(POP_DIR, "state_population_1980_2024.csv")
    if os.path.exists(state_file):
        state_pop = pd.read_csv(state_file)
        if 'state_abbrev' in state_pop.columns:
            state_pop['state_abbrev'] = state_pop['state_abbrev'].apply(normalize_state)
        elif 'state' in state_pop.columns:
            state_pop['state_abbrev'] = state_pop['state'].apply(normalize_state)
        print(f"  Loaded state population: {len(state_pop):,} records")

    return county_pop, state_pop


def merge_with_population(df_county_year):
    """
    Merge crime data with population data to calculate rates per 100K.

    Strategy:
      1. Try county-level population match (state_abbrev + county name)
      2. Fall back to state-level population for unmatched rows
      3. Use nearest available year for time gaps
    """
    print("\n" + "=" * 70)
    print("MERGING WITH POPULATION DATA")
    print("=" * 70)

    county_pop, state_pop = load_population_data()

    df = df_county_year.copy()
    df['population'] = np.nan

    # --- County-level merge ---
    if county_pop is not None:
        # Build a lookup: (state_abbrev, county_name, year) -> population
        county_pop_lookup = county_pop.set_index(
            ['state_abbrev', 'county', 'year']
        )['population']

        matched = 0
        for idx, row in df.iterrows():
            key = (row['State'], row['County'], int(row['year']))
            if key in county_pop_lookup.index:
                df.at[idx, 'population'] = county_pop_lookup[key]
                matched += 1
            else:
                # Try nearest year within +/- 3 years
                state_county = county_pop[
                    (county_pop['state_abbrev'] == row['State']) &
                    (county_pop['county'] == row['County'])
                ]
                if len(state_county) > 0:
                    nearest_idx = (state_county['year'] - row['year']).abs().idxmin()
                    df.at[idx, 'population'] = state_county.loc[nearest_idx, 'population']
                    matched += 1

        print(f"  County-level match: {matched:,} / {len(df):,} rows")

    # --- State-level fallback for remaining NaN ---
    if state_pop is not None:
        still_missing = df['population'].isna()
        if still_missing.any():
            state_pop_lookup = state_pop.set_index(
                ['state_abbrev', 'year']
            )['population']

            fallback_count = 0
            for idx in df[still_missing].index:
                row = df.loc[idx]
                key = (row['State'], int(row['year']))
                if key in state_pop_lookup.index:
                    df.at[idx, 'population'] = state_pop_lookup[key]
                    fallback_count += 1
                else:
                    # Nearest year
                    state_data = state_pop[
                        state_pop['state_abbrev'] == row['State']
                    ]
                    if len(state_data) > 0:
                        nearest_idx = (state_data['year'] - row['year']).abs().idxmin()
                        df.at[idx, 'population'] = state_data.loc[nearest_idx, 'population']
                        fallback_count += 1

            print(f"  State-level fallback: {fallback_count:,} rows")

    # --- Calculate per-100K rates ---
    still_missing = df['population'].isna() | (df['population'] == 0)
    df.loc[~still_missing, 'missing_per_100k'] = (
        df.loc[~still_missing, 'missing_count'] /
        df.loc[~still_missing, 'population'] * 100_000
    )
    df.loc[~still_missing, 'bodies_per_100k'] = (
        df.loc[~still_missing, 'bodies_count'] /
        df.loc[~still_missing, 'population'] * 100_000
    )

    # Fill remaining with 0
    df['missing_per_100k'] = df['missing_per_100k'].fillna(0)
    df['bodies_per_100k'] = df['bodies_per_100k'].fillna(0)
    df['population'] = df['population'].fillna(0)

    # Summary
    has_pop = (df['population'] > 0).sum()
    has_rate = (df['missing_per_100k'] > 0).sum()
    print(f"\n  Rows with population > 0: {has_pop:,} / {len(df):,} ({has_pop/len(df)*100:.1f}%)")
    print(f"  Rows with missing_per_100k > 0: {has_rate:,}")

    return df


def calculate_standard_deviation_scores(df):
    """
    Calculate Z-scores (standard deviations from mean)
    for both missing persons and unidentified bodies rates.
    """
    print("\n" + "=" * 70)
    print("CALCULATING STANDARD DEVIATION SCORES")
    print("=" * 70)

    # Only compute baselines from rows with actual population data
    has_pop = df['population'] > 0

    mp_mean = df.loc[has_pop, 'missing_per_100k'].mean()
    mp_std = df.loc[has_pop, 'missing_per_100k'].std()

    bodies_mean = df.loc[has_pop, 'bodies_per_100k'].mean()
    bodies_std = df.loc[has_pop, 'bodies_per_100k'].std()

    print(f"\nNational Baselines (from {has_pop.sum():,} rows with population):")
    print(f"  Missing Persons: {mp_mean:.2f} +/- {mp_std:.2f} per 100K")
    print(f"  Unidentified Bodies: {bodies_mean:.2f} +/- {bodies_std:.2f} per 100K")

    # Calculate Z-scores (avoid division by zero)
    if mp_std > 0:
        df['mp_z_score'] = (df['missing_per_100k'] - mp_mean) / mp_std
    else:
        df['mp_z_score'] = 0.0

    if bodies_std > 0:
        df['bodies_z_score'] = (df['bodies_per_100k'] - bodies_mean) / bodies_std
    else:
        df['bodies_z_score'] = 0.0

    # Composite score
    df['composite_z_score'] = (df['mp_z_score'] + df['bodies_z_score']) / 2

    # Set z-scores to 0 for rows without population data
    df.loc[~has_pop, ['mp_z_score', 'bodies_z_score', 'composite_z_score']] = 0.0

    print(f"\nCalculated Z-scores for {len(df):,} county-year combinations")

    return df


def classify_alert_level(row):
    """
    Classify alert level based on Z-scores.
    GREEN: < 1 sigma (normal)
    YELLOW: 1-2 sigma (moderate outlier)
    ORANGE: 2-3 sigma (significant outlier)
    RED: > 3 sigma (extreme outlier)
    """
    mp_z = row['mp_z_score']
    bodies_z = row['bodies_z_score']
    composite_z = row['composite_z_score']

    # Both elevated (classic serial killer pattern)
    if mp_z > 1 and bodies_z > 1:
        if composite_z > 3:
            return 'RED', 'Both elevated >3 sigma - URGENT'
        elif composite_z > 2:
            return 'ORANGE', 'Both elevated >2 sigma - High Priority'
        elif composite_z > 1:
            return 'YELLOW', 'Both elevated >1 sigma - Monitor'
    # High MP, low bodies (destroyer pattern)
    elif mp_z > 2 and bodies_z < 1:
        return 'ORANGE', 'High MP / Low Bodies - Destroyer Pattern'
    # High bodies, low MP (transient/unreported pattern)
    elif bodies_z > 2 and mp_z < 1:
        return 'ORANGE', 'High Bodies / Low MP - Transient Pattern'
    # Single elevated
    elif composite_z > 3:
        return 'RED', 'Extreme outlier >3 sigma'
    elif composite_z > 2:
        return 'ORANGE', 'Significant outlier >2 sigma'
    elif composite_z > 1:
        return 'YELLOW', 'Moderate outlier >1 sigma'

    return 'GREEN', 'Normal'


def apply_alert_classification(df):
    """Apply alert classification to all records."""
    print("\n" + "=" * 70)
    print("APPLYING ALERT CLASSIFICATION")
    print("=" * 70)

    df[['alert_level', 'alert_reason']] = df.apply(
        classify_alert_level, axis=1, result_type='expand'
    )

    alert_counts = df['alert_level'].value_counts()
    print("\nAlert Level Distribution:")
    for level in ['RED', 'ORANGE', 'YELLOW', 'GREEN']:
        count = alert_counts.get(level, 0)
        print(f"  {level}: {count:,} counties")

    return df


def aggregate_by_decade(df):
    """
    Aggregate county-year data into county-decade data.
    Produces the county_decade_outliers.csv used by the dashboard.
    """
    print("\n" + "=" * 70)
    print("AGGREGATING BY COUNTY AND DECADE")
    print("=" * 70)

    df = df.copy()
    df['decade'] = (df['year'] // 10) * 10

    # Aggregate by State, County, decade
    decade_agg = df.groupby(['State', 'County', 'state_name', 'decade']).agg(
        missing_count=('missing_count', 'sum'),
        bodies_count=('bodies_count', 'sum'),
        population=('population', 'mean'),  # average population over the decade
        years_of_data=('year', 'count'),
    ).reset_index()

    # Calculate per-100K rates for decade
    has_pop = decade_agg['population'] > 0
    decade_agg.loc[has_pop, 'missing_per_100k'] = (
        decade_agg.loc[has_pop, 'missing_count'] /
        decade_agg.loc[has_pop, 'population'] * 100_000
    )
    decade_agg.loc[has_pop, 'bodies_per_100k'] = (
        decade_agg.loc[has_pop, 'bodies_count'] /
        decade_agg.loc[has_pop, 'population'] * 100_000
    )
    decade_agg['missing_per_100k'] = decade_agg['missing_per_100k'].fillna(0)
    decade_agg['bodies_per_100k'] = decade_agg['bodies_per_100k'].fillna(0)

    # Calculate z-scores at decade level
    mp_mean = decade_agg.loc[has_pop, 'missing_per_100k'].mean()
    mp_std = decade_agg.loc[has_pop, 'missing_per_100k'].std()
    bodies_mean = decade_agg.loc[has_pop, 'bodies_per_100k'].mean()
    bodies_std = decade_agg.loc[has_pop, 'bodies_per_100k'].std()

    if mp_std > 0:
        decade_agg['mp_z_score'] = (decade_agg['missing_per_100k'] - mp_mean) / mp_std
    else:
        decade_agg['mp_z_score'] = 0.0

    if bodies_std > 0:
        decade_agg['bodies_z_score'] = (decade_agg['bodies_per_100k'] - bodies_mean) / bodies_std
    else:
        decade_agg['bodies_z_score'] = 0.0

    decade_agg['composite_z_score'] = (
        decade_agg['mp_z_score'] + decade_agg['bodies_z_score']
    ) / 2

    # Zero out scores for rows without population
    decade_agg.loc[~has_pop, ['mp_z_score', 'bodies_z_score', 'composite_z_score']] = 0.0

    # Alert classification
    decade_agg[['alert_level', 'alert_reason']] = decade_agg.apply(
        classify_alert_level, axis=1, result_type='expand'
    )

    # Save
    output_file = os.path.join(ANALYSIS_DIR, "county_decade_outliers.csv")
    decade_agg.to_csv(output_file, index=False)

    print(f"\nAggregated to {len(decade_agg):,} county-decade combinations")
    print(f"Decades: {sorted(decade_agg['decade'].unique())}")

    alert_counts = decade_agg['alert_level'].value_counts()
    print("\nDecade-Level Alert Distribution:")
    for level in ['RED', 'ORANGE', 'YELLOW', 'GREEN']:
        count = alert_counts.get(level, 0)
        print(f"  {level}: {count:,}")

    print(f"\nSaved to: {output_file}")

    return decade_agg


def validate_with_known_killers(df):
    """
    Validate the outlier detection system using known serial killers.
    """
    print("\n" + "=" * 70)
    print("VALIDATING WITH KNOWN SERIAL KILLERS")
    print("=" * 70)

    test_cases = [
        {
            'name': 'Green River Killer (Ridgway)',
            'state': 'WA', 'county': 'King',
            'years': range(1982, 1999),
        },
        {
            'name': 'John Wayne Gacy',
            'state': 'IL', 'county': 'Cook',
            'years': [1978],
        },
        {
            'name': 'Jeffrey Dahmer',
            'state': 'WI', 'county': 'Milwaukee',
            'years': range(1978, 1992),
        },
    ]

    for test in test_cases:
        print(f"\n{test['name']}:")
        subset = df[
            (df['State'] == test['state']) &
            (df['County'].str.contains(test['county'], case=False, na=False)) &
            (df['year'].isin(test['years']))
        ]

        if len(subset) > 0:
            avg_mp_z = subset['mp_z_score'].mean()
            avg_bodies_z = subset['bodies_z_score'].mean()
            alerts = subset['alert_level'].value_counts()

            print(f"  Years analyzed: {len(subset)}")
            print(f"  Avg MP Z-score: {avg_mp_z:.2f} sigma")
            print(f"  Avg Bodies Z-score: {avg_bodies_z:.2f} sigma")
            print(f"  Alerts: {dict(alerts)}")

            if avg_mp_z > 1 or avg_bodies_z > 1:
                print(f"  DETECTED as outlier")
            else:
                print(f"  NOT detected (may validate multi-tier need)")
        else:
            print(f"  No data found for this period")

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

    # Save county-year results
    output_file = os.path.join(ANALYSIS_DIR, "county_outlier_scores.csv")
    df_validated.to_csv(output_file, index=False)
    print(f"\nSaved outlier scores to: {output_file}")

    # Aggregate by decade
    df_decade = aggregate_by_decade(df_validated)

    # Show top 20 RED alerts
    red_alerts = df_validated[
        df_validated['alert_level'] == 'RED'
    ].sort_values('composite_z_score', ascending=False).head(20)

    if len(red_alerts) > 0:
        print(f"\nTOP 20 RED ALERTS (Extreme Outliers):")
        print("=" * 70)
        for _, row in red_alerts.iterrows():
            print(f"{row['County']}, {row['State']} ({int(row['year'])})")
            print(f"  MP: {row['missing_count']:.0f} ({row['missing_per_100k']:.1f}/100K, {row['mp_z_score']:.2f} sigma)")
            print(f"  Bodies: {row['bodies_count']:.0f} ({row['bodies_per_100k']:.1f}/100K, {row['bodies_z_score']:.2f} sigma)")
            print(f"  Alert: {row['alert_reason']}")
            print()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
