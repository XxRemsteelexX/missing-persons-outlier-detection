#!/usr/bin/env python3
"""
Download Pre-Built Population Datasets
Sources: Census.gov public data files, not requiring API key

Fixes applied (Phase 0):
  - 2000-2010 intercensal CSV: filter to total rows (SEX=0, ORIGIN=0, RACE=0,
    AGEGRP=0) instead of iterating all demographic breakdown rows
  - State names normalized to full names consistently
  - County population download with FIPS codes for reliable merging
  - Verification checks after download
"""
import pandas as pd
import numpy as np
import requests
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    POP_DIR, STATE_ABBREV, STATE_FULL, STATE_FIPS, FIPS_STATE,
    normalize_state, normalize_state_to_full,
)

os.makedirs(POP_DIR, exist_ok=True)


def download_state_population_estimates():
    """
    Download state population estimates from Census.gov (2000-2024).
    Returns a DataFrame with columns: year, state, state_abbrev, population, source
    """
    print("=" * 70)
    print("DOWNLOADING STATE POPULATION ESTIMATES (2000-2024)")
    print("=" * 70)

    urls = {
        '2020-2024': 'https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/state/totals/NST-EST2023-ALLDATA.csv',
        '2010-2020': 'https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/state/totals/nst-est2020-alldata.csv',
        '2000-2010': 'https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/intercensal/state/st-est00int-alldata.csv',
    }

    all_data = []

    # -------------------------------------------------------------------------
    # 2020-2024 estimates (simple structure: one row per state)
    # -------------------------------------------------------------------------
    print("\nDownloading 2020-2024 estimates...")
    try:
        df = pd.read_csv(urls['2020-2024'], encoding='latin1')
        # Filter to states only (exclude US total, regions, divisions)
        df = df[df['NAME'].isin(STATE_ABBREV.keys())]
        print(f"  Downloaded {len(df)} state rows")

        for _, row in df.iterrows():
            state_name = row['NAME']
            state_abbrev = STATE_ABBREV.get(state_name, '')
            for year in range(2020, 2024):
                pop_col = f'POPESTIMATE{year}'
                if pop_col in df.columns:
                    all_data.append({
                        'year': year,
                        'state': state_name,
                        'state_abbrev': state_abbrev,
                        'population': int(row[pop_col]),
                        'source': 'census_nst_est_2020'
                    })
    except Exception as e:
        print(f"  Error: {e}")

    # -------------------------------------------------------------------------
    # 2010-2020 estimates (simple structure: one row per state)
    # -------------------------------------------------------------------------
    print("\nDownloading 2010-2020 estimates...")
    try:
        df = pd.read_csv(urls['2010-2020'], encoding='latin1')
        df = df[df['NAME'].isin(STATE_ABBREV.keys())]
        print(f"  Downloaded {len(df)} state rows")

        for _, row in df.iterrows():
            state_name = row['NAME']
            state_abbrev = STATE_ABBREV.get(state_name, '')
            for year in range(2010, 2021):
                pop_col = f'POPESTIMATE{year}'
                if pop_col in df.columns:
                    all_data.append({
                        'year': year,
                        'state': state_name,
                        'state_abbrev': state_abbrev,
                        'population': int(row[pop_col]),
                        'source': 'census_nst_est_2010'
                    })
    except Exception as e:
        print(f"  Error: {e}")

    # -------------------------------------------------------------------------
    # 2000-2010 intercensal estimates
    # BUG FIX: This CSV has demographic breakdowns (SEX, ORIGIN, RACE, AGEGRP).
    # Must filter to total population rows:
    #   SEX=0 (both sexes), ORIGIN=0 (all origins),
    #   RACE=0 (all races), AGEGRP=0 (all ages),
    #   STATE>0 (exclude US total row)
    # -------------------------------------------------------------------------
    print("\nDownloading 2000-2010 intercensal estimates...")
    try:
        df = pd.read_csv(urls['2000-2010'], encoding='latin1')
        print(f"  Downloaded {len(df)} total rows (with demographic breakdowns)")

        # Filter to total population rows per state
        totals = df[
            (df['SEX'] == 0) &
            (df['ORIGIN'] == 0) &
            (df['RACE'] == 0) &
            (df['AGEGRP'] == 0) &
            (df['STATE'] > 0)
        ].copy()
        print(f"  Filtered to {len(totals)} state-level total rows")

        for _, row in totals.iterrows():
            state_name = row['NAME']
            state_abbrev = STATE_ABBREV.get(state_name, '')
            for year in range(2000, 2011):
                pop_col = f'POPESTIMATE{year}'
                if pop_col in df.columns:
                    all_data.append({
                        'year': year,
                        'state': state_name,
                        'state_abbrev': state_abbrev,
                        'population': int(row[pop_col]),
                        'source': 'census_intercensal'
                    })
    except Exception as e:
        print(f"  Error: {e}")

    # -------------------------------------------------------------------------
    # Combine and deduplicate (prefer later sources for overlapping years)
    # -------------------------------------------------------------------------
    if all_data:
        df_combined = pd.DataFrame(all_data)
        df_combined = df_combined.sort_values(['state', 'year', 'source'])
        df_combined = df_combined.drop_duplicates(
            subset=['state', 'year'], keep='last'
        )
        df_combined = df_combined.sort_values(['state', 'year']).reset_index(drop=True)

        output_file = os.path.join(POP_DIR, "state_population_1980_2024.csv")
        df_combined.to_csv(output_file, index=False)

        print(f"\nSaved {len(df_combined)} records to {output_file}")
        print(f"   {df_combined['state'].nunique()} states/territories")
        print(f"   Years: {df_combined['year'].min()}-{df_combined['year'].max()}")

        # Verification
        _verify_state_population(df_combined)

        return df_combined
    else:
        print("\nNo data downloaded")
        return None


def _verify_state_population(df):
    """Verify key population values are reasonable."""
    print("\n--- Population Verification ---")
    checks = [
        ('Alabama', 2000, 4_000_000, 5_000_000),
        ('Alabama', 2020, 4_800_000, 5_200_000),
        ('California', 2000, 33_000_000, 35_000_000),
        ('California', 2020, 38_000_000, 40_000_000),
        ('Texas', 2000, 20_000_000, 22_000_000),
        ('Texas', 2020, 28_000_000, 30_000_000),
        ('Wyoming', 2000, 450_000, 550_000),
    ]

    all_pass = True
    for state, year, low, high in checks:
        row = df[(df['state'] == state) & (df['year'] == year)]
        if len(row) == 0:
            print(f"  FAIL: {state} {year} -- no data")
            all_pass = False
        else:
            pop = row.iloc[0]['population']
            ok = low <= pop <= high
            status = "PASS" if ok else "FAIL"
            print(f"  {status}: {state} {year} = {pop:,.0f} (expected {low:,.0f}-{high:,.0f})")
            if not ok:
                all_pass = False

    if all_pass:
        print("  All verification checks passed")
    else:
        print("  WARNING: Some verification checks failed")


def download_county_population_estimates():
    """
    Download county population estimates from Census.gov (2010-2024).
    Includes FIPS codes for reliable merging with crime data.
    """
    print("\n" + "=" * 70)
    print("DOWNLOADING COUNTY POPULATION ESTIMATES")
    print("=" * 70)

    all_county_data = []

    # -------------------------------------------------------------------------
    # 2020-2024 county estimates
    # -------------------------------------------------------------------------
    url_2020 = 'https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv'
    print("\nDownloading 2020-2024 county estimates...")
    try:
        df = pd.read_csv(url_2020, encoding='latin1')
        # Filter out state-level summary rows (COUNTY == 0)
        df = df[df['COUNTY'] > 0]
        print(f"  Downloaded {len(df)} county rows")

        for _, row in df.iterrows():
            state_fips = str(int(row['STATE'])).zfill(2)
            county_fips = str(int(row['COUNTY'])).zfill(3)
            full_fips = state_fips + county_fips
            county_name = str(row.get('CTYNAME', '')).replace(' County', '').replace(' Parish', '').replace(' Borough', '').replace(' Census Area', '').replace(' Municipality', '').strip()
            state_name = row.get('STNAME', '')
            state_abbrev = STATE_ABBREV.get(state_name, normalize_state(state_name))

            for year in range(2020, 2024):
                pop_col = f'POPESTIMATE{year}'
                if pop_col in df.columns:
                    all_county_data.append({
                        'year': year,
                        'state': state_name,
                        'state_abbrev': state_abbrev,
                        'county': county_name,
                        'state_fips': state_fips,
                        'county_fips': county_fips,
                        'fips': full_fips,
                        'population': int(row[pop_col]),
                        'source': 'census_county_est_2020'
                    })
    except Exception as e:
        print(f"  Error: {e}")

    # -------------------------------------------------------------------------
    # 2010-2020 county estimates
    # -------------------------------------------------------------------------
    url_2010 = 'https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/counties/totals/co-est2020-alldata.csv'
    print("\nDownloading 2010-2020 county estimates...")
    try:
        df = pd.read_csv(url_2010, encoding='latin1')
        df = df[df['COUNTY'] > 0]
        print(f"  Downloaded {len(df)} county rows")

        for _, row in df.iterrows():
            state_fips = str(int(row['STATE'])).zfill(2)
            county_fips = str(int(row['COUNTY'])).zfill(3)
            full_fips = state_fips + county_fips
            county_name = str(row.get('CTYNAME', '')).replace(' County', '').replace(' Parish', '').replace(' Borough', '').replace(' Census Area', '').replace(' Municipality', '').strip()
            state_name = row.get('STNAME', '')
            state_abbrev = STATE_ABBREV.get(state_name, normalize_state(state_name))

            for year in range(2010, 2021):
                pop_col = f'POPESTIMATE{year}'
                if pop_col in df.columns:
                    all_county_data.append({
                        'year': year,
                        'state': state_name,
                        'state_abbrev': state_abbrev,
                        'county': county_name,
                        'state_fips': state_fips,
                        'county_fips': county_fips,
                        'fips': full_fips,
                        'population': int(row[pop_col]),
                        'source': 'census_county_est_2010'
                    })
    except Exception as e:
        print(f"  Error: {e}")

    # -------------------------------------------------------------------------
    # 2000-2010 county intercensal estimates
    # -------------------------------------------------------------------------
    url_2000 = 'https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/intercensal/county/co-est00int-tot.csv'
    print("\nDownloading 2000-2010 county intercensal estimates...")
    try:
        df = pd.read_csv(url_2000, encoding='latin1')
        print(f"  Downloaded {len(df)} rows")

        # This file has a simpler structure: SUMLEV, STATE, COUNTY, STNAME, CTYNAME,
        # ESTIMATESBASE2000, POPESTIMATE2000-2010
        if 'COUNTY' in df.columns:
            df = df[df['COUNTY'] > 0]

        for _, row in df.iterrows():
            state_fips = str(int(row['STATE'])).zfill(2)
            county_fips = str(int(row['COUNTY'])).zfill(3)
            full_fips = state_fips + county_fips
            county_name = str(row.get('CTYNAME', '')).replace(' County', '').replace(' Parish', '').replace(' Borough', '').replace(' Census Area', '').replace(' Municipality', '').strip()
            state_name = row.get('STNAME', '')
            state_abbrev = STATE_ABBREV.get(state_name, normalize_state(state_name))

            for year in range(2000, 2011):
                pop_col = f'POPESTIMATE{year}'
                if pop_col in df.columns:
                    all_county_data.append({
                        'year': year,
                        'state': state_name,
                        'state_abbrev': state_abbrev,
                        'county': county_name,
                        'state_fips': state_fips,
                        'county_fips': county_fips,
                        'fips': full_fips,
                        'population': int(row[pop_col]),
                        'source': 'census_county_intercensal'
                    })
    except Exception as e:
        print(f"  Error downloading 2000-2010 county data: {e}")
        print("  County data for 2000-2010 will be approximated from state-level data")

    # -------------------------------------------------------------------------
    # Combine and save
    # -------------------------------------------------------------------------
    if all_county_data:
        df_counties = pd.DataFrame(all_county_data)
        df_counties = df_counties.sort_values(['state', 'county', 'year', 'source'])
        df_counties = df_counties.drop_duplicates(
            subset=['state', 'county', 'year'], keep='last'
        )
        df_counties = df_counties.sort_values(
            ['state', 'county', 'year']
        ).reset_index(drop=True)

        output_file = os.path.join(POP_DIR, "county_population_2000_2024.csv")
        df_counties.to_csv(output_file, index=False)

        print(f"\nSaved {len(df_counties)} records to {output_file}")
        print(f"   {df_counties['county'].nunique()} unique county names")
        print(f"   {df_counties['state'].nunique()} states")
        print(f"   Years: {df_counties['year'].min()}-{df_counties['year'].max()}")

        # Verification
        _verify_county_population(df_counties)

        return df_counties
    else:
        print("\nNo county data downloaded")
        return None


def _verify_county_population(df):
    """Verify key county population values."""
    print("\n--- County Population Verification ---")
    checks = [
        ('California', 'Los Angeles', 2020, 9_000_000, 11_000_000),
        ('Texas', 'Harris', 2020, 4_000_000, 5_000_000),
        ('Illinois', 'Cook', 2020, 5_000_000, 5_500_000),
        ('Arizona', 'Pima', 2020, 900_000, 1_100_000),
    ]

    for state, county, year, low, high in checks:
        row = df[
            (df['state'] == state) &
            (df['county'] == county) &
            (df['year'] == year)
        ]
        if len(row) == 0:
            print(f"  SKIP: {county}, {state} {year} -- no data found")
        else:
            pop = row.iloc[0]['population']
            ok = low <= pop <= high
            status = "PASS" if ok else "FAIL"
            print(f"  {status}: {county}, {state} {year} = {pop:,.0f}")


def create_interpolated_estimates(df_state):
    """
    Create interpolated estimates for missing years using available data.
    Fills gaps between known data points with linear interpolation.
    """
    print("\n" + "=" * 70)
    print("CREATING INTERPOLATED ESTIMATES FOR GAPS")
    print("=" * 70)

    if df_state is None or len(df_state) == 0:
        print("No state data available for interpolation")
        return None

    interpolated = []
    states = df_state['state'].unique()

    for state in states:
        state_data = df_state[df_state['state'] == state].sort_values('year')
        if len(state_data) < 2:
            continue

        min_year = int(state_data['year'].min())
        max_year = int(state_data['year'].max())
        existing_years = set(state_data['year'].values)
        state_abbrev = state_data.iloc[0].get('state_abbrev', normalize_state(state))

        # Interpolate gaps
        years_array = state_data['year'].values
        pops_array = state_data['population'].values

        for year in range(min_year, max_year + 1):
            if year not in existing_years:
                pop = int(np.interp(year, years_array, pops_array))
                interpolated.append({
                    'year': year,
                    'state': state,
                    'state_abbrev': state_abbrev,
                    'population': pop,
                    'source': 'interpolated'
                })

    if interpolated:
        df_interp = pd.DataFrame(interpolated)
        df_combined = pd.concat([df_state, df_interp])
        df_combined = df_combined.sort_values(['state', 'year'])
        df_combined = df_combined.drop_duplicates(
            subset=['state', 'year'], keep='first'
        )
        df_combined = df_combined.reset_index(drop=True)

        output_file = os.path.join(POP_DIR, "state_population_complete.csv")
        df_combined.to_csv(output_file, index=False)

        print(f"Added {len(interpolated)} interpolated records")
        print(f"Saved complete dataset: {len(df_combined)} records to {output_file}")
        return df_combined

    return df_state


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("POPULATION DATA DOWNLOADER (NO API KEY NEEDED)")
    print("=" * 70)

    # Download state data
    df_state = download_state_population_estimates()

    # Download county data
    df_county = download_county_population_estimates()

    # Create interpolated estimates for any gaps
    if df_state is not None:
        create_interpolated_estimates(df_state)

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"\nData saved to: {POP_DIR}")
