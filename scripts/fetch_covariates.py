#!/usr/bin/env python3
"""
Fetch Socioeconomic Covariates from Census ACS 5-Year Estimates

Downloads county-level socioeconomic data for covariate adjustment:
  - Poverty rate
  - Median household income
  - Unemployment rate
  - Percent foreign-born
  - Population density (requires land area)
  - Urbanization proxy (population in urban areas)

Uses the Census API (no key required for basic queries, but rate-limited).
Falls back to pre-built ACS data files if API is unavailable.
"""
import pandas as pd
import numpy as np
import requests
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    COVARIATES_DIR, DATA_DIR, POP_DIR,
    STATE_FIPS, FIPS_STATE, STATE_FULL,
    normalize_state,
)

os.makedirs(COVARIATES_DIR, exist_ok=True)

# ACS 5-year variables of interest
# See https://api.census.gov/data/2022/acs/acs5/variables.html
ACS_VARIABLES = {
    'B17001_001E': 'total_poverty_universe',    # Total population for poverty status
    'B17001_002E': 'below_poverty',             # Below poverty level
    'B19013_001E': 'median_household_income',   # Median household income
    'B23025_003E': 'civilian_labor_force',       # In civilian labor force
    'B23025_005E': 'unemployed',                 # Unemployed
    'B05002_001E': 'total_nativity_universe',   # Total for nativity
    'B05002_013E': 'foreign_born',              # Foreign born
    'B01003_001E': 'total_population',          # Total population
}


def fetch_acs_county_data(year=2022):
    """
    Fetch ACS 5-year county-level data from Census API.
    No API key required for basic access.
    """
    print(f"Fetching ACS 5-year data for {year}...")

    variables = ','.join(ACS_VARIABLES.keys())
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        'get': f'NAME,{variables}',
        'for': 'county:*',
        'in': 'state:*',
    }

    try:
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Error fetching ACS data: {e}")
        return None

    # Parse response: first row is header, rest is data
    header = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=header)

    # Rename columns
    rename_map = {v: k for k, v in ACS_VARIABLES.items()}
    # Actually reverse: ACS code -> our name
    for acs_code, our_name in ACS_VARIABLES.items():
        if acs_code in df.columns:
            df = df.rename(columns={acs_code: our_name})

    # Convert numeric columns and replace Census sentinel values
    for col in ACS_VARIABLES.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Census uses -666666666 as sentinel for missing/suppressed data
            df[col] = df[col].replace(-666666666, np.nan)
            df.loc[df[col] < 0, col] = np.nan

    # Build FIPS
    df['state_fips'] = df['state'].str.zfill(2)
    df['county_fips'] = df['county'].str.zfill(3)
    df['fips'] = df['state_fips'] + df['county_fips']

    # Extract state abbreviation and county name from NAME
    # NAME format: "County Name, State Name"
    df['county_name'] = df['NAME'].str.split(',').str[0].str.replace(
        ' County', ''
    ).str.replace(' Parish', '').str.replace(' Borough', '').str.replace(
        ' Census Area', ''
    ).str.replace(' Municipality', '').str.strip()

    df['state_name'] = df['NAME'].str.split(',').str[-1].str.strip()
    df['state_abbrev'] = df['state_name'].apply(normalize_state)

    print(f"  Downloaded {len(df)} county records for {year}")

    return df


def compute_derived_variables(df):
    """
    Compute derived socioeconomic indicators from raw ACS variables.
    """
    print("Computing derived variables...")

    # Poverty rate
    df['poverty_rate'] = np.where(
        df['total_poverty_universe'] > 0,
        df['below_poverty'] / df['total_poverty_universe'] * 100,
        np.nan
    )

    # Unemployment rate
    df['unemployment_rate'] = np.where(
        df['civilian_labor_force'] > 0,
        df['unemployed'] / df['civilian_labor_force'] * 100,
        np.nan
    )

    # Percent foreign-born
    df['pct_foreign_born'] = np.where(
        df['total_nativity_universe'] > 0,
        df['foreign_born'] / df['total_nativity_universe'] * 100,
        np.nan
    )

    # Log population (proxy for urbanization)
    df['log_population'] = np.where(
        df['total_population'] > 0,
        np.log10(df['total_population']),
        np.nan
    )

    return df


def fetch_land_area():
    """
    Fetch county land area from Census gazetteer files for population density.
    """
    print("Fetching county land area from Census gazetteer...")

    urls = [
        'https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_counties_national.txt',
        'https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.txt',
        'https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.txt',
    ]

    for url in urls:
        try:
            df = pd.read_csv(url, sep='\t', dtype={'GEOID': str}, encoding='latin1')
            df.columns = df.columns.str.strip()
            df['fips'] = df['GEOID'].str.zfill(5)
            area_col = [c for c in df.columns if 'ALAND' in c and 'SQMI' in c]
            if area_col:
                df = df[['fips', area_col[0]]].rename(columns={area_col[0]: 'land_area_sqmi'})
                print(f"  Downloaded land area for {len(df)} counties from {url.split('/')[-1]}")
                return df
        except Exception as e:
            print(f"  Failed {url.split('/')[-1]}: {e}")
            continue

    # Fallback: compute density from population/area later if possible
    print("  Could not download gazetteer data; population density will be unavailable")
    return None


def build_covariate_dataset():
    """
    Build the complete county-level covariate dataset.
    """
    print("=" * 70)
    print("BUILDING COUNTY SOCIOECONOMIC COVARIATE DATASET")
    print("=" * 70)

    # Fetch ACS data (try 2022, fall back to 2021)
    df = fetch_acs_county_data(2022)
    if df is None:
        print("Trying 2021 ACS data...")
        df = fetch_acs_county_data(2021)

    if df is None:
        print("Could not fetch ACS data from Census API")
        return None

    # Compute derived variables
    df = compute_derived_variables(df)

    # Fetch land area for population density
    land_area = fetch_land_area()
    if land_area is not None:
        df = df.merge(land_area, on='fips', how='left')
        df['pop_density'] = np.where(
            df['land_area_sqmi'] > 0,
            df['total_population'] / df['land_area_sqmi'],
            np.nan
        )
        df['log_pop_density'] = np.where(
            df['pop_density'] > 0,
            np.log10(df['pop_density']),
            np.nan
        )
    else:
        df['land_area_sqmi'] = np.nan
        df['pop_density'] = np.nan
        df['log_pop_density'] = np.nan

    # Select output columns
    output_cols = [
        'fips', 'state_fips', 'county_fips', 'state_abbrev', 'state_name',
        'county_name', 'total_population',
        'poverty_rate', 'median_household_income', 'unemployment_rate',
        'pct_foreign_born', 'log_population', 'pop_density', 'log_pop_density',
        'land_area_sqmi',
    ]

    df_out = df[[c for c in output_cols if c in df.columns]].copy()

    # Save
    output_file = os.path.join(COVARIATES_DIR, "county_socioeconomic.csv")
    df_out.to_csv(output_file, index=False)

    print(f"\nSaved {len(df_out)} county records to {output_file}")

    # Summary statistics
    print("\n--- Covariate Summary ---")
    for col in ['poverty_rate', 'median_household_income', 'unemployment_rate',
                'pct_foreign_born', 'pop_density']:
        if col in df_out.columns:
            vals = df_out[col].dropna()
            print(f"  {col}: mean={vals.mean():.2f}, median={vals.median():.2f}, "
                  f"min={vals.min():.2f}, max={vals.max():.2f}")

    return df_out


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CENSUS ACS COVARIATE FETCHER")
    print("=" * 70)

    df = build_covariate_dataset()

    if df is not None:
        print("\n" + "=" * 70)
        print("COVARIATE FETCH COMPLETE")
        print("=" * 70)
    else:
        print("\nFailed to fetch covariates")
