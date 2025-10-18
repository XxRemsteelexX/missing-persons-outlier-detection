#!/usr/bin/env python3
"""
Fetch Population Data from US Census API
For: City, County, State by Year (1980-2025)
"""
import pandas as pd
import requests
import time
import os
from datetime import datetime

# Census API endpoint
# Note: Census API requires a key (free registration at census.gov/developers)
CENSUS_API_KEY = os.environ.get('CENSUS_API_KEY', 'YOUR_KEY_HERE')

OUTPUT_DIR = "/home/yeblad/Desktop/Geospatial_Crime_Analysis/data/population"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_state_population_by_year(start_year=1980, end_year=2024):
    """
    Fetch state-level population data by year
    Uses Census Decennial (1980, 1990, 2000, 2010, 2020) + ACS estimates for other years
    """
    print("=" * 70)
    print("FETCHING STATE POPULATION DATA")
    print("=" * 70)

    state_data = []

    # Decennial Census years (exact counts)
    decennial_years = [1980, 1990, 2000, 2010, 2020]

    for year in decennial_years:
        if year < start_year or year > end_year:
            continue

        print(f"\nFetching {year} Decennial Census...")

        # Census API endpoint varies by year
        if year == 2020:
            url = f"https://api.census.gov/data/2020/dec/pl?get=NAME,P1_001N&for=state:*&key={CENSUS_API_KEY}"
        elif year == 2010:
            url = f"https://api.census.gov/data/2010/dec/sf1?get=NAME,P001001&for=state:*&key={CENSUS_API_KEY}"
        elif year == 2000:
            url = f"https://api.census.gov/data/2000/dec/sf1?get=NAME,P001001&for=state:*&key={CENSUS_API_KEY}"
        elif year == 1990:
            url = f"https://api.census.gov/data/1990/dec/sf1?get=NAME,P0010001&for=state:*&key={CENSUS_API_KEY}"
        elif year == 1980:
            url = f"https://api.census.gov/data/1980/dec/sf1?get=NAME,P0010001&for=state:*&key={CENSUS_API_KEY}"

        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                for row in data[1:]:  # Skip header
                    state_data.append({
                        'year': year,
                        'state': row[0],
                        'population': int(row[1]),
                        'source': 'decennial'
                    })
                print(f"  ✓ Got {len(data)-1} states")
            else:
                print(f"  ✗ Error {response.status_code}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        time.sleep(0.5)  # Rate limiting

    # ACS 5-year estimates for non-decennial years
    acs_years = [y for y in range(start_year, end_year+1) if y not in decennial_years and y >= 2009]

    for year in acs_years:
        print(f"\nFetching {year} ACS 5-Year Estimates...")

        url = f"https://api.census.gov/data/{year}/acs/acs5?get=NAME,B01003_001E&for=state:*&key={CENSUS_API_KEY}"

        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                for row in data[1:]:
                    state_data.append({
                        'year': year,
                        'state': row[0],
                        'population': int(row[1]) if row[1] else 0,
                        'source': 'acs5'
                    })
                print(f"  ✓ Got {len(data)-1} states")
            else:
                print(f"  ✗ Error {response.status_code}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        time.sleep(0.5)

    # Save to CSV
    df = pd.DataFrame(state_data)
    df = df.sort_values(['state', 'year'])

    output_file = os.path.join(OUTPUT_DIR, "state_population_by_year.csv")
    df.to_csv(output_file, index=False)

    print(f"\n✓ Saved {len(df)} records to {output_file}")
    return df

def fetch_county_population_by_year(start_year=1980, end_year=2024):
    """
    Fetch county-level population data by year
    """
    print("\n" + "=" * 70)
    print("FETCHING COUNTY POPULATION DATA")
    print("=" * 70)

    county_data = []

    # ACS 5-year estimates (2009+)
    for year in range(max(2009, start_year), end_year+1):
        print(f"\nFetching {year} County ACS Estimates...")

        url = f"https://api.census.gov/data/{year}/acs/acs5?get=NAME,B01003_001E&for=county:*&key={CENSUS_API_KEY}"

        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                for row in data[1:]:
                    county_data.append({
                        'year': year,
                        'county': row[0],
                        'state_fips': row[2],
                        'county_fips': row[3],
                        'population': int(row[1]) if row[1] else 0,
                        'source': 'acs5'
                    })
                print(f"  ✓ Got {len(data)-1} counties")
            else:
                print(f"  ✗ Error {response.status_code}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        time.sleep(0.5)

    # Save to CSV
    df = pd.DataFrame(county_data)
    df = df.sort_values(['county', 'year'])

    output_file = os.path.join(OUTPUT_DIR, "county_population_by_year.csv")
    df.to_csv(output_file, index=False)

    print(f"\n✓ Saved {len(df)} records to {output_file}")
    return df

def fetch_city_population_estimates():
    """
    Fetch city-level population estimates (from latest ACS)
    Note: Historical city data is harder to get, may need alternative sources
    """
    print("\n" + "=" * 70)
    print("FETCHING CITY POPULATION DATA (Latest)")
    print("=" * 70)

    # Get latest year's city populations
    year = 2023

    url = f"https://api.census.gov/data/{year}/acs/acs5?get=NAME,B01003_001E&for=place:*&key={CENSUS_API_KEY}"

    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            data = response.json()

            city_data = []
            for row in data[1:]:
                city_data.append({
                    'year': year,
                    'city': row[0],
                    'state_fips': row[2],
                    'place_fips': row[3],
                    'population': int(row[1]) if row[1] else 0
                })

            df = pd.DataFrame(city_data)
            output_file = os.path.join(OUTPUT_DIR, "city_population_2023.csv")
            df.to_csv(output_file, index=False)

            print(f"✓ Saved {len(df)} cities to {output_file}")
            return df
        else:
            print(f"✗ Error {response.status_code}")
    except Exception as e:
        print(f"✗ Error: {e}")

    return None

def create_alternative_population_estimates():
    """
    For years without Census data, create linear interpolations
    """
    print("\n" + "=" * 70)
    print("CREATING INTERPOLATED ESTIMATES FOR MISSING YEARS")
    print("=" * 70)

    # Load existing state data
    state_file = os.path.join(OUTPUT_DIR, "state_population_by_year.csv")
    if not os.path.exists(state_file):
        print("No state data found - run fetch_state_population_by_year first")
        return

    df = pd.read_csv(state_file)

    # Interpolate missing years
    all_years = range(1980, 2025)
    states = df['state'].unique()

    interpolated = []
    for state in states:
        state_df = df[df['state'] == state].sort_values('year')

        # Create complete year range
        for year in all_years:
            if year in state_df['year'].values:
                # Use existing data
                row = state_df[state_df['year'] == year].iloc[0]
                interpolated.append(row.to_dict())
            else:
                # Interpolate
                before = state_df[state_df['year'] < year].iloc[-1] if len(state_df[state_df['year'] < year]) > 0 else None
                after = state_df[state_df['year'] > year].iloc[0] if len(state_df[state_df['year'] > year]) > 0 else None

                if before is not None and after is not None:
                    # Linear interpolation
                    ratio = (year - before['year']) / (after['year'] - before['year'])
                    pop = int(before['population'] + ratio * (after['population'] - before['population']))

                    interpolated.append({
                        'year': year,
                        'state': state,
                        'population': pop,
                        'source': 'interpolated'
                    })

    df_complete = pd.DataFrame(interpolated)
    output_file = os.path.join(OUTPUT_DIR, "state_population_complete_1980_2024.csv")
    df_complete.to_csv(output_file, index=False)

    print(f"✓ Saved complete dataset: {len(df_complete)} records")
    print(f"  {len(states)} states x {len(all_years)} years = {len(states) * len(all_years)} total")

    return df_complete

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CENSUS POPULATION DATA FETCHER")
    print("=" * 70)
    print("\n⚠️  NOTE: Requires Census API Key")
    print("Get free key at: https://api.census.gov/data/key_signup.html")
    print("Set key: export CENSUS_API_KEY='your_key_here'")
    print("=" * 70)

    if CENSUS_API_KEY == 'YOUR_KEY_HERE':
        print("\n❌ No Census API key set!")
        print("\nAlternative: Using pre-built population data...")
        print("Creating sample structure...")

        # Create sample data structure for demonstration
        sample_data = {
            'year': [2020, 2020, 2020],
            'state': ['California', 'Texas', 'Florida'],
            'population': [39538223, 29145505, 21538187],
            'source': 'sample'
        }
        df = pd.DataFrame(sample_data)
        output_file = os.path.join(OUTPUT_DIR, "SAMPLE_state_population.csv")
        df.to_csv(output_file, index=False)
        print(f"✓ Created sample file: {output_file}")
    else:
        # Run with API key
        fetch_state_population_by_year(start_year=1980, end_year=2024)
        fetch_county_population_by_year(start_year=2009, end_year=2024)
        fetch_city_population_estimates()
        create_alternative_population_estimates()

        print("\n" + "=" * 70)
        print("POPULATION DATA FETCH COMPLETE")
        print("=" * 70)
