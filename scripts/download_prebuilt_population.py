#!/usr/bin/env python3
"""
Download Pre-Built Population Datasets
Sources: Census.gov public data files, not requiring API key
"""
import pandas as pd
import requests
import os

OUTPUT_DIR = "/home/yeblad/Desktop/Geospatial_Crime_Analysis/data/population"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_state_population_estimates():
    """
    Download state population estimates from Census.gov
    Uses NST-EST (National/State Population Totals)
    """
    print("=" * 70)
    print("DOWNLOADING STATE POPULATION ESTIMATES (1980-2024)")
    print("=" * 70)

    # Census provides annual state population estimates as CSV
    # https://www2.census.gov/programs-surveys/popest/tables/

    urls = {
        '2020-2024': 'https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/state/totals/NST-EST2023-ALLDATA.csv',
        '2010-2020': 'https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/state/totals/nst-est2020-alldata.csv',
        '2000-2010': 'https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/intercensal/state/st-est00int-alldata.csv',
        '1990-2000': 'https://www2.census.gov/programs-surveys/popest/tables/1990-2000/state/totals/st-99-03.txt',
        '1980-1990': 'https://www2.census.gov/programs-surveys/popest/tables/1980-1990/state/asrh/st8090ts.txt'
    }

    all_data = []

    # Download 2020-2024 estimates
    print("\nDownloading 2020-2024 estimates...")
    try:
        df = pd.read_csv(urls['2020-2024'], encoding='latin1')
        print(f"  ✓ Downloaded {len(df)} rows")

        # Extract relevant columns
        for idx, row in df.iterrows():
            state_name = row.get('NAME', '')
            for year in range(2020, 2024):
                pop_col = f'POPESTIMATE{year}'
                if pop_col in df.columns:
                    all_data.append({
                        'year': year,
                        'state': state_name,
                        'population': row[pop_col],
                        'source': 'census_nst_est'
                    })
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Download 2010-2020 estimates
    print("\nDownloading 2010-2020 estimates...")
    try:
        df = pd.read_csv(urls['2010-2020'], encoding='latin1')
        print(f"  ✓ Downloaded {len(df)} rows")

        for idx, row in df.iterrows():
            state_name = row.get('NAME', '')
            for year in range(2010, 2021):
                pop_col = f'POPESTIMATE{year}'
                if pop_col in df.columns:
                    all_data.append({
                        'year': year,
                        'state': state_name,
                        'population': row[pop_col],
                        'source': 'census_nst_est'
                    })
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Download 2000-2010 intercensal estimates
    print("\nDownloading 2000-2010 estimates...")
    try:
        df = pd.read_csv(urls['2000-2010'], encoding='latin1')
        print(f"  ✓ Downloaded {len(df)} rows")

        for idx, row in df.iterrows():
            state_name = row.get('NAME', '')
            for year in range(2000, 2011):
                pop_col = f'POPESTIMATE{year}'
                if pop_col in df.columns:
                    all_data.append({
                        'year': year,
                        'state': state_name,
                        'population': row[pop_col],
                        'source': 'census_intercensal'
                    })
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Save combined data
    if all_data:
        df_combined = pd.DataFrame(all_data)
        df_combined = df_combined.sort_values(['state', 'year'])
        df_combined = df_combined.drop_duplicates(subset=['state', 'year'], keep='last')

        output_file = os.path.join(OUTPUT_DIR, "state_population_1980_2024.csv")
        df_combined.to_csv(output_file, index=False)

        print(f"\n✅ Saved {len(df_combined)} records to {output_file}")
        print(f"   {len(df_combined['state'].unique())} states")
        print(f"   {len(df_combined['year'].unique())} years")

        return df_combined
    else:
        print("\n❌ No data downloaded")
        return None

def download_county_population_estimates():
    """
    Download county population estimates from Census.gov
    """
    print("\n" + "=" * 70)
    print("DOWNLOADING COUNTY POPULATION ESTIMATES (2010-2024)")
    print("=" * 70)

    # County estimates (more limited time range)
    url = 'https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv'

    print("\nDownloading county estimates...")
    try:
        df = pd.read_csv(url, encoding='latin1')
        print(f"  ✓ Downloaded {len(df)} counties")

        # Extract data by year
        county_data = []
        for idx, row in df.iterrows():
            state_fips = str(row.get('STATE', '')).zfill(2)
            county_fips = str(row.get('COUNTY', '')).zfill(3)
            county_name = row.get('CTYNAME', '')
            state_name = row.get('STNAME', '')

            for year in range(2020, 2024):
                pop_col = f'POPESTIMATE{year}'
                if pop_col in df.columns:
                    county_data.append({
                        'year': year,
                        'state': state_name,
                        'county': county_name,
                        'state_fips': state_fips,
                        'county_fips': county_fips,
                        'population': row[pop_col],
                        'source': 'census_county_est'
                    })

        df_counties = pd.DataFrame(county_data)
        output_file = os.path.join(OUTPUT_DIR, "county_population_2020_2024.csv")
        df_counties.to_csv(output_file, index=False)

        print(f"\n✅ Saved {len(df_counties)} records to {output_file}")
        print(f"   {len(df_counties['county'].unique())} counties")

        return df_counties
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None

def create_interpolated_estimates(df_state):
    """
    Create interpolated estimates for missing years (1980-1999)
    """
    print("\n" + "=" * 70)
    print("CREATING INTERPOLATED ESTIMATES (1980-1999)")
    print("=" * 70)

    if df_state is None or len(df_state) == 0:
        print("No state data available for interpolation")
        return None

    # Get decennial census benchmarks
    benchmarks = {
        1980: {},  # Will need to add manually or estimate
        1990: {},
        2000: {}
    }

    # Extract 1990, 2000 from existing data
    for state in df_state['state'].unique():
        state_data = df_state[df_state['state'] == state]

        for year in [1990, 2000]:
            year_data = state_data[state_data['year'] == year]
            if len(year_data) > 0:
                benchmarks[year][state] = year_data.iloc[0]['population']

    # Interpolate between decades
    interpolated = []

    for state in df_state['state'].unique():
        # 1990-2000 interpolation
        if state in benchmarks[1990] and state in benchmarks[2000]:
            pop_1990 = benchmarks[1990][state]
            pop_2000 = benchmarks[2000][state]

            for year in range(1990, 2000):
                if not ((df_state['state'] == state) & (df_state['year'] == year)).any():
                    ratio = (year - 1990) / (2000 - 1990)
                    pop = int(pop_1990 + ratio * (pop_2000 - pop_1990))

                    interpolated.append({
                        'year': year,
                        'state': state,
                        'population': pop,
                        'source': 'interpolated'
                    })

    if interpolated:
        # Combine with existing data
        df_interp = pd.DataFrame(interpolated)
        df_combined = pd.concat([df_state, df_interp])
        df_combined = df_combined.sort_values(['state', 'year'])
        df_combined = df_combined.drop_duplicates(subset=['state', 'year'], keep='first')

        output_file = os.path.join(OUTPUT_DIR, "state_population_complete_1990_2024.csv")
        df_combined.to_csv(output_file, index=False)

        print(f"✅ Created complete dataset: {len(df_combined)} records")
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

    # Create interpolated estimates
    if df_state is not None:
        create_interpolated_estimates(df_state)

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"\nData saved to: {OUTPUT_DIR}")
