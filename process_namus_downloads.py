#!/usr/bin/env python3
"""
Auto-Process NamUs Downloads - Separate Multi-State CSVs
Watches Downloads folder and automatically separates by state
"""
import pandas as pd
import os
import glob
from pathlib import Path
from datetime import datetime

# Paths
DOWNLOADS_DIR = "/home/yeblad/Downloads"
PROJECT_DIR = "/home/yeblad/Desktop/Geospatial_Crime_Analysis/data/raw"
PROCESSED_LOG = os.path.join(PROJECT_DIR, "processed_files.log")

# State name to abbreviation mapping
STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC',
    'Puerto Rico': 'PR', 'Guam': 'GU', 'Virgin Islands': 'VI'
}

def load_processed_files():
    """Load list of already processed files"""
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def mark_file_processed(filename):
    """Mark file as processed"""
    with open(PROCESSED_LOG, 'a') as f:
        f.write(f"{filename}\n")

def detect_data_type(df):
    """
    Detect if this is Missing Persons or Unidentified Bodies
    """
    # Check column names to determine type
    if 'DBF' in df.columns:
        return 'unidentified_bodies'
    elif 'DLC' in df.columns or 'Date Missing' in df.columns or 'Last Seen' in df.columns or 'Missing Age' in df.columns or 'Case Number' in df.columns:
        return 'missing_persons'
    else:
        # Try to infer from first few rows
        print("WARNING: Could not auto-detect data type. Assuming unidentified_bodies.")
        return 'unidentified_bodies'

def separate_by_state(csv_path):
    """
    Separate multi-state CSV into individual state files
    """
    print(f"\n{'='*70}")
    print(f"Processing: {os.path.basename(csv_path)}")
    print(f"{'='*70}")

    # Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR: Could not load CSV: {e}")
        return []

    print(f"Total records: {len(df)}")

    # Check if 'State' column exists
    if 'State' not in df.columns:
        print("ERROR: No 'State' column found!")
        print(f"Available columns: {', '.join(df.columns)}")
        return []

    # Detect data type
    data_type = detect_data_type(df)
    print(f"Data type: {data_type}")

    # Group by state
    states = df['State'].unique()
    print(f"States found: {len(states)}")
    print(f"States: {', '.join(sorted(states))}")

    saved_files = []

    for state in states:
        state_df = df[df['State'] == state]
        count = len(state_df)

        # Get state abbreviation (use full name if not in dict)
        state_abbrev = STATE_ABBREV.get(state, state.replace(' ', '_'))

        # Create filename
        state_lower = state.lower().replace(' ', '_')
        filename = f"{state_lower}_{data_type}.csv"
        filepath = os.path.join(PROJECT_DIR, filename)

        # Check if file already exists
        if os.path.exists(filepath):
            # Append to existing file (avoid duplicates)
            existing_df = pd.read_csv(filepath)

            # Combine and drop duplicates (based on Case or Case Number column)
            if 'Case' in state_df.columns:
                combined_df = pd.concat([existing_df, state_df])
                combined_df = combined_df.drop_duplicates(subset=['Case'], keep='last')
                combined_df.to_csv(filepath, index=False)
                print(f"  ✓ {state} ({state_abbrev}): {count} new records (merged with existing {len(existing_df)} → {len(combined_df)} total)")
            elif 'Case Number' in state_df.columns:
                combined_df = pd.concat([existing_df, state_df])
                combined_df = combined_df.drop_duplicates(subset=['Case Number'], keep='last')
                combined_df.to_csv(filepath, index=False)
                print(f"  ✓ {state} ({state_abbrev}): {count} new records (merged with existing {len(existing_df)} → {len(combined_df)} total)")
            else:
                # No Case column, just append
                combined_df = pd.concat([existing_df, state_df])
                combined_df.to_csv(filepath, index=False)
                print(f"  ✓ {state} ({state_abbrev}): {count} new records (appended to existing)")
        else:
            # New file
            state_df.to_csv(filepath, index=False)
            print(f"  ✓ {state} ({state_abbrev}): {count} records saved to {filename}")

        saved_files.append((state, filepath, count))

    return saved_files

def generate_summary():
    """
    Generate summary of all collected data
    """
    print(f"\n{'='*70}")
    print("CURRENT DATA SUMMARY")
    print(f"{'='*70}\n")

    # Unidentified Bodies
    unidentified_files = glob.glob(os.path.join(PROJECT_DIR, "*_unidentified_bodies.csv"))
    print(f"UNIDENTIFIED BODIES: {len(unidentified_files)} states")
    print("-" * 70)

    total_bodies = 0
    state_summary = []

    for filepath in sorted(unidentified_files):
        state_name = os.path.basename(filepath).replace('_unidentified_bodies.csv', '').replace('_', ' ').title()
        df = pd.read_csv(filepath)
        count = len(df)
        total_bodies += count
        state_summary.append((state_name, count))

    # Sort by count descending
    state_summary.sort(key=lambda x: x[1], reverse=True)

    print(f"{'State':<25} {'Bodies':>10}")
    print("-" * 40)
    for state, count in state_summary:
        print(f"{state:<25} {count:>10,}")

    print("-" * 40)
    print(f"{'TOTAL':<25} {total_bodies:>10,}")

    # Missing Persons
    print(f"\n{'='*70}")
    missing_files = glob.glob(os.path.join(PROJECT_DIR, "*_missing_persons.csv"))
    print(f"MISSING PERSONS: {len(missing_files)} states")
    print("-" * 70)

    if missing_files:
        total_missing = 0
        missing_summary = []

        for filepath in sorted(missing_files):
            state_name = os.path.basename(filepath).replace('_missing_persons.csv', '').replace('_', ' ').title()
            df = pd.read_csv(filepath)
            count = len(df)
            total_missing += count
            missing_summary.append((state_name, count))

        missing_summary.sort(key=lambda x: x[1], reverse=True)

        print(f"{'State':<25} {'Missing':>10}")
        print("-" * 40)
        for state, count in missing_summary:
            print(f"{state:<25} {count:>10,}")

        print("-" * 40)
        print(f"{'TOTAL':<25} {total_missing:>10,}")
    else:
        print("No missing persons data collected yet.")

    print(f"\n{'='*70}")
    print(f"Data stored in: {PROJECT_DIR}")
    print(f"{'='*70}\n")

def main():
    """
    Main processing function
    """
    print("\n" + "="*70)
    print("NamUs Download Auto-Processor")
    print("="*70)

    # Create project directory if needed
    os.makedirs(PROJECT_DIR, exist_ok=True)

    # Find all download CSVs
    download_pattern = os.path.join(DOWNLOADS_DIR, "download_10-18-2025*.csv")
    csv_files = glob.glob(download_pattern)

    if not csv_files:
        print("\nNo new download files found.")
        print(f"Looking for: {download_pattern}")
        generate_summary()
        return

    print(f"\nFound {len(csv_files)} download file(s)")

    # Load processed files log
    processed = load_processed_files()

    # Process each file
    total_processed = 0
    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)

        # Skip if already processed
        if filename in processed:
            print(f"\nSkipping (already processed): {filename}")
            continue

        # Separate by state
        saved_files = separate_by_state(csv_file)

        if saved_files:
            # Mark as processed
            mark_file_processed(filename)
            total_processed += 1

    print(f"\n{'='*70}")
    print(f"Processed {total_processed} new file(s)")
    print(f"{'='*70}")

    # Generate summary
    generate_summary()

if __name__ == "__main__":
    main()
