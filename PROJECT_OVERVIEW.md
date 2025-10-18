# Geospatial Temporal Crime Pattern Analysis System
## Detecting Serial Crime Patterns Through Population-Normalized Anomaly Detection

**Project Goal:** Build an interactive geospatial-temporal analysis system to identify statistical outliers in missing persons cases across space and time, enabling detection of serial crime patterns and investigative leads.

**Status:** Planning / Data Collection
**Timeline:** 2-3 days to working demo
**Target Application:** FBI Field Specialist Data Scientist role + Portfolio piece

---

## Table of Contents
1. [The Core Idea](#the-core-idea)
2. [Why This Works](#why-this-works)
3. [Data Sources](#data-sources)
4. [Methodology](#methodology)
5. [Technical Architecture](#technical-architecture)
6. [Validation Strategy](#validation-strategy)
7. [Deliverables](#deliverables)
8. [Implementation Plan](#implementation-plan)
9. [Code Structure](#code-structure)

---

## The Core Idea

### The Problem:
**Serial criminals leave geographic and temporal patterns, but they're hidden in noisy data.**

- Missing persons vary by population (NYC has more than small-town Iowa)
- Different area types have different baselines (urban vs. rural)
- Patterns emerge over YEARS, not single snapshots
- Traditional analysis compares raw numbers (wrong)

### The Solution:
**Population-normalized, area-stratified, temporal outlier detection**

**Step 1: Normalize by population**
- Don't compare raw counts (NYC will always be higher)
- Use: Missing persons per 100,000 residents
- Accounts for population differences

**Step 2: Stratify by area type**
- Don't compare cities to rural areas
- Categories: Metro / Small City / Small Town / Rural
- Create separate baselines for each type

**Step 3: Temporal outlier detection**
- Don't look at single years (too noisy)
- Use multi-year windows (5-year or 10-year)
- Identify statistical outliers using box plots (IQR method)

**Step 4: Geospatial correlation**
- Map outliers over time
- Look for patterns that MOVE (suspect relocates)
- Example: Hudson, Iowa outlier in 1980s → Wisconsin outlier in 1990s
- Correlation suggests a mobile suspect

**Step 5: Interactive visualization**
- Map with time slider (select decades)
- Highlight statistical outliers in red
- Click county → show temporal trend
- Identify current AND historical hotspots

### The Insight:
**This doesn't FIND the person. It shows WHERE to look and WHEN patterns occurred.**

**Use cases:**
- Cold case prioritization (which regions need investigation?)
- Pattern correlation (did suspect move? follow the outliers)
- Resource allocation (where to deploy investigators?)
- Hotspot identification (child kidnapping rings, trafficking routes)

---

## Why This Works

### Statistical Foundation:
**Box Plot Outlier Detection:**
- Calculate Q1 (25th percentile), Q3 (75th percentile)
- IQR = Q3 - Q1
- Outliers: Values > Q3 + 1.5*IQR or < Q1 - 1.5*IQR
- Robust to extreme values, works with skewed distributions

**Stratification Prevents False Positives:**
- Rural Iowa baseline: ~5 missing per 100K
- Urban NYC baseline: ~15 missing per 100K
- If you don't stratify, rural areas NEVER appear as outliers
- Stratification: Compare Iowa to similar rural areas, NYC to similar metros

**Temporal Smoothing Reduces Noise:**
- Single year: High variance (1 missing person in small town = 50 per 100K)
- 10-year average: Smooths variance, reveals true patterns
- Box plot across years: Identifies sustained elevation, not random spikes

### Real-World Examples (Hypothetical):

**Example 1: The Moving Pattern**
```
1980s: Hudson, Iowa → 15 missing per 100K (rural baseline: 5)
        Statistical outlier: Yes (3× baseline)

1990s: Hudson, Iowa → 4 missing per 100K (back to normal)
        La Crosse, Wisconsin → 18 missing per 100K (rural baseline: 5)
        Statistical outlier: Yes (3.6× baseline)

Correlation: Suspect may have moved Iowa → Wisconsin
              Check: Did anyone move from Hudson to La Crosse area in late 1980s?
```

**Example 2: The Persistent Hotspot**
```
Seattle-Tacoma corridor: Elevated missing persons 1982-1998
Statistical outlier: Yes (sustained, not spike)
Known case: Gary Ridgway (Green River Killer) active 1982-1998
Validation: System would have flagged this region
```

**Example 3: The Network**
```
Multiple rural counties in same state: Simultaneous elevation
Pattern: Not random, clustered geographically
Hypothesis: Organized ring (trafficking, kidnapping)
Investigative lead: Look for connections between counties
```

---

## Data Sources

### Primary Data Sources (All Public):

#### 1. **NamUs - National Missing and Unidentified Persons System**
**URL:** https://www.namus.gov/
**API:** https://www.namus.gov/api/ (may require registration)
**Data Available:**
- Missing persons cases (name, date, location)
- Geographic data (county, state, coordinates)
- Temporal data (date reported missing)
- Demographics (age, sex, race)

**Access Method:**
- Web scraping (if API restricted)
- Manual download (CSV exports available)
- Historical data back to 2007+

**Data Fields Needed:**
```
- Case ID
- Date reported missing
- County FIPS code
- State
- Lat/Lon (if available)
```

#### 2. **FBI UCR - Uniform Crime Reporting**
**URL:** https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/home
**Data Available:**
- Crime statistics by jurisdiction
- Missing persons (limited, but available)
- Historical data (1960-present)

**Access Method:**
- FBI Crime Data Explorer API
- CSV downloads by year
- State-level and county-level aggregations

**API Endpoint:**
```
https://api.usa.gov/crime/fbi/sapi/
```

#### 3. **US Census Bureau - Population Data**
**URL:** https://www.census.gov/data/developers/data-sets.html
**API:** https://api.census.gov/data.html
**Data Available:**
- Population by county (annual estimates)
- Historical decennial census (1980, 1990, 2000, 2010, 2020)
- American Community Survey (ACS) - annual updates

**Access Method:**
```python
# Census API (free, requires key)
import requests

API_KEY = "your_key_here"
url = f"https://api.census.gov/data/2020/dec/pl?get=P1_001N&for=county:*&key={API_KEY}"
response = requests.get(url)
population_data = response.json()
```

**Data Fields Needed:**
```
- Year
- County FIPS code
- State FIPS code
- Total population
```

#### 4. **OMB Metro/Micro Area Definitions**
**URL:** https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html
**Data Available:**
- Metropolitan Statistical Area (MSA) definitions
- Micropolitan Statistical Area definitions
- Rural classification

**Access Method:**
- Download Excel file
- Parse into area type categories

**Categories:**
```
- Metropolitan (>50K population)
- Micropolitan (10K-50K population)
- Rural (<10K population)
- OR use custom bins: [0-10K, 10K-50K, 50K-250K, 250K+]
```

#### 5. **USGS Geocoding Service** (Optional)
**URL:** https://geocoding.geo.census.gov/geocoder/
**Use:** Convert addresses to lat/lon if not provided in source data

---

### Secondary Data Sources (For Validation):

#### 6. **Wikipedia / News Archives - Known Serial Crime Cases**
**Use:** Validation dataset
**Cases to test:**
```
- Ted Bundy (WA, UT, CO, FL - 1974-1978)
- Gary Ridgway (WA - 1982-1998)
- John Wayne Gacy (IL - 1972-1978)
- Jeffrey Dahmer (WI - 1978-1991)
- Dennis Rader (KS - 1974-1991)
```

**Question:** Would the system flag these counties as outliers during active periods?

#### 7. **State-Level Missing Persons Databases**
**Examples:**
- California DOJ Missing Persons
- Texas DPS Missing Persons
- Florida FDLE Missing Persons

**Use:** Supplement federal data where available

---

### Data Structure (Target Schema)

```python
# Final analysis dataset
{
    'year': int,                    # 1980-2024
    'county_fips': str,             # '19001' (5-digit FIPS)
    'state': str,                   # 'Iowa'
    'state_fips': str,              # '19'
    'county_name': str,             # 'Adair County'
    'population': int,              # Census data
    'missing_persons': int,         # Count from NamUs/FBI
    'missing_per_100k': float,      # Normalized rate
    'area_type': str,               # 'metro' / 'micro' / 'rural'
    'lat': float,                   # County centroid
    'lon': float,                   # County centroid
    'is_outlier': bool,             # Statistical outlier flag
    'outlier_score': float,         # Z-score or IQR distance
    'decade': str                   # '1980s', '1990s', etc.
}
```

---

## Methodology

### Phase 1: Data Collection & Cleaning

**Step 1: Gather population data**
```python
# Census API for 1980-2024 county population
for year in range(1980, 2025):
    if year % 10 == 0:  # Decennial census
        endpoint = f"census/{year}/dec/pl"
    else:  # Annual estimates
        endpoint = f"pep/{year}/population"

    # Get county population
    population_df = fetch_census_data(year, endpoint)
```

**Step 2: Gather missing persons data**
```python
# NamUs data
namus_df = scrape_namus_cases()  # All historical cases
# OR
namus_df = pd.read_csv("namus_export.csv")  # Manual download

# Aggregate by county + year
missing_counts = namus_df.groupby(['county_fips', 'year']).size()
```

**Step 3: Merge datasets**
```python
# Join population + missing persons
df = population_df.merge(missing_counts,
                         on=['county_fips', 'year'],
                         how='left')
df['missing_persons'] = df['missing_persons'].fillna(0)
```

**Step 4: Classify area types**
```python
# Load OMB metro definitions
metro_df = pd.read_excel("metro_delineation.xlsx")

# Classify counties
def classify_area_type(population):
    if population >= 250000:
        return 'metro'
    elif population >= 50000:
        return 'small_city'
    elif population >= 10000:
        return 'small_town'
    else:
        return 'rural'

df['area_type'] = df['population'].apply(classify_area_type)
```

### Phase 2: Normalize & Calculate Rates

```python
# Calculate missing persons per 100K
df['missing_per_100k'] = (df['missing_persons'] / df['population']) * 100000

# Handle edge cases
df['missing_per_100k'] = df['missing_per_100k'].replace([np.inf, -np.inf], np.nan)
df['missing_per_100k'] = df['missing_per_100k'].fillna(0)
```

### Phase 3: Temporal Aggregation

```python
# Create decade bins
df['decade'] = (df['year'] // 10) * 10
df['decade_label'] = df['decade'].astype(str) + 's'

# Rolling averages (5-year windows)
df = df.sort_values(['county_fips', 'year'])
df['missing_per_100k_5yr_avg'] = df.groupby('county_fips')['missing_per_100k'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)
```

### Phase 4: Outlier Detection (Stratified by Area Type)

```python
def detect_outliers_by_area_type(df, area_type, decade):
    """
    Detect outliers using IQR method within area type + decade
    """
    subset = df[(df['area_type'] == area_type) & (df['decade'] == decade)]

    # Calculate quartiles
    Q1 = subset['missing_per_100k_5yr_avg'].quantile(0.25)
    Q3 = subset['missing_per_100k_5yr_avg'].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Flag outliers
    subset['is_outlier'] = (
        (subset['missing_per_100k_5yr_avg'] < lower_bound) |
        (subset['missing_per_100k_5yr_avg'] > upper_bound)
    )

    # Calculate outlier score (how many IQRs away from median)
    median = subset['missing_per_100k_5yr_avg'].median()
    subset['outlier_score'] = (subset['missing_per_100k_5yr_avg'] - median) / IQR

    return subset

# Apply to all area types and decades
results = []
for area_type in ['metro', 'small_city', 'small_town', 'rural']:
    for decade in [1980, 1990, 2000, 2010, 2020]:
        outliers = detect_outliers_by_area_type(df, area_type, decade)
        results.append(outliers)

df_with_outliers = pd.concat(results)
```

### Phase 5: Geospatial Hotspot Analysis

```python
from scipy.spatial import distance
from scipy.stats import zscore

# Z-score for geospatial context
df['zscore'] = df.groupby(['decade', 'area_type'])['missing_per_100k_5yr_avg'].transform(zscore)

# Identify high-confidence hotspots (z > 2.0)
hotspots = df[df['zscore'] > 2.0]

# Cluster nearby hotspots (potential networks)
from sklearn.cluster import DBSCAN

coords = hotspots[['lat', 'lon']].values
clustering = DBSCAN(eps=1.0, min_samples=3).fit(coords)  # eps in degrees (~100km)
hotspots['cluster_id'] = clustering.labels_
```

---

## Technical Architecture

### Tech Stack:

**Data Processing:**
- Python 3.10+
- pandas (data manipulation)
- numpy (numerical operations)
- scipy (statistical analysis)

**Geospatial:**
- geopandas (geospatial operations)
- shapely (geometric operations)
- folium OR plotly (mapping)

**Visualization:**
- Plotly (interactive plots + maps)
- Streamlit (dashboard)
- Tableau Public (alternative for dashboard)

**Data Sources:**
- requests (API calls)
- BeautifulSoup (web scraping if needed)

### File Structure:

```
Geospatial_Crime_Analysis/
├── README.md
├── PROJECT_OVERVIEW.md (this file)
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── namus_cases.csv
│   │   ├── census_population.csv
│   │   ├── metro_definitions.xlsx
│   │   └── fbi_ucr_data.csv
│   ├── processed/
│   │   ├── county_analysis.csv
│   │   ├── outliers_by_decade.csv
│   │   └── geospatial_hotspots.geojson
│   └── validation/
│       └── known_cases.csv
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_outlier_detection.ipynb
│   ├── 04_geospatial_analysis.ipynb
│   └── 05_validation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_collection.py
│   ├── data_processing.py
│   ├── outlier_detection.py
│   ├── geospatial_analysis.py
│   └── visualization.py
├── dashboard/
│   └── streamlit_app.py
├── results/
│   ├── figures/
│   ├── maps/
│   └── report.pdf
└── validation/
    └── case_studies.md
```

---

## Validation Strategy

### Test Cases (Known Serial Criminals):

**Case 1: Gary Ridgway (Green River Killer)**
- **Location:** King County, WA (Seattle-Tacoma)
- **Active Period:** 1982-1998
- **Expected Result:** King County should appear as outlier in 1980s-1990s
- **Validation:** Compare missing persons rate 1982-1998 vs. baseline

**Case 2: Ted Bundy**
- **Locations:**
  - WA (1974): King County, Thurston County
  - UT (1974-1975): Salt Lake County
  - CO (1975): Multiple counties
  - FL (1978): Leon County, Seminole County
- **Expected Result:** Sequential geographic outliers following his movements
- **Validation:** Do outliers appear in correct sequence and locations?

**Case 3: John Wayne Gacy**
- **Location:** Cook County, IL (Chicago suburbs)
- **Active Period:** 1972-1978
- **Expected Result:** Cook County outlier in 1970s
- **Validation:** Was there statistical elevation during this period?

**Case 4: Jeffrey Dahmer**
- **Location:** Milwaukee County, WI
- **Active Period:** 1978-1991
- **Expected Result:** Milwaukee County outlier in 1980s-1990s
- **Validation:** Sustained elevation across two decades?

### Validation Metrics:

**True Positive Rate:**
```
# Of 10 known serial crime locations
# How many appear as statistical outliers during active periods?
# Target: >70% detection rate
```

**False Positive Analysis:**
```
# Of counties flagged as outliers
# How many have alternative explanations? (poverty, drug epidemic, etc.)
# Investigate top 10 current outliers: What's causing elevation?
```

**Temporal Accuracy:**
```
# Do outliers appear DURING active periods (not before/after)?
# Compare: Ridgway active 1982-1998 → outlier appears 1985-2000 (yes)
#          Ridgway active 1982-1998 → outlier appears 2005-2010 (no, false positive)
```

---

## Deliverables

### 1. Interactive Dashboard (Streamlit)

**Features:**
- County-level choropleth map
- Time slider (select decade: 1980s, 1990s, 2000s, 2010s, 2020s)
- Color coding: Red = outlier, Yellow = elevated, Green = baseline
- Click county → popup with:
  - Missing persons per 100K (time series graph)
  - Area type classification
  - Outlier status by decade
  - Comparison to area type baseline
- Filter by area type (metro / city / town / rural)
- Search by county or state

**Technical:**
- Streamlit + Plotly
- Deployed to Streamlit Cloud (free hosting)
- Public URL for demo

### 2. Technical Report (PDF, 10-15 pages)

**Sections:**
1. Executive Summary (1 page)
   - Key findings in 3 bullets
   - Top 10 current outliers
   - Validation results (% of known cases detected)

2. Methodology (3 pages)
   - Population normalization approach
   - Area type stratification
   - Temporal outlier detection (IQR method)
   - Geospatial clustering

3. Data Sources (1 page)
   - NamUs, FBI UCR, Census
   - Date ranges, completeness

4. Analysis Results (5 pages)
   - Outlier maps by decade
   - Temporal trends (line graphs)
   - Geospatial clusters (hotspot maps)
   - Statistical tables

5. Validation (2 pages)
   - Known case detection rate
   - False positive analysis
   - Limitations and caveats

6. Investigative Applications (2 pages)
   - Cold case prioritization
   - Resource allocation recommendations
   - Correlation analysis (do patterns move?)

7. Conclusion & Future Work (1 page)
   - Summary of capabilities
   - Potential enhancements (add crime type, victim demographics, etc.)

### 3. Code Repository (GitHub)

**Contents:**
- Clean, documented Python code
- Jupyter notebooks showing analysis
- README with:
  - Project description
  - How to run the code
  - Data sources and access methods
  - Example outputs
- requirements.txt for reproducibility
- MIT License (or similar)

**Repository Name:** `geospatial-crime-analysis`

### 4. Presentation Slides (PowerPoint, 10-12 slides)

**Outline:**
1. Title slide (your name, project title)
2. Problem statement (serial crimes leave patterns)
3. Solution approach (population-normalized outlier detection)
4. Data sources (NamUs, Census, FBI)
5. Methodology diagram (data flow)
6. Example: Map showing outliers in 1980s
7. Example: Map showing outliers in 1990s
8. Validation: Known case detection results
9. Demo: Link to interactive dashboard
10. Applications: How FBI could use this
11. About you: RSNA performance, infrastructure, skills
12. Contact & GitHub link

---

## Implementation Plan

### Day 1: Data Collection & Setup

**Morning (4 hours):**
- [ ] Set up project structure (folders, files)
- [ ] Install dependencies (requirements.txt)
- [ ] Register for Census API key
- [ ] Download NamUs data (manual or scrape)
- [ ] Download FBI UCR data
- [ ] Download OMB metro definitions

**Afternoon (4 hours):**
- [ ] Clean Census population data (1980-2024)
- [ ] Clean NamUs missing persons data
- [ ] Merge datasets by county + year
- [ ] Geocode counties (lat/lon from Census or USGS)

**Evening (2 hours):**
- [ ] Exploratory data analysis (Jupyter notebook)
- [ ] Check data completeness by decade
- [ ] Identify any gaps or issues

### Day 2: Analysis & Visualization

**Morning (4 hours):**
- [ ] Calculate missing persons per 100K
- [ ] Classify area types (metro/city/town/rural)
- [ ] Temporal aggregation (5-year rolling averages)
- [ ] Outlier detection (IQR method, stratified)

**Afternoon (4 hours):**
- [ ] Geospatial clustering (DBSCAN)
- [ ] Z-score analysis for hotspots
- [ ] Create choropleth maps (Plotly)
- [ ] Build time-series graphs

**Evening (2 hours):**
- [ ] Start Streamlit dashboard
- [ ] Basic map with time slider
- [ ] County click functionality

### Day 3: Validation & Packaging

**Morning (4 hours):**
- [ ] Validate against known cases (Bundy, Ridgway, etc.)
- [ ] Calculate detection rate
- [ ] Analyze false positives
- [ ] Document findings

**Afternoon (4 hours):**
- [ ] Finish Streamlit dashboard
- [ ] Deploy to Streamlit Cloud
- [ ] Write technical report (PDF)
- [ ] Create presentation slides

**Evening (2 hours):**
- [ ] Polish GitHub README
- [ ] Record quick demo video (optional)
- [ ] Prepare application email
- [ ] Submit to FBI Field Specialist role

---

## Code Structure

### requirements.txt

```txt
# Data processing
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Geospatial
geopandas>=0.13.0
shapely>=2.0.0
folium>=0.14.0

# Visualization
plotly>=5.14.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Dashboard
streamlit>=1.22.0

# Data collection
requests>=2.31.0
beautifulsoup4>=4.12.0

# ML/Stats
scikit-learn>=1.2.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.65.0
```

### src/data_collection.py

```python
"""
Data collection utilities for geospatial crime analysis
"""
import pandas as pd
import requests
from typing import Dict, List
import os
from dotenv import load_dotenv

load_dotenv()

CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")

def fetch_census_population(year: int, state_fips: str = "*") -> pd.DataFrame:
    """
    Fetch county population data from Census API

    Args:
        year: Census year (1980, 1990, 2000, 2010, 2020 or intermediate)
        state_fips: State FIPS code or "*" for all states

    Returns:
        DataFrame with columns: year, state_fips, county_fips, population
    """
    # Determine API endpoint based on year
    if year % 10 == 0 and year >= 1990:
        # Decennial census
        endpoint = f"https://api.census.gov/data/{year}/dec/pl"
        pop_var = "P001001"  # Total population variable
    else:
        # Population estimates
        endpoint = f"https://api.census.gov/data/{year}/pep/population"
        pop_var = "POP"

    # Build API request
    params = {
        "get": f"{pop_var},NAME",
        "for": "county:*",
        "key": CENSUS_API_KEY
    }

    if state_fips != "*":
        params["in"] = f"state:{state_fips}"

    response = requests.get(endpoint, params=params)

    if response.status_code != 200:
        raise Exception(f"Census API error: {response.status_code}")

    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])
    df['year'] = year
    df['population'] = pd.to_numeric(df[pop_var])
    df['county_fips'] = df['state'] + df['county']

    return df[['year', 'state', 'county', 'county_fips', 'NAME', 'population']]


def load_namus_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean NamUs missing persons data

    Args:
        filepath: Path to NamUs CSV export

    Returns:
        DataFrame with columns: case_id, date_missing, county_fips, state, lat, lon
    """
    df = pd.read_csv(filepath)

    # Clean and standardize
    # (Actual column names depend on NamUs export format)
    df['date_missing'] = pd.to_datetime(df['DateMissing'], errors='coerce')
    df['year'] = df['date_missing'].dt.year

    # Extract location info
    # This will depend on NamUs data format
    # May need geocoding if county FIPS not provided

    return df


def geocode_counties() -> pd.DataFrame:
    """
    Get lat/lon centroids for all US counties

    Returns:
        DataFrame with county_fips, lat, lon
    """
    # Option 1: Use Census TIGER/Line shapefiles
    # Option 2: Use pre-computed centroids from GitHub

    url = "https://raw.githubusercontent.com/btskinner/spatial/master/data/county_centers.csv"
    df = pd.read_csv(url)

    return df


def classify_area_type(population: int) -> str:
    """
    Classify county by population

    Args:
        population: County population

    Returns:
        Area type: 'metro', 'small_city', 'small_town', or 'rural'
    """
    if population >= 250000:
        return 'metro'
    elif population >= 50000:
        return 'small_city'
    elif population >= 10000:
        return 'small_town'
    else:
        return 'rural'
```

### src/outlier_detection.py

```python
"""
Outlier detection for geospatial crime analysis
"""
import pandas as pd
import numpy as np
from typing import Tuple

def calculate_outlier_bounds(series: pd.Series) -> Tuple[float, float, float]:
    """
    Calculate IQR-based outlier bounds

    Args:
        series: Pandas Series of values

    Returns:
        (lower_bound, upper_bound, IQR)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return lower_bound, upper_bound, IQR


def detect_outliers_stratified(df: pd.DataFrame,
                               value_col: str = 'missing_per_100k',
                               stratify_cols: List[str] = ['area_type', 'decade']) -> pd.DataFrame:
    """
    Detect outliers within each stratum (area type + decade)

    Args:
        df: DataFrame with crime data
        value_col: Column to analyze for outliers
        stratify_cols: Columns to stratify by

    Returns:
        DataFrame with added columns: is_outlier, outlier_score
    """
    results = []

    for name, group in df.groupby(stratify_cols):
        lower, upper, IQR = calculate_outlier_bounds(group[value_col])

        group = group.copy()
        group['is_outlier'] = (
            (group[value_col] < lower) |
            (group[value_col] > upper)
        )

        # Calculate outlier score (distance from median in IQR units)
        median = group[value_col].median()
        if IQR > 0:
            group['outlier_score'] = (group[value_col] - median) / IQR
        else:
            group['outlier_score'] = 0

        results.append(group)

    return pd.concat(results)


def temporal_smoothing(df: pd.DataFrame,
                      window: int = 5,
                      value_col: str = 'missing_per_100k') -> pd.DataFrame:
    """
    Apply rolling average to smooth temporal noise

    Args:
        df: DataFrame with temporal data
        window: Rolling window size (years)
        value_col: Column to smooth

    Returns:
        DataFrame with smoothed column
    """
    df = df.sort_values(['county_fips', 'year'])

    df[f'{value_col}_smoothed'] = df.groupby('county_fips')[value_col].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )

    return df
```

### src/visualization.py

```python
"""
Visualization utilities for geospatial crime analysis
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
from typing import Optional

def create_choropleth_map(df: pd.DataFrame,
                          decade: int,
                          area_type: Optional[str] = None) -> go.Figure:
    """
    Create interactive choropleth map of outliers

    Args:
        df: DataFrame with analysis results
        decade: Decade to visualize (1980, 1990, etc.)
        area_type: Filter by area type (None = all)

    Returns:
        Plotly Figure
    """
    # Filter data
    plot_df = df[df['decade'] == decade].copy()
    if area_type:
        plot_df = plot_df[plot_df['area_type'] == area_type]

    # Load county geometries
    # This requires a GeoJSON or shapefile of US counties
    # You can get this from Census TIGER/Line or use built-in plotly data

    fig = px.choropleth(
        plot_df,
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations='county_fips',
        color='missing_per_100k_smoothed',
        color_continuous_scale='RdYlGn_r',
        scope='usa',
        labels={'missing_per_100k_smoothed': 'Missing per 100K'},
        hover_data=['county_name', 'area_type', 'is_outlier', 'outlier_score'],
        title=f'Missing Persons Analysis - {decade}s'
    )

    fig.update_layout(
        geo=dict(
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'
        )
    )

    return fig


def create_temporal_plot(df: pd.DataFrame, county_fips: str) -> go.Figure:
    """
    Create time series plot for a specific county

    Args:
        df: DataFrame with temporal data
        county_fips: County FIPS code

    Returns:
        Plotly Figure
    """
    county_data = df[df['county_fips'] == county_fips].sort_values('year')

    fig = go.Figure()

    # Add line for missing persons rate
    fig.add_trace(go.Scatter(
        x=county_data['year'],
        y=county_data['missing_per_100k'],
        mode='lines+markers',
        name='Missing per 100K',
        line=dict(color='blue')
    ))

    # Add smoothed trend
    fig.add_trace(go.Scatter(
        x=county_data['year'],
        y=county_data['missing_per_100k_smoothed'],
        mode='lines',
        name='5-Year Average',
        line=dict(color='red', dash='dash')
    ))

    # Highlight outlier years
    outlier_years = county_data[county_data['is_outlier']]
    fig.add_trace(go.Scatter(
        x=outlier_years['year'],
        y=outlier_years['missing_per_100k'],
        mode='markers',
        name='Outlier',
        marker=dict(color='red', size=10, symbol='x')
    ))

    county_name = county_data['county_name'].iloc[0] if len(county_data) > 0 else county_fips
    fig.update_layout(
        title=f'Missing Persons Trend - {county_name}',
        xaxis_title='Year',
        yaxis_title='Missing Persons per 100,000',
        hovermode='x unified'
    )

    return fig
```

### dashboard/streamlit_app.py (Skeleton)

```python
"""
Streamlit dashboard for geospatial crime analysis
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
sys.path.append('../src')

from visualization import create_choropleth_map, create_temporal_plot

# Page config
st.set_page_config(
    page_title="Geospatial Crime Analysis",
    page_icon="🗺️",
    layout="wide"
)

# Title
st.title("🗺️ Geospatial Temporal Crime Pattern Analysis")
st.markdown("**Detecting Serial Crime Patterns Through Population-Normalized Anomaly Detection**")

# Load data
@st.cache_data
def load_data():
    # Load your processed data here
    df = pd.read_csv("../data/processed/county_analysis.csv")
    return df

df = load_data()

# Sidebar controls
st.sidebar.header("Filters")

decade = st.sidebar.select_slider(
    "Select Decade",
    options=[1980, 1990, 2000, 2010, 2020],
    value=2010,
    format_func=lambda x: f"{x}s"
)

area_type = st.sidebar.selectbox(
    "Area Type",
    options=["All", "Metro", "Small City", "Small Town", "Rural"],
)

# Main map
st.header(f"Missing Persons Analysis - {decade}s")

area_filter = None if area_type == "All" else area_type.lower().replace(" ", "_")
fig_map = create_choropleth_map(df, decade, area_filter)
st.plotly_chart(fig_map, use_container_width=True)

# County selector
st.header("County Deep Dive")

county_list = df['county_name'].unique()
selected_county = st.selectbox("Select County", sorted(county_list))

if selected_county:
    county_fips = df[df['county_name'] == selected_county]['county_fips'].iloc[0]
    fig_temporal = create_temporal_plot(df, county_fips)
    st.plotly_chart(fig_temporal, use_container_width=True)

    # Show stats
    county_data = df[df['county_fips'] == county_fips]
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_rate = county_data['missing_per_100k'].mean()
        st.metric("Avg Missing per 100K", f"{avg_rate:.2f}")

    with col2:
        outlier_years = county_data['is_outlier'].sum()
        st.metric("Outlier Years", outlier_years)

    with col3:
        area_type_val = county_data['area_type'].iloc[0]
        st.metric("Area Type", area_type_val.title())

# Footer
st.markdown("---")
st.markdown("**Data Sources:** NamUs, FBI UCR, US Census Bureau")
st.markdown("**Created by:** Glenn Dalbey | [GitHub](https://github.com/XxRemsteelexX)")
```

---

## Next Steps

**Immediate Actions:**
1. [ ] Register for Census API key (census.gov/developers)
2. [ ] Download NamUs data or determine scraping approach
3. [ ] Set up Python environment and install dependencies
4. [ ] Create folder structure as outlined above

**Data Collection (Day 1):**
1. [ ] Fetch Census population data (1980-2024)
2. [ ] Load missing persons data
3. [ ] Geocode counties
4. [ ] Merge datasets

**Analysis (Day 2):**
1. [ ] Calculate normalized rates
2. [ ] Detect outliers
3. [ ] Create visualizations
4. [ ] Build dashboard

**Validation & Delivery (Day 3):**
1. [ ] Test against known cases
2. [ ] Write report
3. [ ] Deploy dashboard
4. [ ] Apply to FBI role

---

## Key Success Metrics

**Technical:**
- ✅ Data coverage: All US counties, 1980-2024
- ✅ Validation: >70% detection rate on known serial crime cases
- ✅ Performance: Dashboard loads in <5 seconds
- ✅ Usability: Non-technical users can explore data

**Career:**
- ✅ Demonstrates geospatial analysis skills
- ✅ Shows systems thinking and methodology
- ✅ Proves domain understanding (crime analysis)
- ✅ Portfolio piece for FBI or law enforcement roles
- ✅ Can be discussed in interviews (take-home quality)

---

**Created:** 2025-10-17
**Author:** Glenn Dalbey
**Status:** Ready to execute
**Target Completion:** 3 days

**Let's catch some patterns.**
