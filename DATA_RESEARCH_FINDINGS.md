# Data Research Findings
## NamUs Access & Serial Killer Operating Patterns

**Created:** 2025-10-18
**Status:** Research Complete, Ready for Implementation

---

## 1. NamUs Data Access

### Access Method
- **Registration Required:** Yes - must register at namus.nij.ojp.gov
- **User Types:**
  - **Professional users** (law enforcement, medical examiners) - full access to sensitive data
  - **Public users** (family members, researchers) - limited access to public data
- **No Official API:** NamUs does not provide a public REST API for programmatic access

### Data Export Capabilities
- **CSV Export:** Available to all registered users
- **Batch Limit:** 10,000 cases per export
- **Workaround:** Export in batches by state to exceed 10K limit
- **Fields Available:**
  - Case ID
  - Date missing
  - Location (city, county, state)
  - Demographics (age, sex, race)
  - Case status (active, resolved, archived)

### Historical Data Range
- **NamUs Launch:** 2007 (Unidentified Persons), 2008 (Missing Persons)
- **Oldest Cases:** Date back to 1975 in some records
- **Coverage:** Primarily 2007-present, with some historical backfill
- **Data Completeness:** Varies by state (voluntary reporting)

### Critical Filter Needed
**MUST filter for unsolved cases only:**
- "Active" status = still missing (what we want)
- "Resolved" status = found alive/deceased (exclude these)
- Need to aggregate only cases that remained missing for extended periods

---

## 2. Serial Killer Geographic Operating Radius

### Key Research Findings

#### Distance from Home Base
**Source:** Criminology research on U.S. serial killers

| Metric | Distance | Notes |
|--------|----------|-------|
| **Mean distance to first encounter (PFE)** | **1.46 miles** | Where killer encounters victim |
| **Mean distance to body dump (BD)** | **14.3 miles** | Where body is discovered |
| **63% of German serial killers** | **<10 km (<6.2 miles)** | Lived within 10km of crime scenes |
| **Most U.S. serial killers** | **<10 miles** | Lived within 10 miles of victims |

#### Circle Hypothesis (Canter's Theory)
**Definition:** Serial killers operate within a circle defined by the diameter between their two farthest offenses.

**Research Validation:** Studies confirm serial killers live and dump victims within the search area established by the Circle Hypothesis.

**Practical Application:**
1. Find all missing persons in a cluster
2. Draw circle using two farthest points as diameter endpoints
3. Center of circle = likely home base of offender
4. Adjustable radius: **1-200 miles** (most will be <50 miles)

#### Distance Decay Function
**Pattern:** Crimes are more likely to occur closer to offender's home base, with probability decreasing with distance.

**Evolution:** As serial killers gain confidence, they venture further from home over time.

**Implication:** Early victims should cluster tightly around home base (1-10 miles), later victims may extend to 50-200 miles.

---

## 3. Child Trafficking Ring Geographic Patterns

### Spatial Clustering
**Source:** Research on sex trafficking spatial patterns

| Finding | Value | Notes |
|---------|-------|-------|
| **Spatial clustering** | **30% of activity in 4% of areas** | Highly concentrated |
| **Geographic correlation** | **Interstate highways** | Proximity to I-highways = risk factor |
| **Clustering tools** | **Kernel density estimation** | Used to map hotspots |
| **Catchment area** | **Not well-defined in literature** | Research gap |

### Risk Factors (Geographic)
- Proximity to interstate highways
- Cheaper hotels/motels (budget accommodations)
- Sexually oriented businesses (strip clubs, etc.)
- Concentrated disadvantage (poverty, drug use)

### Known Trafficking Cases for Validation
**Research Note:** Specific catchment radius data is limited in published literature.

**Approach:** Use known busted trafficking rings to empirically measure radius:
- Map all missing persons in ring's operation area
- Calculate radius from center to farthest victim
- Compare to serial killer radius (likely smaller for trafficking)

**Hypothesis:** Trafficking rings may have larger radius than serial killers (organized networks vs. lone actors)

---

## 4. Validation Cases (Known Serial Killers)

### Case 1: Gary Ridgway (Green River Killer)
- **Location:** King County, WA (Seattle-Tacoma corridor)
- **Active Period:** 1982-1998 (16 years)
- **Victims:** 49 confirmed, 71+ suspected
- **Operating Radius:** Stayed within King County / Pierce County area (~50 mile radius)
- **Expected Result:** Sustained outlier in King County throughout 1980s-1990s

### Case 2: Ted Bundy
- **Multi-State Offender:**
  - **Washington (1974):** King County, Thurston County
  - **Utah (1974-1975):** Salt Lake County
  - **Colorado (1975):** Multiple counties (Aspen, Vail areas)
  - **Florida (1978):** Leon County (Tallahassee), Seminole County
- **Pattern:** Sequential geographic outliers as he moved
- **Expected Result:** Outliers appear in correct temporal sequence across states

### Case 3: John Wayne Gacy
- **Location:** Cook County, IL (Des Plaines, Chicago suburbs)
- **Active Period:** 1972-1978 (6 years)
- **Victims:** 33 confirmed (all buried at home or nearby)
- **Operating Radius:** <10 miles (stayed very close to home)
- **Expected Result:** Cook County outlier in 1970s

### Case 4: Jeffrey Dahmer
- **Location:** Milwaukee County, WI
- **Active Period:** 1978-1991 (13 years, with gaps)
- **Victims:** 17 confirmed
- **Operating Radius:** <5 miles (apartment kills, stayed local)
- **Expected Result:** Milwaukee County outlier in 1980s-1990s

### Case 5: Dennis Rader (BTK Killer)
- **Location:** Sedgwick County, KS (Wichita)
- **Active Period:** 1974-1991 (17 years, sporadic)
- **Victims:** 10 confirmed
- **Operating Radius:** <20 miles (Wichita metro area)
- **Expected Result:** Sedgwick County outlier in 1970s-1980s

---

## 5. Radial Search Tool Design

### Circle Hypothesis Implementation

**User Interface:**
1. Select counties or draw region on map
2. Adjust radius slider: 1-200 miles (default: 50 miles)
3. System calculates:
   - Center point (weighted by missing persons density)
   - Circle encompassing selected points
   - All missing persons within radius

**Algorithm:**
```python
def find_circle_center_and_radius(missing_persons_coords):
    """
    Implement Circle Hypothesis:
    1. Find two farthest points in cluster
    2. Center = midpoint between them
    3. Radius = half the distance
    """
    max_distance = 0
    point_a, point_b = None, None

    # Find two farthest points
    for i, p1 in enumerate(missing_persons_coords):
        for j, p2 in enumerate(missing_persons_coords[i+1:]):
            dist = haversine_distance(p1, p2)
            if dist > max_distance:
                max_distance = dist
                point_a, point_b = p1, p2

    # Center is midpoint
    center_lat = (point_a[0] + point_b[0]) / 2
    center_lon = (point_a[1] + point_b[1]) / 2

    # Radius is half the max distance
    radius = max_distance / 2

    return (center_lat, center_lon), radius
```

**Adjustable Radius Feature:**
- User can override calculated radius
- Test different radii (10, 25, 50, 100, 200 miles)
- Highlight clusters that emerge at each radius
- Look for "hot centers" that appear consistently

**Visualization:**
- Red circle on map showing search radius
- Red dot at calculated center (potential home base)
- Missing persons plotted as points
- Density heatmap overlay

---

## 6. Data Collection Strategy (Updated)

### Phase 1: Register and Export NamUs Data

**Step 1: Create NamUs Account**
- Go to https://namus.nij.ojp.gov/
- Register as public user (or professional if you have credentials)
- Wait for account approval (may take 1-2 days)

**Step 2: Export Missing Persons Data**
- Search Missing Persons database
- Filter:
  - **Status:** Active only (still missing)
  - **Date Range:** 1975-2024 (get maximum historical data)
  - **All states**
- Export in batches of 10K cases per state
- Combine CSVs

**Step 3: Data Quality Check**
```python
# After loading NamUs data
print(f"Total cases: {len(namus_df)}")
print(f"Date range: {namus_df['date_missing'].min()} to {namus_df['date_missing'].max()}")
print(f"States covered: {namus_df['state'].nunique()}")
print(f"Counties covered: {namus_df['county'].nunique()}")
print(f"Cases with coordinates: {namus_df['lat'].notna().sum()}")
```

**Expected Data Coverage:**
- **Best case:** 2007-2024 (NamUs era)
- **Partial coverage:** 1990-2006 (some historical backfill)
- **Sparse:** Pre-1990 (very limited, may not be usable)

**Fallback Plan:**
If NamUs data doesn't go back far enough, we **cannot validate against 1970s-1980s serial killers** (Bundy, Gacy, Dahmer).

**Alternative Validation:**
- Focus on 1990s-2020s cases (Ridgway, modern cases)
- Use FBI UCR "Missing Persons" reports for historical context
- Acknowledge limitation in report: "System would be more powerful with complete historical data"

---

## 7. Population Data (Census API)

### Census API Key
- **Get Key:** https://api.census.gov/data/key_signup.html (instant approval)
- **Free Tier:** 500 requests/day (sufficient for this project)

### Data Endpoints

**Decennial Census (1980, 1990, 2000, 2010, 2020):**
```python
# Example: 2020 Census
url = "https://api.census.gov/data/2020/dec/pl"
params = {
    "get": "P1_001N,NAME",  # Total population + name
    "for": "county:*",       # All counties
    "key": CENSUS_API_KEY
}
```

**Annual Population Estimates (2011-2024):**
```python
# Example: 2023 estimates
url = "https://api.census.gov/data/2023/pep/population"
params = {
    "get": "POP,NAME",
    "for": "county:*",
    "key": CENSUS_API_KEY
}
```

**Date Range We Can Get:**
- 1980, 1990, 2000, 2010, 2020 (decennial)
- 2010-2024 (annual estimates)
- **Gap:** 1981-1989, 1991-1999, 2001-2009 (interpolate if needed)

---

## 8. Key Implementation Decisions

### Decision 1: Date Range
**Given NamUs limitation (2007+) and Census availability (1980+):**

**Recommended Approach:**
- **Primary Analysis:** 2010-2024 (best data quality, annual population estimates)
- **Extended Analysis:** 2007-2024 (full NamUs era)
- **Historical Context:** 1980, 1990, 2000 (decennial census only, acknowledge sparse missing persons data)

**Trade-off:** Cannot validate against Bundy, Gacy, Dahmer (pre-2007), but CAN validate against:
- Gary Ridgway (active until 1998, but victims discovered 2001-2003)
- Modern cases (2010s-2020s)

### Decision 2: Geographic Granularity
**County-level is correct choice:**
- NamUs provides county data
- Census provides county population
- Not too fine (zip code = too noisy)
- Not too coarse (state = patterns lost)

### Decision 3: Radial Search Default Radius
**Based on research, set defaults:**
- **Serial killers:** 50 miles (covers most cases, per research)
- **Trafficking rings:** 100 miles (hypothesis: larger networks)
- **User adjustable:** 1-200 miles

### Decision 4: Temporal Window
**Use 5-year rolling average (not 10-year):**
- Research shows serial killers active for 5-15 years
- 5-year window captures pattern without over-smoothing
- Can still detect 2-3 year sprees if sustained

---

## 9. Next Steps (Updated)

### Immediate (Today):
1. **Register for NamUs account** (may take 1-2 days for approval)
2. **Get Census API key** (instant)
3. **Set up project environment** (Python venv, install packages)

### While Waiting for NamUs Approval:
1. **Fetch Census population data** (1980-2024, all counties)
2. **Geocode counties** (get lat/lon centroids)
3. **Build outlier detection code** (IQR method, stratification)
4. **Build radial search tool** (Circle Hypothesis algorithm)

### After NamUs Access:
1. **Export missing persons data** (active cases only, all states)
2. **Merge with population data**
3. **Run analysis and validation**
4. **Build dashboard**

---

## 10. Known Limitations

**Data Limitations:**
1. NamUs only goes back to ~2007 (cannot validate against 1970s-1980s killers)
2. Voluntary reporting (some counties may under-report)
3. Data quality varies by state

**Methodological Limitations:**
1. Correlation ≠ causation (outliers may be poverty, drug epidemics, etc.)
2. Circle Hypothesis assumes home-based offenders (not applicable to transient/trucker killers)
3. Small population counties = high variance (1 missing person = 100 per 100K)

**Scope Limitations:**
1. Only detects patterns, not individuals
2. Cannot distinguish serial crime from other causes (natural disasters, migration, etc.)
3. Requires manual investigation of outliers

**Acknowledge These in Report:**
"This system is an investigative tool for pattern detection and resource allocation, not a predictive model for identifying specific offenders."

---

## 11. Expected Deliverables (Updated)

### What We CAN Deliver (3 Days):
✅ Interactive map of missing persons outliers (2010-2024)
✅ Radial search tool with Circle Hypothesis implementation
✅ Validation against modern cases (2000s-2020s)
✅ Streamlit dashboard with county deep-dive
✅ Technical report (10-15 pages)
✅ GitHub repo with clean code
✅ PowerPoint presentation

### What We CANNOT Deliver:
❌ Validation against Bundy, Gacy, Dahmer (pre-2007 data gap)
❌ Complete 1975-2024 timeline (NamUs data doesn't go back that far)
❌ Child trafficking ring validation (no public data on busted rings)

### How to Address Limitations:
**In Report:**
"Due to NamUs data availability (2007-present), this analysis focuses on 2010-2024. Historical validation against 1970s-1980s cases was not possible. Future work could incorporate FBI UCR historical data or state-level missing persons archives to extend analysis back to 1980."

**In FBI Application:**
"This demonstrates the methodology. With access to FBI databases (NCIC, ViCAP), this system could analyze complete historical data (1960-present) and cross-reference with known criminal cases."

---

## Summary: What We Learned

### NamUs:
- ✅ CSV export available (10K batches)
- ✅ Registration required
- ⚠️ Limited to 2007-present
- ❌ No official API

### Serial Killer Radius:
- ✅ Mean distance: 1.5 miles (encounter), 14 miles (body dump)
- ✅ Circle Hypothesis validated in research
- ✅ Default radius: 50 miles

### Trafficking Rings:
- ✅ Spatially clustered (30% in 4% of areas)
- ✅ Correlated with highways, cheap motels
- ⚠️ Specific catchment radius not well-studied

### Validation:
- ✅ Can validate against Ridgway (2000s discoveries)
- ❌ Cannot validate against Bundy, Gacy, Dahmer (pre-2007)

### Timeline:
- **Realistic:** 3 days for 2010-2024 analysis
- **Stretch:** 4-5 days if we add FBI UCR historical data

---

**Status:** Ready to start implementation
**Blocker:** NamUs account approval (1-2 days)
**Parallel Work:** Census data collection, outlier detection code, radial search tool

**Let's get the account registered and start building while we wait.**
