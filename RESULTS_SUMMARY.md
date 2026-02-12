# Geospatial Crime Analysis System - Results Summary

**Project:** Multi-Tier Anomaly Detection for Serial Crime Patterns
**Author:** Glenn Dalbey
**Purpose:** Predictive Crime Pattern Analysis
**Date:** February 2026
**Status:** Full statistical + ML pipeline validated

---

## Executive Summary

Built a statistical and machine learning outlier detection system that identifies anomalous crime patterns across the United States using:
- **41,200 total cases** (15,457 unidentified bodies + 25,743 missing persons)
- **55 states/territories** with complete coverage
- **7 statistical methods** (standard z, robust z, FDR, empirical Bayes, Poisson, NB, percentile)
- **3 ML methods** (Isolation Forest, LOF, Random Forest)
- **Spatial autocorrelation** (Moran's I, LISA clustering)
- **ARIMA time series forecasting** with backtesting
- **Population-normalized rates** (per 100K) using Census county data

**Key Finding:** Texas border counties (Kenedy, Brooks) are the most extreme outliers across every method tested. I-35 corridor is accelerating at +10.80 MP/year. All 6 geographic zones show increasing missing persons trends.

---

## Top Outliers Discovered

### Extreme Outliers (Covariate-Adjusted)

| Rank | County | State | Decade | MP/100K | Bodies/100K | Composite Z | ML Class |
|------|--------|-------|--------|---------|-------------|-------------|----------|
| 1 | Kenedy | TX | 2020s | 4,617.6 | 2,597.4 | 46.86 | Extreme Anomaly |
| 2 | Kenedy | TX | 2010s | 1,174.0 | 7,513.5 | 37.50 | Extreme Anomaly |
| 3 | Kenedy | TX | 1990s | 0.0 | 5,714.3 | 21.31 | Extreme Anomaly |
| 4 | Brooks | TX | 2010s | 793.1 | 3,214.2 | 18.35 | Extreme Anomaly |
| 5 | Kenedy | TX | 2000s | 0.0 | 4,272.2 | 15.92 | Extreme Anomaly |

These counties remain extreme outliers even after:
- Empirical Bayes shrinkage (Kenedy: 4,617 -> 60/100K shrunk rate, weight=0.012)
- Covariate adjustment (controlling for poverty, demographics, urbanization)
- FDR multiple comparison correction
- Random Forest nonlinear modeling (Kenedy RF residual z = 40.93)

### Spatial Hotspot Clusters (LISA)

Texas border counties form statistically significant High-High (HH) spatial clusters:
- Kenedy, Brooks, Jeff Davis, Hudspeth, Terrell, Dimmit, Maverick, Webb, Zapata (all TX)
- Alaska interior: Dillingham, Denali, Nome, Lake and Peninsula, Kodiak Island, Aleutians

86 total HH hotspot observations across all decades. Top states: Alaska (46), Texas (33).

Global Moran's I = 0.22 for 2000s missing persons rates (z=26.03, p = 0.001) -- spatial clustering is highly significant.

---

## Alert Classification

### Statistical Alert Levels

| Alert Level | Threshold | County-Decades | Percentage |
|-------------|-----------|----------------|------------|
| RED | Composite z > 3, both MP and bodies elevated | 4 | 0.05% |
| ORANGE | z > 2 either metric, or pattern match | 27 | 0.37% |
| GREEN | Below thresholds | 7,265 | 99.57% |

Total county-decade observations: 7,296

### After FDR Correction
- 915 county-decades remain statistically significant at alpha=0.05
- 16.5% of county-decades changed alert level with robust (Median/MAD) baselines

### ML Classification (Ensemble)
- Normal: 6,560 (90.0%)
- Suspicious: 364 (5.0%)
- Anomalous: 292 (4.0%)
- Extreme Anomaly: 73 (1.0%)

Concordance: 100% of statistical RED/ORANGE alerts are captured by ML. ML additionally flags 698 county-decades not caught by z-scores alone.

---

## Method Comparison

### Outlier Detection Methods

| Method | Threshold | Counties Flagged | Notes |
|--------|-----------|-----------------|-------|
| Standard z > 2 | 2 sigma | ~31 | Suppressed by right-skewed data inflating std |
| Robust z > 2 | 2 MAD | ~1,223 | More sensitive, median-based baseline |
| Log z > 2 | 2 sigma (log scale) | ~604 | Better for skewed distributions |
| Poisson exceedance < 0.01 | p < 0.01 | ~989 | Count-appropriate model |
| NB exceedance < 0.01 | p < 0.01 | ~180 | Handles overdispersion |
| Percentile > 99th | top 1% | ~131 | Non-parametric |
| FDR significant | BH-adjusted p < 0.05 | 915 | Multiple comparison corrected |
| IF + LOF ensemble | top 1% | 73 | ML unsupervised |

### Empirical Bayes Shrinkage Effect

Small counties with extreme raw rates are shrunk toward the national mean:
- Kenedy TX (pop=346): 4,617/100K -> 60/100K (weight=0.012)
- Brooks TX (pop=7,200): 793/100K -> 699/100K (weight=0.198)
- Large counties barely affected: Harris TX (pop=4.7M): weight=0.994

### Forecasting Comparison (ARIMA vs Linear)

| Zone | Best Model (MP) | Improvement |
|------|----------------|-------------|
| US-Mexico Border | ARIMA | 9.2% |
| I-35 Corridor | ARIMA | 10.6% |
| Pacific Northwest | Linear | 2.0% |
| Midwest Metro | ARIMA | 2.2% |
| Northeast Corridor | ARIMA | 40.2% |
| Southern California | Linear | 80.7% |

Overall: ARIMA wins 7/12 comparisons across zones and metrics.

---

## Temporal Trends

### Mann-Kendall Trend Tests (All 6 Zones)

| Zone | MP Trend | p-value | Bodies Trend | Structural Break |
|------|----------|---------|-------------|-----------------|
| US-Mexico Border | Increasing | < 0.001 | Increasing | 2022 |
| I-35 Corridor | Increasing | < 0.001 | Increasing | 2020 |
| Pacific Northwest | Increasing | < 0.001 | No trend | 2013 |
| Midwest Metro | Increasing | < 0.001 | No trend | 2017 |
| Northeast Corridor | Increasing | < 0.001 | No trend | 2016 |
| Southern California | Increasing | < 0.001 | Decreasing | 1998 |

All 6 zones show statistically significant increasing trends in missing persons. Only US-Mexico Border and I-35 Corridor show increasing body recovery trends. Southern California shows decreasing bodies since structural break in 1998.

---

## Covariate-Adjusted Analysis

### OLS Regression (MP Rate ~ Covariates)
- R-squared: 0.083
- Strongest predictors: pct_foreign_born (t=11.69), log_population (t=-11.60)
- 6 counties with extreme residuals (>3 sigma after adjustment)

### Random Forest Overperformance
- Nonlinear model captures interactions OLS cannot
- 5-fold cross-validated R-squared assessed
- Top overperformers remain Kenedy TX (RF residual z = 40.93) and Brooks TX even after RF adjustment
- Feature importance: log_population and pct_foreign_born dominate

---

## Validation

### System Design Scope

This system detects large-scale structural anomalies: trafficking corridors, cartel activity zones, mass disappearance patterns, and spatial clusters. Individual serial killers in large metro areas (King County pop 1.7M, Cook County pop 5.5M) do not produce statistically detectable signals at county-decade resolution -- their victim counts are too small relative to the population denominator.

| Test Case | Location | Decade | Composite Z | Alert | Result |
|-----------|----------|--------|-------------|-------|--------|
| Kenedy County (border zone) | TX | 2020s | 46.86 | RED | Detected across all methods |
| Brooks County (border zone) | TX | 2010s | 18.35 | RED | Detected across all methods |
| Gary Ridgway (49 victims) | King Co, WA | 1980s | -0.05 | GREEN | Not detectable at this resolution |
| John Wayne Gacy (33 victims) | Cook Co, IL | 1970s | -0.09 | GREEN | Not detectable at this resolution |
| Jeffrey Dahmer (17 victims) | Milwaukee, WI | 1990s | -0.08 | GREEN | Not detectable at this resolution |

### Pattern Types Detected

1. **Classic Serial:** High MP + High bodies (border counties)
2. **Destroyer:** High MP + Low bodies (remains destroyed)
3. **Transient Victims:** Low MP + High bodies (undocumented migrants, Kenedy 1990s-2000s)
4. **Trafficking:** Very high MP in connected metro areas (I-35 Corridor)
5. **Spatial Cluster:** Neighboring counties all elevated (TX border HH cluster)
6. **Accelerating Zone:** Trend slope increasing over time (I-35 structural break 2020)

---

## Technical Specifications

- **Data:** 41,200 cases, 55 states/territories, 7,296 county-decade observations
- **Pipeline:** 16 Python scripts, 16 output CSVs, 37 columns in enriched dataset
- **Statistical tests:** Chi-squared, Kruskal-Wallis, Poisson rate ratios, ADF, Mann-Kendall
- **ML models:** Isolation Forest (200 trees), LOF (k=20), Random Forest (200 trees, depth=10), DBSCAN, Ward clustering
- **Dashboard:** 11-page Streamlit app with Plotly visualizations
- **Dependencies:** pandas, numpy, scipy, scikit-learn, statsmodels, streamlit, plotly

---

**Last Updated:** February 2026
