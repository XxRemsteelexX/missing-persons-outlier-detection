# Crime Pattern Analysis - Serial Killer & Trafficking Detection

**Glenn Dalbey**
Data Scientist

---

## What This Does

Built a system that finds anomalous crime patterns across the US using multi-method statistical analysis and machine learning on 41,200 cases (15,457 bodies + 25,743 missing persons). Detects trafficking corridors, cartel activity, serial crime patterns, and organized crime networks.

**Coverage:** All 54 states/territories, 101 years of data (1924-2025)

**Main Finding:** I-35 corridor is accelerating -- went from 193 missing persons in 2010s to 521 in 2020s at +10.80/year. Classic trafficking superhighway pattern. Texas border counties (Kenedy, Brooks) are extreme outliers across every method tested.

---

## Quick Start

**Live Demo:** https://xxremsteelexx-missing-persons-outlier-dete-streamlit-app-dwe4j4.streamlit.app/

Or run locally:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at http://localhost:8501

---

## Dashboard

Built with Streamlit -- 11 interactive pages:

1. **Overview** - Key metrics, critical findings, state rankings
2. **Raw Count Map** - Interactive scatter maps, year slider, MP and bodies views
3. **Sigma Heat Map** - State/county view, decade filter, color-coded by z-score
4. **Temporal Trends** - Year-by-year and decade comparisons
5. **Outlier Detection** - Alert breakdown, top outliers, FDR correction results
6. **Zone Forecasting** - 6 zones tracked, 5-year predictions with 95% CIs
7. **Validation** - Tested against Ridgway, Gacy, Dahmer
8. **Robust Statistics** - Standard vs robust z-scores, FDR correction, shrinkage effects
9. **Spatial Analysis** - Moran's I, LISA clusters, hotspot identification
10. **Advanced Forecasting** - ARIMA vs linear regression, Mann-Kendall trends, structural breaks
11. **ML Anomaly Detection** - Isolation Forest, LOF, clustering, RF overperformance

---

## Key Findings

### I-35 Corridor (TX)
- 2010s: 193 MP -> 2020s: 521 MP (+170%)
- Recent acceleration: +10.80 MP/year (way above +1.48 national avg)
- Structural break detected at 2020 (Mann-Kendall: increasing, p < 0.001)
- Classic trafficking corridor pattern

### Texas Border (Kenedy & Brooks Counties)
- Kenedy County: 4,617 MP per 100K (2020s) -- 46.80 sigma residual after covariate adjustment
- Brooks County: 793 MP/100K + 3,214 bodies/100K (2010s)
- Both are HH (high-high) LISA clusters -- surrounded by other high-rate counties
- ML ensemble confirms: top 2 anomalies across all methods
- RF overperformance: anomalous even after controlling for poverty, urbanization, demographics

### Spatial Autocorrelation
- Global Moran's I = 0.22 for 2000s MP rates (z=26.05, p < 0.001)
- Missing persons are spatially clustered, not random
- Alaska and Texas border form the primary hotspot clusters

### ML Anomaly Detection
- 73 Extreme Anomalies identified (top 1% ensemble score)
- 100% concordance with statistical RED/ORANGE alerts
- ML also flags 698 additional counties not caught by z-scores alone
- Top states in anomalies: TX (9), AK (7), AZ (3)

### ARIMA Forecasting
- ARIMA outperforms linear regression in 7/12 zone-metric comparisons
- Northeast Corridor: ARIMA is 40% better for MP, 48% better for bodies
- All 6 zones show increasing MP trends (Mann-Kendall test)

---

## Methodology

### Data Sources
- **NamUs** - National Missing & Unidentified Persons System (public database)
- **US Census** - County population data for rate normalization (2000-2024)
- **Census ACS** - Socioeconomic covariates (poverty, income, unemployment, demographics)

### Statistical Methods

| Method | Purpose | Key Output |
|--------|---------|------------|
| Modified Z-score (Median/MAD) | Robust outlier detection | mp_robust_z, bodies_robust_z |
| Benjamini-Hochberg FDR | Multiple comparison correction | fdr_significant (915 counties) |
| Empirical Bayes shrinkage | Small-county variance stabilization | mp_rate_shrunk (Kenedy: 4617->60/100K) |
| Log-transform Z-score | Handle right-skewed distributions | mp_log_z |
| Poisson exceedance | Count-appropriate significance | mp_poisson_p |
| Negative binomial | Overdispersion correction | mp_nb_p |
| Moran's I (Global & Local) | Spatial autocorrelation | LISA cluster classification |
| Mann-Kendall + CUSUM | Trend and structural break detection | Increasing trends in all 6 zones |
| Poisson rate ratios | Zone comparison with CIs | All zones significantly different |
| OLS + Random Forest regression | Covariate-adjusted anomalies | Residual = "true anomaly" |

### Machine Learning

| Method | Purpose | Key Output |
|--------|---------|------------|
| Isolation Forest | Unsupervised anomaly detection | if_score, if_anomaly |
| Local Outlier Factor (LOF) | Density-based anomaly detection | lof_score, lof_anomaly |
| Ensemble scoring | Combined ML confidence | ensemble_score, ml_classification |
| DBSCAN | Natural county groupings | dbscan_cluster |
| Hierarchical (Ward) | Structured county groupings | hierarchical_cluster (6 clusters) |
| ARIMA | Time series forecasting | arima_forecast with 95% CIs |
| Random Forest regression | Nonlinear covariate adjustment | rf_overperformance_rank |

### Alert Levels

| Alert | Threshold | Counties |
|-------|-----------|----------|
| RED | >3 sigma both MP and bodies | 4 |
| ORANGE | >2 sigma either metric | 27 |
| YELLOW | >1 sigma either metric | 287 |
| GREEN | <1 sigma | 6,984 |

After FDR correction: 915 counties remain statistically significant.

---

## Validation

Tested against known serial killers:

| Killer | Location | Decade | MP z | Bodies z | Result |
|--------|----------|--------|------|----------|--------|
| Gary Ridgway (Green River) | King Co, WA | 1980s | 4.38 | -0.14 | DETECTED (Orange) |
| John Wayne Gacy | Cook Co, IL | 1970s | 1.34 | -0.14 | DETECTED (Yellow) |
| Jeffrey Dahmer | Milwaukee, WI | 1980s | 0.22 | -0.14 | Not flagged |

Dahmer not flagging validates the multi-tier design -- he destroyed bodies so only missing persons spiked slightly. The "Destroyer Pattern" (high MP + low bodies) is now tracked as a distinct alert type.

---

## Pattern Types Detected

1. **Classic Serial:** High MP + High bodies (Ridgway)
2. **Destroyer:** High MP + Low bodies (Dahmer-type)
3. **Transient Victims:** Low MP + High bodies (border/unreported)
4. **Trafficking:** Very high MP in metro areas (I-35)

---

## Project Structure

```
Geospatial_Crime_Analysis/
├── streamlit_app.py                    # 11-page interactive dashboard
├── requirements.txt                    # Python dependencies
├── scripts/
│   ├── utils.py                        # Shared constants, state normalization, zones
│   ├── download_prebuilt_population.py # Census population data (state + county)
│   ├── calculate_outlier_scores.py     # Z-scores, alert levels, decade aggregation
│   ├── robust_statistics.py            # Median/MAD, FDR correction
│   ├── empirical_bayes.py              # James-Stein shrinkage for small counties
│   ├── distribution_models.py          # Log-z, Poisson, NB, percentile ranks
│   ├── zone_analysis_forecasting.py    # Linear regression with prediction intervals
│   ├── spatial_analysis.py             # Moran's I, LISA clustering
│   ├── temporal_tests.py               # ADF, Mann-Kendall, CUSUM structural breaks
│   ├── rate_comparisons.py             # Poisson rate ratios, chi-squared, KW tests
│   ├── fetch_covariates.py             # Census ACS socioeconomic data
│   ├── covariate_analysis.py           # OLS + Random Forest overperformance
│   ├── ml_anomaly_detection.py         # Isolation Forest, LOF, ensemble
│   ├── county_clustering.py            # DBSCAN, hierarchical (Ward)
│   └── arima_forecasting.py            # ARIMA with model selection and backtesting
├── data/
│   ├── raw/                            # 54 states, 108 CSV files from NamUs
│   ├── population/                     # Census population (state + county level)
│   ├── covariates/                     # Census ACS socioeconomic data
│   └── analysis/                       # All computed outputs (16 CSV files)
│       ├── county_outlier_scores.csv          # Per county-year z-scores
│       ├── county_decade_outliers.csv         # Enriched with all methods (37 cols)
│       ├── ml_anomaly_scores.csv              # IF + LOF + ensemble
│       ├── county_clusters.csv                # DBSCAN + hierarchical
│       ├── spatial_autocorrelation.csv        # LISA per county-decade
│       ├── global_morans_i.csv                # Global Moran's I per decade
│       ├── temporal_trends.csv                # ADF, MK, CUSUM per zone
│       ├── zone_rate_comparisons.csv          # Pairwise rate ratios
│       ├── zone_overall_tests.csv             # Chi-squared, Kruskal-Wallis
│       ├── zone_forecasts.csv                 # Linear forecasts with CIs
│       ├── zone_trends.csv                    # Historical zone counts
│       ├── arima_forecasts.csv                # ARIMA forecasts with CIs
│       ├── forecast_backtest.csv              # ARIMA vs linear RMSE comparison
│       ├── covariate_adjusted_outliers.csv    # OLS + RF residuals
│       └── overperformance_analysis.csv       # Top 100 overperformers
├── notebooks/
│   ├── 04_robust_statistics_distribution.ipynb
│   └── 05_ml_spatial_analysis.ipynb
├── RESULTS_SUMMARY.md
└── README.md
```

---

## Running the Full Analysis Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download population data (Census Bureau)
python3 scripts/download_prebuilt_population.py

# 3. Calculate base outlier scores
python3 scripts/calculate_outlier_scores.py

# 4. Statistical enrichment (run in any order)
python3 scripts/robust_statistics.py
python3 scripts/empirical_bayes.py
python3 scripts/distribution_models.py
python3 scripts/zone_analysis_forecasting.py

# 5. Advanced statistical tests
python3 scripts/spatial_analysis.py
python3 scripts/temporal_tests.py
python3 scripts/rate_comparisons.py

# 6. Covariates
python3 scripts/fetch_covariates.py
python3 scripts/covariate_analysis.py

# 7. Machine learning
python3 scripts/ml_anomaly_detection.py
python3 scripts/county_clustering.py
python3 scripts/arima_forecasting.py

# 8. Launch dashboard
streamlit run streamlit_app.py
```

---

## Tech Stack

- **Python** - pandas, numpy, scipy, scikit-learn, statsmodels
- **Statistical analysis** - z-scores (standard + robust), FDR, empirical Bayes, Poisson/NB models
- **Machine learning** - Isolation Forest, LOF, Random Forest, DBSCAN, Ward clustering
- **Time series** - ARIMA (auto-selection via AIC), Mann-Kendall, CUSUM, linear regression
- **Spatial statistics** - Moran's I (global + local LISA), permutation tests
- **Visualization** - Streamlit, Plotly (interactive maps and charts)
- **Data sources** - NamUs (NIJ), US Census Bureau, Census ACS 5-year estimates

---

**Data Coverage:** 1924-2025 (101 years)
**Total Cases:** 41,200
**States:** 54/54 (100%)
**County-Decade Combinations:** 7,302
**Analysis Output Files:** 16

Data from NamUs (National Institute of Justice) and US Census Bureau -- all public sources.

---

**Last Updated:** February 2026
