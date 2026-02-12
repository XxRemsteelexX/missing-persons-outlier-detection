#!/usr/bin/env python3
"""
ML-Based Anomaly Detection for Missing Persons Outlier Detection
================================================================

This script applies unsupervised machine learning methods (Isolation Forest
and Local Outlier Factor) to identify anomalous county-decade observations
in the missing persons / unidentified bodies dataset.  Results are compared
with the purely statistical z-score outliers to assess concordance.

Pipeline
--------
1. Load county_decade_outliers.csv
2. Build a feature matrix from empirical-Bayes-shrunk rates, robust z-scores,
   log-transformed z-scores, population, and decade.
3. Standardize features to zero mean / unit variance.
4. Isolation Forest  (contamination=0.05, n_estimators=200)
5. Local Outlier Factor (n_neighbors=20, contamination=0.05)
6. Ensemble scoring and tiered classification.
7. Concordance analysis against statistical alert levels.
8. Save results to ml_anomaly_scores.csv
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from utils import ANALYSIS_DIR

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
INPUT_PATH = os.path.join(ANALYSIS_DIR, "county_decade_outliers.csv")
OUTPUT_PATH = os.path.join(ANALYSIS_DIR, "ml_anomaly_scores.csv")

print("=" * 72)
print("  ML Anomaly Detection -- Missing Persons Outlier Detection")
print("=" * 72)
print()

print(f"Loading data from {INPUT_PATH} ...")
df = pd.read_csv(INPUT_PATH)
print(f"  Raw dataset: {len(df):,} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. Build feature matrix
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "mp_rate_shrunk",
    "bodies_rate_shrunk",
    "mp_log_z",
    "bodies_log_z",
    "mp_robust_z",
    "bodies_robust_z",
    "population",
    "decade",
]

# Filter: population > 0 and all feature columns non-null
mask = (df["population"] > 0) & df[FEATURE_COLS].notna().all(axis=1)
df_valid = df.loc[mask].copy().reset_index(drop=True)
print(f"  After filtering (population > 0, features non-null): {len(df_valid):,} rows")
print()

X = df_valid[FEATURE_COLS].values.astype(np.float64)

# ---------------------------------------------------------------------------
# 3. Standardize features
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Feature matrix standardized (zero mean, unit variance).")
print(f"  Shape: {X_scaled.shape}")
print()

# ---------------------------------------------------------------------------
# 4. Isolation Forest
# ---------------------------------------------------------------------------
print("Running Isolation Forest (n_estimators=200, contamination=0.05) ...")
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1,
)
if_labels = iso_forest.fit_predict(X_scaled)          # -1 = anomaly, 1 = normal
if_raw_scores = iso_forest.decision_function(X_scaled) # lower = more anomalous

# Convert so that *higher* score = more anomalous (negate decision function)
if_scores = -if_raw_scores

df_valid["if_score"] = if_scores
df_valid["if_anomaly"] = (if_labels == -1).astype(int)

n_if_anom = df_valid["if_anomaly"].sum()
print(f"  Isolation Forest anomalies: {n_if_anom:,} "
      f"({100 * n_if_anom / len(df_valid):.1f}%)")
print()

# ---------------------------------------------------------------------------
# 5. Local Outlier Factor
# ---------------------------------------------------------------------------
print("Running Local Outlier Factor (n_neighbors=20, contamination=0.05) ...")
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
    n_jobs=-1,
)
lof_labels = lof.fit_predict(X_scaled)                  # -1 = anomaly, 1 = normal
lof_raw_scores = lof.negative_outlier_factor_            # negative, closer to -1 = normal

# Convert so that *higher* score = more anomalous (negate LOF scores)
lof_scores = -lof_raw_scores

df_valid["lof_score"] = lof_scores
df_valid["lof_anomaly"] = (lof_labels == -1).astype(int)

n_lof_anom = df_valid["lof_anomaly"].sum()
print(f"  LOF anomalies: {n_lof_anom:,} "
      f"({100 * n_lof_anom / len(df_valid):.1f}%)")
print()

# ---------------------------------------------------------------------------
# 6. Ensemble score and tiered classification
# ---------------------------------------------------------------------------
print("Computing ensemble score and classification tiers ...")

# Min-max normalize both scores to [0, 1] before averaging
def minmax(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

if_norm = minmax(if_scores)
lof_norm = minmax(lof_scores)
ensemble = (if_norm + lof_norm) / 2.0

df_valid["ensemble_score"] = ensemble

# Classification thresholds based on percentiles of the ensemble score
p90 = np.percentile(ensemble, 90)
p95 = np.percentile(ensemble, 95)
p99 = np.percentile(ensemble, 99)

def classify(score):
    if score >= p99:
        return "Extreme Anomaly"
    elif score >= p95:
        return "Anomalous"
    elif score >= p90:
        return "Suspicious"
    else:
        return "Normal"

df_valid["ml_classification"] = df_valid["ensemble_score"].apply(classify)

tier_counts = df_valid["ml_classification"].value_counts()
print("  Classification distribution:")
for tier in ["Normal", "Suspicious", "Anomalous", "Extreme Anomaly"]:
    count = tier_counts.get(tier, 0)
    pct = 100 * count / len(df_valid)
    print(f"    {tier:20s}: {count:>6,}  ({pct:.1f}%)")
print()

# ---------------------------------------------------------------------------
# 7. Concordance analysis: ML anomalies vs statistical alert levels
# ---------------------------------------------------------------------------
print("Concordance analysis: ML classifications vs statistical alert levels ...")

df_valid["statistical_alert"] = df_valid["alert_level"].fillna("GREEN")

# Define concordance: ML flags as non-Normal AND statistical alert is not GREEN
ml_flagged = df_valid["ml_classification"] != "Normal"
stat_flagged = df_valid["statistical_alert"] != "GREEN"

df_valid["is_concordant"] = (ml_flagged & stat_flagged).astype(int)

n_ml_flagged = ml_flagged.sum()
n_stat_flagged = stat_flagged.sum()
n_concordant = df_valid["is_concordant"].sum()
n_ml_only = (ml_flagged & ~stat_flagged).sum()
n_stat_only = (~ml_flagged & stat_flagged).sum()

print(f"  ML flagged (Suspicious+):     {n_ml_flagged:>6,}")
print(f"  Statistical flagged (non-GREEN): {n_stat_flagged:>6,}")
print(f"  Concordant (both flag):        {n_concordant:>6,}")
print(f"  ML-only flags:                 {n_ml_only:>6,}")
print(f"  Statistical-only flags:        {n_stat_only:>6,}")

if n_ml_flagged > 0:
    overlap_ml = 100 * n_concordant / n_ml_flagged
    print(f"  Overlap as pct of ML flags:    {overlap_ml:.1f}%")
if n_stat_flagged > 0:
    overlap_stat = 100 * n_concordant / n_stat_flagged
    print(f"  Overlap as pct of stat flags:  {overlap_stat:.1f}%")
print()

# Cross-tabulation
print("  Cross-tabulation (ML classification vs statistical alert):")
ct = pd.crosstab(
    df_valid["ml_classification"],
    df_valid["statistical_alert"],
    margins=True,
)
# Reorder rows for readability
row_order = [r for r in ["Normal", "Suspicious", "Anomalous", "Extreme Anomaly", "All"]
             if r in ct.index]
ct = ct.reindex(row_order)
print(ct.to_string())
print()

# ---------------------------------------------------------------------------
# 8. Save results
# ---------------------------------------------------------------------------
output_cols = [
    "State",
    "County",
    "state_name",
    "decade",
    "missing_per_100k",
    "bodies_per_100k",
    "population",
    "if_score",
    "if_anomaly",
    "lof_score",
    "lof_anomaly",
    "ensemble_score",
    "ml_classification",
    "statistical_alert",
    "is_concordant",
]

df_out = df_valid[output_cols].copy()
df_out.to_csv(OUTPUT_PATH, index=False)
print(f"Results saved to {OUTPUT_PATH}")
print(f"  Output rows: {len(df_out):,}")
print()

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
print("=" * 72)
print("  SUMMARY STATISTICS")
print("=" * 72)
print()

print("Ensemble score distribution:")
desc = df_out["ensemble_score"].describe()
for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
    print(f"  {stat:>6s}: {desc[stat]:>12.4f}")
print()

print("IF score distribution:")
desc_if = df_out["if_score"].describe()
for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
    print(f"  {stat:>6s}: {desc_if[stat]:>12.4f}")
print()

print("LOF score distribution:")
desc_lof = df_out["lof_score"].describe()
for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
    print(f"  {stat:>6s}: {desc_lof[stat]:>12.4f}")
print()

# ---------------------------------------------------------------------------
# Top 20 anomalies by ensemble score
# ---------------------------------------------------------------------------
print("=" * 72)
print("  TOP 20 ANOMALIES BY ENSEMBLE SCORE")
print("=" * 72)
print()

top20 = df_out.nlargest(20, "ensemble_score")

display_cols = [
    "State", "County", "decade", "missing_per_100k", "bodies_per_100k",
    "population", "ensemble_score", "ml_classification", "statistical_alert",
    "is_concordant",
]

# Format for readable output
for rank, (_, row) in enumerate(top20.iterrows(), start=1):
    print(f"  #{rank:>2d}  {row['State']:>2s} / {row['County']:<25s}  "
          f"decade={int(row['decade'])}  "
          f"MP/100k={row['missing_per_100k']:>9.1f}  "
          f"Bodies/100k={row['bodies_per_100k']:>7.1f}  "
          f"pop={row['population']:>12,.0f}  "
          f"ensemble={row['ensemble_score']:.4f}  "
          f"ML={row['ml_classification']:<17s}  "
          f"stat={row['statistical_alert']:<8s}  "
          f"concordant={int(row['is_concordant'])}")
print()

# Quick summary of top 20 characteristics
top20_states = top20["State"].value_counts()
top20_decades = top20["decade"].value_counts().sort_index()
top20_concordant = top20["is_concordant"].sum()

print("Top 20 anomalies -- breakdown:")
print(f"  States represented: {', '.join(f'{s}({c})' for s, c in top20_states.items())}")
print(f"  Decades: {', '.join(f'{int(d)}({c})' for d, c in top20_decades.items())}")
print(f"  Concordant with statistical flags: {top20_concordant} / 20")
print()

print("=" * 72)
print("  ML Anomaly Detection complete.")
print("=" * 72)
