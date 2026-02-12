#!/usr/bin/env python3
"""
County Clustering for Missing Persons Outlier Detection
========================================================

Aggregates county-decade data into per-county feature vectors and applies
two clustering methods (DBSCAN and Hierarchical/Ward) to identify groups
of counties with similar missing-persons / unidentified-bodies profiles.

Inputs:
    data/analysis/county_decade_outliers.csv

Outputs:
    data/analysis/county_clusters.csv
"""

import sys
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(__file__))
from utils import ANALYSIS_DIR


# =============================================================================
# Configuration
# =============================================================================
INPUT_PATH = os.path.join(ANALYSIS_DIR, "county_decade_outliers.csv")
OUTPUT_PATH = os.path.join(ANALYSIS_DIR, "county_clusters.csv")

FEATURE_COLS = [
    "mean_mp_rate",
    "mean_bodies_rate",
    "mp_trend",
    "bodies_trend",
    "mean_mp_robust_z",
    "mean_bodies_robust_z",
    "log_population",
]

MIN_DECADES = 2
MIN_POPULATION = 10000
DBSCAN_MIN_SAMPLES = 5
K_NEIGHBORS = 5          # for the k-distance heuristic
DEFAULT_EPS = 1.5         # fallback if heuristic fails
HIERARCHICAL_N_CLUSTERS = 6  # matches the number of geographic zones


# =============================================================================
# Helper Functions
# =============================================================================

def compute_slope(series):
    """
    Compute the OLS slope of a series indexed by decade.
    Returns 0.0 if there are fewer than 2 valid points.
    """
    vals = series.dropna()
    if len(vals) < 2:
        return 0.0
    x = np.arange(len(vals), dtype=float)
    y = vals.values.astype(float)
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope


def build_county_features(df):
    """
    Aggregate the decade-level data into a single row per (State, County)
    with the required feature columns.

    Features:
        mean_mp_rate         -- mean of missing_per_100k across decades
        mean_bodies_rate     -- mean of bodies_per_100k across decades
        mp_trend             -- slope of missing_per_100k across decades
        bodies_trend         -- slope of bodies_per_100k across decades
        mean_mp_robust_z     -- mean of mp_robust_z across decades
        mean_bodies_robust_z -- mean of bodies_robust_z across decades
        log_population       -- log10 of the most-recent-decade population
    """
    print("Building per-county feature vectors ...")

    # Sort by decade so slopes are computed in chronological order
    df = df.sort_values(["State", "County", "decade"])

    # Group by (State, County)
    grouped = df.groupby(["State", "County"])

    agg = grouped.agg(
        state_name=("state_name", "first"),
        n_decades=("decade", "nunique"),
        mean_mp_rate=("missing_per_100k", "mean"),
        mean_bodies_rate=("bodies_per_100k", "mean"),
        mean_mp_robust_z=("mp_robust_z", "mean"),
        mean_bodies_robust_z=("bodies_robust_z", "mean"),
        latest_population=("population", "last"),
    ).reset_index()

    # Compute trend (slope) per county
    mp_slopes = grouped["missing_per_100k"].apply(compute_slope).reset_index()
    mp_slopes.columns = ["State", "County", "mp_trend"]

    bodies_slopes = grouped["bodies_per_100k"].apply(compute_slope).reset_index()
    bodies_slopes.columns = ["State", "County", "bodies_trend"]

    agg = agg.merge(mp_slopes, on=["State", "County"], how="left")
    agg = agg.merge(bodies_slopes, on=["State", "County"], how="left")

    # Log population (base-10)
    agg["log_population"] = np.log10(agg["latest_population"].clip(lower=1))

    print(f"  Total unique counties: {len(agg)}")
    return agg


def filter_counties(agg):
    """
    Keep only counties that have:
      - at least MIN_DECADES decades of data
      - population > MIN_POPULATION in their most recent decade
    """
    before = len(agg)
    mask = (agg["n_decades"] >= MIN_DECADES) & (agg["latest_population"] > MIN_POPULATION)
    filtered = agg[mask].copy()
    after = len(filtered)
    print(f"  Filtering: {before} -> {after} counties "
          f"(>= {MIN_DECADES} decades, population > {MIN_POPULATION:,})")
    return filtered


def determine_eps(X_scaled, k=K_NEIGHBORS, default=DEFAULT_EPS):
    """
    Use the k-distance plot heuristic to choose DBSCAN eps.

    Steps:
      1. Compute the distance to the k-th nearest neighbor for every point.
      2. Sort those distances in ascending order.
      3. Find the "knee" -- the point of maximum curvature.
      4. Return the distance at the knee as eps.

    Falls back to default if the heuristic produces an unreasonable value.
    """
    print(f"  Computing {k}-nearest-neighbor distances for eps selection ...")
    nn = NearestNeighbors(n_neighbors=k + 1)  # +1 because point is its own neighbor
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, k])  # k-th neighbor column, sorted

    # Find knee: largest second derivative (discrete approximation)
    if len(k_distances) < 5:
        print(f"  Too few data points for heuristic; using default eps={default}")
        return default

    # Second derivative of the sorted k-distance curve
    first_deriv = np.diff(k_distances)
    second_deriv = np.diff(first_deriv)

    if len(second_deriv) == 0:
        print(f"  Could not compute curvature; using default eps={default}")
        return default

    knee_idx = np.argmax(second_deriv) + 1  # +1 to offset diff
    eps_candidate = k_distances[knee_idx]

    # Sanity check: eps should be positive and not absurdly large
    if eps_candidate <= 0 or eps_candidate > 10 * np.median(k_distances):
        print(f"  Heuristic eps={eps_candidate:.4f} looks unreasonable; "
              f"using default eps={default}")
        return default

    print(f"  Knee found at index {knee_idx}, eps={eps_candidate:.4f}")
    return eps_candidate


# =============================================================================
# Clustering
# =============================================================================

def run_dbscan(X_scaled, eps, min_samples=DBSCAN_MIN_SAMPLES):
    """Run DBSCAN and return labels."""
    print(f"\n  DBSCAN: eps={eps:.4f}, min_samples={min_samples}")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels) - {-1})
    n_noise = np.sum(labels == -1)
    print(f"    Clusters found: {n_clusters}")
    print(f"    Noise points:   {n_noise}")
    return labels


def run_hierarchical(X_scaled, n_clusters=HIERARCHICAL_N_CLUSTERS):
    """Run Ward hierarchical clustering and return labels."""
    print(f"\n  Hierarchical (Ward): n_clusters={n_clusters}")
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X_scaled)
    for c in range(n_clusters):
        print(f"    Cluster {c}: {np.sum(labels == c)} counties")
    return labels


def compute_silhouette(X_scaled, labels, method_name):
    """
    Compute the mean silhouette score.
    Returns -1.0 if the score cannot be computed (e.g., only one cluster).
    """
    unique_labels = set(labels)
    # Need at least 2 clusters and no single-cluster-only assignment
    n_actual = len(unique_labels - {-1})
    if n_actual < 2:
        print(f"    Silhouette ({method_name}): N/A (fewer than 2 clusters)")
        return -1.0

    # For DBSCAN, exclude noise points from silhouette computation
    mask = labels != -1
    if mask.sum() < 2:
        print(f"    Silhouette ({method_name}): N/A (too few non-noise points)")
        return -1.0

    score = silhouette_score(X_scaled[mask], labels[mask])
    print(f"    Silhouette ({method_name}): {score:.4f}")
    return score


def per_sample_silhouette(X_scaled, labels):
    """
    Compute per-sample silhouette values.
    Noise points (label == -1) get a score of NaN.
    """
    scores = np.full(len(labels), np.nan)
    mask = labels != -1
    unique_non_noise = set(labels[mask])
    if len(unique_non_noise) >= 2 and mask.sum() >= 2:
        scores[mask] = silhouette_samples(X_scaled[mask], labels[mask])
    return scores


# =============================================================================
# Cluster Profiling
# =============================================================================

def profile_clusters(df, labels, feature_cols, method_name, top_n=5):
    """
    Print the mean feature values and top counties for each cluster.
    """
    print(f"\n{'=' * 70}")
    print(f"CLUSTER PROFILES -- {method_name}")
    print(f"{'=' * 70}")

    df_tmp = df.copy()
    df_tmp["_cluster"] = labels

    unique_clusters = sorted(set(labels))
    for cl in unique_clusters:
        label_str = f"Cluster {cl}" if cl != -1 else "Noise (-1)"
        subset = df_tmp[df_tmp["_cluster"] == cl]
        print(f"\n--- {label_str} ({len(subset)} counties) ---")

        # Mean feature values
        print("  Mean feature values:")
        for col in feature_cols:
            print(f"    {col:30s} = {subset[col].mean():10.4f}")

        # Top counties by mean_mp_rate
        top = subset.nlargest(top_n, "mean_mp_rate")
        print(f"  Top {top_n} counties by mean missing-persons rate:")
        for _, row in top.iterrows():
            print(f"    {row['County']:25s} {row['State']}  "
                  f"mp_rate={row['mean_mp_rate']:.2f}  "
                  f"bodies_rate={row['mean_bodies_rate']:.2f}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("COUNTY CLUSTERING -- Missing Persons Outlier Detection")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\nLoading data from {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  Rows: {len(df)}, Columns: {df.shape[1]}")
    print(f"  Decades present: {sorted(df['decade'].unique())}")
    print(f"  Unique counties (State+County): "
          f"{df.groupby(['State', 'County']).ngroups}")

    # ------------------------------------------------------------------
    # 2. Build per-county feature vectors
    # ------------------------------------------------------------------
    agg = build_county_features(df)

    # ------------------------------------------------------------------
    # 3. Filter counties
    # ------------------------------------------------------------------
    agg = filter_counties(agg)

    if len(agg) < DBSCAN_MIN_SAMPLES + 1:
        print("\nERROR: Not enough counties after filtering to run clustering.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Standardize features
    # ------------------------------------------------------------------
    print("\nStandardizing features ...")
    # Replace any infinities that may have crept in
    for col in FEATURE_COLS:
        agg[col] = agg[col].replace([np.inf, -np.inf], np.nan)

    # Fill remaining NaN with column median (robust to outliers)
    for col in FEATURE_COLS:
        median_val = agg[col].median()
        agg[col] = agg[col].fillna(median_val)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(agg[FEATURE_COLS].values)
    print(f"  Feature matrix shape: {X_scaled.shape}")

    # ------------------------------------------------------------------
    # 5. DBSCAN (eps via k-distance heuristic)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("DBSCAN CLUSTERING")
    print("-" * 70)
    eps = determine_eps(X_scaled, k=K_NEIGHBORS, default=DEFAULT_EPS)
    dbscan_labels = run_dbscan(X_scaled, eps=eps, min_samples=DBSCAN_MIN_SAMPLES)
    sil_dbscan = compute_silhouette(X_scaled, dbscan_labels, "DBSCAN")

    # ------------------------------------------------------------------
    # 6. Hierarchical Clustering (Ward)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("HIERARCHICAL CLUSTERING (Ward)")
    print("-" * 70)

    # Also compute full linkage matrix for reference (scipy)
    Z = linkage(X_scaled, method="ward")
    print(f"  Linkage matrix shape: {Z.shape}")

    hier_labels = run_hierarchical(X_scaled, n_clusters=HIERARCHICAL_N_CLUSTERS)
    sil_hier = compute_silhouette(X_scaled, hier_labels, "Hierarchical")

    # ------------------------------------------------------------------
    # 7. Per-sample silhouette scores
    # ------------------------------------------------------------------
    sil_dbscan_samples = per_sample_silhouette(X_scaled, dbscan_labels)
    sil_hier_samples = per_sample_silhouette(X_scaled, hier_labels)

    # ------------------------------------------------------------------
    # 8. Profile clusters
    # ------------------------------------------------------------------
    profile_clusters(agg, dbscan_labels, FEATURE_COLS, "DBSCAN", top_n=5)
    profile_clusters(agg, hier_labels, FEATURE_COLS, "Hierarchical (Ward)", top_n=5)

    # ------------------------------------------------------------------
    # 9. Build output dataframe and save
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SAVING RESULTS")
    print("-" * 70)

    out = agg[["State", "County", "state_name"] + FEATURE_COLS].copy()
    out["dbscan_cluster"] = dbscan_labels
    out["hierarchical_cluster"] = hier_labels
    out["silhouette_dbscan"] = sil_dbscan_samples
    out["silhouette_hierarchical"] = sil_hier_samples

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved {len(out)} rows to {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\n  Counties clustered:           {len(out)}")
    print(f"  DBSCAN eps used:              {eps:.4f}")
    n_db_clusters = len(set(dbscan_labels) - {-1})
    n_db_noise = int(np.sum(dbscan_labels == -1))
    print(f"  DBSCAN clusters:              {n_db_clusters}")
    print(f"  DBSCAN noise points:          {n_db_noise} "
          f"({100.0 * n_db_noise / len(out):.1f}%)")
    print(f"  DBSCAN silhouette (overall):  "
          f"{sil_dbscan:.4f}" if sil_dbscan != -1.0 else "  DBSCAN silhouette (overall):  N/A")
    print(f"  Hierarchical clusters:        {HIERARCHICAL_N_CLUSTERS}")
    print(f"  Hierarchical silhouette:      {sil_hier:.4f}" if sil_hier != -1.0
          else f"  Hierarchical silhouette:      N/A")

    # Feature-level summary
    print(f"\n  Feature summary (after standardization):")
    print(f"    {'Feature':30s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print(f"    {'-' * 70}")
    for i, col in enumerate(FEATURE_COLS):
        vals = X_scaled[:, i]
        print(f"    {col:30s} {vals.mean():10.4f} {vals.std():10.4f} "
              f"{vals.min():10.4f} {vals.max():10.4f}")

    print(f"\n{'=' * 70}")
    print("DONE -- County clustering complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
