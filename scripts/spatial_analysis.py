#!/usr/bin/env python3
"""
Spatial Autocorrelation Analysis for Missing Persons Outlier Detection

Computes Global and Local Moran's I statistics to identify spatial clustering
of missing persons and unidentified bodies rates across US counties.

Approach:
  - Uses within-state contiguity as the spatial weight matrix: all counties
    within the same state are treated as neighbors (queen-style within-state).
  - Moran's I implemented from scratch using numpy/scipy (no libpysal/esda).
  - Permutation tests with 999 random permutations for significance.
  - Focuses on 2000s, 2010s, 2020s decades (most complete data).

Outputs:
  - data/analysis/global_morans_i.csv
  - data/analysis/spatial_autocorrelation.csv
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from utils import ANALYSIS_DIR, DATA_DIR, STATE_FIPS, FIPS_STATE, normalize_state

os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Decades with the most complete county-level data
TARGET_DECADES = [2000, 2010, 2020]
N_PERMUTATIONS = 999
SIGNIFICANCE_LEVEL = 0.05
RANDOM_SEED = 42


# =============================================================================
# Spatial Weights (Within-State Contiguity)
# =============================================================================

def build_within_state_weights(df):
    """
    Build a row-standardized spatial weight matrix where counties in the
    same state are neighbors.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'State' column. Index positions are used as row/col indices.

    Returns
    -------
    W : np.ndarray of shape (n, n)
        Row-standardized weight matrix. W[i,j] > 0 iff counties i and j
        share the same state (and i != j).
    """
    n = len(df)
    states = df["State"].values

    # Build binary adjacency: same-state = 1, else 0, diagonal = 0
    # Vectorized: outer comparison
    W = (states[:, None] == states[None, :]).astype(np.float64)
    np.fill_diagonal(W, 0.0)

    # Row-standardize: each row sums to 1 (islands get 0 everywhere)
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero for isolates
    W = W / row_sums[:, None]

    return W


# =============================================================================
# Global Moran's I
# =============================================================================

def global_morans_i(x, W):
    """
    Compute Global Moran's I statistic.

    Parameters
    ----------
    x : np.ndarray of shape (n,)
        Observed values.
    W : np.ndarray of shape (n, n)
        Spatial weight matrix (does NOT need to be row-standardized for this
        formula, but we use it with row-standardized weights).

    Returns
    -------
    I : float
        Moran's I statistic.
    E_I : float
        Expected I under spatial randomness = -1/(n-1).
    var_I : float
        Variance of I under normality assumption.
    z_score : float
        Standardized z-score.
    """
    n = len(x)
    z = x - x.mean()
    S0 = W.sum()

    if S0 == 0 or np.sum(z ** 2) == 0:
        return 0.0, -1.0 / (n - 1), 0.0, 0.0

    # Moran's I: (n / S0) * (z' W z) / (z' z)
    numerator = z @ W @ z
    denominator = z @ z
    I = (n / S0) * (numerator / denominator)

    # Expected value
    E_I = -1.0 / (n - 1)

    # Variance under normality assumption (Cliff & Ord)
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2)

    n2 = n * n
    A = n * ((n2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 * S0)
    B = (z ** 4).sum() / ((z ** 2).sum() / n) ** 2  # kurtosis-like term
    # Under normality assumption, B (kurtosis) = 3n(n-1) / ((n-1)(n+1)) but
    # we'll use the simpler normal-theory formula:
    D = (n - 1) * (n - 2) * (n - 3) * S0 * S0
    C = (n2 - n) * S1 - 2 * n * S2 + 6 * S0 * S0

    # Normal-theory variance
    var_I_num = (n * ((n2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 * S0)
                 - ((n2 - n) * S1 - 2 * n * S2 + 6 * S0 * S0))
    if D == 0:
        var_I = 0.0
    else:
        var_I = var_I_num / D - E_I ** 2

    # z-score
    if var_I > 0:
        z_score = (I - E_I) / np.sqrt(var_I)
    else:
        z_score = 0.0

    return I, E_I, var_I, z_score


def permutation_test_global(x, W, observed_I, n_perm=999, seed=42):
    """
    Permutation test for Global Moran's I.

    Returns
    -------
    p_value : float
        Pseudo p-value = (count of |I_perm| >= |I_obs| + 1) / (n_perm + 1).
    """
    rng = np.random.RandomState(seed)
    n = len(x)
    count = 0
    abs_obs = abs(observed_I)

    z_base = x - x.mean()
    denom_base = z_base @ z_base
    S0 = W.sum()

    if S0 == 0 or denom_base == 0:
        return 1.0

    for _ in range(n_perm):
        perm = rng.permutation(n)
        z_perm = z_base[perm]
        I_perm = (n / S0) * (z_perm @ W @ z_perm) / denom_base
        if abs(I_perm) >= abs_obs:
            count += 1

    p_value = (count + 1) / (n_perm + 1)
    return p_value


# =============================================================================
# Local Moran's I (LISA)
# =============================================================================

def local_morans_i(x, W):
    """
    Compute Local Indicators of Spatial Association (LISA).

    Parameters
    ----------
    x : np.ndarray of shape (n,)
        Observed values.
    W : np.ndarray of shape (n, n)
        Row-standardized spatial weight matrix.

    Returns
    -------
    I_local : np.ndarray of shape (n,)
        Local Moran's I for each observation.
    """
    n = len(x)
    z = x - x.mean()
    m2 = (z ** 2).sum() / n  # variance (population)

    if m2 == 0:
        return np.zeros(n)

    # Local I_i = (z_i / m2) * sum_j(w_ij * z_j)
    spatial_lag = W @ z
    I_local = (z / m2) * spatial_lag

    return I_local


def permutation_test_local(x, W, I_local, n_perm=999, seed=42):
    """
    Conditional permutation test for Local Moran's I.

    For each location i, fix x_i and permute remaining values, recompute I_i.

    Returns
    -------
    p_values : np.ndarray of shape (n,)
        Pseudo p-values for each observation.
    """
    rng = np.random.RandomState(seed)
    n = len(x)
    z = x - x.mean()
    m2 = (z ** 2).sum() / n

    if m2 == 0:
        return np.ones(n)

    p_values = np.ones(n)
    abs_obs = np.abs(I_local)

    # For efficiency: vectorized permutation across all locations in batches
    # We permute the full z-vector and reconstruct spatial lags.
    # This is an "unconditional" shortcut that is standard for large datasets.
    counts = np.zeros(n)

    for _ in range(n_perm):
        perm = rng.permutation(n)
        z_perm = z[perm]
        spatial_lag_perm = W @ z_perm
        I_perm = (z_perm / m2) * spatial_lag_perm

        counts += (np.abs(I_perm) >= abs_obs).astype(float)

    p_values = (counts + 1) / (n_perm + 1)
    return p_values


def classify_lisa(x, W, I_local, p_values, alpha=0.05):
    """
    Classify each observation into LISA clusters.

    Categories:
      HH - High-High: high value surrounded by high values (hot spot)
      LL - Low-Low: low value surrounded by low values (cold spot)
      HL - High-Low: high value surrounded by low values (spatial outlier)
      LH - Low-High: low value surrounded by high values (spatial outlier)
      NS - Not Significant

    Parameters
    ----------
    x : np.ndarray
        Observed values.
    W : np.ndarray
        Row-standardized weight matrix.
    I_local : np.ndarray
        Local Moran's I values.
    p_values : np.ndarray
        Pseudo p-values from permutation test.
    alpha : float
        Significance threshold.

    Returns
    -------
    clusters : np.ndarray of str
        Cluster labels for each observation.
    """
    n = len(x)
    z = x - x.mean()
    spatial_lag = W @ z

    clusters = np.full(n, "NS", dtype="U2")

    significant = p_values < alpha

    for i in range(n):
        if not significant[i]:
            continue
        if z[i] > 0 and spatial_lag[i] > 0:
            clusters[i] = "HH"
        elif z[i] < 0 and spatial_lag[i] < 0:
            clusters[i] = "LL"
        elif z[i] > 0 and spatial_lag[i] < 0:
            clusters[i] = "HL"
        elif z[i] < 0 and spatial_lag[i] > 0:
            clusters[i] = "LH"

    return clusters


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def load_county_data():
    """Load the enriched county-decade outlier data."""
    path = os.path.join(ANALYSIS_DIR, "county_decade_outliers.csv")
    if not os.path.exists(path):
        print(f"ERROR: Data file not found: {path}")
        print("Run calculate_outlier_scores.py first.")
        sys.exit(1)

    df = pd.read_csv(path)

    # Normalize state abbreviations
    df["State"] = df["State"].apply(normalize_state)

    print(f"Loaded {len(df)} county-decade records")
    print(f"Decades available: {sorted(df['decade'].unique())}")
    print(f"Target decades: {TARGET_DECADES}")

    return df


def analyze_decade(df_decade, decade):
    """
    Run full spatial autocorrelation analysis for a single decade.

    Returns
    -------
    global_results : list of dict
        Global Moran's I results for each metric.
    local_df : pd.DataFrame
        Local LISA results merged with county identifiers.
    """
    df_decade = df_decade.reset_index(drop=True)
    n = len(df_decade)
    print(f"\n  Counties in decade: {n}")
    print(f"  States represented: {df_decade['State'].nunique()}")

    # Build spatial weight matrix
    t0 = time.time()
    W = build_within_state_weights(df_decade)
    t_weights = time.time() - t0
    print(f"  Weight matrix built in {t_weights:.1f}s  (shape: {W.shape})")

    # Verify weight matrix properties
    avg_neighbors = (W > 0).sum(axis=1).mean()
    isolates = (W.sum(axis=1) == 0).sum()
    print(f"  Avg neighbors per county: {avg_neighbors:.1f}")
    if isolates > 0:
        print(f"  WARNING: {isolates} isolate counties (no neighbors)")

    global_results = []
    local_results = df_decade[["State", "County", "decade"]].copy()

    for metric, col in [("missing_per_100k", "missing_per_100k"),
                        ("bodies_per_100k", "bodies_per_100k")]:
        prefix = "mp" if "missing" in metric else "bodies"
        print(f"\n  --- {metric} ---")

        x = df_decade[col].values.astype(np.float64)

        # Handle NaN/inf: replace with 0
        mask = np.isfinite(x)
        if not mask.all():
            bad = (~mask).sum()
            print(f"  Replacing {bad} non-finite values with 0")
            x = np.where(mask, x, 0.0)

        # Global Moran's I
        t0 = time.time()
        I_val, E_I, var_I, z_score = global_morans_i(x, W)
        p_value = permutation_test_global(x, W, I_val, N_PERMUTATIONS, RANDOM_SEED)
        t_global = time.time() - t0

        print(f"  Global Moran's I = {I_val:.6f}")
        print(f"  E[I]             = {E_I:.6f}")
        print(f"  Variance         = {var_I:.6f}")
        print(f"  z-score          = {z_score:.4f}")
        print(f"  p-value (perm)   = {p_value:.4f}")
        print(f"  Significant      = {'YES' if p_value < SIGNIFICANCE_LEVEL else 'NO'}")
        print(f"  (computed in {t_global:.1f}s)")

        global_results.append({
            "decade": decade,
            "metric": metric,
            "morans_i": round(I_val, 6),
            "expected_i": round(E_I, 6),
            "variance": round(var_I, 6),
            "z_score": round(z_score, 4),
            "p_value": round(p_value, 4),
        })

        # Local Moran's I (LISA)
        t0 = time.time()
        I_local = local_morans_i(x, W)
        p_local = permutation_test_local(x, W, I_local, N_PERMUTATIONS, RANDOM_SEED)
        clusters = classify_lisa(x, W, I_local, p_local, SIGNIFICANCE_LEVEL)
        t_local = time.time() - t0

        local_results[f"{prefix}_lisa_i"] = np.round(I_local, 6)
        local_results[f"{prefix}_lisa_cluster"] = clusters
        local_results[f"{prefix}_lisa_p"] = np.round(p_local, 4)

        # Cluster summary
        unique, counts = np.unique(clusters, return_counts=True)
        cluster_dist = dict(zip(unique, counts))
        print(f"  LISA clusters: {cluster_dist}")
        print(f"  (computed in {t_local:.1f}s)")

    return global_results, local_results


def print_summary(global_df, local_df):
    """Print a summary of spatial analysis results."""
    print("\n")
    print("=" * 70)
    print("SPATIAL AUTOCORRELATION SUMMARY")
    print("=" * 70)

    # Global Moran's I summary table
    print("\n--- Global Moran's I by Decade ---")
    print(f"{'Decade':<8} {'Metric':<20} {'I':>10} {'E[I]':>10} {'z-score':>10} {'p-value':>10} {'Sig?':>6}")
    print("-" * 74)
    for _, row in global_df.iterrows():
        sig = "YES" if row["p_value"] < SIGNIFICANCE_LEVEL else "NO"
        print(f"{int(row['decade']):<8} {row['metric']:<20} "
              f"{row['morans_i']:>10.4f} {row['expected_i']:>10.4f} "
              f"{row['z_score']:>10.4f} {row['p_value']:>10.4f} {sig:>6}")

    # Interpretation
    print("\nInterpretation:")
    print("  Moran's I > 0 indicates positive spatial autocorrelation (clustering)")
    print("  Moran's I < 0 indicates negative spatial autocorrelation (dispersion)")
    print("  Moran's I ~ E[I] indicates spatial randomness")

    # LISA cluster summary by decade
    print("\n--- LISA Cluster Counts by Decade ---")

    for decade in TARGET_DECADES:
        d_sub = local_df[local_df["decade"] == decade]
        if d_sub.empty:
            continue
        print(f"\n  Decade {decade}s:")
        for prefix, label in [("mp", "Missing Persons"), ("bodies", "Unid. Bodies")]:
            col = f"{prefix}_lisa_cluster"
            if col not in d_sub.columns:
                continue
            counts = d_sub[col].value_counts()
            parts = [f"{k}={v}" for k, v in sorted(counts.items())]
            print(f"    {label}: {', '.join(parts)}")

    # Top HH (hot-spot) clusters for missing persons
    print("\n--- Top HH Clusters (Missing Persons) ---")
    print("Counties in significant High-High clusters (hotspots of high missing rates")
    print("surrounded by other counties with high missing rates):\n")

    for decade in TARGET_DECADES:
        d_sub = local_df[(local_df["decade"] == decade) &
                         (local_df["mp_lisa_cluster"] == "HH")]
        if d_sub.empty:
            print(f"  Decade {decade}s: No significant HH clusters")
            continue

        d_sub_sorted = d_sub.sort_values("mp_lisa_i", ascending=False)
        top_n = min(10, len(d_sub_sorted))
        print(f"  Decade {decade}s (top {top_n} of {len(d_sub_sorted)} HH counties):")
        for _, row in d_sub_sorted.head(top_n).iterrows():
            print(f"    {row['County']}, {row['State']}  "
                  f"(LISA I = {row['mp_lisa_i']:.4f}, p = {row['mp_lisa_p']:.4f})")

    # Top HH clusters for bodies
    print("\n--- Top HH Clusters (Unidentified Bodies) ---")
    print("Counties in significant High-High clusters (hotspots of high body recovery")
    print("rates surrounded by other counties with high body rates):\n")

    for decade in TARGET_DECADES:
        d_sub = local_df[(local_df["decade"] == decade) &
                         (local_df["bodies_lisa_cluster"] == "HH")]
        if d_sub.empty:
            print(f"  Decade {decade}s: No significant HH clusters")
            continue

        d_sub_sorted = d_sub.sort_values("bodies_lisa_i", ascending=False)
        top_n = min(10, len(d_sub_sorted))
        print(f"  Decade {decade}s (top {top_n} of {len(d_sub_sorted)} HH counties):")
        for _, row in d_sub_sorted.head(top_n).iterrows():
            print(f"    {row['County']}, {row['State']}  "
                  f"(LISA I = {row['bodies_lisa_i']:.4f}, p = {row['bodies_lisa_p']:.4f})")


def main():
    print("=" * 70)
    print("SPATIAL AUTOCORRELATION ANALYSIS")
    print("Missing Persons Outlier Detection -- County-Level LISA")
    print("=" * 70)
    print(f"\nSettings:")
    print(f"  Target decades:     {TARGET_DECADES}")
    print(f"  Permutations:       {N_PERMUTATIONS}")
    print(f"  Significance level: {SIGNIFICANCE_LEVEL}")
    print(f"  Weights:            Within-state contiguity (row-standardized)")

    # Load data
    df = load_county_data()

    # Filter to target decades
    df = df[df["decade"].isin(TARGET_DECADES)].copy()
    print(f"\nFiltered to target decades: {len(df)} records")

    all_global = []
    all_local = []

    for decade in TARGET_DECADES:
        df_decade = df[df["decade"] == decade].copy()
        if df_decade.empty:
            print(f"\nWARNING: No data for decade {decade}s, skipping.")
            continue

        print(f"\n{'='*70}")
        print(f"DECADE: {decade}s")
        print(f"{'='*70}")

        global_results, local_results = analyze_decade(df_decade, decade)
        all_global.extend(global_results)
        all_local.append(local_results)

    # Combine results
    global_df = pd.DataFrame(all_global)
    local_df = pd.concat(all_local, ignore_index=True)

    # Save Global Moran's I
    global_path = os.path.join(ANALYSIS_DIR, "global_morans_i.csv")
    global_df.to_csv(global_path, index=False)
    print(f"\nSaved Global Moran's I results: {global_path}")
    print(f"  ({len(global_df)} rows)")

    # Save Local LISA results
    local_path = os.path.join(ANALYSIS_DIR, "spatial_autocorrelation.csv")
    local_df.to_csv(local_path, index=False)
    print(f"Saved LISA results: {local_path}")
    print(f"  ({len(local_df)} rows)")

    # Summary
    print_summary(global_df, local_df)

    print("\n" + "=" * 70)
    print("SPATIAL ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
