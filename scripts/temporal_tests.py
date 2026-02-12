#!/usr/bin/env python3
"""
Temporal Statistical Tests for Zone Trend Data
Runs ADF stationarity tests, Mann-Kendall monotonic trend tests, and
CUSUM structural break detection on missing persons and bodies time series
for each geographic zone.

Tests:
  1. Augmented Dickey-Fuller (ADF) stationarity test (via statsmodels)
  2. Mann-Kendall monotonic trend test (manual implementation)
  3. CUSUM structural break detection (cumulative sum of linear residuals)

Input:  data/analysis/zone_trends.csv
Output: data/analysis/temporal_trends.csv
"""
import os
import sys
import math

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, os.path.dirname(__file__))
from utils import ANALYSIS_DIR, ZONES, RAW_DIR, normalize_state

MIN_YEARS = 10  # Minimum number of observations required for tests


# =============================================================================
# Mann-Kendall Trend Test (manual implementation)
# =============================================================================
def mann_kendall_test(x, alpha=0.05):
    """
    Perform the Mann-Kendall monotonic trend test.

    Parameters
    ----------
    x : array-like
        Time series values (ordered by time).
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict with keys:
        z_stat : float  -- standardized test statistic
        p_value : float -- two-sided p-value
        trend : str     -- 'increasing', 'decreasing', or 'no trend'
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    # Calculate S statistic: sum of sgn(x_j - x_i) for all j > i
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
            # diff == 0 contributes 0

    # Calculate variance of S, accounting for tied groups
    # Count tied groups
    unique, counts = np.unique(x, return_counts=True)
    tied_groups = counts[counts > 1]

    # Var(S) = [n(n-1)(2n+5) - sum(t_i(t_i-1)(2t_i+5))] / 18
    var_s = (n * (n - 1) * (2 * n + 5))
    for t in tied_groups:
        var_s -= t * (t - 1) * (2 * t + 5)
    var_s = var_s / 18.0

    # Avoid division by zero for constant series
    if var_s == 0:
        return {'z_stat': 0.0, 'p_value': 1.0, 'trend': 'no trend'}

    # Calculate Z statistic with continuity correction
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    # Two-sided p-value from standard normal
    p_value = 2.0 * sp_stats.norm.sf(abs(z))

    # Determine trend direction
    if p_value <= alpha:
        trend = 'increasing' if z > 0 else 'decreasing'
    else:
        trend = 'no trend'

    return {'z_stat': z, 'p_value': p_value, 'trend': trend}


# =============================================================================
# CUSUM Structural Break Detection
# =============================================================================
def cusum_break_test(x, years, alpha=0.05):
    """
    Detect structural breaks using CUSUM of residuals from a linear trend.

    Fits a linear trend to the series, computes the cumulative sum of
    standardized residuals, and identifies the year with maximum cumulative
    deviation. Significance is assessed using the Brown-Durbin-Evans
    approximation: threshold = c(alpha) * sqrt(n), where c(0.05) ~ 0.948.

    Parameters
    ----------
    x : array-like
        Time series values.
    years : array-like
        Corresponding year labels.
    alpha : float
        Significance level.

    Returns
    -------
    dict with keys:
        break_year : int or None -- year of maximum CUSUM deviation
        is_significant : bool    -- whether the break exceeds the threshold
        max_cusum : float        -- maximum absolute CUSUM value
        threshold : float        -- significance threshold
    """
    x = np.asarray(x, dtype=float)
    years = np.asarray(years, dtype=float)
    n = len(x)

    # Fit linear trend
    slope, intercept, _, _, _ = sp_stats.linregress(years, x)
    predicted = slope * years + intercept
    residuals = x - predicted

    # Standardize residuals
    std_resid = np.std(residuals, ddof=1)
    if std_resid == 0:
        return {
            'break_year': None,
            'is_significant': False,
            'max_cusum': 0.0,
            'threshold': 0.0,
        }

    standardized = residuals / std_resid

    # Cumulative sum of standardized residuals
    cusum = np.cumsum(standardized)

    # Find year with maximum absolute CUSUM
    max_idx = np.argmax(np.abs(cusum))
    max_cusum = np.abs(cusum[max_idx])
    break_year = int(years[max_idx])

    # Brown-Durbin-Evans significance threshold
    # Critical values for CUSUM test at alpha=0.05 approx 0.948
    # Threshold scales with sqrt(n)
    if alpha <= 0.01:
        c_alpha = 1.143
    elif alpha <= 0.05:
        c_alpha = 0.948
    else:
        c_alpha = 0.850
    threshold = c_alpha * math.sqrt(n)

    is_significant = max_cusum > threshold

    return {
        'break_year': break_year,
        'is_significant': is_significant,
        'max_cusum': max_cusum,
        'threshold': threshold,
    }


# =============================================================================
# ADF Test Wrapper
# =============================================================================
def run_adf_test(x):
    """
    Run the Augmented Dickey-Fuller test for stationarity.

    Parameters
    ----------
    x : array-like
        Time series values.

    Returns
    -------
    dict with keys:
        adf_stat : float
        adf_p : float
        is_stationary : bool (True if p < 0.05)
    """
    x = np.asarray(x, dtype=float)

    try:
        result = adfuller(x, autolag='AIC')
        adf_stat = result[0]
        adf_p = result[1]
        is_stationary = adf_p < 0.05
    except Exception as e:
        print(f"    ADF test failed: {e}")
        adf_stat = np.nan
        adf_p = np.nan
        is_stationary = False

    return {
        'adf_stat': adf_stat,
        'adf_p': adf_p,
        'is_stationary': is_stationary,
    }


# =============================================================================
# Main Analysis
# =============================================================================
def run_temporal_tests():
    """Load zone_trends.csv and run all temporal tests for each zone."""
    print("=" * 70)
    print("TEMPORAL STATISTICAL TESTS")
    print("=" * 70)

    # Load zone trend data
    trends_path = os.path.join(ANALYSIS_DIR, "zone_trends.csv")
    if not os.path.exists(trends_path):
        print(f"ERROR: {trends_path} not found.")
        print("Run zone_analysis_forecasting.py first to generate zone trends.")
        sys.exit(1)

    df = pd.read_csv(trends_path)
    print(f"Loaded {len(df)} records from {trends_path}")
    print(f"Zones found: {df['zone'].nunique()}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")

    results = []
    zones = df['zone'].unique()

    for zone in zones:
        print(f"\n{'=' * 70}")
        print(f"ZONE: {zone}")
        print(f"{'=' * 70}")

        zone_df = df[df['zone'] == zone].sort_values('year').reset_index(drop=True)
        n_years = len(zone_df)
        print(f"  Years of data: {n_years} ({int(zone_df['year'].min())}-{int(zone_df['year'].max())})")

        if n_years < MIN_YEARS:
            print(f"  SKIPPED: Insufficient data (need at least {MIN_YEARS} years, have {n_years})")
            continue

        years = zone_df['year'].values
        metrics = {
            'mp': zone_df['mp_count'].values.astype(float),
            'bodies': zone_df['bodies_count'].values.astype(float),
        }

        for metric_name, values in metrics.items():
            label = "Missing Persons" if metric_name == "mp" else "Unidentified Bodies"
            print(f"\n  --- {label} ---")

            # 1. ADF Stationarity Test
            adf_result = run_adf_test(values)
            stationarity_str = "STATIONARY" if adf_result['is_stationary'] else "NON-STATIONARY"
            print(f"  ADF Test: stat={adf_result['adf_stat']:.4f}, p={adf_result['adf_p']:.4f} -> {stationarity_str}")

            # 2. Mann-Kendall Trend Test
            mk_result = mann_kendall_test(values)
            print(f"  Mann-Kendall: Z={mk_result['z_stat']:.4f}, p={mk_result['p_value']:.4f} -> {mk_result['trend']}")

            # 3. CUSUM Structural Break Detection
            cusum_result = cusum_break_test(values, years)
            if cusum_result['break_year'] is not None:
                sig_str = "SIGNIFICANT" if cusum_result['is_significant'] else "not significant"
                print(f"  CUSUM Break: year={cusum_result['break_year']}, "
                      f"max_cusum={cusum_result['max_cusum']:.3f}, "
                      f"threshold={cusum_result['threshold']:.3f} -> {sig_str}")
            else:
                print(f"  CUSUM Break: no break detected (constant series)")

            # Collect row
            results.append({
                'zone': zone,
                'metric': metric_name,
                'adf_stat': round(adf_result['adf_stat'], 6) if not np.isnan(adf_result['adf_stat']) else np.nan,
                'adf_p': round(adf_result['adf_p'], 6) if not np.isnan(adf_result['adf_p']) else np.nan,
                'is_stationary': adf_result['is_stationary'],
                'mk_z': round(mk_result['z_stat'], 6),
                'mk_p': round(mk_result['p_value'], 6),
                'mk_trend': mk_result['trend'],
                'break_year': cusum_result['break_year'],
                'break_significance': cusum_result['is_significant'],
            })

    # Save results
    if results:
        df_results = pd.DataFrame(results)
        output_path = os.path.join(ANALYSIS_DIR, "temporal_trends.csv")
        df_results.to_csv(output_path, index=False)
        print(f"\n{'=' * 70}")
        print(f"RESULTS SAVED")
        print(f"{'=' * 70}")
        print(f"Saved {len(df_results)} test results to {output_path}")

        # Print summary table
        print(f"\n{'=' * 70}")
        print(f"SUMMARY")
        print(f"{'=' * 70}")
        print(f"\n{'Zone':<25} {'Metric':<8} {'ADF p':<10} {'Stationary':<12} "
              f"{'MK Z':<10} {'MK Trend':<14} {'Break Yr':<10} {'Break Sig':<10}")
        print("-" * 109)
        for _, row in df_results.iterrows():
            adf_p_str = f"{row['adf_p']:.4f}" if not pd.isna(row['adf_p']) else "N/A"
            stat_str = "Yes" if row['is_stationary'] else "No"
            break_yr_str = str(int(row['break_year'])) if pd.notna(row['break_year']) else "N/A"
            break_sig_str = "Yes" if row['break_significance'] else "No"
            print(f"{row['zone']:<25} {row['metric']:<8} {adf_p_str:<10} {stat_str:<12} "
                  f"{row['mk_z']:<10.4f} {row['mk_trend']:<14} {break_yr_str:<10} {break_sig_str:<10}")
    else:
        print("\nNo results to save -- all zones had insufficient data.")

    print(f"\n{'=' * 70}")
    print("TEMPORAL TESTS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_temporal_tests()
