#!/usr/bin/env python3
"""
Robust Statistics for Missing Persons Outlier Detection
========================================================

Enhances the existing outlier detection pipeline by replacing fragile
mean/std z-scores with robust alternatives and applying multiple-
comparison correction.

New columns added to county_decade_outliers.csv:
  - mp_robust_z        Modified z-score for missing persons rate
  - bodies_robust_z    Modified z-score for unidentified bodies rate
  - robust_composite_z Average of the two robust z-scores
  - mp_p_value         Two-tailed p-value from robust z (normal approx)
  - bodies_p_value     Two-tailed p-value from robust z (normal approx)
  - mp_adjusted_p      Benjamini-Hochberg adjusted p-value
  - bodies_adjusted_p  Benjamini-Hochberg adjusted p-value
  - fdr_significant    True if EITHER adjusted p-value < 0.05
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import median_abs_deviation, norm

sys.path.insert(0, os.path.dirname(__file__))
from utils import ANALYSIS_DIR


# ============================================================================
# Robust z-score helpers
# ============================================================================

def modified_z_score(series):
    """
    Compute the modified z-score using the median and MAD.

    Formula: (x - median) / (1.4826 * MAD)

    The constant 1.4826 makes MAD a consistent estimator of the standard
    deviation for normally distributed data.

    Parameters
    ----------
    series : pd.Series
        Numeric series (NaN values are ignored in the baseline calculation).

    Returns
    -------
    pd.Series
        Modified z-scores (same index as input). Returns 0.0 where the
        denominator would be zero (all identical values).
    """
    med = series.median()
    mad = median_abs_deviation(series.dropna(), nan_policy="omit")
    denominator = 1.4826 * mad
    if denominator == 0:
        return pd.Series(0.0, index=series.index)
    return (series - med) / denominator


def trimmed_stats(series, proportiontocut=0.05):
    """
    Return the trimmed mean and trimmed standard deviation.

    Parameters
    ----------
    series : pd.Series
        Numeric series.
    proportiontocut : float
        Fraction to cut from each end (default 5%).

    Returns
    -------
    tuple of (float, float)
        (trimmed_mean, trimmed_std)
    """
    clean = series.dropna().values
    tmean = stats.trim_mean(clean, proportiontocut)
    # scipy.stats.trimboth returns the trimmed array
    trimmed = stats.trimboth(clean, proportiontocut)
    tstd = trimmed.std(ddof=1) if len(trimmed) > 1 else 0.0
    return tmean, tstd


# ============================================================================
# Benjamini-Hochberg FDR correction
# ============================================================================

def benjamini_hochberg(p_values, alpha=0.05):
    """
    Apply the Benjamini-Hochberg procedure for FDR correction.

    Tries statsmodels first; falls back to a manual implementation.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    alpha : float
        Desired FDR level.

    Returns
    -------
    np.ndarray
        Adjusted p-values (same length as input).
    """
    p_values = np.asarray(p_values, dtype=float)

    try:
        from statsmodels.stats.multitest import multipletests
        _, adjusted, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
        return adjusted
    except ImportError:
        pass

    # Manual Benjamini-Hochberg
    n = len(p_values)
    if n == 0:
        return np.array([])

    order = np.argsort(p_values)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)

    adjusted = p_values * n / ranked
    # Enforce monotonicity (step-down) by walking from largest rank to smallest
    sorted_idx = np.argsort(ranked)[::-1]
    cummin = np.inf
    result = np.empty(n)
    for idx in sorted_idx:
        cummin = min(cummin, adjusted[idx])
        result[idx] = cummin

    return np.clip(result, 0.0, 1.0)


# ============================================================================
# Main pipeline
# ============================================================================

def load_decade_data():
    """Load the county-decade outlier CSV produced by calculate_outlier_scores.py."""
    path = os.path.join(ANALYSIS_DIR, "county_decade_outliers.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run calculate_outlier_scores.py first.")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from county_decade_outliers.csv")
    return df


def compute_robust_scores(df):
    """
    Compute robust z-scores, p-values, and FDR-adjusted p-values.

    Only rows with population > 0 participate in baseline estimation and
    receive robust scores. Rows with population <= 0 get NaN for robust
    columns but are preserved in the output.
    """
    print("\n" + "=" * 70)
    print("COMPUTING ROBUST STATISTICS")
    print("=" * 70)

    mask = df["population"] > 0
    print(f"Rows with population > 0: {mask.sum():,} / {len(df):,}")

    # ---- Modified z-scores ---------------------------------------------------
    mp_rates = df.loc[mask, "missing_per_100k"]
    bodies_rates = df.loc[mask, "bodies_per_100k"]

    df["mp_robust_z"] = np.nan
    df["bodies_robust_z"] = np.nan

    df.loc[mask, "mp_robust_z"] = modified_z_score(mp_rates).values
    df.loc[mask, "bodies_robust_z"] = modified_z_score(bodies_rates).values

    df["robust_composite_z"] = (df["mp_robust_z"] + df["bodies_robust_z"]) / 2

    # ---- Trimmed-mean comparison stats (informational) -----------------------
    mp_tmean, mp_tstd = trimmed_stats(mp_rates)
    bodies_tmean, bodies_tstd = trimmed_stats(bodies_rates)

    mp_median = mp_rates.median()
    mp_mad = median_abs_deviation(mp_rates.dropna(), nan_policy="omit")
    bodies_median = bodies_rates.median()
    bodies_mad = median_abs_deviation(bodies_rates.dropna(), nan_policy="omit")

    print(f"\nMissing Persons per 100K baselines:")
    print(f"  Original    mean={mp_rates.mean():.4f}  std={mp_rates.std():.4f}")
    print(f"  Trimmed(5%) mean={mp_tmean:.4f}  std={mp_tstd:.4f}")
    print(f"  Robust      median={mp_median:.4f}  MAD={mp_mad:.4f}")

    print(f"\nUnidentified Bodies per 100K baselines:")
    print(f"  Original    mean={bodies_rates.mean():.4f}  std={bodies_rates.std():.4f}")
    print(f"  Trimmed(5%) mean={bodies_tmean:.4f}  std={bodies_tstd:.4f}")
    print(f"  Robust      median={bodies_median:.4f}  MAD={bodies_mad:.4f}")

    # ---- P-values from robust z-scores (two-tailed, normal assumption) -------
    df["mp_p_value"] = np.nan
    df["bodies_p_value"] = np.nan

    df.loc[mask, "mp_p_value"] = 2 * norm.sf(np.abs(df.loc[mask, "mp_robust_z"]))
    df.loc[mask, "bodies_p_value"] = 2 * norm.sf(np.abs(df.loc[mask, "bodies_robust_z"]))

    # ---- Benjamini-Hochberg FDR correction -----------------------------------
    print("\nApplying Benjamini-Hochberg FDR correction (alpha=0.05) ...")

    df["mp_adjusted_p"] = np.nan
    df["bodies_adjusted_p"] = np.nan

    mp_pvals = df.loc[mask, "mp_p_value"].values
    bodies_pvals = df.loc[mask, "bodies_p_value"].values

    df.loc[mask, "mp_adjusted_p"] = benjamini_hochberg(mp_pvals, alpha=0.05)
    df.loc[mask, "bodies_adjusted_p"] = benjamini_hochberg(bodies_pvals, alpha=0.05)

    # ---- FDR significance flag -----------------------------------------------
    df["fdr_significant"] = False
    df.loc[mask, "fdr_significant"] = (
        (df.loc[mask, "mp_adjusted_p"] < 0.05) |
        (df.loc[mask, "bodies_adjusted_p"] < 0.05)
    )

    sig_count = df["fdr_significant"].sum()
    print(f"FDR-significant counties (at least one rate): {sig_count:,} / {mask.sum():,}")

    return df


def print_comparison(df):
    """
    Print a side-by-side comparison of original vs. robust alert counts.
    """
    print("\n" + "=" * 70)
    print("COMPARISON: ORIGINAL vs ROBUST Z-SCORES")
    print("=" * 70)

    mask = df["population"] > 0
    sub = df.loc[mask].copy()

    # Reclassify alerts using robust z-scores
    def robust_alert(row):
        mp_z = row["mp_robust_z"]
        bodies_z = row["bodies_robust_z"]
        composite_z = row["robust_composite_z"]

        if mp_z > 1 and bodies_z > 1:
            if composite_z > 3:
                return "RED"
            elif composite_z > 2:
                return "ORANGE"
            elif composite_z > 1:
                return "YELLOW"
        elif mp_z > 2 and bodies_z < 1:
            return "ORANGE"
        elif bodies_z > 2 and mp_z < 1:
            return "ORANGE"
        elif composite_z > 3:
            return "RED"
        elif composite_z > 2:
            return "ORANGE"
        elif composite_z > 1:
            return "YELLOW"
        return "GREEN"

    sub["robust_alert_level"] = sub.apply(robust_alert, axis=1)

    print("\nAlert level distribution (rows with population > 0):")
    print(f"  {'Level':<10} {'Original':>10} {'Robust':>10} {'Change':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for level in ["RED", "ORANGE", "YELLOW", "GREEN"]:
        orig = (sub["alert_level"] == level).sum()
        robust = (sub["robust_alert_level"] == level).sum()
        diff = robust - orig
        sign = "+" if diff > 0 else ""
        print(f"  {level:<10} {orig:>10,} {robust:>10,} {sign}{diff:>9,}")

    # How many rows changed alert level
    changed = (sub["alert_level"] != sub["robust_alert_level"]).sum()
    total = len(sub)
    print(f"\nRows with alert level change: {changed:,} / {total:,} ({changed/total*100:.1f}%)")

    # FDR filter impact
    fdr_sig = sub["fdr_significant"].sum()
    orig_flagged = (sub["alert_level"] != "GREEN").sum()
    robust_flagged = (sub["robust_alert_level"] != "GREEN").sum()
    both = ((sub["robust_alert_level"] != "GREEN") & sub["fdr_significant"]).sum()

    print(f"\nFDR-significant rows:           {fdr_sig:,}")
    print(f"Original non-GREEN alerts:      {orig_flagged:,}")
    print(f"Robust non-GREEN alerts:        {robust_flagged:,}")
    print(f"Robust non-GREEN AND FDR-sig:   {both:,}")


def main():
    print("=" * 70)
    print("ROBUST STATISTICS FOR MISSING PERSONS OUTLIER DETECTION")
    print("=" * 70)

    df = load_decade_data()
    df = compute_robust_scores(df)
    print_comparison(df)

    # Save enriched data back (overwrite with additional columns)
    output_path = os.path.join(ANALYSIS_DIR, "county_decade_outliers.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved enriched data to: {output_path}")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
