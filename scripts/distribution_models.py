#!/usr/bin/env python3
"""
Distribution Models for Missing Persons Outlier Detection
----------------------------------------------------------
Standard z-scores assume normality, but crime count data is typically
right-skewed.  This script computes several alternative outlier measures:

  1. Log-transform z-scores  (stabilize right-skew)
  2. Poisson exceedance probabilities  (count-based model)
  3. Negative binomial exceedance  (handles overdispersion)
  4. Percentile ranks  (non-parametric)

Results are appended as new columns to county_decade_outliers.csv so
downstream analyses can compare detection methods.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from utils import ANALYSIS_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_FILE = os.path.join(ANALYSIS_DIR, "county_decade_outliers.csv")
SHAPIRO_MAX_N = 5000
SEPARATOR = "=" * 70


# ===================================================================
# 1. Data loading
# ===================================================================
def load_data():
    """Load the county-decade outlier data."""
    df = pd.read_csv(INPUT_FILE)
    print(SEPARATOR)
    print("DISTRIBUTION MODEL ANALYSIS")
    print(SEPARATOR)
    print(f"Loaded {len(df):,} county-decade records from")
    print(f"  {INPUT_FILE}")
    return df


# ===================================================================
# 2. Normality diagnostics
# ===================================================================
def run_normality_tests(df):
    """
    Run Shapiro-Wilk normality tests on rate distributions.
    If n > SHAPIRO_MAX_N, draw a random sample (Shapiro-Wilk has an
    upper-bound on sample size and becomes overly sensitive for large n).
    """
    print(f"\n{SEPARATOR}")
    print("NORMALITY DIAGNOSTICS")
    print(SEPARATOR)

    mask = df["population"] > 0
    results = {}

    for col_label, col_name in [
        ("Missing Persons per 100K", "missing_per_100k"),
        ("Unidentified Bodies per 100K", "bodies_per_100k"),
    ]:
        values = df.loc[mask, col_name].dropna()
        n = len(values)

        skew = float(stats.skew(values))
        kurt = float(stats.kurtosis(values))

        # Shapiro-Wilk (sample if needed)
        if n > SHAPIRO_MAX_N:
            sample = values.sample(n=SHAPIRO_MAX_N, random_state=42)
            stat, p_value = stats.shapiro(sample)
            sampled = True
        else:
            stat, p_value = stats.shapiro(values)
            sampled = False

        results[col_name] = {
            "n": n,
            "skewness": skew,
            "kurtosis": kurt,
            "shapiro_stat": float(stat),
            "shapiro_p": float(p_value),
            "sampled": sampled,
        }

        print(f"\n  {col_label} (n={n:,}):")
        print(f"    Skewness:  {skew:>8.3f}")
        print(f"    Kurtosis:  {kurt:>8.3f}")
        sample_note = f" (sampled {SHAPIRO_MAX_N:,})" if sampled else ""
        print(f"    Shapiro-Wilk W = {stat:.6f},  p = {p_value:.4e}{sample_note}")
        if p_value < 0.05:
            print("    --> Distribution is NOT normal (p < 0.05)")
        else:
            print("    --> Distribution is consistent with normality (p >= 0.05)")

    return results


# ===================================================================
# 3. Log-transform z-scores
# ===================================================================
def compute_log_z_scores(df):
    """
    Compute z-scores on the log(rate + 1) transformed distributions.
    The +1 offset handles zero counts gracefully.
    """
    print(f"\n{SEPARATOR}")
    print("LOG-TRANSFORM Z-SCORES")
    print(SEPARATOR)

    mask = df["population"] > 0

    for rate_col, out_col in [
        ("missing_per_100k", "mp_log_z"),
        ("bodies_per_100k", "bodies_log_z"),
    ]:
        log_values = np.log(df.loc[mask, rate_col] + 1)
        mean_log = log_values.mean()
        std_log = log_values.std()

        # Compute for all rows, but only meaningful where population > 0
        df[out_col] = np.nan
        if std_log > 0:
            df.loc[mask, out_col] = (
                np.log(df.loc[mask, rate_col] + 1) - mean_log
            ) / std_log
        else:
            df.loc[mask, out_col] = 0.0

        df[out_col] = df[out_col].fillna(0.0)

        print(f"\n  {rate_col}:")
        print(f"    log mean = {mean_log:.4f},  log std = {std_log:.4f}")
        print(f"    log-z range: [{df.loc[mask, out_col].min():.3f}, "
              f"{df.loc[mask, out_col].max():.3f}]")

    return df


# ===================================================================
# 4. Poisson exceedance probability
# ===================================================================
def compute_poisson_exceedance(df):
    """
    For each county-decade, compute P(X >= observed | Poisson(lambda))
    where lambda = national_rate * county_population / 100_000.

    A small p-value means the observed count is unexpectedly high
    given the national baseline rate.
    """
    print(f"\n{SEPARATOR}")
    print("POISSON EXCEEDANCE PROBABILITIES")
    print(SEPARATOR)

    mask = df["population"] > 0

    for count_col, rate_col, out_col in [
        ("missing_count", "missing_per_100k", "mp_poisson_p"),
        ("bodies_count", "bodies_per_100k", "bodies_poisson_p"),
    ]:
        # National rate: total counts / total population * 100K
        total_count = df.loc[mask, count_col].sum()
        total_pop = df.loc[mask, "population"].sum()
        national_rate = total_count / total_pop * 100_000 if total_pop > 0 else 0

        df[out_col] = np.nan

        if national_rate > 0:
            # Expected count per county-decade
            expected = df.loc[mask, "population"] * national_rate / 100_000
            observed = df.loc[mask, count_col].astype(int)

            # P(X >= observed) = 1 - P(X <= observed - 1) = sf(observed - 1)
            # sf(k) = P(X > k), so P(X >= k) = sf(k - 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_values = stats.poisson.sf(observed - 1, mu=expected)

            df.loc[mask, out_col] = p_values

        df[out_col] = df[out_col].fillna(1.0)

        sig_001 = (df.loc[mask, out_col] < 0.001).sum()
        sig_01 = (df.loc[mask, out_col] < 0.01).sum()
        sig_05 = (df.loc[mask, out_col] < 0.05).sum()

        print(f"\n  {count_col}:")
        print(f"    National rate: {national_rate:.4f} per 100K")
        print(f"    Significant at p < 0.001: {sig_001:,}")
        print(f"    Significant at p < 0.01:  {sig_01:,}")
        print(f"    Significant at p < 0.05:  {sig_05:,}")

    return df


# ===================================================================
# 5. Negative binomial exceedance probability
# ===================================================================
def compute_negative_binomial_exceedance(df):
    """
    Negative binomial model accounts for overdispersion (variance > mean)
    which is common in crime count data.

    We estimate the overdispersion parameter from the data, then compute
    P(X >= observed) under the fitted NB model.

    Parameterization: NB(n, p) where
      - mean  = n * (1-p) / p
      - var   = n * (1-p) / p^2
      - n (dispersion) = mean^2 / (var - mean)  [when var > mean]
    """
    print(f"\n{SEPARATOR}")
    print("NEGATIVE BINOMIAL EXCEEDANCE PROBABILITIES")
    print(SEPARATOR)

    mask = df["population"] > 0

    for count_col, rate_col, out_col in [
        ("missing_count", "missing_per_100k", "mp_nb_p"),
        ("bodies_count", "bodies_per_100k", "bodies_nb_p"),
    ]:
        # National rate
        total_count = df.loc[mask, count_col].sum()
        total_pop = df.loc[mask, "population"].sum()
        national_rate = total_count / total_pop * 100_000 if total_pop > 0 else 0

        df[out_col] = np.nan

        if national_rate > 0:
            expected = df.loc[mask, "population"] * national_rate / 100_000
            observed = df.loc[mask, count_col].astype(int)

            # Estimate overdispersion from residuals
            # Use the ratio of variance to mean of (observed / expected) as a
            # proxy, then map back to NB parameters per row.
            ratios = observed / expected.replace(0, np.nan)
            ratios = ratios.dropna()

            global_mean = ratios.mean()
            global_var = ratios.var()

            if global_var > global_mean and global_mean > 0:
                # Overdispersed -- use NB
                # For each row: mu = expected, and the common dispersion r
                # r = mu^2 / (sigma^2 - mu) at the global level
                overall_mu = expected.mean()
                overall_var = observed.var()

                if overall_var > overall_mu and overall_mu > 0:
                    r_global = overall_mu ** 2 / (overall_var - overall_mu)
                    r_global = max(r_global, 0.01)  # floor to avoid degenerate case
                else:
                    r_global = 1.0

                # Per-row NB parameters
                # mu_i = expected_i, r = r_global
                # p_i = r / (r + mu_i)
                mu_i = expected.values
                r = r_global
                p_i = r / (r + mu_i)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # scipy nbinom: P(X = k) with params (n, p)
                    # sf(k-1) = P(X >= k)
                    p_values = stats.nbinom.sf(
                        observed.values - 1, n=r, p=p_i
                    )

                df.loc[mask, out_col] = p_values
                print(f"\n  {count_col}:")
                print(f"    Overdispersion detected (var/mean ratio = "
                      f"{global_var / global_mean:.2f})")
                print(f"    Estimated r (dispersion) = {r_global:.4f}")
            else:
                # Not overdispersed -- fall back to Poisson
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p_values = stats.poisson.sf(
                        observed.values - 1, mu=expected.values
                    )
                df.loc[mask, out_col] = p_values
                print(f"\n  {count_col}:")
                print(f"    No overdispersion detected -- using Poisson fallback")

        df[out_col] = df[out_col].fillna(1.0)

        sig_001 = (df.loc[mask, out_col] < 0.001).sum()
        sig_01 = (df.loc[mask, out_col] < 0.01).sum()
        sig_05 = (df.loc[mask, out_col] < 0.05).sum()

        print(f"    Significant at p < 0.001: {sig_001:,}")
        print(f"    Significant at p < 0.01:  {sig_01:,}")
        print(f"    Significant at p < 0.05:  {sig_05:,}")

    return df


# ===================================================================
# 6. Percentile ranks (non-parametric)
# ===================================================================
def compute_percentile_ranks(df):
    """
    Compute percentile rank for each county-decade rate within the
    population-weighted distribution.  Pure rank-based, no distributional
    assumptions required.
    """
    print(f"\n{SEPARATOR}")
    print("PERCENTILE RANKS (NON-PARAMETRIC)")
    print(SEPARATOR)

    mask = df["population"] > 0

    for rate_col, out_col in [
        ("missing_per_100k", "mp_percentile"),
        ("bodies_per_100k", "bodies_percentile"),
    ]:
        df[out_col] = np.nan

        values = df.loc[mask, rate_col]
        # rankdata with 'average' method handles ties
        ranks = stats.rankdata(values, method="average")
        percentiles = ranks / len(ranks) * 100.0
        df.loc[mask, out_col] = percentiles

        df[out_col] = df[out_col].fillna(0.0)

        p90 = (df.loc[mask, out_col] >= 90).sum()
        p95 = (df.loc[mask, out_col] >= 95).sum()
        p99 = (df.loc[mask, out_col] >= 99).sum()

        print(f"\n  {rate_col}:")
        print(f"    >= 90th percentile: {p90:,}")
        print(f"    >= 95th percentile: {p95:,}")
        print(f"    >= 99th percentile: {p99:,}")

    return df


# ===================================================================
# 7. Method comparison diagnostics
# ===================================================================
def print_method_comparison(df):
    """
    Print a comparison of how different methods flag outliers so the
    analyst can judge which is most appropriate.
    """
    print(f"\n{SEPARATOR}")
    print("METHOD COMPARISON SUMMARY")
    print(SEPARATOR)

    mask = df["population"] > 0
    sub = df.loc[mask].copy()

    # Define thresholds for "flagged as outlier" under each method
    methods = {
        "Standard z-score (|z| > 2)": (
            (sub["mp_z_score"].abs() > 2) | (sub["bodies_z_score"].abs() > 2)
        ),
        "Log z-score (|z| > 2)": (
            (sub["mp_log_z"].abs() > 2) | (sub["bodies_log_z"].abs() > 2)
        ),
        "Poisson (p < 0.01)": (
            (sub["mp_poisson_p"] < 0.01) | (sub["bodies_poisson_p"] < 0.01)
        ),
        "Neg. Binomial (p < 0.01)": (
            (sub["mp_nb_p"] < 0.01) | (sub["bodies_nb_p"] < 0.01)
        ),
        "Percentile (>= 99th)": (
            (sub["mp_percentile"] >= 99) | (sub["bodies_percentile"] >= 99)
        ),
    }

    print(f"\n  {'Method':<35} {'Flagged':>8} {'% of total':>12}")
    print(f"  {'-' * 55}")
    for name, flagged in methods.items():
        n_flagged = flagged.sum()
        pct = n_flagged / len(sub) * 100 if len(sub) > 0 else 0
        print(f"  {name:<35} {n_flagged:>8,} {pct:>11.2f}%")

    # Overlap: flagged by ALL methods
    all_flagged = None
    for flagged in methods.values():
        if all_flagged is None:
            all_flagged = flagged
        else:
            all_flagged = all_flagged & flagged

    if all_flagged is not None:
        n_all = all_flagged.sum()
        print(f"\n  Flagged by ALL methods: {n_all:,}")

    # Correlation between z-score and log-z
    for prefix, label in [("mp", "Missing Persons"), ("bodies", "Unidentified Bodies")]:
        z_col = f"{prefix}_z_score"
        lz_col = f"{prefix}_log_z"
        corr = sub[z_col].corr(sub[lz_col])
        print(f"\n  Correlation (standard z vs log-z) for {label}: {corr:.4f}")


# ===================================================================
# 8. Save results
# ===================================================================
def save_results(df):
    """Save the enriched DataFrame back to the same CSV."""
    df.to_csv(INPUT_FILE, index=False)
    print(f"\n{SEPARATOR}")
    print("RESULTS SAVED")
    print(SEPARATOR)
    print(f"  Output: {INPUT_FILE}")
    print(f"  Rows:   {len(df):,}")
    new_cols = [
        "mp_log_z", "bodies_log_z",
        "mp_poisson_p", "bodies_poisson_p",
        "mp_nb_p", "bodies_nb_p",
        "mp_percentile", "bodies_percentile",
    ]
    print(f"  New columns added: {', '.join(new_cols)}")


# ===================================================================
# Main
# ===================================================================
def main():
    df = load_data()

    # Normality diagnostics (informational only)
    run_normality_tests(df)

    # Compute alternative outlier measures
    df = compute_log_z_scores(df)
    df = compute_poisson_exceedance(df)
    df = compute_negative_binomial_exceedance(df)
    df = compute_percentile_ranks(df)

    # Diagnostics
    print_method_comparison(df)

    # Persist
    save_results(df)

    print(f"\n{SEPARATOR}")
    print("DISTRIBUTION MODEL ANALYSIS COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
