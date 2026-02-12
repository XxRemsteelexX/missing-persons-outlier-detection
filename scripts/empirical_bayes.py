#!/usr/bin/env python3
"""
Empirical Bayes Shrinkage for County-Level Missing Persons Rates
================================================================

Problem:
  Small counties with tiny populations produce extreme per-100K rates from
  small counts. For example, 1 missing person in a county of 354 people
  yields 282 per 100K -- a statistical artifact, not a meaningful signal.

Solution:
  James-Stein style empirical Bayes shrinkage pulls small-county rates toward
  the national mean, weighted by population reliability. Large counties keep
  their observed rates nearly unchanged; small counties get regularized.

  shrunk_rate = w * observed_rate + (1 - w) * grand_mean
  where w = population / (population + shrinkage_constant)

Outputs:
  - Adds columns: mp_rate_shrunk, bodies_rate_shrunk, shrinkage_weight,
    small_county_flag, mp_eb_z, bodies_eb_z
  - Overwrites county_decade_outliers.csv with enriched data
"""
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from utils import ANALYSIS_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_POPULATION_THRESHOLD = 10_000  # per decade; below this is flagged unreliable
INPUT_FILE = os.path.join(ANALYSIS_DIR, "county_decade_outliers.csv")


def estimate_shrinkage_constant(populations, observed_rates):
    """
    Estimate the shrinkage constant (B) from the data.

    Uses a method-of-moments approach:
      - The variance of observed rates has two components:
        (1) true between-county variance (signal)
        (2) sampling variance that scales as ~1/population (noise)
      - We approximate B as the harmonic mean of the populations, which
        represents the "typical" population at which sampling noise equals
        the signal variance.

    For very skewed population distributions the harmonic mean is a robust
    choice because it naturally down-weights the enormous counties that
    don't need shrinkage anyway.
    """
    valid = populations > 0
    if valid.sum() == 0:
        return 1.0

    pops = populations[valid].values.astype(float)
    harmonic_mean = len(pops) / np.sum(1.0 / pops)

    return harmonic_mean


def apply_empirical_bayes(df):
    """
    Apply empirical Bayes shrinkage to missing persons and bodies rates.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: population, missing_per_100k, bodies_per_100k

    Returns
    -------
    pd.DataFrame
        Original dataframe with added columns:
        mp_rate_shrunk, bodies_rate_shrunk, shrinkage_weight,
        small_county_flag, mp_eb_z, bodies_eb_z
    """
    df = df.copy()

    # Only work with rows that have valid population
    has_pop = df["population"] > 0

    # ------------------------------------------------------------------
    # Step 1: Estimate grand means (national baselines)
    # ------------------------------------------------------------------
    mp_grand_mean = df.loc[has_pop, "missing_per_100k"].mean()
    bodies_grand_mean = df.loc[has_pop, "bodies_per_100k"].mean()

    print("=" * 70)
    print("EMPIRICAL BAYES SHRINKAGE")
    print("=" * 70)
    print(f"\nNational grand mean (missing persons): {mp_grand_mean:.4f} per 100K")
    print(f"National grand mean (unidentified bodies): {bodies_grand_mean:.4f} per 100K")
    print(f"County-decades with population > 0: {has_pop.sum():,} / {len(df):,}")

    # ------------------------------------------------------------------
    # Step 2: Estimate shrinkage constant from the data
    # ------------------------------------------------------------------
    B_mp = estimate_shrinkage_constant(
        df.loc[has_pop, "population"],
        df.loc[has_pop, "missing_per_100k"],
    )
    B_bodies = estimate_shrinkage_constant(
        df.loc[has_pop, "population"],
        df.loc[has_pop, "bodies_per_100k"],
    )
    # Use a single shrinkage constant (average) for interpretability
    B = (B_mp + B_bodies) / 2.0

    print(f"\nEstimated shrinkage constant (B): {B:,.0f}")
    print(f"  (Harmonic mean of populations across county-decades)")
    print(f"  Counties with pop << {B:,.0f} get pulled strongly toward the mean")
    print(f"  Counties with pop >> {B:,.0f} keep their observed rates")

    # ------------------------------------------------------------------
    # Step 3: Compute shrinkage weights and shrunk rates
    # ------------------------------------------------------------------
    populations = df["population"].values.astype(float)

    # Weight = population / (population + B); ranges from 0 to 1
    weights = np.where(
        populations > 0,
        populations / (populations + B),
        0.0,
    )
    df["shrinkage_weight"] = weights

    # Shrunk rates: w * observed + (1 - w) * grand_mean
    df["mp_rate_shrunk"] = np.where(
        has_pop,
        weights * df["missing_per_100k"].values + (1.0 - weights) * mp_grand_mean,
        0.0,
    )
    df["bodies_rate_shrunk"] = np.where(
        has_pop,
        weights * df["bodies_per_100k"].values + (1.0 - weights) * bodies_grand_mean,
        0.0,
    )

    # ------------------------------------------------------------------
    # Step 4: Flag small counties
    # ------------------------------------------------------------------
    df["small_county_flag"] = np.where(
        df["population"] < MIN_POPULATION_THRESHOLD,
        True,
        False,
    )
    df["small_county_flag"] = df["small_county_flag"].astype(bool)

    n_small = df["small_county_flag"].sum()
    n_total = len(df)
    print(f"\nSmall county flag (population < {MIN_POPULATION_THRESHOLD:,}):")
    print(f"  Flagged: {n_small:,} / {n_total:,} ({n_small / n_total * 100:.1f}%)")
    print(f"  Reliable: {n_total - n_small:,} / {n_total:,} ({(n_total - n_small) / n_total * 100:.1f}%)")

    # ------------------------------------------------------------------
    # Step 5: Recompute z-scores on shrunk rates
    # ------------------------------------------------------------------
    mp_shrunk_mean = df.loc[has_pop, "mp_rate_shrunk"].mean()
    mp_shrunk_std = df.loc[has_pop, "mp_rate_shrunk"].std()

    bodies_shrunk_mean = df.loc[has_pop, "bodies_rate_shrunk"].mean()
    bodies_shrunk_std = df.loc[has_pop, "bodies_rate_shrunk"].std()

    if mp_shrunk_std > 0:
        df["mp_eb_z"] = np.where(
            has_pop,
            (df["mp_rate_shrunk"] - mp_shrunk_mean) / mp_shrunk_std,
            0.0,
        )
    else:
        df["mp_eb_z"] = 0.0

    if bodies_shrunk_std > 0:
        df["bodies_eb_z"] = np.where(
            has_pop,
            (df["bodies_rate_shrunk"] - bodies_shrunk_mean) / bodies_shrunk_std,
            0.0,
        )
    else:
        df["bodies_eb_z"] = 0.0

    print(f"\nShrunk rate statistics (rows with population > 0):")
    print(f"  MP shrunk:     {mp_shrunk_mean:.4f} +/- {mp_shrunk_std:.4f} per 100K")
    print(f"  Bodies shrunk: {bodies_shrunk_mean:.4f} +/- {bodies_shrunk_std:.4f} per 100K")

    return df


def print_before_after_comparison(df):
    """
    Print a comparison of top outliers before and after shrinkage.
    Shows how small-county artifacts get corrected.
    """
    has_pop = df["population"] > 0
    df_valid = df[has_pop].copy()

    print("\n" + "=" * 70)
    print("BEFORE vs AFTER SHRINKAGE -- TOP 15 MISSING PERSONS OUTLIERS")
    print("=" * 70)

    # Before: top by raw rate
    top_raw = df_valid.nlargest(15, "missing_per_100k")
    print("\n[BEFORE] Top 15 by raw missing_per_100k:")
    print(f"  {'County':<25} {'State':<6} {'Decade':<7} {'Pop':>10} {'Raw Rate':>10} {'Z-score':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*10} {'-'*10} {'-'*8}")
    for _, row in top_raw.iterrows():
        print(
            f"  {row['County']:<25} {row['State']:<6} {int(row['decade']):<7} "
            f"{row['population']:>10,.0f} {row['missing_per_100k']:>10.1f} "
            f"{row['mp_z_score']:>8.2f}"
        )

    # After: top by shrunk rate
    top_shrunk = df_valid.nlargest(15, "mp_rate_shrunk")
    print(f"\n[AFTER] Top 15 by shrunk mp_rate_shrunk:")
    print(f"  {'County':<25} {'State':<6} {'Decade':<7} {'Pop':>10} {'Shrunk Rate':>11} {'EB Z':>8} {'Weight':>7}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*10} {'-'*11} {'-'*8} {'-'*7}")
    for _, row in top_shrunk.iterrows():
        print(
            f"  {row['County']:<25} {row['State']:<6} {int(row['decade']):<7} "
            f"{row['population']:>10,.0f} {row['mp_rate_shrunk']:>11.1f} "
            f"{row['mp_eb_z']:>8.2f} {row['shrinkage_weight']:>7.3f}"
        )

    # Show biggest drops (counties most affected by shrinkage)
    df_valid["mp_rate_drop"] = df_valid["missing_per_100k"] - df_valid["mp_rate_shrunk"]
    top_drops = df_valid.nlargest(10, "mp_rate_drop")

    print(f"\n[BIGGEST CORRECTIONS] Counties most shrunk toward the mean:")
    print(f"  {'County':<25} {'State':<6} {'Pop':>10} {'Raw':>8} {'Shrunk':>8} {'Drop':>8} {'Weight':>7}")
    print(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
    for _, row in top_drops.iterrows():
        print(
            f"  {row['County']:<25} {row['State']:<6} "
            f"{row['population']:>10,.0f} {row['missing_per_100k']:>8.1f} "
            f"{row['mp_rate_shrunk']:>8.1f} {row['mp_rate_drop']:>8.1f} "
            f"{row['shrinkage_weight']:>7.3f}"
        )


def print_weight_distribution(df):
    """Print the distribution of shrinkage weights for interpretability."""
    has_pop = df["population"] > 0
    weights = df.loc[has_pop, "shrinkage_weight"]

    print("\n" + "=" * 70)
    print("SHRINKAGE WEIGHT DISTRIBUTION")
    print("=" * 70)
    print(f"  Min:    {weights.min():.4f}")
    print(f"  25th:   {weights.quantile(0.25):.4f}")
    print(f"  Median: {weights.median():.4f}")
    print(f"  75th:   {weights.quantile(0.75):.4f}")
    print(f"  Max:    {weights.max():.4f}")
    print(f"\n  Weight < 0.5 (heavily shrunk): {(weights < 0.5).sum():,} county-decades")
    print(f"  Weight >= 0.5 and < 0.9:       {((weights >= 0.5) & (weights < 0.9)).sum():,} county-decades")
    print(f"  Weight >= 0.9 (minimal shrinkage): {(weights >= 0.9).sum():,} county-decades")


def main():
    """Load data, apply empirical Bayes shrinkage, save enriched results."""
    print("\n" + "=" * 70)
    print("MISSING PERSONS OUTLIER DETECTION")
    print("Empirical Bayes Shrinkage for Small-County Rate Stabilization")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if not os.path.exists(INPUT_FILE):
        print(f"\nERROR: Input file not found: {INPUT_FILE}")
        print("Run calculate_outlier_scores.py first to generate decade-level data.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded {len(df):,} county-decade records from:")
    print(f"  {INPUT_FILE}")

    pop_stats = df.loc[df["population"] > 0, "population"]
    print(f"\nPopulation range: {pop_stats.min():,.0f} to {pop_stats.max():,.0f}")
    print(f"Median population: {pop_stats.median():,.0f}")

    # ------------------------------------------------------------------
    # Apply empirical Bayes shrinkage
    # ------------------------------------------------------------------
    print()
    df = apply_empirical_bayes(df)

    # ------------------------------------------------------------------
    # Print diagnostics
    # ------------------------------------------------------------------
    print_weight_distribution(df)
    print_before_after_comparison(df)

    # ------------------------------------------------------------------
    # Save enriched data
    # ------------------------------------------------------------------
    df.to_csv(INPUT_FILE, index=False)
    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)
    print(f"Saved enriched data ({len(df):,} rows, {len(df.columns)} columns) to:")
    print(f"  {INPUT_FILE}")
    print(f"\nNew columns added:")
    for col in ["mp_rate_shrunk", "bodies_rate_shrunk", "shrinkage_weight",
                 "small_county_flag", "mp_eb_z", "bodies_eb_z"]:
        print(f"  - {col}")

    print("\n" + "=" * 70)
    print("EMPIRICAL BAYES SHRINKAGE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
