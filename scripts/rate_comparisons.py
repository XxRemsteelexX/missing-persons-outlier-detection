#!/usr/bin/env python3
"""
Zone Rate Comparisons -- Poisson rate ratios, chi-squared, and Kruskal-Wallis
tests for missing-persons counts across geographic zones.

Outputs:
  data/analysis/zone_rate_comparisons.csv  (pairwise Poisson rate-ratio tests)
  data/analysis/zone_overall_tests.csv     (chi-squared + Kruskal-Wallis)
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from utils import ANALYSIS_DIR, ZONES, RAW_DIR, normalize_state, STATE_FULL

CONTINUITY = 0.5  # added to zero counts to keep log-transform finite


# ---------------------------------------------------------------------------
# 1. Load and aggregate
# ---------------------------------------------------------------------------
def load_zone_totals():
    """Return a DataFrame with one row per zone: zone, total_mp, total_bodies, n_years."""
    path = os.path.join(ANALYSIS_DIR, "zone_trends.csv")
    df = pd.read_csv(path)
    agg = (
        df.groupby("zone")
        .agg(total_mp=("mp_count", "sum"),
             total_bodies=("bodies_count", "sum"),
             n_years=("year", "nunique"))
        .reset_index()
    )
    return agg, df


# ---------------------------------------------------------------------------
# 2. Poisson pairwise rate ratios
# ---------------------------------------------------------------------------
def poisson_rate_ratios(totals):
    """Compute pairwise Poisson rate ratios with 95 % CIs and z-test p-values.

    For zones i, j the rate is total_mp / n_years.  Rate ratio RR = r_i / r_j.
    CI via log-transform: exp(log(RR) +/- 1.96 * sqrt(1/c_i + 1/c_j)).
    """
    rows = []
    zones = totals["zone"].tolist()
    for z1, z2 in itertools.combinations(zones, 2):
        for metric in ("mp_count", "bodies_count"):
            col = "total_mp" if metric == "mp_count" else "total_bodies"
            r1_row = totals.loc[totals["zone"] == z1].iloc[0]
            r2_row = totals.loc[totals["zone"] == z2].iloc[0]

            c1 = r1_row[col]
            c2 = r2_row[col]
            ny1 = r1_row["n_years"]
            ny2 = r2_row["n_years"]

            # Continuity correction for zero counts
            c1_adj = c1 if c1 > 0 else CONTINUITY
            c2_adj = c2 if c2 > 0 else CONTINUITY

            rate1 = c1_adj / ny1
            rate2 = c2_adj / ny2
            rr = rate1 / rate2
            log_rr = np.log(rr)
            se = np.sqrt(1.0 / c1_adj + 1.0 / c2_adj)

            rr_lower = np.exp(log_rr - 1.96 * se)
            rr_upper = np.exp(log_rr + 1.96 * se)

            z_stat = log_rr / se
            p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

            rows.append({
                "zone_1": z1,
                "zone_2": z2,
                "metric": metric,
                "rate_1": round(rate1, 4),
                "rate_2": round(rate2, 4),
                "rate_ratio": round(rr, 4),
                "rr_lower_95": round(rr_lower, 4),
                "rr_upper_95": round(rr_upper, 4),
                "p_value": round(p_val, 6),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Chi-squared goodness-of-fit
# ---------------------------------------------------------------------------
def chi_squared_test(totals, metric_col="total_mp"):
    """Chi-squared test: observed zone counts vs expected (proportional to n_years)."""
    observed = totals[metric_col].values.astype(float)
    n_years = totals["n_years"].values.astype(float)

    total_count = observed.sum()
    total_years = n_years.sum()
    expected = total_count * (n_years / total_years)

    # Continuity: bump any zero expected cell
    expected = np.where(expected == 0, CONTINUITY, expected)

    chi2, p_val = stats.chisquare(observed, f_exp=expected)
    dof = len(observed) - 1
    return chi2, p_val, dof


# ---------------------------------------------------------------------------
# 4. Kruskal-Wallis (ANOVA-style non-parametric)
# ---------------------------------------------------------------------------
def kruskal_wallis_test(trends_df, metric="mp_count"):
    """Kruskal-Wallis H-test of yearly counts across zones."""
    groups = [g[metric].values for _, g in trends_df.groupby("zone")]
    if len(groups) < 2:
        return np.nan, np.nan
    h_stat, p_val = stats.kruskal(*groups)
    return h_stat, p_val


# ---------------------------------------------------------------------------
# 5. Pretty-print summary
# ---------------------------------------------------------------------------
def print_summary(totals, pairwise, overall):
    sep = "=" * 74
    print(f"\n{sep}")
    print("ZONE RATE COMPARISONS -- SUMMARY")
    print(sep)

    # Per-zone totals
    print("\n-- Aggregated totals per zone --")
    print(f"{'Zone':<25s} {'MP':>7s} {'Bodies':>8s} {'Years':>6s} {'MP/yr':>8s}")
    print("-" * 58)
    for _, r in totals.iterrows():
        mp_rate = r["total_mp"] / r["n_years"] if r["n_years"] else 0
        print(f"{r['zone']:<25s} {int(r['total_mp']):>7d} {int(r['total_bodies']):>8d} "
              f"{int(r['n_years']):>6d} {mp_rate:>8.2f}")

    # Top pairwise results (mp_count only, sorted by p-value)
    mp_pairs = pairwise[pairwise["metric"] == "mp_count"].sort_values("p_value")
    print(f"\n-- Pairwise Poisson rate ratios (missing persons, top 10) --")
    print(f"{'Zone 1':<25s} {'Zone 2':<25s} {'RR':>7s} {'95% CI':>18s} {'p-value':>10s}")
    print("-" * 88)
    for _, r in mp_pairs.head(10).iterrows():
        ci = f"[{r['rr_lower_95']:.3f}, {r['rr_upper_95']:.3f}]"
        print(f"{r['zone_1']:<25s} {r['zone_2']:<25s} {r['rate_ratio']:>7.3f} "
              f"{ci:>18s} {r['p_value']:>10.6f}")

    # Overall tests
    print(f"\n-- Overall tests --")
    print(f"{'Test':<30s} {'Statistic':>12s} {'p-value':>12s} {'DoF':>6s}")
    print("-" * 62)
    for _, r in overall.iterrows():
        dof_str = str(int(r["dof"])) if pd.notna(r["dof"]) else "--"
        print(f"{r['test']:<30s} {r['statistic']:>12.4f} {r['p_value']:>12.6f} {dof_str:>6s}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    totals, trends = load_zone_totals()
    print(f"Loaded zone trends: {len(trends)} rows, {len(totals)} zones")

    # Pairwise rate ratios (both metrics)
    pairwise = poisson_rate_ratios(totals)
    out_pw = os.path.join(ANALYSIS_DIR, "zone_rate_comparisons.csv")
    pairwise.to_csv(out_pw, index=False)
    print(f"Saved pairwise comparisons -> {out_pw}")

    # Overall tests
    overall_rows = []

    # Chi-squared for MP
    chi2_mp, p_chi_mp, dof_mp = chi_squared_test(totals, "total_mp")
    overall_rows.append({"test": "Chi-squared (MP)", "statistic": chi2_mp,
                         "p_value": p_chi_mp, "dof": dof_mp})

    # Chi-squared for Bodies
    chi2_bd, p_chi_bd, dof_bd = chi_squared_test(totals, "total_bodies")
    overall_rows.append({"test": "Chi-squared (Bodies)", "statistic": chi2_bd,
                         "p_value": p_chi_bd, "dof": dof_bd})

    # Kruskal-Wallis for MP
    h_mp, p_kw_mp = kruskal_wallis_test(trends, "mp_count")
    overall_rows.append({"test": "Kruskal-Wallis (MP)", "statistic": h_mp,
                         "p_value": p_kw_mp, "dof": np.nan})

    # Kruskal-Wallis for Bodies
    h_bd, p_kw_bd = kruskal_wallis_test(trends, "bodies_count")
    overall_rows.append({"test": "Kruskal-Wallis (Bodies)", "statistic": h_bd,
                         "p_value": p_kw_bd, "dof": np.nan})

    overall = pd.DataFrame(overall_rows)
    out_ov = os.path.join(ANALYSIS_DIR, "zone_overall_tests.csv")
    overall.to_csv(out_ov, index=False)
    print(f"Saved overall tests       -> {out_ov}")

    print_summary(totals, pairwise, overall)


if __name__ == "__main__":
    main()
