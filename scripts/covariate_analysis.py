#!/usr/bin/env python3
"""
Covariate-Adjusted Outlier Detection

Regresses missing persons rates on socioeconomic covariates to identify
counties that are anomalous AFTER controlling for poverty, urbanization,
population density, and other factors.

Residual = observed_rate - predicted_rate = "true anomaly"
Counties with high positive residuals are anomalous even after context.
"""
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn is not installed. Random Forest overperformance "
          "analysis will be skipped. Install it with: pip install scikit-learn")

sys.path.insert(0, os.path.dirname(__file__))
from utils import ANALYSIS_DIR, COVARIATES_DIR, normalize_state

os.makedirs(ANALYSIS_DIR, exist_ok=True)


def load_data():
    """Load county decade outliers and socioeconomic covariates."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # Load crime data
    crime_file = os.path.join(ANALYSIS_DIR, "county_decade_outliers.csv")
    df_crime = pd.read_csv(crime_file)
    print(f"  Crime data: {len(df_crime)} county-decade records")

    # Load covariates
    cov_file = os.path.join(COVARIATES_DIR, "county_socioeconomic.csv")
    if not os.path.exists(cov_file):
        print(f"  Covariates not found at {cov_file}")
        print("  Run fetch_covariates.py first")
        return None, None

    df_cov = pd.read_csv(cov_file)
    print(f"  Covariates: {len(df_cov)} county records")

    return df_crime, df_cov


def merge_crime_with_covariates(df_crime, df_cov):
    """
    Merge crime data with covariates.
    Uses state abbreviation + county name for matching.
    """
    print("\n" + "=" * 70)
    print("MERGING CRIME DATA WITH COVARIATES")
    print("=" * 70)

    # Normalize state codes
    df_cov['state_abbrev'] = df_cov['state_abbrev'].apply(normalize_state)

    # Focus on recent decades (2010s, 2020s) where covariates are most relevant
    df_recent = df_crime[df_crime['decade'].isin([2010, 2020])].copy()
    print(f"  Crime data for 2010s-2020s: {len(df_recent)} records")

    # Merge on state + county
    df_merged = df_recent.merge(
        df_cov,
        left_on=['State', 'County'],
        right_on=['state_abbrev', 'county_name'],
        how='left',
        suffixes=('', '_cov')
    )

    matched = df_merged['poverty_rate'].notna().sum()
    print(f"  Matched with covariates: {matched} / {len(df_merged)} ({matched/len(df_merged)*100:.1f}%)")

    return df_merged


def run_multiple_regression(df):
    """
    Multiple regression: missing_per_100k ~ covariates
    Identifies which socioeconomic factors predict MP rates.
    """
    print("\n" + "=" * 70)
    print("MULTIPLE REGRESSION: MP RATE ~ COVARIATES")
    print("=" * 70)

    # Select features -- only include covariates that have data
    candidate_cols = [
        'poverty_rate', 'median_household_income', 'unemployment_rate',
        'pct_foreign_born', 'log_pop_density', 'log_population',
    ]

    # Only use covariates that exist and have sufficient non-null values
    available_covs = []
    for col in candidate_cols:
        if col in df.columns and df[col].notna().sum() > 100:
            available_covs.append(col)

    print(f"  Available covariates: {available_covs}")

    # Filter to rows with all available covariates
    mask = df['population'] > 0
    for col in available_covs:
        mask = mask & df[col].notna()

    df_reg = df[mask].copy()
    print(f"  Rows for regression: {len(df_reg)}")
    if not available_covs:
        print("  No covariates available for regression")
        return df, None

    X = df_reg[available_covs].values
    y = df_reg['missing_per_100k'].values

    # Standardize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # avoid division by zero
    X_scaled = (X - X_mean) / X_std

    # Add intercept
    n = len(X_scaled)
    X_design = np.column_stack([np.ones(n), X_scaled])

    # OLS: beta = (X'X)^-1 X'y
    try:
        XtX = X_design.T @ X_design
        Xty = X_design.T @ y
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        print("  Regression failed (singular matrix)")
        return df, None

    # Predictions and residuals
    y_pred = X_design @ beta
    residuals = y - y_pred

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Residual standard error
    df_resid = n - len(beta)
    mse = ss_res / df_resid if df_resid > 0 else 0
    se_resid = np.sqrt(mse)

    # Standard errors of coefficients
    try:
        cov_matrix = mse * np.linalg.inv(XtX)
        se_beta = np.sqrt(np.diag(cov_matrix))
        t_stats = beta / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))
    except np.linalg.LinAlgError:
        se_beta = np.full_like(beta, np.nan)
        t_stats = np.full_like(beta, np.nan)
        p_values = np.full_like(beta, np.nan)

    # Print results
    print(f"\n  R-squared: {r_squared:.4f}")
    print(f"  Residual SE: {se_resid:.4f}")
    print(f"  N: {n}")

    print(f"\n  {'Variable':<30} {'Coef':>10} {'SE':>10} {'t':>10} {'p':>10}")
    print(f"  {'-'*70}")
    print(f"  {'(Intercept)':<30} {beta[0]:>10.4f} {se_beta[0]:>10.4f} {t_stats[0]:>10.4f} {p_values[0]:>10.4f}")
    for i, col in enumerate(available_covs):
        j = i + 1
        sig = '*' if p_values[j] < 0.05 else ''
        print(f"  {col:<30} {beta[j]:>10.4f} {se_beta[j]:>10.4f} {t_stats[j]:>10.4f} {p_values[j]:>10.4f} {sig}")

    # Store predictions and residuals back into the full dataframe
    df.loc[df_reg.index, 'predicted_mp_rate'] = y_pred
    df.loc[df_reg.index, 'mp_residual'] = residuals
    df.loc[df_reg.index, 'mp_residual_z'] = (residuals - residuals.mean()) / residuals.std()

    # Do the same for bodies
    y_bodies = df_reg['bodies_per_100k'].values
    try:
        Xty_b = X_design.T @ y_bodies
        beta_b = np.linalg.solve(XtX, Xty_b)
        y_pred_b = X_design @ beta_b
        resid_b = y_bodies - y_pred_b

        ss_res_b = np.sum(resid_b ** 2)
        ss_tot_b = np.sum((y_bodies - y_bodies.mean()) ** 2)
        r2_b = 1 - ss_res_b / ss_tot_b if ss_tot_b > 0 else 0

        print(f"\n  Bodies regression R-squared: {r2_b:.4f}")

        df.loc[df_reg.index, 'predicted_bodies_rate'] = y_pred_b
        df.loc[df_reg.index, 'bodies_residual'] = resid_b
        df.loc[df_reg.index, 'bodies_residual_z'] = (resid_b - resid_b.mean()) / resid_b.std()
    except Exception:
        pass

    reg_info = {
        'r_squared': r_squared,
        'coefficients': dict(zip(['intercept'] + available_covs, beta)),
        'p_values': dict(zip(['intercept'] + available_covs, p_values)),
        'n': n,
    }

    return df, reg_info


def identify_overperformers(df):
    """
    Identify counties that are anomalous even after covariate adjustment.
    Positive residual = more MP than predicted by socioeconomic context.
    """
    print("\n" + "=" * 70)
    print("OVERPERFORMANCE ANALYSIS (COVARIATE-ADJUSTED OUTLIERS)")
    print("=" * 70)

    has_resid = df['mp_residual'].notna()
    df_adj = df[has_resid].copy()

    if len(df_adj) == 0:
        print("  No residuals available")
        return df

    # Top overperformers (highest positive residuals = most unexplained MP)
    print(f"\nTop 20 Counties with UNEXPLAINED High MP Rates:")
    print(f"  (Observed rate far exceeds what socioeconomic context predicts)")
    print(f"  {'County':<25} {'State':<6} {'Decade':<8} {'Obs Rate':>10} {'Pred Rate':>10} {'Residual':>10} {'Resid Z':>8}")
    print(f"  {'-'*85}")

    top = df_adj.nlargest(20, 'mp_residual')
    for _, row in top.iterrows():
        print(f"  {row['County']:<25} {row['State']:<6} {int(row['decade']):<8} "
              f"{row['missing_per_100k']:>10.1f} {row['predicted_mp_rate']:>10.1f} "
              f"{row['mp_residual']:>10.1f} {row['mp_residual_z']:>8.2f}")

    # Top underperformers (negative residuals = fewer MP than expected)
    print(f"\nTop 10 Counties with LOWER THAN EXPECTED MP Rates:")
    bottom = df_adj.nsmallest(10, 'mp_residual')
    for _, row in bottom.iterrows():
        print(f"  {row['County']:<25} {row['State']:<6} {int(row['decade']):<8} "
              f"{row['missing_per_100k']:>10.1f} {row['predicted_mp_rate']:>10.1f} "
              f"{row['mp_residual']:>10.1f} {row['mp_residual_z']:>8.2f}")

    # Classify
    df.loc[has_resid, 'adj_anomaly'] = 'Normal'
    df.loc[has_resid & (df['mp_residual_z'] > 2), 'adj_anomaly'] = 'High Residual (>2 sigma)'
    df.loc[has_resid & (df['mp_residual_z'] > 3), 'adj_anomaly'] = 'Extreme Residual (>3 sigma)'

    adj_counts = df.loc[has_resid, 'adj_anomaly'].value_counts()
    print(f"\nCovariate-Adjusted Anomaly Classification:")
    for level, count in adj_counts.items():
        print(f"  {level}: {count}")

    return df


def run_rf_overperformance(df):
    """
    Step 3.4: Random Forest overperformance analysis.

    Trains a Random Forest regressor on socioeconomic covariates to predict
    missing persons and unidentified bodies rates. Counties with the largest
    positive residuals (observed - predicted) represent the most anomalous
    locations after accounting for socioeconomic context via a nonlinear model.

    This complements the linear regression by capturing nonlinear covariate
    interactions that OLS cannot model.
    """
    if not SKLEARN_AVAILABLE:
        print("\n" + "=" * 70)
        print("SKIPPING RANDOM FOREST ANALYSIS (scikit-learn not installed)")
        print("=" * 70)
        return df

    print("\n" + "=" * 70)
    print("RANDOM FOREST OVERPERFORMANCE ANALYSIS (Step 3.4)")
    print("=" * 70)

    # --- Select covariates (same logic as the linear regression) ---
    candidate_cols = [
        'poverty_rate', 'median_household_income', 'unemployment_rate',
        'pct_foreign_born', 'log_population',
    ]

    available_covs = []
    for col in candidate_cols:
        if col in df.columns and df[col].notna().sum() > 100:
            available_covs.append(col)

    print(f"  Available covariates for RF: {available_covs}")

    if not available_covs:
        print("  No covariates available -- skipping RF analysis.")
        return df

    # --- Build clean analysis subset ---
    mask = df['missing_per_100k'].notna()
    for col in available_covs:
        mask = mask & df[col].notna()

    df_rf = df[mask].copy()
    print(f"  Rows with complete data for RF: {len(df_rf)}")

    if len(df_rf) < 30:
        print("  Too few rows for a meaningful Random Forest model.")
        return df

    X = df_rf[available_covs].values
    y_mp = df_rf['missing_per_100k'].values

    # ------------------------------------------------------------------
    # Part A: Random Forest for missing_per_100k
    # ------------------------------------------------------------------
    print("\n  --- Missing Persons Rate (missing_per_100k) ---")

    rf_mp = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=10,
        n_jobs=-1,
    )
    rf_mp.fit(X, y_mp)

    # 5-fold cross-validated R-squared
    cv_scores_mp = cross_val_score(
        RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10, n_jobs=-1),
        X, y_mp, cv=5, scoring='r2',
    )
    print(f"  5-fold CV R-squared (MP):  mean={cv_scores_mp.mean():.4f}  "
          f"std={cv_scores_mp.std():.4f}  "
          f"folds={[round(s, 4) for s in cv_scores_mp]}")

    y_mp_pred = rf_mp.predict(X)
    resid_mp = y_mp - y_mp_pred
    resid_mp_z = (resid_mp - resid_mp.mean()) / resid_mp.std() if resid_mp.std() > 0 else np.zeros_like(resid_mp)

    # Store into the full dataframe
    df.loc[df_rf.index, 'rf_predicted_mp'] = y_mp_pred
    df.loc[df_rf.index, 'rf_residual_mp'] = resid_mp
    df.loc[df_rf.index, 'rf_residual_mp_z'] = resid_mp_z

    # Feature importances for MP model
    importances_mp = rf_mp.feature_importances_
    print(f"\n  Feature importances (MP model):")
    for col, imp in sorted(zip(available_covs, importances_mp), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"    {col:<30} {imp:.4f}  {bar}")

    # ------------------------------------------------------------------
    # Part B: Random Forest for bodies_per_100k
    # ------------------------------------------------------------------
    has_bodies = df_rf['bodies_per_100k'].notna()
    if has_bodies.sum() > 30:
        print("\n  --- Unidentified Bodies Rate (bodies_per_100k) ---")

        y_bodies = df_rf.loc[has_bodies, 'bodies_per_100k'].values
        X_bodies = df_rf.loc[has_bodies, available_covs].values

        rf_bodies = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=10,
            n_jobs=-1,
        )
        rf_bodies.fit(X_bodies, y_bodies)

        cv_scores_bodies = cross_val_score(
            RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10, n_jobs=-1),
            X_bodies, y_bodies, cv=5, scoring='r2',
        )
        print(f"  5-fold CV R-squared (Bodies):  mean={cv_scores_bodies.mean():.4f}  "
              f"std={cv_scores_bodies.std():.4f}  "
              f"folds={[round(s, 4) for s in cv_scores_bodies]}")

        y_bodies_pred = rf_bodies.predict(X_bodies)
        resid_bodies = y_bodies - y_bodies_pred
        resid_bodies_z = (
            (resid_bodies - resid_bodies.mean()) / resid_bodies.std()
            if resid_bodies.std() > 0
            else np.zeros_like(resid_bodies)
        )

        bodies_idx = df_rf.index[has_bodies]
        df.loc[bodies_idx, 'rf_predicted_bodies'] = y_bodies_pred
        df.loc[bodies_idx, 'rf_residual_bodies'] = resid_bodies
        df.loc[bodies_idx, 'rf_residual_bodies_z'] = resid_bodies_z

        # Feature importances for bodies model
        importances_b = rf_bodies.feature_importances_
        print(f"\n  Feature importances (Bodies model):")
        for col, imp in sorted(zip(available_covs, importances_b), key=lambda x: -x[1]):
            bar = "#" * int(imp * 50)
            print(f"    {col:<30} {imp:.4f}  {bar}")
    else:
        print("\n  Insufficient bodies data for RF model.")

    # ------------------------------------------------------------------
    # Rank counties by RF residual (highest positive = most unexplained)
    # ------------------------------------------------------------------
    rf_mask = df['rf_residual_mp'].notna()
    df.loc[rf_mask, 'rf_overperformance_rank'] = (
        df.loc[rf_mask, 'rf_residual_mp']
        .rank(ascending=False, method='min')
        .astype(int)
    )

    # ------------------------------------------------------------------
    # Print top 20 overperformers by RF residual
    # ------------------------------------------------------------------
    df_ranked = df[rf_mask].copy()
    if len(df_ranked) > 0:
        print(f"\n  Top 20 RF Overperformers (highest positive residual = most unexplained):")
        print(f"  {'Rank':<6} {'County':<25} {'State':<6} {'Decade':<8} "
              f"{'Observed':>10} {'RF Pred':>10} {'Residual':>10} {'Resid Z':>8}")
        print(f"  {'-'*90}")

        top20 = df_ranked.nsmallest(20, 'rf_overperformance_rank')
        for _, row in top20.iterrows():
            decade_str = str(int(row['decade'])) if pd.notna(row.get('decade')) else '?'
            print(f"  {int(row['rf_overperformance_rank']):<6} "
                  f"{row['County']:<25} {row['State']:<6} {decade_str:<8} "
                  f"{row['missing_per_100k']:>10.1f} "
                  f"{row['rf_predicted_mp']:>10.1f} "
                  f"{row['rf_residual_mp']:>10.1f} "
                  f"{row['rf_residual_mp_z']:>8.2f}")

    return df


def save_results(df):
    """Save the covariate-adjusted analysis results."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save full adjusted dataset
    output_cols = [
        'State', 'County', 'state_name', 'decade',
        'missing_count', 'bodies_count', 'population',
        'missing_per_100k', 'bodies_per_100k',
        'mp_z_score', 'bodies_z_score',
        'poverty_rate', 'median_household_income', 'unemployment_rate',
        'pct_foreign_born', 'log_pop_density',
        'predicted_mp_rate', 'mp_residual', 'mp_residual_z',
        'predicted_bodies_rate', 'bodies_residual', 'bodies_residual_z',
        'adj_anomaly',
        'rf_predicted_mp', 'rf_residual_mp', 'rf_residual_mp_z',
        'rf_predicted_bodies', 'rf_residual_bodies', 'rf_residual_bodies_z',
        'rf_overperformance_rank',
    ]

    available_cols = [c for c in output_cols if c in df.columns]
    df_out = df[available_cols].copy()

    # Filter to rows with residuals
    df_out = df_out[df_out['mp_residual'].notna()]

    output_file = os.path.join(ANALYSIS_DIR, "covariate_adjusted_outliers.csv")
    df_out.to_csv(output_file, index=False)
    print(f"  Saved {len(df_out)} records to {output_file}")

    # Also save overperformance ranking
    df_rank = df_out.sort_values('mp_residual', ascending=False).head(100)
    rank_file = os.path.join(ANALYSIS_DIR, "overperformance_analysis.csv")
    df_rank.to_csv(rank_file, index=False)
    print(f"  Saved top 100 overperformers to {rank_file}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COVARIATE-ADJUSTED OUTLIER ANALYSIS")
    print("=" * 70)

    df_crime, df_cov = load_data()

    if df_crime is None or df_cov is None:
        print("Cannot proceed without both crime data and covariates")
        sys.exit(1)

    # Merge
    df_merged = merge_crime_with_covariates(df_crime, df_cov)

    # Initialize missing columns
    df_merged['predicted_mp_rate'] = np.nan
    df_merged['mp_residual'] = np.nan
    df_merged['mp_residual_z'] = np.nan
    df_merged['predicted_bodies_rate'] = np.nan
    df_merged['bodies_residual'] = np.nan
    df_merged['bodies_residual_z'] = np.nan

    # Regression
    df_result, reg_info = run_multiple_regression(df_merged)

    # Random Forest overperformance analysis (Step 3.4)
    df_result = run_rf_overperformance(df_result)

    # Overperformance
    df_result = identify_overperformers(df_result)

    # Save
    save_results(df_result)

    print("\n" + "=" * 70)
    print("COVARIATE ANALYSIS COMPLETE")
    print("=" * 70)
