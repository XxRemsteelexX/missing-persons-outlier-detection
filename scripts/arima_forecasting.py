#!/usr/bin/env python3
"""
ARIMA-Based Forecasting for Zone Crime Trends

Fits ARIMA models alongside the existing linear regression forecasts produced
by zone_analysis_forecasting.py.  For each zone and metric (mp_count,
bodies_count) the script:

  1. Auto-selects the best (p,d,q) order by AIC grid search.
  2. Generates a 5-year forecast (2026-2030) with 95% prediction intervals.
  3. Back-tests both ARIMA and linear regression on the last 5 years of
     historical data and compares hold-out RMSE.

Outputs
-------
  data/analysis/arima_forecasts.csv
      zone, year, metric, arima_forecast, arima_lower_95, arima_upper_95,
      arima_order, arima_aic
  data/analysis/forecast_backtest.csv
      zone, metric, linear_rmse, arima_rmse, better_model, improvement_pct
"""

import warnings
import itertools
import sys
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

sys.path.insert(0, os.path.dirname(__file__))
from utils import ANALYSIS_DIR

os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FORECAST_YEARS = np.arange(2026, 2031)
HOLDOUT_SIZE = 5                          # years held out for back-testing
P_RANGE = range(0, 4)                     # p = 0..3
D_RANGE = range(0, 3)                     # d = 0..2
Q_RANGE = range(0, 4)                     # q = 0..3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_arima(series, order):
    """Fit a single ARIMA(p,d,q) model, return the fitted result or None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ARIMA(series, order=order)
            result = model.fit()
            return result
        except Exception:
            return None


def auto_arima(series):
    """
    Grid-search over (p,d,q) combinations and return the fit with lowest AIC.

    Returns
    -------
    best_result : ARIMAResults or None
    best_order  : tuple (p, d, q)
    best_aic    : float
    """
    best_aic = np.inf
    best_result = None
    best_order = (0, 0, 0)

    for p, d, q in itertools.product(P_RANGE, D_RANGE, Q_RANGE):
        if p == 0 and q == 0:
            # ARIMA(0,d,0) is just differencing -- skip
            continue
        result = _fit_arima(series, order=(p, d, q))
        if result is not None and np.isfinite(result.aic):
            if result.aic < best_aic:
                best_aic = result.aic
                best_result = result
                best_order = (p, d, q)

    # Fallback: if nothing converged, try the simplest non-trivial model
    if best_result is None:
        result = _fit_arima(series, order=(1, 0, 0))
        if result is not None:
            best_result = result
            best_order = (1, 0, 0)
            best_aic = result.aic

    return best_result, best_order, best_aic


def arima_forecast(series, steps=5):
    """
    Auto-select ARIMA order, fit on full series, forecast `steps` ahead.

    Returns dict with forecast values, confidence bounds, order, and AIC,
    or None if fitting fails entirely.
    """
    result, order, aic = auto_arima(series)
    if result is None:
        return None

    fc = result.get_forecast(steps=steps)
    predicted_mean = fc.predicted_mean
    predicted = np.asarray(predicted_mean)
    ci = fc.conf_int(alpha=0.05)
    # conf_int may return a DataFrame or ndarray depending on statsmodels version
    if hasattr(ci, "iloc"):
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
    else:
        ci = np.asarray(ci)
        lower = ci[:, 0]
        upper = ci[:, 1]

    # Floor at 0 -- counts cannot be negative
    predicted = np.maximum(predicted, 0.0)
    lower = np.maximum(lower, 0.0)
    upper = np.maximum(upper, 0.0)

    return {
        "forecast": predicted,
        "lower_95": lower,
        "upper_95": upper,
        "order": order,
        "aic": aic,
    }


def linear_forecast(years_train, values_train, years_test):
    """
    Simple linear regression forecast (mirrors zone_analysis_forecasting.py).
    Returns predicted values for years_test.
    """
    if len(years_train) < 3:
        return np.full(len(years_test), np.nan)
    slope, intercept, _, _, _ = stats.linregress(years_train, values_train)
    predicted = slope * np.array(years_test, dtype=float) + intercept
    return np.maximum(predicted, 0.0)


def rmse(actual, predicted):
    """Root mean squared error, ignoring NaN entries."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    trends_path = os.path.join(ANALYSIS_DIR, "zone_trends.csv")
    forecasts_path = os.path.join(ANALYSIS_DIR, "zone_forecasts.csv")

    if not os.path.exists(trends_path):
        print("ERROR: zone_trends.csv not found at %s" % trends_path)
        print("Run zone_analysis_forecasting.py first.")
        sys.exit(1)
    if not os.path.exists(forecasts_path):
        print("ERROR: zone_forecasts.csv not found at %s" % forecasts_path)
        print("Run zone_analysis_forecasting.py first.")
        sys.exit(1)

    df_trends = pd.read_csv(trends_path)
    df_linear = pd.read_csv(forecasts_path)

    zones = df_trends["zone"].unique()
    metrics = ["mp_count", "bodies_count"]

    print("=" * 70)
    print("ARIMA FORECASTING")
    print("=" * 70)
    print("Loaded %d trend records across %d zones" % (len(df_trends), len(zones)))
    print("Metrics: %s" % ", ".join(metrics))
    print()

    # ------------------------------------------------------------------
    # 2. Forecast and back-test for each zone / metric
    # ------------------------------------------------------------------
    arima_rows = []       # rows for arima_forecasts.csv
    backtest_rows = []    # rows for forecast_backtest.csv

    for zone in zones:
        zone_data = df_trends[df_trends["zone"] == zone].sort_values("year")
        years = zone_data["year"].values.astype(float)

        print("-" * 70)
        print("Zone: %s  (%d years of data)" % (zone, len(years)))
        print("-" * 70)

        for metric in metrics:
            values = zone_data[metric].values.astype(float)

            # -- Full-sample ARIMA forecast (2026-2030) ------------------
            fc = arima_forecast(values, steps=len(FORECAST_YEARS))
            if fc is None:
                print("  [%s] ARIMA fitting failed -- skipping" % metric)
                continue

            order_str = "(%d,%d,%d)" % fc["order"]
            print("  [%s] Best ARIMA order: %s  AIC=%.1f"
                  % (metric, order_str, fc["aic"]))

            for i, yr in enumerate(FORECAST_YEARS):
                arima_rows.append({
                    "zone": zone,
                    "year": int(yr),
                    "metric": metric,
                    "arima_forecast": round(fc["forecast"][i], 4),
                    "arima_lower_95": round(fc["lower_95"][i], 4),
                    "arima_upper_95": round(fc["upper_95"][i], 4),
                    "arima_order": order_str,
                    "arima_aic": round(fc["aic"], 4),
                })

            # -- Back-test: hold out last HOLDOUT_SIZE years -------------
            if len(values) <= HOLDOUT_SIZE + 5:
                # Not enough data for a meaningful holdout
                print("    Backtest skipped -- too few data points")
                continue

            train_values = values[:-HOLDOUT_SIZE]
            test_values = values[-HOLDOUT_SIZE:]
            train_years = years[:-HOLDOUT_SIZE]
            test_years = years[-HOLDOUT_SIZE:]

            # ARIMA on training set
            fc_bt = arima_forecast(train_values, steps=HOLDOUT_SIZE)
            if fc_bt is not None:
                arima_rmse_val = rmse(test_values, fc_bt["forecast"])
            else:
                arima_rmse_val = np.nan

            # Linear regression on training set
            linear_pred = linear_forecast(train_years, train_values, test_years)
            linear_rmse_val = rmse(test_values, linear_pred)

            # Determine the better model
            if np.isnan(arima_rmse_val):
                better = "linear"
                improvement = 0.0
            elif np.isnan(linear_rmse_val):
                better = "arima"
                improvement = 0.0
            elif arima_rmse_val < linear_rmse_val:
                better = "arima"
                if linear_rmse_val > 0:
                    improvement = (linear_rmse_val - arima_rmse_val) / linear_rmse_val * 100.0
                else:
                    improvement = 0.0
            else:
                better = "linear"
                if arima_rmse_val > 0:
                    improvement = (arima_rmse_val - linear_rmse_val) / arima_rmse_val * 100.0
                else:
                    improvement = 0.0

            backtest_rows.append({
                "zone": zone,
                "metric": metric,
                "linear_rmse": round(linear_rmse_val, 4) if np.isfinite(linear_rmse_val) else np.nan,
                "arima_rmse": round(arima_rmse_val, 4) if np.isfinite(arima_rmse_val) else np.nan,
                "better_model": better,
                "improvement_pct": round(improvement, 2),
            })

            print("    Backtest RMSE -- Linear: %.2f  ARIMA: %.2f  --> %s (%.1f%% improvement)"
                  % (linear_rmse_val, arima_rmse_val, better, improvement))

    # ------------------------------------------------------------------
    # 3. Save outputs
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    if arima_rows:
        df_arima = pd.DataFrame(arima_rows)
        out_arima = os.path.join(ANALYSIS_DIR, "arima_forecasts.csv")
        df_arima.to_csv(out_arima, index=False)
        print("Saved %d ARIMA forecast records to %s" % (len(df_arima), out_arima))
    else:
        print("WARNING: No ARIMA forecasts were generated.")

    if backtest_rows:
        df_bt = pd.DataFrame(backtest_rows)
        out_bt = os.path.join(ANALYSIS_DIR, "forecast_backtest.csv")
        df_bt.to_csv(out_bt, index=False)
        print("Saved %d backtest records to %s" % (len(df_bt), out_bt))
    else:
        print("WARNING: No backtest results were generated.")

    # ------------------------------------------------------------------
    # 4. Print comparison summary table
    # ------------------------------------------------------------------
    if backtest_rows:
        print()
        print("=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)

        df_bt = pd.DataFrame(backtest_rows)

        header = "%-25s %-15s %12s %12s  %-8s %10s" % (
            "Zone", "Metric", "Linear RMSE", "ARIMA RMSE", "Better", "Improv%")
        print(header)
        print("-" * len(header))

        for _, row in df_bt.iterrows():
            lin_str = "%.2f" % row["linear_rmse"] if pd.notna(row["linear_rmse"]) else "N/A"
            ari_str = "%.2f" % row["arima_rmse"] if pd.notna(row["arima_rmse"]) else "N/A"
            print("%-25s %-15s %12s %12s  %-8s %9.1f%%" % (
                row["zone"], row["metric"],
                lin_str, ari_str,
                row["better_model"], row["improvement_pct"]))

        # Overall tally
        n_arima = (df_bt["better_model"] == "arima").sum()
        n_linear = (df_bt["better_model"] == "linear").sum()
        print()
        print("Overall: ARIMA wins %d / %d comparisons, Linear wins %d / %d"
              % (n_arima, len(df_bt), n_linear, len(df_bt)))

    print()
    print("=" * 70)
    print("ARIMA FORECASTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run()
