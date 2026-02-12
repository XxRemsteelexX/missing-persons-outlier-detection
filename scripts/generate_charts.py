#!/usr/bin/env python3
"""Generate README chart images for the Geospatial Crime Analysis project."""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(PROJECT_DIR, "data", "analysis")
CHARTS_DIR = os.path.join(PROJECT_DIR, "docs", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 12,
}
plt.rcParams.update(STYLE)


def chart_alert_distribution():
    """Alert distribution -- focused on flagged counties only, with full-context inset."""
    df = pd.read_csv(os.path.join(ANALYSIS_DIR, "county_decade_outliers.csv"))
    counts = df['alert_level'].value_counts()
    total = len(df)
    red = counts.get('RED', 0)
    orange = counts.get('ORANGE', 0)
    green = counts.get('GREEN', 0)
    flagged = red + orange

    fig, (ax_context, ax_detail) = plt.subplots(1, 2, figsize=(14, 5),
                                                 gridspec_kw={'width_ratios': [1, 2]})

    # Left panel: stacked percentage bar showing how rare flagged counties are
    ax_context.barh(['All\nCounty-Decades'], [green], color='#2ecc71', edgecolor='white', height=0.5,
                    label='GREEN (%d)' % green)
    ax_context.barh(['All\nCounty-Decades'], [flagged], left=[green], color='#e74c3c', edgecolor='white',
                    height=0.5, label='Flagged (%d)' % flagged)
    ax_context.set_xlim(0, total)
    ax_context.set_xlabel('County-Decade Observations')
    ax_context.set_title('Full Distribution (n = {:,})'.format(total), fontsize=13, fontweight='bold')
    ax_context.legend(loc='center right', fontsize=10)
    # Annotate the flagged sliver -- position below the bar to avoid title overlap
    ax_context.annotate('{:,} flagged ({:.2f}%)'.format(flagged, 100 * flagged / total),
                        xy=(green + flagged / 2, 0), xytext=(total * 0.7, -0.3),
                        fontsize=11, fontweight='bold', color='#c0392b',
                        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5),
                        ha='center')
    ax_context.set_ylim(-0.5, 0.6)

    # Right panel: zoomed bar chart of just RED and ORANGE
    categories = ['ORANGE', 'RED']
    values = [orange, red]
    colors = ['#e67e22', '#e74c3c']
    bars = ax_detail.barh(categories, values, color=colors, edgecolor='white', height=0.5)
    for bar, val in zip(bars, values):
        ax_detail.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                       str(val), va='center', fontsize=14, fontweight='bold')
    ax_detail.set_xlabel('Number of County-Decades')
    ax_detail.set_title('Flagged County-Decades by Alert Level', fontsize=13, fontweight='bold')
    ax_detail.set_xlim(0, max(values) * 1.3)

    # Add threshold descriptions next to bar labels
    ax_detail.text(orange * 0.5, 0, 'z > 2 either metric', ha='center', va='center',
                   fontsize=10, color='white', fontweight='bold')
    ax_detail.text(red + max(values) * 0.12, 1, 'z > 3 both metrics', ha='left', va='center',
                   fontsize=10, color='#e74c3c', fontweight='bold')

    fig.suptitle('Statistical Alert Classification Results', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "alert_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: %s" % path)


def chart_ml_classification():
    """ML classification -- exclude Normal, focus on the anomaly tiers."""
    df = pd.read_csv(os.path.join(ANALYSIS_DIR, "ml_anomaly_scores.csv"))
    total = len(df)
    counts = df['ml_classification'].value_counts()
    normal = counts.get('Normal', 0)
    suspicious = counts.get('Suspicious', 0)
    anomalous = counts.get('Anomalous', 0)
    extreme = counts.get('Extreme Anomaly', 0)
    flagged = suspicious + anomalous + extreme

    fig, (ax_context, ax_detail) = plt.subplots(1, 2, figsize=(14, 5),
                                                 gridspec_kw={'width_ratios': [1, 2]})

    # Left panel: donut chart showing Normal vs Flagged split
    sizes = [normal, flagged]
    donut_colors = ['#2ecc71', '#e74c3c']
    wedges, texts = ax_context.pie(sizes, colors=donut_colors, startangle=90,
                                    wedgeprops=dict(width=0.4, edgecolor='white'))
    ax_context.text(0, 0, '{:.0f}%\nflagged'.format(100 * flagged / total),
                    ha='center', va='center', fontsize=14, fontweight='bold', color='#c0392b')
    ax_context.set_title('Normal vs Flagged\n(n = {:,})'.format(total), fontsize=13, fontweight='bold')
    legend_labels = ['Normal ({:,})'.format(normal), 'Flagged ({:,})'.format(flagged)]
    ax_context.legend(wedges, legend_labels, loc='lower center', fontsize=10)

    # Right panel: horizontal bar chart of the 3 anomaly tiers
    categories = ['Extreme Anomaly', 'Anomalous', 'Suspicious']
    values = [extreme, anomalous, suspicious]
    colors = ['#c0392b', '#e67e22', '#f39c12']
    bars = ax_detail.barh(categories, values, color=colors, edgecolor='white', height=0.5)
    for bar, val in zip(bars, values):
        pct = 100 * val / total
        ax_detail.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                       '{:,}  ({:.1f}%)'.format(val, pct), va='center', fontsize=12, fontweight='bold')
    ax_detail.set_xlabel('Number of County-Decades')
    ax_detail.set_title('ML Ensemble Anomaly Tiers\n(Isolation Forest + LOF)', fontsize=13, fontweight='bold')
    ax_detail.set_xlim(0, max(values) * 1.4)

    fig.suptitle('Machine Learning Classification Results', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "ml_classification.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: %s" % path)


def chart_arima_vs_linear():
    """ARIMA vs Linear -- dumbbell chart, split by metric, full zone names."""
    df = pd.read_csv(os.path.join(ANALYSIS_DIR, "forecast_backtest.csv"))

    fig, (ax_mp, ax_bod) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for ax, metric, title in [(ax_mp, 'mp_count', 'Missing Persons'),
                               (ax_bod, 'bodies_count', 'Unidentified Bodies')]:
        sub = df[df['metric'] == metric].sort_values('linear_rmse', ascending=True)
        zones = sub['zone'].tolist()
        linear = sub['linear_rmse'].tolist()
        arima = sub['arima_rmse'].tolist()
        winners = sub['better_model'].tolist()
        improvements = sub['improvement_pct'].tolist()

        y_pos = range(len(zones))

        # Draw connecting lines (dumbbells)
        for i, (z, l, a) in enumerate(zip(zones, linear, arima)):
            ax.plot([l, a], [i, i], color='#bdc3c7', linewidth=2, zorder=1)

        # Plot dots
        ax.scatter(linear, y_pos, color='#7f8c8d', s=100, zorder=2, label='Linear Regression')
        ax.scatter(arima, y_pos, color='#2980b9', s=100, zorder=2, label='ARIMA')

        # Annotate winner and improvement
        for i, (w, imp, l, a) in enumerate(zip(winners, improvements, linear, arima)):
            best_val = a if w == 'arima' else l
            color = '#2980b9' if w == 'arima' else '#7f8c8d'
            label = 'ARIMA' if w == 'arima' else 'Linear'
            x_offset = max(l, a)
            ax.text(x_offset + 1, i, ' {} wins (-{:.0f}%)'.format(label, imp),
                    va='center', fontsize=9, fontweight='bold', color=color)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(zones, fontsize=11)
        ax.set_xlabel('RMSE (lower is better)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        # Add some padding on right for annotations
        xmax = max(max(linear), max(arima))
        ax.set_xlim(0, xmax * 1.6)

    fig.suptitle('ARIMA vs Linear Regression: 5-Year Holdout Backtesting',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "arima_vs_linear.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: %s" % path)


if __name__ == "__main__":
    print("Generating charts...")
    chart_alert_distribution()
    chart_ml_classification()
    chart_arima_vs_linear()
    print("Done.")
