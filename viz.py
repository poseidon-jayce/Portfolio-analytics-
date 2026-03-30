from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt  # pyright: ignore
import matplotlib.ticker as mticker
import pandas as pd


# =============================================================================
# GLOBAL STYLE
# Common settings for all charts for visual consistency.
# =============================================================================
COLORS = {
    "portfolio": "#2563EB",  # blue
    "benchmark": "#DC2626",  # red
    "positive": "#16A34A",  # green
    "negative": "#DC2626",  # red
    "neutral": "#6B7280",  # gray
    "beta": "#9333EA",  # purple
}


def _apply_common_style(ax: plt.Axes, title: str) -> None:
    """Applies common style to all charts: grid, title, background."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# STANDARD CHARTS
# =============================================================================


def plot_equity_curves(
    port_eq: pd.Series,
    bench_eq: pd.Series,
    outpath: Path,
) -> None:
    """
    Equity curve chart: portfolio vs S&P 500 Total Return.
    Both series are indexed to 1.0 on the first day.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    port_clean = port_eq.dropna()
    bench_clean = bench_eq.dropna()

    ax.plot(
        port_clean.index,
        port_clean.values,
        label="Portfolio",
        color=COLORS["portfolio"],
        linewidth=2,
    )
    ax.plot(
        bench_clean.index,
        bench_clean.values,
        label="S&P 500 TR",
        color=COLORS["benchmark"],
        linewidth=2,
        linestyle="--",
    )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.legend(fontsize=11)
    _apply_common_style(ax, "Equity Curve — Portfolio vs S&P 500 TR")
    ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_drawdown(dd: pd.Series, outpath: Path) -> None:
    """
    Portfolio drawdown chart over time.
    The area below zero is shaded red to highlight loss periods.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    dd_clean = dd.dropna()
    ax.fill_between(
        dd_clean.index, dd_clean.values, 0, color=COLORS["negative"], alpha=0.3
    )
    ax.plot(dd_clean.index, dd_clean.values, color=COLORS["negative"], linewidth=1.5)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    _apply_common_style(ax, "Portfolio Drawdown")
    ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_contribution_bar(
    contrib: pd.Series,
    title: str,
    outpath: Path,
    top_n: int = 10,
) -> None:
    """
    Bar chart of total return contribution by ticker.
    Positive bars are green, negative bars are red.
    Shows the top_n securities by absolute contribution.
    """
    s = contrib.dropna().sort_values(ascending=False).head(top_n)
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in s]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(s.index, s.values, color=colors, edgecolor="white", linewidth=0.5)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.tick_params(axis="x", rotation=30)
    _apply_common_style(ax, title)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# =============================================================================
# ROLLING CHARTS
# =============================================================================


def plot_rolling_sharpe(
    rolling_df: pd.DataFrame,
    outpath: Path,
    window: int = 180,
) -> None:
    """
    Rolling Sharpe Ratio chart for the portfolio.

    Shows how the risk/return profile has changed over time.
    The dashed line at 0 separates positive and negative periods.
    The line at 1.0 indicates a generally good level.
    """
    col = f"rolling_sharpe_{window}d"
    if col not in rolling_df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    series = rolling_df[col].dropna()
    ax.plot(series.index, series.values, color=COLORS["portfolio"], linewidth=1.8)

    # Reference lines
    ax.axhline(0, color=COLORS["negative"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(1.0, color=COLORS["positive"], linestyle="--", linewidth=1.0, alpha=0.7)

    # Shade area: green above 0, red below 0
    ax.fill_between(
        series.index,
        series.values,
        0,
        where=(series.values >= 0),
        color=COLORS["positive"],
        alpha=0.15,
    )
    ax.fill_between(
        series.index,
        series.values,
        0,
        where=(series.values < 0),
        color=COLORS["negative"],
        alpha=0.15,
    )

    _apply_common_style(ax, f"Rolling Sharpe Ratio ({window}-day window)")
    ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_rolling_vol(
    rolling_df: pd.DataFrame,
    outpath: Path,
    window: int = 180,
) -> None:
    """
    Rolling volatility chart: portfolio vs benchmark.

    Shows how volatile the portfolio has been compared to the S&P 500
    over the last `window` days. When the blue line is below the red one,
    the portfolio is less volatile than the benchmark.

    FIX: uses ax.plot() instead of series.plot(ax=ax) to avoid the warning
    "This axis already has a converter set" caused by overlapping handling
    of datetime axes between pandas and matplotlib.
    """
    col_port = f"rolling_vol_port_{window}d"
    col_bench = f"rolling_vol_bench_{window}d"

    if col_port not in rolling_df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    port_clean = rolling_df[col_port].dropna()
    ax.plot(
        port_clean.index,
        port_clean.values,
        label="Portfolio",
        color=COLORS["portfolio"],
        linewidth=2,
    )

    if col_bench in rolling_df.columns:
        bench_clean = rolling_df[col_bench].dropna()
        ax.plot(
            bench_clean.index,
            bench_clean.values,
            label="S&P 500 TR",
            color=COLORS["benchmark"],
            linewidth=2,
            linestyle="--",
        )

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=11)
    _apply_common_style(ax, f"Rolling Volatility ({window}-day window)")
    ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_rolling_beta(
    rolling_df: pd.DataFrame,
    outpath: Path,
    window: int = 180,
) -> None:
    """
    Rolling beta chart of the portfolio vs S&P 500.

    Shows how market sensitivity has changed over time.
    The dashed line at 1.0 represents market beta.
    Beta < 1 = defensive portfolio, beta > 1 = aggressive portfolio.
    """
    col = f"rolling_beta_{window}d"
    if col not in rolling_df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    series = rolling_df[col].dropna()
    ax.plot(series.index, series.values, color=COLORS["beta"], linewidth=1.8)

    # Reference lines
    ax.axhline(
        1.0,
        color=COLORS["neutral"],
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="Beta = 1 (market)",
    )
    ax.axhline(0.0, color=COLORS["neutral"], linestyle=":", linewidth=0.8, alpha=0.5)

    ax.legend(fontsize=10)
    _apply_common_style(ax, f"Rolling Beta vs S&P 500 ({window}-day window)")
    ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)