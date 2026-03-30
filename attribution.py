from __future__ import annotations

import numpy as np
import pandas as pd


def compute_weights(
    shares: pd.DataFrame,
    prices_eur: pd.DataFrame,
    mv_assets: pd.Series,
) -> pd.DataFrame:
    """
    Computes portfolio weights for attribution.

    w_{i,t} = MV_{i,t} / MV_assets_t

    CRITICAL FIX compared to the previous version:
    The denominator was mv_total (assets + cash). This was incorrect:
    - If you have 50% cash, all weights get halved and sum to ~0.5
    - The sum of contributions does not converge to portfolio return
    - A stock rising 10% with true weight 20% contributes 2%,
      but with distorted weight 10% (due to cash) contributes only 1%: clear error

    Now the denominator is mv_assets (only invested assets),
    so weights always sum to 1 on days when you are invested.

    Cash drag (impact of holding cash on performance) is not captured
    in ticker-level attribution, but is visible in portfolio vs benchmark comparison.
    """
    shares_al = shares.reindex(prices_eur.index).ffill(limit=5).fillna(0.0)
    mv_asset_matrix = shares_al * prices_eur

    # Use mv_assets (only assets, excluding cash) as denominator
    w = mv_asset_matrix.div(mv_assets, axis=0)

    # Where mv_assets is zero or NaN (no open positions), weights are NaN
    return w


def contribution_daily(
    shares: pd.DataFrame,
    prices_eur: pd.DataFrame,
    mv_assets: pd.Series,
) -> pd.DataFrame:
    """
    Daily attribution by ticker (log-return method).

    contrib_{i,t} = w_{i,t-1} * ln(1 + r_{i,t})

    Summing log-contributions over time is consistent with compounding:
    exp(sum(contrib_i)) - 1 ≈ total contribution of asset i.

    NOTE: attribution is computed on invested assets only.
    Cash drag (lost return due to holding cash instead of investing)
    is not attributed to any specific ticker.
    """
    r_assets = prices_eur.pct_change()
    lr_assets = np.log1p(r_assets)  # log(1 + r) for consistency in compounding

    w = compute_weights(shares=shares, prices_eur=prices_eur, mv_assets=mv_assets)
    w_prev = w.shift(1)  # yesterday's weights determine today's contribution

    contrib = w_prev * lr_assets
    return contrib


def contribution_total(contrib_daily: pd.DataFrame) -> pd.Series:
    """
    Total contribution per ticker over the entire period.

    Converts sum of log-contributions into percentage return:
    total_i = exp(sum_t contrib_{i,t}) - 1
    """
    s = contrib_daily.sum(skipna=True)
    return np.expm1(s).sort_values(ascending=False)


def contribution_by_year(contrib_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregated attribution by year.

    Returns a DataFrame with:
    - one row per year
    - one column per ticker
    - an additional column `is_partial_year` (True for the current year)

    FIX compared to previous version:
    Added `is_partial_year` column to indicate that the current year
    has fewer than 252 trading days: the return shown is YTD, not annualized.

    Without this flag, current-year numbers may appear artificially low
    and could be incorrectly compared with full years.
    """
    df = contrib_daily.copy()
    df.index = pd.to_datetime(df.index)

    current_year = pd.Timestamp.today().year

    summed = df.groupby(df.index.year).sum(min_count=1)
    result = np.expm1(summed)

    # Count actual trading days per year
    trading_days_per_year = df.groupby(df.index.year).count().max(axis=1)
    result["is_partial_year"] = trading_days_per_year < 200  # ~full-year threshold
    result.loc[current_year, "is_partial_year"] = True  # current year always partial

    return result


def reconcile_lastday_table(
    shares: pd.DataFrame,
    prices_eur: pd.DataFrame,
    cash: pd.Series,
    ticker_ccy: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Reconciliation table for the latest available day.

    Shows for each ticker: shares held, latest EUR price, total value.
    Adds summary rows for cash, total assets, and total portfolio.
    """
    ticker_ccy = ticker_ccy or {}

    shares_al = shares.reindex(prices_eur.index).ffill(limit=5).fillna(0.0)
    last_day = prices_eur.index.max()

    shares_last = shares_al.loc[last_day]
    prices_last = prices_eur.loc[last_day]
    values_last = shares_last * prices_last

    out = pd.DataFrame(
        {
            "ticker": prices_eur.columns,
            "shares": shares_last.reindex(prices_eur.columns).values,
            "last_price_eur": prices_last.reindex(prices_eur.columns).values,
            "value_eur": values_last.reindex(prices_eur.columns).values,
            "ccy_hint": [ticker_ccy.get(t, "EUR") for t in prices_eur.columns],
        }
    ).sort_values("value_eur", ascending=False)

    cash_last = float(cash.reindex(prices_eur.index).ffill(limit=5).iloc[-1])
    total_assets = float(out["value_eur"].sum())
    total_portfolio = total_assets + cash_last

    summary = pd.DataFrame(
        [
            {
                "ticker": "__CASH__",
                "shares": None,
                "last_price_eur": None,
                "value_eur": cash_last,
                "ccy_hint": "EUR",
            },
            {
                "ticker": "__TOTAL_ASSETS__",
                "shares": None,
                "last_price_eur": None,
                "value_eur": total_assets,
                "ccy_hint": "EUR",
            },
            {
                "ticker": "__TOTAL_PORTFOLIO__",
                "shares": None,
                "last_price_eur": None,
                "value_eur": total_portfolio,
                "ccy_hint": "EUR",
            },
        ]
    )

    return pd.concat([out, summary], ignore_index=True)