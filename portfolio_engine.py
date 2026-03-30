from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioSeries:
    calendar: pd.DatetimeIndex
    shares: pd.DataFrame  # shares held for each ticker, each day
    cash: pd.Series  # total cash (EUR)
    mv_assets: pd.Series  # market value of assets (EUR)
    mv_total: pd.Series  # total market value = assets + cash (EUR)
    external_flows: pd.Series  # external deposits/withdrawals (EUR)
    r_twr_daily: pd.Series  # daily TWR return


def build_daily_positions(
    ledger: pd.DataFrame, calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Builds the daily positions table (shares held per ticker).
    Each cell = number of shares held at end of day.
    """
    trades = ledger[ledger["shares_delta"].ne(0)].copy()
    if trades.empty:
        return pd.DataFrame(index=calendar)

    piv = trades.pivot_table(
        index="date", columns="ticker", values="shares_delta", aggfunc="sum"
    ).fillna(0.0)

    piv = piv.reindex(calendar).fillna(0.0)
    return piv.cumsum()


def build_cash_total(
    ledger: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    external_flows: pd.Series,
) -> pd.Series:
    """
    Computes the daily cash balance of the portfolio.

    Internal flows = cash generated/used by trades and dividends (from ledger).
    External flows = deposits and withdrawals (from external_cashflows.csv).
    The cumulative sum gives the available cash each day.

    NOTE: INITIAL_CASH removed. Initial balance must be added as the first
    record in external_cashflows.csv with the deposit date.
    """
    internal = (
        ledger.groupby("date")["cash_flow_eur"].sum().reindex(calendar).fillna(0.0)
    )
    external = external_flows.reindex(calendar).fillna(0.0)
    cash = (internal + external).cumsum()
    cash.name = "cash_eur"
    return cash


def compute_market_values(
    shares: pd.DataFrame, close_eur: pd.DataFrame, cash: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """
    Computes the market value of assets and total portfolio.

    FIX compared to the previous version:
    - ffill with limit=5: if a price is missing for more than 5 consecutive days,
      the value becomes NaN instead of using stale prices.
      5 days = tolerance for holidays and short trading suspensions.
      Beyond that threshold, the data likely indicates delisting or error.
    """
    # Forward fill with limit: max 5 days without price before flagging NaN
    shares_al = shares.reindex(close_eur.index).ffill(limit=5).fillna(0.0)
    prices_al = close_eur.reindex(close_eur.index).ffill(limit=5)

    # Days where positions exist but price is missing → invalid
    mask_missing_price = (shares_al > 0) & prices_al.isna()
    invalid_days = mask_missing_price.any(axis=1)

    mv_assets = (shares_al * prices_al).sum(axis=1)
    mv_assets[invalid_days] = pd.NA
    mv_assets.name = "mv_assets_eur"

    cash_al = cash.reindex(close_eur.index).ffill(limit=5).fillna(0.0)
    mv_total = mv_assets + cash_al
    mv_total[invalid_days] = pd.NA
    mv_total.name = "mv_total_eur"

    return mv_assets, mv_total


def compute_twr_returns(mv_total: pd.Series, external_flows: pd.Series) -> pd.Series:
    """
    Computes daily TWR (Time-Weighted Return).

    Formula: r_t = (MV_t - CF_t) / MV_{t-1} - 1

    CASH FLOW TIMING ASSUMPTION:
    External flows (deposits/withdrawals) are assumed to arrive at END of day,
    so they are subtracted from the numerator (MV_t).
    If in reality a deposit arrives at the start of the day, the denominator
    should be MV_{t-1} + CF_t — but this requires intraday timestamps,
    which are typically unavailable in brokerage statements.
    The resulting approximation error is negligible for small flows relative to NAV.
    """
    mv = mv_total.astype(float)
    cf = external_flows.reindex(mv.index).fillna(0.0).astype(float)
    prev = mv.shift(1)

    # Return can be computed only where both prev and mv are valid
    valid = prev.notna() & mv.notna()

    r = pd.Series(np.nan, index=mv.index, name="r_twr_daily", dtype=float)
    r[valid] = (mv[valid] - cf[valid]) / prev[valid] - 1.0

    return r


def build_portfolio_series(
    ledger: pd.DataFrame,
    close_eur: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    external_flows: pd.Series,
) -> PortfolioSeries:
    """
    Main entry point: builds all portfolio time series.

    NOTE: the initial_cash parameter has been removed.
    Add the initial deposit in external_cashflows.csv.
    """
    shares = build_daily_positions(ledger, calendar)
    cash = build_cash_total(ledger, calendar, external_flows=external_flows)

    close_eur = close_eur.reindex(calendar).ffill(limit=5)
    mv_assets, mv_total = compute_market_values(shares, close_eur, cash)
    r_twr = compute_twr_returns(mv_total, external_flows)

    return PortfolioSeries(
        calendar=calendar,
        shares=shares.reindex(calendar).ffill(limit=5).fillna(0.0),
        cash=cash.reindex(calendar).ffill(limit=5).fillna(0.0),
        mv_assets=mv_assets,
        mv_total=mv_total,
        external_flows=external_flows.reindex(calendar).fillna(0.0),
        r_twr_daily=r_twr,
    )