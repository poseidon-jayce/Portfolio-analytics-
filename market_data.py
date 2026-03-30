from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf


@dataclass
class PriceBundle:
    close_native: pd.DataFrame   
    close_eur: pd.DataFrame      
    fx_to_eur: pd.DataFrame      


def _normalize_index_to_date(df: pd.DataFrame) -> pd.DataFrame:
   
    out = df.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out


def download_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    
    tickers = sorted(
        {t for t in tickers if isinstance(t, str) and t.strip() and t != "nan"}
    )
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].copy()
        close.columns = tickers

    close = _normalize_index_to_date(close).sort_index()
    return close


def cache_read_parquet(path: Path) -> pd.DataFrame | None:
   
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df
    return None


def cache_write_parquet(df: pd.DataFrame, path: Path) -> None:
   
    df.to_parquet(path)


def build_trading_calendar(
    reference_series: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    
    idx = pd.to_datetime(reference_series.index).tz_localize(None).normalize()
    idx = idx[(idx >= start) & (idx <= end)]
    return pd.DatetimeIndex(idx).unique().sort_values()


def align_to_calendar(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """
    Aligns a DataFrame to the trading calendar with limited forward fill.

    FIX compared to the previous version:
    - Added parameter `ffill_limit` (default 5 days).
      Previously the forward fill was unlimited: if a ticker stopped trading
      (delisting, long suspension), the last known price would be propagated
      indefinitely, distorting P&L and attribution calculations.
      With limit=5, after 5 days without price the value becomes NaN,
      explicitly signaling the issue instead of hiding it.

    5 days is used to cover holidays and international market gaps.
    """
    if df.empty:
        return pd.DataFrame(index=calendar)
    return df.reindex(calendar).ffill(limit=ffill_limit)


def build_fx_to_eur(calendar: pd.DatetimeIndex, ccy_set: set[str]) -> pd.DataFrame:
    """
    Downloads and builds the FX rates table to EUR.

    Yahoo Finance provides:
      EURUSD=X → USD per 1 EUR → we invert to get EUR per 1 USD
      EURGBP=X → GBP per 1 EUR → we invert to get EUR per 1 GBP
    """
    fx_tickers: list[str] = []
    if "USD" in ccy_set:
        fx_tickers.append("EURUSD=X")
    if "GBP" in ccy_set:
        fx_tickers.append("EURGBP=X")

    if not fx_tickers:
        return pd.DataFrame(index=calendar)

    fx = download_close(
        fx_tickers,
        start=str(calendar.min().date()),
        end=str((calendar.max() + pd.Timedelta(days=1)).date()),
    )
    fx = align_to_calendar(fx, calendar)

    out = pd.DataFrame(index=calendar)

    if "EURUSD=X" in fx.columns:
        out["USD"] = 1.0 / fx["EURUSD=X"]
    if "EURGBP=X" in fx.columns:
        out["GBP"] = 1.0 / fx["EURGBP=X"]

    return out


def make_price_bundle_eur(
    close_native: pd.DataFrame,
    ticker_currency: dict[str, str],
    fx_cache_path: Path,
) -> PriceBundle:
    """
    Converts all prices to EUR using FX rates.

    ticker_currency: dictionary ticker → quotation currency.
      Supported values: "EUR", "USD", "GBP", "GBp" (pence sterling).
      "GBp" is converted to GBP by dividing the price by 100,
      then converted to EUR like normal GBP.

    The currency mapping must come from isin_ticker.csv (column `currency`),
    NOT hardcoded in the code — see main.py for how to build it.
    """
    calendar = close_native.index
    ticker_ccy = dict(ticker_currency)
    close_eur = close_native.copy()

    # Pence → GBP at price level
    for tkr, ccy in list(ticker_ccy.items()):
        if ccy == "GBp" and tkr in close_eur.columns:
            close_eur[tkr] = close_eur[tkr] / 100.0
            ticker_ccy[tkr] = "GBP"

    ccy_set = set(ticker_ccy.values())

    fx_cached = cache_read_parquet(fx_cache_path)
    if fx_cached is None:
        fx_to_eur = build_fx_to_eur(calendar, ccy_set)
        cache_write_parquet(fx_to_eur, fx_cache_path)
    else:
        fx_to_eur = fx_cached.reindex(calendar).ffill(limit=5)

    # Convert each non-EUR ticker by multiplying with FX rate
    for tkr in close_eur.columns:
        ccy = ticker_ccy.get(tkr, "EUR")
        if ccy == "EUR":
            continue
        if ccy not in fx_to_eur.columns:
            raise ValueError(
                f"Missing FX rate for currency '{ccy}' (ticker: {tkr}). "
                f"Check the 'currency' column in isin_ticker.csv."
            )
        close_eur[tkr] = close_eur[tkr] * fx_to_eur[ccy]

    return PriceBundle(
        close_native=close_native, close_eur=close_eur, fx_to_eur=fx_to_eur
    )