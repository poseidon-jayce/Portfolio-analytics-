from __future__ import annotations

import json

import pandas as pd

from .attribution import (
    contribution_by_year,
    contribution_daily,
    contribution_total,
    reconcile_lastday_table,
)
from .config import RISK_FREE_ANNUAL, get_paths
from .io_fineco import load_fineco_movements_xlsx
from .market_data import (
    align_to_calendar,
    build_trading_calendar,
    cache_read_parquet,
    cache_write_parquet,
    download_close,
    make_price_bundle_eur,
)
from .metrics import (
    annualized_return,
    annualized_vol,
    beta_alpha,
    calmar_ratio,
    drawdown,
    equity_curve,
    information_ratio,
    max_drawdown,
    rolling_metrics_table,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
    xirr,
)
from .portfolio_engine import build_portfolio_series
from .viz import (
    plot_contribution_bar,
    plot_drawdown,
    plot_equity_curves,
    plot_rolling_beta,
    plot_rolling_sharpe,
    plot_rolling_vol,
)

BENCHMARK_TICKER = "^SP500TR"
BENCHMARK_CCY = "USD"
ROLLING_WINDOW = 180  # trading days in the rolling window (~9 months)


def _build_ticker_currency_map(
    portfolio_tickers: list[str], isin_map_df: pd.DataFrame | None = None
) -> dict[str, str]:
    """
    Builds the mapping ticker → quotation currency.

    Reads the `currency` column from isin_ticker.csv if present.
    Otherwise uses EUR as default with overrides for known USD tickers.

    TODO: add `currency` column to isin_ticker.csv to eliminate
    the hardcoded overrides below.
    """
    ticker_ccy: dict[str, str] = {}

    if isin_map_df is not None and "currency" in isin_map_df.columns:
        for _, row in isin_map_df.iterrows():
            ticker = str(row.get("ticker", "")).strip()
            ccy = str(row.get("currency", "EUR")).strip()
            if ticker:
                ticker_ccy[ticker] = ccy
    else:
        for t in portfolio_tickers:
            ticker_ccy[t] = "EUR"

        # Override for USD tickers in the portfolio
        # TODO: move to CSV when you add the currency column
        usd_stocks = {"AAPL", "KO", "META", "KHC", "WFC", "BABA"}
        for t in usd_stocks:
            if t in ticker_ccy:
                ticker_ccy[t] = "USD"

    ticker_ccy[BENCHMARK_TICKER] = BENCHMARK_CCY
    return ticker_ccy


def run() -> None:
    p = get_paths()

    # -------------------------------------------------------------------------
    # 1. INPUT: read the Fineco ledger and external cash flows
    # -------------------------------------------------------------------------
    isin_map_df = pd.read_csv(p.data / "isin_ticker.csv")
    isin_map_df["isin"] = isin_map_df["isin"].astype(str).str.strip()
    isin_map_df["ticker"] = isin_map_df["ticker"].astype(str).str.strip()
    isin_to_ticker = dict(zip(isin_map_df["isin"], isin_map_df["ticker"], strict=False))

    ledger = load_fineco_movements_xlsx(
        str(p.data / "Lista_Titoli.xlsx"), isin_to_ticker
    )

    ext = pd.read_csv(p.data / "external_cashflows.csv")
    ext["date"] = pd.to_datetime(ext["date"]).dt.normalize()
    ext["amount_eur"] = pd.to_numeric(ext["amount_eur"], errors="coerce").fillna(0.0)

    # -------------------------------------------------------------------------
    # 2. START/END DATES
    # The analysis starts from the first trade (not the first deposit).
    # Deposits before the first trade are carried forward to the first
    # available trading day in step 5.
    # -------------------------------------------------------------------------
    portfolio_tickers = sorted(set(ledger["ticker"].dropna().astype(str)))
    tickers_all = sorted(set(portfolio_tickers + [BENCHMARK_TICKER]))

    first_trade = pd.to_datetime(ledger["date"]).min()
    start = str(first_trade.date())
    end = str((pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date())

    # -------------------------------------------------------------------------
    # 3. PRICES: cache + incremental download
    # -------------------------------------------------------------------------
    prices_cache = p.cache / "prices_close.parquet"
    close_cached = cache_read_parquet(prices_cache)

    if close_cached is None:
        close_native = download_close(tickers_all, start=start, end=end)
        cache_write_parquet(close_native, prices_cache)
    else:
        close_native = close_cached
        missing = [t for t in tickers_all if t not in close_native.columns]
        if missing:
            new = download_close(missing, start=start, end=end)
            close_native = close_native.join(new, how="outer").sort_index()
            cache_write_parquet(close_native, prices_cache)

    close_native = close_native.sort_index()

    # -------------------------------------------------------------------------
    # 4. TRADING CALENDAR (from S&P 500 benchmark)
    # -------------------------------------------------------------------------
    bench_native = close_native[BENCHMARK_TICKER].dropna()
    cal = build_trading_calendar(
        bench_native, pd.to_datetime(start), pd.to_datetime(end)
    )
    close_native = align_to_calendar(close_native, cal)

    # -------------------------------------------------------------------------
    # 5. EXTERNAL FLOWS aligned to calendar
    # Deposits before the first calendar date are summed
    # and assigned to the first available day to avoid losing them.
    # -------------------------------------------------------------------------
    ext_grouped = ext.groupby("date")["amount_eur"].sum()

    pre_cal = ext_grouped[ext_grouped.index < cal.min()].sum()
    external_flows = ext_grouped.reindex(cal).fillna(0.0)
    if pre_cal != 0:
        external_flows.iloc[0] += pre_cal

    external_flows.name = "external_flows_eur"

    # -------------------------------------------------------------------------
    # 6. CURRENCY MAP (from CSV, with hardcoded fallback)
    # -------------------------------------------------------------------------
    ticker_ccy = _build_ticker_currency_map(portfolio_tickers, isin_map_df)

    # -------------------------------------------------------------------------
    # 7. FX CONVERSION → EUR
    # -------------------------------------------------------------------------
    fx_cache = p.cache / "fx_to_eur.parquet"
    bundle = make_price_bundle_eur(close_native, ticker_ccy, fx_cache)

    close_eur = bundle.close_eur
    bench_eur = close_eur[BENCHMARK_TICKER]
    port_close_eur = close_eur.drop(columns=[BENCHMARK_TICKER])

    # -------------------------------------------------------------------------
    # 8. PORTFOLIO ENGINE
    # -------------------------------------------------------------------------
    ps = build_portfolio_series(
        ledger=ledger,
        close_eur=port_close_eur,
        calendar=cal,
        external_flows=external_flows,
    )

    port_ret = ps.r_twr_daily
    bench_ret = bench_eur.pct_change()

    port_eq = equity_curve(port_ret, base=1.0)
    bench_eq = equity_curve(bench_ret, base=1.0)

    # -------------------------------------------------------------------------
    # 9. RECONCILIATION TABLE
    # -------------------------------------------------------------------------
    rec = reconcile_lastday_table(
        shares=ps.shares,
        prices_eur=port_close_eur,
        cash=ps.cash,
        ticker_ccy=ticker_ccy,
    )
    rec.to_csv(p.outputs / "reconcile_lastday_by_ticker.csv", index=False)

    # -------------------------------------------------------------------------
    # 10. PERFORMANCE METRICS (full period)
    # -------------------------------------------------------------------------
    metrics = {
        "start_date": str(cal.min().date()),
        "end_date": str(cal.max().date()),
        "mv_last_eur": float(ps.mv_total.dropna().iloc[-1]),
        # Portfolio
        "port_ann_return": annualized_return(port_ret),
        "port_ann_vol": annualized_vol(port_ret),
        "port_sharpe": sharpe_ratio(port_ret, risk_free_annual=RISK_FREE_ANNUAL),
        "port_sortino": sortino_ratio(port_ret, risk_free_annual=RISK_FREE_ANNUAL),
        "port_calmar": calmar_ratio(port_ret),
        "port_max_drawdown": max_drawdown(port_eq),
        # Benchmark
        "bench_ann_return": annualized_return(bench_ret),
        "bench_ann_vol": annualized_vol(bench_ret),
        "bench_sharpe": sharpe_ratio(bench_ret, risk_free_annual=RISK_FREE_ANNUAL),
        "bench_sortino": sortino_ratio(bench_ret, risk_free_annual=RISK_FREE_ANNUAL),
        "bench_calmar": calmar_ratio(bench_ret),
        "bench_max_drawdown": max_drawdown(bench_eq),
        # Active metrics
        "tracking_error": tracking_error(port_ret, bench_ret),
        "information_ratio": information_ratio(port_ret, bench_ret),
    }

    # CAPM
    capm = beta_alpha(
        pd.to_numeric(port_ret, errors="coerce"),
        pd.to_numeric(bench_ret, errors="coerce"),
    )
    metrics.update({f"capm_{k}": v for k, v in capm.items()})

    # XIRR
    cfs = [
        (pd.Timestamp(d), -float(v)) for d, v in external_flows.items() if abs(v) > 1e-9
    ]
    cfs.sort(key=lambda x: x[0])
    cfs.append((pd.Timestamp(cal.max()), float(ps.mv_total.dropna().iloc[-1])))
    metrics["xirr_mwr"] = xirr(cfs)

    (p.outputs / "report_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # 11. DAILY REPORT
    # -------------------------------------------------------------------------
    daily = pd.DataFrame(
        {
            "mv_total_eur": ps.mv_total,
            "mv_assets_eur": ps.mv_assets,
            "cash_eur": ps.cash,
            "external_flows_eur": ps.external_flows,
            "port_ret": port_ret,
            "bench_ret": bench_ret,
            "port_eq": port_eq,
            "bench_eq": bench_eq,
        },
        index=cal,
    )
    daily.index.name = "Date"
    daily.to_csv(p.outputs / "report_daily.csv", index=True)

    # -------------------------------------------------------------------------
    # 12. ROLLING METRICS
    # Computed over a rolling window of ROLLING_WINDOW days (default 180).
    # Helps analyze how risk/return profile evolves over time
    # instead of a single full-period number.
    # -------------------------------------------------------------------------
    rolling_df = rolling_metrics_table(
        port_returns=port_ret,
        bench_returns=bench_ret,
        window=ROLLING_WINDOW,
        risk_free_annual=RISK_FREE_ANNUAL,
    )
    rolling_df.to_csv(p.outputs / "report_rolling.csv", index=True)

    # -------------------------------------------------------------------------
    # 13. ATTRIBUTION / CONTRIBUTION BY TICKER
    # -------------------------------------------------------------------------
    contrib_d = contribution_daily(
        shares=ps.shares,
        prices_eur=port_close_eur,
        mv_assets=ps.mv_assets,
    )
    contrib_d.to_csv(p.outputs / "contrib_daily.csv", index=True)

    contrib_tot = contribution_total(contrib_d)
    contrib_tot.to_csv(p.outputs / "contrib_total.csv")

    contrib_y = contribution_by_year(contrib_d)
    contrib_y.to_csv(p.outputs / "contrib_by_year.csv")

    # -------------------------------------------------------------------------
    # 14. CHARTS
    # -------------------------------------------------------------------------
    plot_equity_curves(port_eq, bench_eq, p.charts / "equity_port_vs_sp500tr.png")
    plot_drawdown(drawdown(port_eq), p.charts / "drawdown_portfolio.png")
    plot_contribution_bar(
        contrib_tot,
        "Total Return Contribution (top 10)",
        p.charts / "contrib_total_top10.png",
        top_n=10,
    )

    # Rolling charts
    plot_rolling_sharpe(
        rolling_df, p.charts / "rolling_sharpe.png", window=ROLLING_WINDOW
    )
    plot_rolling_vol(rolling_df, p.charts / "rolling_vol.png", window=ROLLING_WINDOW)
    plot_rolling_beta(rolling_df, p.charts / "rolling_beta.png", window=ROLLING_WINDOW)

    print("DONE")
    print(f"Outputs: {p.outputs}")
    print(f"Metrics: {p.outputs / 'report_metrics.json'}")
    print(f"Rolling:  {p.outputs / 'report_rolling.csv'}")
    print(f"Charts:  {p.charts}")


if __name__ == "__main__":
    run()