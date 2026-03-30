from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, newton

from .config import TRADING_DAYS_PER_YEAR


def equity_curve(returns: pd.Series, base: float = 1.0) -> pd.Series:
    """
    Builds the equity curve from daily returns.
    The first day is forced to `base` (e.g., 1.0) instead of being NaN.
    """
    r = returns.copy()
    if r.empty:
        return r

    eq = pd.Series(np.nan, index=r.index)
    first_valid = r.first_valid_index()
    if first_valid is None:
        return eq

    r_valid = r.loc[first_valid:]
    compound = (1.0 + r_valid.fillna(0.0)).cumprod() * base
    compound.iloc[0] = base
    eq.loc[first_valid:] = compound
    return eq


def drawdown(equity: pd.Series) -> pd.Series:
    """Percentage drawdown from previous peak."""
    eq = equity.dropna()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd.reindex(equity.index)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown (negative number, e.g., -0.25 = -25%)."""
    return float(drawdown(equity).min())


def annualized_return(returns: pd.Series) -> float:
    """Geometric annualized return (CAGR)."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    r = r[r > -1.0]
    if r.empty:
        return np.nan
    total = float((1.0 + r).prod())
    years = len(r) / TRADING_DAYS_PER_YEAR
    if years <= 0 or total <= 0:
        return np.nan
    return float(total ** (1.0 / years) - 1.0)


def annualized_vol(returns: pd.Series) -> float:
    """Annualized volatility (standard deviation * sqrt(252))."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    """
    Annualized Sharpe Ratio.
    Uses ddof=1 (sample standard deviation), standard convention
    in practical implementations (e.g., pyfolio, QuantLib).
    """
    r = returns.dropna()
    if r.empty:
        return np.nan
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess = r - rf_daily
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    """
    Annualized Sortino Ratio.
    Similar to Sharpe but considers only downside volatility (downside deviation).
    More appropriate for asymmetric return distributions.
    """
    r = returns.dropna()
    if r.empty:
        return np.nan
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess = r - rf_daily
    downside = excess[excess < 0]
    if downside.empty:
        return np.nan
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def calmar_ratio(returns: pd.Series) -> float:
    """
    Calmar Ratio = annualized return / |max drawdown|.
    Measures how much return you achieve per unit of maximum drawdown endured.
    Values > 1 are generally considered good.
    """
    ann_ret = annualized_return(returns)
    eq = equity_curve(returns)
    mdd = max_drawdown(eq)
    if np.isnan(ann_ret) or np.isnan(mdd) or mdd == 0:
        return np.nan
    return float(ann_ret / abs(mdd))


def beta_alpha(port_returns: pd.Series, bench_returns: pd.Series) -> dict[str, float]:
    """
    OLS estimation of beta and alpha relative to the benchmark.
    Minimum 60 observations required for a stable estimate.
    Annualized alpha should be interpreted cautiously with daily data.
    """
    df = pd.concat(
        [port_returns.rename("p"), bench_returns.rename("b")], axis=1
    ).dropna()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 60:
        return {"beta": np.nan, "alpha_annual": np.nan, "r2": np.nan}
    cov = np.cov(df["p"], df["b"], ddof=1)[0, 1]
    var = np.var(df["b"], ddof=1)
    beta = cov / var if var != 0 else np.nan
    alpha_daily = df["p"].mean() - beta * df["b"].mean()
    alpha_annual = (1.0 + alpha_daily) ** TRADING_DAYS_PER_YEAR - 1.0
    corr = np.corrcoef(df["p"], df["b"])[0, 1]
    r2 = corr**2
    return {"beta": float(beta), "alpha_annual": float(alpha_annual), "r2": float(r2)}


def tracking_error(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """
    Annualized Tracking Error = volatility of active returns (port - bench).
    High = you are making decisions very different from the benchmark.
    """
    active = (port_returns - bench_returns).dropna()
    if active.empty:
        return np.nan
    return float(active.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def information_ratio(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """
    Information Ratio = annualized active return / Tracking Error.
    Values > 0.5 are considered good. Negative = underperformance vs benchmark.
    """
    active = (port_returns - bench_returns).dropna()
    if active.empty:
        return np.nan
    te = active.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if te == 0 or np.isnan(te):
        return np.nan
    return float(active.mean() * TRADING_DAYS_PER_YEAR / te)


# =============================================================================
# ROLLING METRICS
# =============================================================================
# Rolling metrics compute Sharpe, volatility, and beta over a moving window
# of N days (default 180 ≈ 9 months of trading).
#
# Instead of a single number for the entire historical period, you get a
# time series showing how the risk/return profile has evolved over time.
# This is crucial for:
#
#   - Identifying stress periods (e.g., Sharpe collapsing in 2022)
#   - Observing whether beta is increasing or decreasing over time
#   - Understanding if volatility regimes have structurally changed
#
# The first `window` days are NaN because there are not enough observations
# to fill the window.
# =============================================================================


def rolling_sharpe(
    returns: pd.Series,
    window: int = 180,
    risk_free_annual: float = 0.0,
) -> pd.Series:
    """
    Sharpe Ratio over a rolling window of `window` trading days.

    Each point answers: "what was the Sharpe over the last 180 days?"
    A decline in rolling Sharpe signals a period of risk-adjusted underperformance.

    Parameters:
        window: number of trading days in the window (default 180 ≈ 9 months)
        risk_free_annual: annual risk-free rate
    """
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess = returns - rf_daily

    roll_mean = excess.rolling(window=window, min_periods=window).mean()
    roll_std = excess.rolling(window=window, min_periods=window).std(ddof=1)

    sharpe = roll_mean / roll_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe.name = f"rolling_sharpe_{window}d"
    return sharpe


def rolling_vol(
    returns: pd.Series,
    window: int = 180,
) -> pd.Series:
    """
    Annualized volatility over a rolling window of `window` days.

    Each point answers: "how volatile was the portfolio over the last 180 days?"
    Useful for identifying high/low volatility regimes.

    Parameters:
        window: number of trading days in the window (default 180)
    """
    roll_std = returns.rolling(window=window, min_periods=window).std(ddof=1)
    vol = roll_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    vol.name = f"rolling_vol_{window}d"
    return vol


def rolling_beta(
    port_returns: pd.Series,
    bench_returns: pd.Series,
    window: int = 180,
) -> pd.Series:
    """
    Beta vs benchmark over a rolling window of `window` days.

    Each point answers: "how correlated was the portfolio with the market
    over the last 180 days?" Increasing beta indicates increasing market exposure;
    decreasing beta suggests the portfolio is moving more independently.

    Parameters:
        window: number of trading days in the window (default 180)
    """
    df = pd.concat(
        [port_returns.rename("p"), bench_returns.rename("b")], axis=1
    ).dropna()

    roll_cov = df["p"].rolling(window=window, min_periods=window).cov(df["b"])
    roll_var = df["b"].rolling(window=window, min_periods=window).var(ddof=1)

    beta = (roll_cov / roll_var).reindex(port_returns.index)
    beta.name = f"rolling_beta_{window}d"
    return beta


def rolling_metrics_table(
    port_returns: pd.Series,
    bench_returns: pd.Series,
    window: int = 180,
    risk_free_annual: float = 0.0,
) -> pd.DataFrame:
    """
    Computes all rolling metrics in a single DataFrame ready for CSV export and plotting.

    Returned columns:
        rolling_sharpe_Nd    : portfolio rolling Sharpe ratio
        rolling_vol_port_Nd  : portfolio rolling volatility
        rolling_vol_bench_Nd : benchmark rolling volatility (for comparison)
        rolling_beta_Nd      : portfolio rolling beta vs benchmark

    Parameters:
        port_returns   : daily portfolio returns
        bench_returns  : daily benchmark returns
        window         : rolling window length (default 180)
        risk_free_annual: annual risk-free rate
    """
    rs = rolling_sharpe(port_returns, window=window, risk_free_annual=risk_free_annual)

    rv_port = rolling_vol(port_returns, window=window)
    rv_port.name = f"rolling_vol_port_{window}d"

    rv_bench = rolling_vol(bench_returns, window=window)
    rv_bench.name = f"rolling_vol_bench_{window}d"

    rb = rolling_beta(port_returns, bench_returns, window=window)

    df = pd.concat([rs, rv_port, rv_bench, rb], axis=1)
    df.index.name = "Date"
    return df


# =============================================================================
# XIRR
# =============================================================================


def xnpv(rate: float, cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    """Net Present Value with irregular dates (365-day basis)."""
    t0 = cashflows[0][0]
    total = 0.0
    for t, cf in cashflows:
        days = (t - t0).days
        total += cf / ((1.0 + rate) ** (days / 365.0))
    return total


def xirr(cashflows: list[tuple[pd.Timestamp, float]], guess: float = 0.1) -> float:
    """
    XIRR: internal rate of return for irregular cash flows.

    First attempt uses Newton-Raphson (fast), fallback uses Brent (robust).
    If both fail, returns NaN.
    """
    cashflows = sorted(cashflows, key=lambda x: x[0])
    if len(cashflows) < 2:
        return np.nan

    def f(r: float) -> float:
        return xnpv(r, cashflows)

    try:
        result = float(newton(f, guess, maxiter=200))
        if -1.0 < result < 100.0:
            return result
    except Exception:
        pass

    try:
        return float(brentq(f, -0.999, 100.0, maxiter=500))
    except Exception:
        return np.nan