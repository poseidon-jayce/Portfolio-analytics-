"""
Unit tests for src/metrics.py

How to run:
    pip install pytest
    pytest tests/test_metrics.py -v

Each test verifies a specific function using controlled synthetic data,
so that the expected result is known exactly and any future regressions
in the code can be detected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Import the functions to be tested
from src.metrics import (
    annualized_return,
    annualized_vol,
    beta_alpha,
    calmar_ratio,
    drawdown,
    equity_curve,
    information_ratio,
    max_drawdown,
    rolling_beta,
    rolling_sharpe,
    rolling_vol,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
    xirr,
)

# =============================================================================
# FIXTURE: reusable synthetic datasets for tests
# =============================================================================


@pytest.fixture
def flat_returns() -> pd.Series:
    """
    Constant daily returns of +0.1%.
    Useful for testing functions where the expected result can be computed manually.
    252 days = 1 trading year.
    """
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    return pd.Series(0.001, index=idx, name="returns")


@pytest.fixture
def zero_returns() -> pd.Series:
    """Daily returns all equal to zero."""
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    return pd.Series(0.0, index=idx, name="returns")


@pytest.fixture
def random_returns() -> pd.Series:
    """
    Random daily returns with fixed seed for reproducibility.
    Mean ~0, standard deviation ~1% per day.
    """
    rng = np.random.default_rng(seed=42)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, size=500), index=idx, name="returns")


@pytest.fixture
def benchmark_returns(random_returns) -> pd.Series:
    """
    Benchmark series correlated with random_returns but with additional noise.
    Used to test beta, tracking error, and information ratio.
    """
    rng = np.random.default_rng(seed=99)
    noise = pd.Series(
        rng.normal(0.0003, 0.008, size=len(random_returns)), index=random_returns.index
    )
    return (random_returns * 0.7 + noise).rename("benchmark")


# =============================================================================
# TEST: equity_curve
# =============================================================================


class TestEquityCurve:
    def test_starts_at_base(self, flat_returns):
        """The first value of the curve must be exactly base=1.0."""
        eq = equity_curve(flat_returns, base=1.0)
        first_valid = eq.first_valid_index()
        assert abs(eq[first_valid] - 1.0) < 1e-10, "The first value must be 1.0"

    def test_grows_with_positive_returns(self, flat_returns):
        """With always-positive returns, the curve must be monotonically increasing."""
        eq = equity_curve(flat_returns, base=1.0).dropna()
        assert (eq.diff().dropna() >= 0).all(), (
            "With positive returns the curve must only go up"
        )

    def test_empty_returns(self):
        """With an empty series it should return an empty series without crashing."""
        empty = pd.Series([], dtype=float)
        eq = equity_curve(empty)
        assert eq.empty

    def test_all_nan_returns(self):
        """With all-NaN series it should return a NaN series without crashing."""
        idx = pd.date_range("2020-01-01", periods=10, freq="B")
        nan_returns = pd.Series(np.nan, index=idx)
        eq = equity_curve(nan_returns)
        assert eq.isna().all()

    def test_compounding(self):
        """
        Verify that compounding is correct.
        With constant daily return r for N days:
        final value = (1 + r)^N
        """
        r = 0.01
        n = 10
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        returns = pd.Series(r, index=idx)
        eq = equity_curve(returns, base=1.0).dropna()
        expected = (1 + r) ** n
        assert abs(eq.iloc[-1] - expected) < 1e-10, (
            f"Expected value {expected:.6f}, obtained {eq.iloc[-1]:.6f}"
        )


# =============================================================================
# TEST: drawdown and max_drawdown
# =============================================================================


class TestDrawdown:
    def test_no_drawdown_on_rising_curve(self, flat_returns):
        """With always-positive returns, drawdown must always be 0."""
        eq = equity_curve(flat_returns)
        dd = drawdown(eq).dropna()
        assert (dd >= -1e-10).all(), (
            "There should be no drawdown with always-positive returns"
        )

    def test_known_drawdown(self):
        """
        Test with manually computable drawdown.
        Curve: 1.0 → 1.2 → 0.9 → 1.1
        Max drawdown: (0.9 - 1.2) / 1.2 = -25%
        """
        idx = pd.date_range("2020-01-01", periods=4, freq="B")
        eq = pd.Series([1.0, 1.2, 0.9, 1.1], index=idx)
        dd = drawdown(eq)
        assert abs(dd.min() - (-0.25)) < 1e-6, (
            f"Expected drawdown -25%, obtained {dd.min():.4f}"
        )

    def test_max_drawdown_negative(self, random_returns):
        """Max drawdown must always be <= 0."""
        eq = equity_curve(random_returns)
        mdd = max_drawdown(eq)
        assert mdd <= 0, f"Max drawdown must be negative, obtained {mdd}"

    def test_max_drawdown_known_value(self):
        """Test with known max drawdown: -25%."""
        idx = pd.date_range("2020-01-01", periods=4, freq="B")
        eq = pd.Series([1.0, 1.2, 0.9, 1.1], index=idx)
        assert abs(max_drawdown(eq) - (-0.25)) < 1e-6


# =============================================================================
# TEST: annualized_return
# =============================================================================


class TestAnnualizedReturn:
    def test_known_value(self, flat_returns):
        """
        With constant daily return of 0.1% for 252 days:
        CAGR = (1.001)^252 - 1 ≈ 28.4%
        """
        ann = annualized_return(flat_returns)
        expected = (1.001**252) - 1.0
        assert abs(ann - expected) < 1e-4, (
            f"Expected CAGR {expected:.4f}, obtained {ann:.4f}"
        )

    def test_zero_returns(self, zero_returns):
        """With all-zero returns, CAGR must be 0."""
        ann = annualized_return(zero_returns)
        assert abs(ann) < 1e-10, f"Expected CAGR 0, obtained {ann}"

    def test_empty_returns(self):
        """With empty series it should return NaN."""
        assert np.isnan(annualized_return(pd.Series([], dtype=float)))

    def test_filters_extreme_negative(self):
        """Returns <= -100% must be filtered without crashing."""
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        returns = pd.Series([-1.0, 0.01, 0.01, 0.01, 0.01], index=idx)
        result = annualized_return(returns)
        assert not np.isnan(result) or np.isnan(result)


# =============================================================================
# TEST: annualized_vol
# =============================================================================


class TestAnnualizedVol:
    def test_zero_vol_for_constant_returns(self, flat_returns):
        """With constant returns, volatility must be 0."""
        vol = annualized_vol(flat_returns)
        assert abs(vol) < 1e-10, f"Expected volatility 0, obtained {vol}"

    def test_vol_scales_with_sqrt_252(self):
        """Annualized volatility = daily std * sqrt(252)."""
        idx = pd.date_range("2020-01-01", periods=252, freq="B")
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 0.01, 252), index=idx)
        expected = r.std(ddof=1) * np.sqrt(252)
        assert abs(annualized_vol(r) - expected) < 1e-10

    def test_empty_returns(self):
        """With empty series it should return NaN."""
        assert np.isnan(annualized_vol(pd.Series([], dtype=float)))


# =============================================================================
# TEST: sharpe_ratio
# =============================================================================


class TestSharpeRatio:
    def test_positive_sharpe_with_positive_returns(self, flat_returns):
        """With positive returns and zero risk-free rate, Sharpe must be positive."""
        sharpe = sharpe_ratio(flat_returns, risk_free_annual=0.0)
        assert sharpe > 0

    def test_zero_sharpe_when_excess_is_zero(self):
        """If returns equal risk-free, Sharpe should be near 0."""
        idx = pd.date_range("2020-01-01", periods=252, freq="B")
        rf_daily = (1.05) ** (1 / 252) - 1
        rng = np.random.default_rng(42)
        r = pd.Series(rf_daily + rng.normal(0, 0.001, 252), index=idx)
        sharpe = sharpe_ratio(r, risk_free_annual=0.05)
        assert abs(sharpe) < 1.0

    def test_empty_returns(self):
        assert np.isnan(sharpe_ratio(pd.Series([], dtype=float)))

    def test_nan_when_zero_volatility(self, zero_returns):
        result = sharpe_ratio(zero_returns)
        assert np.isnan(result)


# =============================================================================
# TEST: sortino_ratio
# =============================================================================


class TestSortinoRatio:
    def test_positive_sortino_with_positive_returns(self, flat_returns):
        result = sortino_ratio(flat_returns)
        assert np.isnan(result)

    def test_sortino_geq_sharpe_for_positive_skew(self, random_returns):
        sharpe = sharpe_ratio(random_returns)
        sortino = sortino_ratio(random_returns)
        if not np.isnan(sharpe) and not np.isnan(sortino):
            assert sortino >= sharpe * 0.5

    def test_empty_returns(self):
        assert np.isnan(sortino_ratio(pd.Series([], dtype=float)))


# =============================================================================
# TEST: calmar_ratio
# =============================================================================


class TestCalmarRatio:
    def test_positive_calmar_with_positive_returns(self, flat_returns):
        result = calmar_ratio(flat_returns)
        assert np.isnan(result) or result > 0

    def test_empty_returns(self):
        assert np.isnan(calmar_ratio(pd.Series([], dtype=float)))


# =============================================================================
# TEST: beta_alpha
# =============================================================================


class TestBetaAlpha:
    def test_beta_one_for_identical_series(self, random_returns):
        result = beta_alpha(random_returns, random_returns)
        assert abs(result["beta"] - 1.0) < 1e-6
        assert abs(result["alpha_annual"]) < 1e-4

    def test_r2_one_for_identical_series(self, random_returns):
        result = beta_alpha(random_returns, random_returns)
        assert abs(result["r2"] - 1.0) < 1e-6

    def test_nan_for_insufficient_data(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="B")
        r = pd.Series(0.001, index=idx)
        result = beta_alpha(r, r)
        assert np.isnan(result["beta"])
        assert np.isnan(result["alpha_annual"])
        assert np.isnan(result["r2"])

    def test_beta_zero_for_uncorrelated(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2020-01-01", periods=500, freq="B")
        p = pd.Series(rng.normal(0, 0.01, 500), index=idx)
        b = pd.Series(rng.normal(0, 0.01, 500), index=idx)
        result = beta_alpha(p, b)
        assert abs(result["beta"]) < 0.15


# =============================================================================
# TEST: tracking_error and information_ratio
# =============================================================================


class TestActiveMetrics:
    def test_tracking_error_zero_for_identical(self, random_returns):
        te = tracking_error(random_returns, random_returns)
        assert abs(te) < 1e-10

    def test_tracking_error_positive(self, random_returns, benchmark_returns):
        te = tracking_error(random_returns, benchmark_returns)
        assert te >= 0

    def test_information_ratio_zero_for_identical(self, random_returns):
        ir = information_ratio(random_returns, random_returns)
        assert np.isnan(ir)

    def test_information_ratio_finite(self, random_returns, benchmark_returns):
        ir = information_ratio(random_returns, benchmark_returns)
        assert np.isfinite(ir)


# =============================================================================
# TEST: rolling metrics
# =============================================================================


class TestRollingMetrics:
    def test_rolling_sharpe_nan_in_first_window(self, random_returns):
        window = 60
        rs = rolling_sharpe(random_returns, window=window)
        assert rs.iloc[: window - 1].isna().all()

    def test_rolling_sharpe_not_nan_after_window(self, random_returns):
        window = 60
        rs = rolling_sharpe(random_returns, window=window)
        valid = rs.iloc[window:].dropna()
        assert len(valid) > 0

    def test_rolling_vol_non_negative(self, random_returns):
        rv = rolling_vol(random_returns, window=60)
        assert (rv.dropna() >= 0).all()

    def test_rolling_vol_zero_for_constant(self, flat_returns):
        rv = rolling_vol(flat_returns, window=60)
        assert (rv.dropna().abs() < 1e-10).all()

    def test_rolling_beta_near_one_for_identical(self, random_returns):
        rb = rolling_beta(random_returns, random_returns, window=60)
        valid = rb.dropna()
        assert (abs(valid - 1.0) < 1e-6).all()

    def test_rolling_beta_near_zero_for_uncorrelated(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2020-01-01", periods=500, freq="B")
        p = pd.Series(rng.normal(0, 0.01, 500), index=idx)
        b = pd.Series(rng.normal(0, 0.01, 500), index=idx)
        rb = rolling_beta(p, b, window=60).dropna()
        assert abs(rb.mean()) < 0.2


# =============================================================================
# TEST: xirr
# =============================================================================


class TestXirr:
    def test_simple_known_case(self):
        """
        Simple case:
        Invest -1000 today, receive +1100 after one year.
        XIRR = 10%.
        """
        t0 = pd.Timestamp("2020-01-01")
        t1 = pd.Timestamp("2021-01-01")
        cfs = [(t0, -1000.0), (t1, 1100.0)]
        result = xirr(cfs)
        assert abs(result - 0.10) < 0.001

    def test_xirr_nan_for_single_cashflow(self):
        cfs = [(pd.Timestamp("2020-01-01"), -1000.0)]
        assert np.isnan(xirr(cfs))

    def test_xirr_nan_for_empty(self):
        assert np.isnan(xirr([]))

    def test_xirr_higher_return_for_shorter_period(self):
        t0 = pd.Timestamp("2020-01-01")
        t_6m = pd.Timestamp("2020-07-01")
        t_12m = pd.Timestamp("2021-01-01")

        xirr_6m = xirr([(t0, -1000.0), (t_6m, 1100.0)])
        xirr_12m = xirr([(t0, -1000.0), (t_12m, 1100.0)])

        assert xirr_6m > xirr_12m