"""Tests for backtest metrics (verified against known values)."""

from __future__ import annotations

import math

import pandas as pd

from stocklab.backtest import metrics


def test_equity_curve_and_cagr():
    returns = pd.Series([0.1, 0.1])
    equity = metrics.equity_curve(returns)
    assert math.isclose(float(equity.iloc[-1]), 1.21, rel_tol=1e-9)
    # 2 periods = exactly 1 year -> CAGR equals the total growth
    assert math.isclose(metrics.cagr(equity, 1.0), 0.21, rel_tol=1e-9)


def test_max_drawdown_known_value():
    equity = pd.Series([1.0, 1.2, 0.9, 1.5])
    # peak 1.2 -> trough 0.9 gives -25%
    assert math.isclose(metrics.max_drawdown(equity), -0.25, rel_tol=1e-9)


def test_profit_factor_and_win_rate():
    returns = pd.Series([0.1, -0.05, 0.05, -0.0])
    # gains 0.15 / losses 0.05 = 3.0
    assert math.isclose(metrics.profit_factor(returns), 3.0, rel_tol=1e-9)
    assert math.isclose(metrics.win_rate(returns), 0.5, rel_tol=1e-9)


def test_sharpe_zero_variance_is_nan():
    returns = pd.Series([0.01, 0.01, 0.01])
    assert math.isnan(metrics.sharpe_ratio(returns, periods_per_year=12))


def test_summarize_keys_complete():
    idx = pd.date_range("2024-01-31", periods=6, freq="ME")
    returns = pd.Series([0.02, -0.01, 0.03, 0.0, 0.01, -0.02], index=idx)
    bench = pd.Series([0.01] * 6, index=idx)
    result = metrics.summarize(returns, bench, holding_days=21)
    assert set(metrics.METRIC_KEYS) <= set(result)
    assert result["n_periods"] == 6.0


def test_summarize_empty_returns_nan():
    result = metrics.summarize(pd.Series(dtype=float), pd.Series(dtype=float), 21)
    assert all(math.isnan(v) for v in result.values())

def test_profit_factor_no_losses_is_inf():
    assert metrics.profit_factor(pd.Series([0.1, 0.2])) == float("inf")
