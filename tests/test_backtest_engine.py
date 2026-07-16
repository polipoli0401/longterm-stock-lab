"""Tests for the backtest engine (trade generation and known returns)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stocklab.backtest.engine import BacktestEngine
from stocklab.config import BacktestConfig


def _prices(days: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    """AAA rises 1%/day, BBB falls 0.5%/day; benchmark is flat."""
    dates = pd.bdate_range("2023-01-02", periods=days)
    rows = []
    for ticker, rate in (("AAA", 0.01), ("BBB", -0.005)):
        close = 100 * np.cumprod(np.full(days, 1 + rate))
        for date, c in zip(dates, close):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": c,
                    "high": c,
                    "low": c,
                    "close": c,
                    "volume": 1e6,
                }
            )
    prices = pd.DataFrame(rows)
    bench = pd.Series(100.0, index=dates)
    return prices, bench


def _predictions(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Always predict AAA above BBB."""
    rows = []
    for date in dates:
        rows.append({"date": date, "ticker": "AAA", "y_pred": 1.0})
        rows.append({"date": date, "ticker": "BBB", "y_pred": -1.0})
    return pd.DataFrame(rows)


def test_top1_return_matches_known_value():
    prices, bench = _prices(days=40)
    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    engine = BacktestEngine(
        BacktestConfig(top_n=1, holding_days=10), prices, bench, filters=None
    )
    result = engine.run(_predictions(dates), model_name="test")
    # Every trade must select AAA with a 10-day return of 1.01**10 - 1.
    expected = 1.01**10 - 1
    assert (result.trades["tickers"] == "AAA").all()
    assert np.allclose(result.trades["return"].to_numpy(), expected, rtol=1e-9)
    # The flat benchmark must yield ~0 return.
    assert np.allclose(result.trades["benchmark_return"].to_numpy(), 0.0, atol=1e-12)
    assert result.metrics["excess_return"] > 0


def test_rotation_count():
    """30 days with a 5-day holding period -> entries occur every 5 days."""
    prices, bench = _prices(days=30)
    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    engine = BacktestEngine(
        BacktestConfig(top_n=1, holding_days=5), prices, bench, filters=None
    )
    result = engine.run(_predictions(dates), model_name="test")
    assert len(result.trades) == 5  # entries at days 0,5,10,15,20 (25 has no exit)
    for prev, cur in zip(result.trades.itertuples(), result.trades.iloc[1:].itertuples()):
        assert prev.exit_date == cur.entry_date


def test_filters_exclude_tickers():
    prices, bench = _prices(days=40)
    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    # Filter that always rejects AAA.
    filters = pd.DataFrame(
        [
            {"date": d, "ticker": t, "filter_pass": t != "AAA"}
            for d in dates
            for t in ("AAA", "BBB")
        ]
    )
    engine = BacktestEngine(
        BacktestConfig(top_n=1, holding_days=10), prices, bench, filters=filters
    )
    result = engine.run(_predictions(dates), model_name="test")
    assert (result.trades["tickers"] == "BBB").all()
    expected = 0.995**10 - 1
    assert math.isclose(result.trades["return"].iloc[0], expected, rel_tol=1e-9)


def test_too_short_period_raises():
    prices, bench = _prices(days=10)
    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    engine = BacktestEngine(
        BacktestConfig(top_n=1, holding_days=30), prices, bench, filters=None
    )
    with pytest.raises(ValueError):
        engine.run(_predictions(dates), model_name="test")
