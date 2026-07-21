"""Tests for JP/US mixed-universe currency handling."""

from __future__ import annotations

import math

import pandas as pd

from stocklab.currency import convert_prices_to_jpy, is_jpy, unit_for


def test_is_jpy_and_unit():
    assert is_jpy("7203.T") is True
    assert is_jpy("^N225") is True
    assert is_jpy("AAPL") is False
    assert is_jpy("BRK-B") is False
    assert unit_for("7203.T", 100) == 100
    assert unit_for("AAPL", 100) == 1


def test_convert_prices_to_jpy():
    dates = pd.to_datetime(["2024-01-04", "2024-01-05"])
    prices = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAPL", "AAPL", "7203.T", "7203.T"],
            "open": [10.0, 11.0, 3000.0, 3010.0],
            "high": [10.0, 11.0, 3000.0, 3010.0],
            "low": [10.0, 11.0, 3000.0, 3010.0],
            "close": [10.0, 11.0, 3000.0, 3010.0],
            "volume": [100.0, 100.0, 100.0, 100.0],
        }
    )
    fx = pd.Series([150.0, 160.0], index=dates)
    out = convert_prices_to_jpy(prices, fx, {"AAPL"})
    aapl = out[out["ticker"] == "AAPL"].sort_values("date")
    assert math.isclose(aapl["close"].iloc[0], 1500.0)
    assert math.isclose(aapl["close"].iloc[1], 1760.0)
    assert math.isclose(aapl["close_native"].iloc[1], 11.0)  # native preserved
    toyota = out[out["ticker"] == "7203.T"]
    assert (toyota["close"] == toyota["close_native"]).all()  # JP untouched
    assert (out["volume"] == 100.0).all()  # share counts never converted
